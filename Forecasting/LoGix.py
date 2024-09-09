import argparse
import datetime
import os

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError


from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



class LocalMixer(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        
        num_scales=args.num_scales
        
        self.convs = nn.ModuleList()
        kernel_sizes = [2**i - 1 for i in range(1, num_scales+1)]
        paddings = [(k - 1) // 2 for k in kernel_sizes]
        
        for k, p in zip(kernel_sizes, paddings):
            self.convs.append(nn.Conv1d(in_features, hidden_features, k, 1, padding=p))
        
        self.weights = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in range(2*(num_scales-1))])
        
        self.mixer = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.num_scales = num_scales


    def forward(self, x):
       
        x = x.transpose(1, 2)
        activations = [self.drop(self.act(conv(x))) for conv in self.convs]
        
        high2low = 0
        low2high = 0
        
        for i in range(self.num_scales - 1):
            high2low += self.weights[i] * self.convs[i](x) * activations[i + 1]
            low2high += self.weights[self.num_scales - 1 + i] * self.convs[-(i + 1)](x) * activations[-(i + 2)]
        
        y = self.mixer(high2low + low2high)
        y = y.transpose(1, 2)
        return y


class GlobalMixer(nn.Module):
    def __init__(self, num_patch, dim):
        super().__init__()
        
        if num_patch % 2 == 0:
            fft_len = int(num_patch / 2 + 1)
        else:
            fft_len = int((num_patch + 1) / 2)
        filter_width = fft_len // 2
        center_patch = fft_len // 2
        
        self.complex_weight_low = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_mid = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_all = nn.Parameter(torch.randn(fft_len, dim, 2, dtype=torch.float32) * 0.02)
        

        with torch.no_grad():
            self.mask_low = torch.zeros(fft_len, dim, device=device)
            self.mask_low[0:filter_width, :] = 1

            self.mask_mid = torch.zeros(fft_len, dim, device=device)
            self.mask_mid[(center_patch - filter_width // 2):(center_patch + filter_width // 2), :] = 1

            self.mask_high = torch.zeros(fft_len, dim, device=device)
            self.mask_high[fft_len - filter_width:fft_len, :] = 1
        
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight_mid, std=.02)
        trunc_normal_(self.complex_weight_low, std=.02)
        trunc_normal_(self.complex_weight_all, std=.02)
    
    def forward(self, x_in):
        B, N, C = x_in.shape
        
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        
        weight_low = torch.view_as_complex(self.complex_weight_low) * self.mask_low
        x_low_filtered = x_fft * weight_low
        weight_mid = torch.view_as_complex(self.complex_weight_mid) * self.mask_mid
        x_mid_filtered = x_fft * weight_mid
        weight_high = torch.view_as_complex(self.complex_weight_high) * self.mask_high
        x_high_filtered = x_fft * weight_high
        weight_all = torch.view_as_complex(self.complex_weight_all)
        x_all_filtered = x_fft * weight_all
        
        x = torch.fft.irfft(x_low_filtered + x_mid_filtered + x_high_filtered + x_all_filtered, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x

class LoGix_layer(L.LightningModule):
    def __init__(self, dim, num_patches, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        self.local_mixer = LocalMixer(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.global_mixer = GlobalMixer(num_patches, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.global_mixer(y) + self.local_mixer(y)
        
        y = x + self.drop_path(y)
        return y

class LoGix(nn.Module):
    def __init__(self):
        super(LoGix, self).__init__()

        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)
        channels = train_data.__getitem__(0)[0].shape[1]

        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.mixer_blocks = nn.ModuleList([
            LoGix_layer(dim=args.emb_dim, num_patches=num_patches, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # Parameters/Embeddings
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

    def pretrain(self, x_in):
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch]
        xb_mask = self.input_layer(xb_mask)

        for mixer_blk in self.mixer_blocks:
            xb_mask = mixer_blk(xb_mask)

        return xb_mask, self.input_layer(x_patched)

    def forward(self, x):
        B, L, M = x.shape # batch, length, feature dimension

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = LoGix()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        _, _, C = batch_x.shape
        batch_x = batch_x.float().to(device)

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = LoGix()
        self.criterion = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.preds = []
        self.trues = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,verbose=True),
            'monitor': 'val_mse',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)
        loss = self.criterion(outputs, batch_y)

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        mse = self.mse(pred.contiguous(), true.contiguous())
        mae = self.mae(pred, true)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._calculate_loss(batch, mode="test")
        self.preds.append(preds)
        self.trues.append(trues)
        return {'test_loss': loss, 'pred': preds, 'true': trues}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mse = self.mse(preds.contiguous(), trues.contiguous())
        mae = self.mae(preds, trues)
        print(f"{mae, mse}")


def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    print(f'train model from {pretrained_model_path}')
    early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=args.train_epochs,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    # Load the best checkpoint after training

    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    mse_result = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
    mae_result = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

    return model, mse_result, mae_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data args...
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='data/ETT-small',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting lengths
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=42)

    # model
    parser.add_argument('--emb_dim', type=int, default=64, help='dimension of model')
    parser.add_argument('--depth', type=int, default=3, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout value')
    parser.add_argument('--patch_size', type=int, default=64, help='size of patches')
    parser.add_argument('--mask_ratio', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--num_scales', type=int, default=4, help='num_scales')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(0))

    torch.set_float32_matmul_precision('medium')

    # load from checkpoint
    run_description = f"{args.data_path.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
    run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}_num_scales{args.num_scales}"
    run_description += f"_lr_{args.lr}_weight_decay_{args.weight_decay}_preTr_{args.load_from_pretrained}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_mse',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    num_channels = train_data.__getitem__(0)[0].shape[1]

    print("Dataset loaded ...")
    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, mse_result, mae_result = train_model(best_model_path)
    print("MSE results", mse_result)
    print("MAE  results", mae_result)

    # Save results into an Excel sheet ...
    df = pd.DataFrame({
        'MSE': mse_result,
        'MAE': mae_result
    })
    df.to_excel(os.path.join(CHECKPOINT_PATH, f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    # Append results into a text file ...
    os.makedirs("textOutput", exist_ok=True)
    f = open(f"textOutput/Mixer_{os.path.basename(args.data_path)}.txt", 'a')
    f.write(run_description + "  \n")
    f.write('MSE:{}, MAE:{}'.format(mse_result, mae_result))
    f.write('\n')
    f.write('\n')
    f.close()
    
    torch.cuda.empty_cache()
