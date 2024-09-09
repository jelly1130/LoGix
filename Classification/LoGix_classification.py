import argparse
import datetime
import os

import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy
from torchmetrics.classification import MulticlassF1Score
from dataloader import get_datasets
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


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
    def __init__(self, dim):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)  * 0.5)
        
    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x

class GlobalMixerNG(nn.Module):
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

class LoGixMixer_layer(L.LightningModule):
    def __init__(self, dim, num_patches, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        self.local_mixer = LocalMixer(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.global_mixer = GlobalMixerNG(num_patches, dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.norm(x)
        y = self.global_mixer(y) + self.local_mixer(y)
        
        y = x + self.drop_path(y)
        return y

class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

class LoGixMixer(nn.Module):
    def __init__(self):
        super(LoGixMixer, self).__init__()
        self.patch_size = args.patch_size
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches
        channels = train_loader.dataset.x_data.shape[1]

        # for classification  
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)

        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.mixer_blocks = nn.ModuleList([
            LoGixMixer_layer(dim=args.emb_dim, num_patches=num_patches, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )
        
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)
        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool() 
        for mixer_blk in self.mixer_blocks:
            xb_mask = mixer_blk(xb_mask)

        return xb_mask, x_patched


    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for mixer_blk in self.mixer_blocks:
            x = mixer_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = LoGixMixer()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4) # hyclassifier 这个原来classifier是1e-4，prediciton是1e-6
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        data = data.to(device)
        preds, target = self.model.pretrain(data)

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
        self.model = LoGixMixer()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


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
    if args.earlystop == 1:
        early_stopping_callback = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        trainer = L.Trainer(
            default_root_dir=CHECKPOINT_PATH,
            accelerator="auto",
            num_sanity_val_steps=0,
            devices=1,
            max_epochs=args.train_epochs,
            callbacks=[
                checkpoint_callback,
                # early_stopping_callback,
                LearningRateMonitor("epoch"),
                TQDMProgressBar(refresh_rate=500)
            ],
    )
    else:
        trainer=L.Trainer(
            default_root_dir=CHECKPOINT_PATH,
            accelerator="auto",
            num_sanity_val_steps=0,
            devices=1,
            max_epochs=args.train_epochs,
            callbacks=[
                checkpoint_callback,
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
    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, acc_result, f1_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')

    # Training parameters:
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
    parser.add_argument('--num_scales', type=int, default=4, help='num_scales')
    parser.add_argument('--earlystop', type=int, default=0, help='if earlystop')
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(0))

    torch.set_float32_matmul_precision('medium')
    
    run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}___"
    run_description += f"_train_epochs_{args.train_epochs}_pretrain_epochs_{args.pretrain_epochs}_patch_size_{args.patch_size}"
    run_description += f"_train_lr_{args.train_lr}_pretrain_lr_{args.pretrain_lr}_weight_decay_{args.weight_decay}_num_scales_{args.num_scales}_earlystop{args.earlystop}"

    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
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
        monitor='val_loss',          
        mode='min'
    )

    save_copy_of_files(pretrain_checkpoint_callback)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets
    train_loader, val_loader, test_loader = get_datasets(args.data_path, args)
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    print("Dataset loaded ...")
    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file
    text_save_dir = "textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"LoGix_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n')
    f.write('\n')
    f.close()

    torch.cuda.empty_cache()
