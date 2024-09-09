export CUDA_VISIBLE_DEVICES=0

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/ETT-small \
  --pred_len $len \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 1 \
  --batch_size 512 \
  --dropout 0.7 \
  --patch_size 32 \
  --train_epochs 100 \
  --pretrain_epochs 100 \
  --lr 1e-3 \
  --weight_decay 1e-3 \
  --num_scales 2
done

