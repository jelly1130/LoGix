export CUDA_VISIBLE_DEVICES=2

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/ETT-small \
  --pred_len $len \
  --data ETTm2 \
  --data_path ETTm2.csv \
  --seq_len 512 \
  --emb_dim 64 \
  --depth 2 \
  --batch_size 512 \
  --dropout 0.7 \
  --patch_size 64 \
  --train_epochs 100 \
  --pretrain_epochs 100 \
  --num_scales 2

done




