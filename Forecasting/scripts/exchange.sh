export CUDA_VISIBLE_DEVICES=1

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/exchange_rate \
  --pred_len $len \
  --data custom \
  --data_path exchange_rate.csv \
  --seq_len 512 \
  --emb_dim 512 \
  --depth 3 \
  --batch_size 40 \
  --dropout 0.7 \
  --patch_size 64 \
  --train_epochs 50 \
  --pretrain_epochs 50 \
  --num_scales 4 

  done



