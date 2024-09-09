export CUDA_VISIBLE_DEVICES=1

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/electricity \
  --pred_len $len \
  --data custom \
  --data_path electricity.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 2 \
  --batch_size 16 \
  --dropout 0.7 \
  --patch_size 32 \
  --train_epochs 50 \
  --pretrain_epochs 50 \
  --num_scales 5
  
  done

