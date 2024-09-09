export CUDA_VISIBLE_DEVICES=3
# for len in 96
for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/traffic \
  --pred_len $len \
  --data custom \
  --data_path traffic.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 4 \
  --batch_size 8 \
  --dropout 0.7 \
  --patch_size 32 \
  --train_epochs 50 \
  --pretrain_epochs 20 \
  --num_scales 6
  
done



