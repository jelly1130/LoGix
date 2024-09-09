export CUDA_VISIBLE_DEVICES=2

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/weather \
  --pred_len $len \
  --data custom \
  --data_path weather.csv \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 3 \
  --batch_size 64 \
  --dropout 0.7 \
  --patch_size 32 \
  --train_epochs 50 \
  --pretrain_epochs 20 \
  --num_scales 4
done


