export CUDA_VISIBLE_DEVICES=3

for len in 96 192 336 720
do
  python -u LoGix.py \
  --root_path ~/all_datasets/Solar \
  --pred_len $len \
  --data Solar \
  --data_path solar_AL.txt \
  --seq_len 512 \
  --emb_dim 128 \
  --depth 4 \
  --batch_size 16 \
  --dropout 0.7 \
  --patch_size 32 \
  --train_epochs 50 \
  --pretrain_epochs 20 \
  --num_scales 5
done


