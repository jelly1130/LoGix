export CUDA_VISIBLE_DEVICES=0

for dataPath in Cricket  EthanolConcentration  Heartbeat  Libras  NATOPS  PEMS-SF PhonemeSpectra
do
  python -u LoGix_classification.py \
  --data_path ~/LoGix/dataset_split/Classification/UEA/$dataPath \
  --emb_dim 256 \
  --depth 3 \
  --train_epochs 100 \
  --pretrain_epochs 100 \
  --patch_size 8 \
  --train_lr 1e-3 \
  --pretrain_lr 1e-4 \
  --weight_decay 1e-6 \
  --num_scales 3 \
  --earlystop 0 \
  --model_id UEA_datasets 
done

for dataPath in sleepedf
do
  python -u LoGix_classification.py \
  --data_path ~/LoGix/dataset_split/Classification/$dataPath \
  --emb_dim 256 \
  --depth 2 \
  --train_epochs 100 \
  --pretrain_epochs 50 \
  --patch_size 8 \
  --train_lr 1e-4 \
  --pretrain_lr 1e-4 \
  --weight_decay 1e-3 \
  --num_scales 4 \
  --earlystop 0 \
  --model_id sleepedf
done

for dataPath in HAR
do
  python -u LoGix_classification.py \
  --data_path ~/LoGix/dataset_split/$dataPath \
  --emb_dim 512 \
  --depth 2 \
  --train_epochs 100 \
  --pretrain_epochs 50 \
  --patch_size 8 \
  --train_lr 1e-4 \
  --pretrain_lr 1e-4 \
  --weight_decay 1e-3 \
  --num_scales 6 \
  --earlystop 0 \
  --model_id HAR
done