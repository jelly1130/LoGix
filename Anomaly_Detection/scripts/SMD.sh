

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model LoGix \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 256 \
  --d_ff 768 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.2 \
  --batch_size 128 \
  --patch_size 1 \
  --stride 1 \
  --depth 5 \
  --dropout 0.4 \
  --num_scales 5 \
  --train_epochs 10 \
  --learning_rate 0.00011
