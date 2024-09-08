# 内存占用情况： batch=8 : 8GB


python main_train.py \
--image_dir data/CT_RATE/ \
--ann_path data/CT_RATE/aligned_annotation.json \
--dataset_name CT_RATE \
--max_seq_length 60 \
--threshold 3 \
--batch_size 4 \
--epochs 100 \
--save_dir results/CT_RATE \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--n_gpu 2 \
--retry_interval 16 \
--min_batch_size 2 \
--max_batch_size 16