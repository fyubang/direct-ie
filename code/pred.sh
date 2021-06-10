export PYTHIONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
export TASK=NYT
python main.py \
    --do_pred \
    --data_dir data/partial/$TASK \
    --model_dir experiments/$TASK/024 \
    --batch_size 128 \
    --max_len 128 \