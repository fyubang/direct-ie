export PYTHIONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=3
export TASK=NYT
python main.py \
    --do_eval \
    --data_dir data/partial/$TASK \
    --model_dir experiments/$TASK/baseline \
    --batch_size 128 \