export PYTHIONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
export TASK=NYT

python main.py \
    --bert_model_dir bert-base-cased \
    --do_train_and_eval \
    --data_dir ../data/partial/$TASK \
    --epoch_num 15 \
    --model_dir experiments/$TASK/baseline \
    --batch_size 32 \
    --max_len 128 \
    --learning_rate 8e-5 \
    --mtl_strategy avg_sum_loss_lr \
    --ema_decay 0.99