import argparse
import os

# warmup_linear_constant
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/WebNLG', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='/home/admin/workspace/fzhao/PTMs/bert-base-cased', help="Directory containing the BERT model in PyTorch")

parser.add_argument('--clip_grad', default=2, help="")
parser.add_argument('--seed', type=int, default=20, help="random seed for initialization") # 8
parser.add_argument('--schedule', default='warmup_linear', help="schedule for optimizer")
parser.add_argument('--weight_decay', default=0.01, help="")
parser.add_argument('--warmup', default=0.1, help="")

parser.add_argument('--model_dir', default='experiments/debug', help="model directory")
parser.add_argument('--mtl_strategy', default='avg', help="strategy of multi-task learning")
parser.add_argument('--do_sampling', action='store_true', help="do sampling or adjust learning rate")
parser.add_argument('--epoch_num', default=5, type=int, help="num of epoch")
parser.add_argument('--batch_size', default=64, type=int, help="batch size")
parser.add_argument('--max_len', default=128, type=int, help="max sequence length")
parser.add_argument('--learning_rate', default=5e-5, type=float, help="learning rate")
parser.add_argument('--te', default=0.9, type=float, help="threshold of extraction")
parser.add_argument('--tc', default=0.5, type=float, help="threshold of classification")
parser.add_argument('--ema_decay', default=0.99, type=float, help="ema decay")

parser.add_argument('--do_train_and_eval', action='store_true', help="do_train_and_eval")
parser.add_argument('--do_eval', action='store_true', help="do_eval")
parser.add_argument('--do_predict', action='store_true', help="do predict")

args = parser.parse_args()

