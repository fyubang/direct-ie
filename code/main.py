import os
import torch
import random
import logging
import pickle
import json

from tqdm import tqdm, trange
from random import shuffle
from time import time
from torch.cuda.amp import autocast
import numpy as np

import utils
from model import Direct
from batch_loader import BatchLoader
from bert.optimization import BertAdam
from config import args

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)

def compute_corrects(pred_spos, gold_spos):
    pred_spos_set = set(pred_spos)
    gold_spos_set = set(gold_spos)
    return len(pred_spos_set), len(gold_spos_set), len(pred_spos_set.intersection(gold_spos_set))

def compute_metrics(num_pred, num_recall, num_correct):
    precision = num_correct / float(num_pred)
    recall = num_correct / float(num_recall)
    f1 = 2*precision*recall/ (precision+recall)
    return precision, recall, f1

def load_train_data(bl, args, dtype='train'):
    if not os.path.isfile(os.path.join(args.data_dir, 'train.pkl')):
        train_data = bl.load_data(dtype)
        s_train_data, o_train_data, p_train_data = bl.tokenize_train_data(train_data)
        with open(os.path.join(args.data_dir, 'train.pkl'), 'wb') as f:
            train_pkl = {'s': s_train_data, 'o': o_train_data, 'p': p_train_data}
            pickle.dump(train_pkl, f)
    else:
        with open(os.path.join(args.data_dir, 'train.pkl'), 'rb') as f:
            train_pkl = pickle.load(f)
            s_train_data, o_train_data, p_train_data = train_pkl['s'], train_pkl['o'], train_pkl['p']
    return s_train_data, o_train_data, p_train_data

def extract(model, bl, data_type, args, batch_size, te=0.5, tc=0.5, n='all'):
    model.eval()
    with torch.no_grad():
        data = bl.load_data(data_type, n)
        # subs extraction
        s_data = bl.tokenize_eval_data(data, 'sub')
        eval_bl = bl.eval_batch_loader(s_data, batch_size=batch_size)
        subs = []
        for i, batch in enumerate(eval_bl):
            batch_tokens = [tmp['tokens'] for tmp in s_data[i*batch_size: (i+1)*batch_size]]
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_subs = model.get_res(token_ids, token_types, batch_tokens, 'sub', te, tc)
            subs += batch_subs
        assert len(subs) == len(data) == len(s_data), (data[:3], s_data[:3], subs[:3])
        for i in range(len(s_data)):
            data[i]['subs'] = subs[i]
        # objs extraction
        o_data = bl.tokenize_eval_data(data, 'obj')
        eval_bl = bl.eval_batch_loader(o_data, batch_size=batch_size)
        objs = []
        for i, batch in enumerate(eval_bl):
            batch_tokens = [tmp['tokens'] for tmp in o_data[i*batch_size: (i+1)*batch_size]]
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_objs = model.get_res(token_ids, token_types, batch_tokens, 'obj', te, tc)
            objs += batch_objs
        assert len(objs) == len(o_data), (o_data[:3], objs[:3])
        for sample in data:
            sample['sub_objs'] = []
        for i in range(len(o_data)):
            sub = o_data[i]['sub']
            idx = o_data[i]['id']
            data[idx]['sub_objs'] += [(sub, obj) for obj in objs[i]]
        # predicates classification
        p_data = bl.tokenize_eval_data(data, 'pre')
        eval_bl = bl.eval_batch_loader(p_data, batch_size=batch_size)
        pres = []
        for i, batch in enumerate(eval_bl):
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_pres = model.get_res(token_ids, token_types, None, 'pre', te, tc)
            pres += batch_pres
        assert len(pres) == len(p_data), (p_data[:3], pres[:3])
        for sample in data:
            sample['pred_spos'] = []
        for i in range(len(p_data)):
            sub, obj = p_data[i]['sub_obj']
            idx = p_data[i]['id']
            data[idx]['pred_spos'] += [(sub, pre, obj) for pre in pres[i]]
    return data

def eval(model, bl, data_type, args, epoch, batch_size=128, te=0.5, tc=0.5, only_bad_cases=False, n='all'):
    pred_data = extract(model, bl, data_type, args, batch_size, te, tc, n)
    with open(os.path.join(args.model_dir, 'epoch_{}'.format(epoch)), 'w', encoding='utf-8') as f:
        num_pred, num_recall, num_correct = 1e-10, 1e-10, 1e-10
        for sample in pred_data:
            gold_spos = sorted(sample['spos'])
            pred_spos = sorted(sample['pred_spos'])
            a, b, c = compute_corrects(pred_spos, gold_spos)
            if not only_bad_cases or not (a == b == c):
                f.write(sample['text']+'\n')
                f.write(str(gold_spos)+'\n')
                f.write(str(pred_spos)+'\n\n')
            num_pred += a
            num_recall += b
            num_correct += c
        precision, recall, f1 = compute_metrics(num_pred, num_recall, num_correct)
        logging.info('{}: Precision: {:5.4f}, Recall: {:5.4f}, F1: {:5.4f}'.format(data_type, precision, recall, f1))
    return f1

def eval_each_task(model, bl, data_type, args, batch_size, te=0.5, tc=0.5):
    model.eval()
    with torch.no_grad():
        data = bl.load_data(data_type)
        # subs extraction
        s_data = bl.tokenize_eval_data(data, 'sub')
        eval_bl = bl.eval_batch_loader(s_data, batch_size=batch_size)
        subs = []
        for i, batch in tqdm(enumerate(eval_bl), desc='sub task'):
            batch_tokens = [tmp['tokens'] for tmp in s_data[i*batch_size: (i+1)*batch_size]]
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_subs = model.get_res(token_ids, token_types, batch_tokens, 'sub', te, tc)
            subs += batch_subs
        assert len(subs) == len(data) == len(s_data), (data[:3], s_data[:3], subs[:3])
        for i in range(len(s_data)):
            data[i]['pred_subs'] = subs[i]
        for i in range(len(s_data)):
            data[i]['subs'] = list(set([spo[0] for spo in data[i]['spos']]))
        pn = rn = cn = 1e-4
        for sample in data:
            pn += len(sample['pred_subs'])
            rn += len(sample['subs'])
            cn += len(list(set(sample['pred_subs']).intersection(set(sample['subs']))))
        logging.info("SUB TASK--pre: {}, recall: {}, f1: {}".format(cn/pn, cn/rn, 2*cn/(pn+rn)))
        # objs extraction
        o_data = bl.tokenize_eval_data(data, 'obj')
        eval_bl = bl.eval_batch_loader(o_data, batch_size=batch_size)
        objs = []
        for i, batch in tqdm(enumerate(eval_bl), desc='obj task'):
            batch_tokens = [tmp['tokens'] for tmp in o_data[i*batch_size: (i+1)*batch_size]]
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_objs = model.get_res(token_ids, token_types, batch_tokens, 'obj', te, tc)
            objs += batch_objs
        assert len(objs) == len(o_data), (o_data[:3], objs[:3])
        for sample in data:
            sample['pred_sub_objs'] = []
        for i in range(len(o_data)):
            sub = o_data[i]['sub']
            idx = o_data[i]['id']
            data[idx]['pred_sub_objs'] += [(sub, obj) for obj in objs[i]]
        for sample in data:
            sample['sub_objs'] = list(set([(spo[0], spo[2]) for spo in sample['spos']]))
        pn = rn = cn = 1e-4
        for sample in data:
            pn += len(sample['pred_sub_objs'])
            rn += len(sample['sub_objs'])
            cn += len(list(set(sample['pred_sub_objs']).intersection(set(sample['sub_objs']))))
        logging.info("OBJ TASK--pre: {}, recall: {}, f1: {}".format(cn/pn, cn/rn, 2*cn/(pn+rn)))
        # predicates classification
        p_data = bl.tokenize_eval_data(data, 'pre')
        eval_bl = bl.eval_batch_loader(p_data, batch_size=batch_size)
        pres = []
        for i, batch in tqdm(enumerate(eval_bl), desc='pre task'):
            token_ids, token_types = tuple(tmp.to(args.device) for tmp in batch)
            batch_pres = model.get_res(token_ids, token_types, None, 'pre', te, tc)
            pres += batch_pres
        assert len(pres) == len(p_data), (p_data[:3], pres[:3])
        for sample in data:
            sample['pred_spos'] = []
        for i in range(len(p_data)):
            sub, obj = p_data[i]['sub_obj']
            idx = p_data[i]['id']
            data[idx]['pred_spos'] += [(sub, pre, obj) for pre in pres[i]]
        pn = rn = cn = 1e-4
        for sample in data:
            pn += len(sample['pred_spos'])
            rn += len(sample['spos'])
            cn += len(list(set(sample['pred_spos']).intersection(set(sample['spos']))))
        logging.info("PRE TASK--pre: {}, recall: {}, f1: {}".format(cn/pn, cn/rn, 2*cn/(pn+rn)))
    return data

if __name__ == '__main__':
    # Use GPUs if available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('device: {}'.format(args.device))
    logging.info('Hyper params:%r'%args.__dict__)
    # Create the input data pipeline
    logging.info('Loading the datasets...')
    bl = BatchLoader(args)
    # Model
    model = Direct.from_pretrained(args.bert_model_dir, predicates=bl.predicates, args=args)
    model.to(args.device)

    train_dtype = 'train'
    valid_dtype = 'valid'
    test_dtype = 'test'

    if args.do_train_and_eval:
        ## Train data
        s_train_data, o_train_data, p_train_data = load_train_data(bl, args, train_dtype)
        train_bls = bl.batch_loader(s_train_data, o_train_data, p_train_data, args.max_len, args.batch_size)
        num_batchs_per_task = [len(train_bl) for train_bl in train_bls]
        logging.info('num of batch per task for train: {}'.format(num_batchs_per_task))
        train_task_ids = sum([[i]*num_batchs_per_task[i] for i in range(3)], [])
        # Optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'names': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'names': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                'weight_decay_rate': 0.0}
        ]
        args.epoch_steps = sum(num_batchs_per_task)
        args.total_steps = args.epoch_steps * args.epoch_num
        optimizer = BertAdam(params=optimizer_grouped_parameters, 
                            lr=args.learning_rate, 
                            warmup=args.warmup, 
                            t_total=args.total_steps, 
                            max_grad_norm=args.clip_grad, 
                            schedule=args.schedule,
                            layer_num=model.layer_num)
        # Train and evaluate
        best_f1 = 0

        for epoch in range(args.epoch_num):
            ## Train
            model.train()
            t = trange(args.epoch_steps, desc='Epoch {} -Train'.format(epoch))
            avg_loss = utils.RunningAverage()
            train_iters = [iter(tmp) for tmp in train_bls] # to use next and reset the iterator
            shuffle(train_task_ids)
            scales = [1, 1, 1]
            tasks_avg_loss = [utils.RunningEMA() for _ in range(len(train_bls))]
            for step in t:
                if epoch == 0 and step < 100:
                    task_id = train_task_ids[step]
                else:
                    if args.do_sampling:
                        task_id = np.random.choice([0,1,2], p=[tmp/3 for tmp in scales])
                    else:
                        task_id = train_task_ids[step]
                try:
                    batch_data = next(train_iters[task_id])
                except StopIteration:
                    train_iters[task_id] = iter(train_bls[task_id])
                    batch_data = next(train_iters[task_id])
                batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                with autocast():
                    loss = model(batch_data, task_id)
                avg_loss.update(loss.item())
                tasks_avg_loss[task_id].update(loss.item())
                if args.mtl_strategy == 'norm':
                    scales = [1, 1, 1]
                elif args.mtl_strategy == 'avg_sum_loss_lr':
                    scales = [a()/b for a, b in zip(model.avg_metrics, num_batchs_per_task)]
                    scales = [3/sum(scales)*tmp for tmp in scales]
                    scales = [tmp / scales[2] for tmp in scales]
                else:
                    raise Exception('wrong mtl strategy')
                if not args.do_sampling:
                    loss = scales[task_id]*loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t.set_postfix(loss='{:5.4f}'.format(loss.item()), \
                    avg_loss='{:5.4f}'.format(avg_loss()), \
                    scales='{:5.2f}, {:5.2f}, {:5.2f}'.format(*[tmp for tmp in scales]), \
                    tasks_avg_loss='{:5.4f}, {:5.4f}, {:5.4f}'.format(*[tmp() for tmp in tasks_avg_loss]))
            ## Evaluate
            f1 = eval(model, bl, valid_dtype, args, epoch, 128)
            eval(model, bl, test_dtype, args, epoch, 128)
            ## Save weights of the network
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict()},
                                is_best=f1>best_f1,
                                checkpoint=args.model_dir)
            best_f1 = max(f1, best_f1)
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
        bl = BatchLoader(args)
        eval(model, bl, valid_dtype, args, 'valid-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True)
        eval(model, bl, test_dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True)
        for i in range(1, 6):
            eval(model, bl, test_dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True, n=i)
        for dtype in ['test_normal', 'test_epo', 'test_seo']:
            eval(model, bl, dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True)

    if args.do_eval:
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
        eval_each_task(model, bl, test_dtype, args, 128, te=args.te, tc=args.tc)
        eval(model, bl, test_dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True)
        for i in range(1, 6):
            logging.info('N={}'.format(i))
            eval(model, bl, test_dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True, n=i)
        for dtype in ['test_normal', 'test_epo', 'test_seo']:
            eval(model, bl, dtype, args, 'test-badcases', 128, te=args.te, tc=args.tc, only_bad_cases=True)