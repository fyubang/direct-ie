import os
import torch
import json
import logging
import re

from random import shuffle
from tqdm import tqdm
from collections import OrderedDict, defaultdict

from config import args
from bert import BertTokenizer
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

WHITESPACE_PLACEHOLDER = '□'

def pad_collate_fn(batch):
    # batch_size == len(batch)
    batch = [torch.stack(x) for x in list(zip(*batch))]
    token_ids = batch[0]
    token_types = batch[1]
    length = token_ids.gt(0).sum(dim=-1).max().item()
    token_ids = token_ids[:, :length]
    token_types = token_types[:, :length]
    if len(batch) == 4:
        heads = batch[2][:, :length]
        tails = batch[3][:, :length]
        return [token_ids, token_types, heads, tails]
    elif len(batch) == 3:
        label = batch[2]
        return [token_ids, token_types, label]
    elif len(batch) == 2:
        return [token_ids, token_types]
    else:
        raise Exception('wrong length of batch')

class BatchLoader(object):
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.predicates = self.load_predicates()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=False)

    def load_predicates(self):
        predicates = []
        with open(os.path.join(self.data_dir, 'predicates.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                p = line.strip()
                predicates.append(p)
        return predicates

    def load_data(self, dtype, n='all'):
        data = []
        with open(os.path.join(self.data_dir, '{}.json'.format(dtype)), 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                sample['id'] = len(data)
                sample['spos'] = [tuple(item) for item in sample['spos']]
                if n == 'all':
                    data.append(sample)
                else:
                    if n < 5:
                        if len(sample['spos']) == n:
                            data.append(sample)
                    else:
                        if len(sample['spos']) >= n:
                            data.append(sample)
        return data

    def spos2dict(self, spos):
        res = {}
        for spo in spos:
            sub, pre, obj = spo
            if sub in res:
                if obj in res[sub]:
                    res[sub][obj].append(pre)
                else:
                    res[sub][obj] = [pre]
            else:
                res[sub] = {obj: [pre]}
        return res

    def tokenize_train_data(self, data):
        s_data = []
        o_data = []
        p_data = []
        for sample in tqdm(data, desc='Tokenizing train data...'):
            text = sample['text']
            spos_dict = self.spos2dict(sample['spos'])
            subs = list(spos_dict.keys())
            tokens, token_ids, token_types,  heads, tails = self._get_extracive_samples(text, subs)
            if 1 not in heads or 1 not in tails:
                continue
            s_item = {'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types, 'heads': heads, 'tails': tails}
            s_data.append(s_item)
            for s in spos_dict:
                objs = list(spos_dict[s].keys())
                tokens, token_ids, token_types, heads, tails = self._get_extracive_samples(text, objs, s)
                o_item = {'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types, 'heads': heads, 'tails': tails, 's': s}
                o_data.append(o_item)
                for o, ps in spos_dict[s].items():
                    tokens, token_ids, token_types, label = self._get_classfied_samples(text, s, o, ps)
                    p_item = {'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types, 'label': label, 'so': (s, o)}
                    p_data.append(p_item)
        return s_data, o_data, p_data

    def tokenize_eval_data(self, data, type):
        res = []
        for idx, sample in enumerate(data):
            text = sample['text']
            idx = sample['id']
            if type == 'sub':
                tokens, token_ids, token_types, _, _ = self._get_extracive_samples(text, [])
                item = {'id': idx, 'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types}
                res.append(item)
            elif type == 'obj':
                subs = sample['subs']
                for sub in subs:
                    tokens, token_ids, token_types, _, _ = self._get_extracive_samples(text, [], sub)
                    item = {'id': idx, 'sub': sub, 'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types}
                    res.append(item)
            elif type == 'pre':
                sub_objs = sample['sub_objs']
                for sub, obj in sub_objs:
                    tokens, token_ids, token_types, _ = self._get_classfied_samples(text, sub, obj, [])
                    item = {'id': idx, 'sub_obj': (sub, obj), 'tokens': tokens, 'token_ids': token_ids, 'token_types': token_types}
                    res.append(item)
            else:
                raise Exception('type must be sub, obj or pre')
        return res

    def batch_loader(self, s_data, o_data, p_data, max_seq_len, batch_size=32):
        s_dataset = self._build_dataset(s_data, 'extract', max_seq_len)
        o_dataset = self._build_dataset(o_data, 'extract', max_seq_len)
        p_dataset = self._build_dataset(p_data, 'classify', max_seq_len)
        dataloaders = []
        for dataset in [s_dataset, o_dataset, p_dataset]:
            dataloaders.append(DataLoader(dataset, batch_size, sampler=RandomSampler(dataset), collate_fn=pad_collate_fn, drop_last=True))
        return dataloaders

    def eval_batch_loader(self, data, max_seq_len=256, batch_size=128):
        dataset = self._build_dataset(data, 'eval', max_seq_len)
        return DataLoader(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler(dataset), collate_fn=pad_collate_fn, drop_last=False)

    def _build_dataset(self, data, task_type, max_seq_len=64):
        dataset = None
        token_ids = torch.tensor(self._padding([item['token_ids'] for item in data], max_seq_len), dtype=torch.long)
        token_types = torch.tensor(self._padding([item['token_types'] for item in data], max_seq_len), dtype=torch.long)
        if task_type == 'extract':
            heads = torch.tensor(self._padding([item['heads'] for item in data], max_seq_len), dtype=torch.float)
            tails = torch.tensor(self._padding([item['tails'] for item in data], max_seq_len), dtype=torch.float)
            dataset = TensorDataset(token_ids, token_types, heads, tails)
        elif task_type == 'classify':
            label = torch.tensor([item['label'] for item in data], dtype=torch.float)
            dataset = TensorDataset(token_ids, token_types, label)
        elif task_type == 'eval':
            dataset = TensorDataset(token_ids, token_types)
        else:
            raise Exception('task type must be extract, classify, eval')
        return dataset

    def _get_token_ids(self, text, add_cls=False, add_sep=False):
        tokens = self.tokenizer.tokenize(text, inference=True)
        tokens = ['[CLS]'] + tokens if add_cls else tokens
        tokens = tokens + ['[SEP]'] if add_sep else tokens
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_ids

    def _get_classfied_samples(self, text, s, o, ps):
        o_tokens, o_token_ids = self._get_token_ids(o, True, True)
        s_tokens, s_token_ids = self._get_token_ids(s, False, True)
        tokens, token_ids = self._get_token_ids(text, False, True)
        tokens, token_ids, token_types = o_tokens + s_tokens + tokens, o_token_ids + s_token_ids + token_ids, [0]*(len(s_tokens+o_tokens)) + [0]*len(tokens)
        label = [0]*len(self.predicates)
        for p in ps:
            label[self.predicates.index(p)] = 1
        return tokens, token_ids, token_types, label

    def _get_extracive_samples(self, text, spans, sub=None):
        if sub is None:
            tokens, token_ids = self._get_token_ids(text, True, True)
            token_types = [0]*len(tokens)
        else:
            sub_tokens, sub_token_ids = self._get_token_ids(sub, True, True)
            tokens, token_ids = self._get_token_ids(text, False, True)
            tokens, token_ids, token_types = sub_tokens + tokens, sub_token_ids + token_ids, [0]*len(sub_tokens) + [0]*len(tokens)
        heads, tails = [0]*len(tokens), [0]*len(tokens)
        for span in spans:
            span_tokens, _ = self._get_token_ids(span)
            head, tail = self._get_head_tail(tokens, span_tokens)
            if head != -1 and tail != -1:
                heads[head] = tails[tail] = 1
            else:
                print('***No span found***')
                print(text)
                print(span)
        return tokens, token_ids, token_types, heads, tails

    def _get_head_tail(self, tokens, span_tokens):
        len_span = len(span_tokens)
        head, tail = -1, -1
        for i in range(len(tokens)-len_span+1):
            if tokens[i:i+len_span] == span_tokens:
                head, tail = i, i+len_span-1
                break
        return head, tail

    def _padding(self, data, max_seq_len, val=0):
        res = []
        for seq in data:
            if len(seq) > max_seq_len:
                res.append(seq[:max_seq_len])
            else:
                res.append(seq + [val]*(max_seq_len-len(seq)))
        return res


def get_spans(tokens, heads, tails):
    len_tokens = len(tokens)
    potential_heads = []
    for i in range(len_tokens):
        if heads[i] > 0.5:
            potential_heads.append(i)
    potential_heads.append(len_tokens)
    spans = []
    for i in range(len(potential_heads)-1):
        max_index, max_val = i, 0
        for j in range(potential_heads[i], potential_heads[i+1]):
            if tails[j] > max_val:
                max_index = j
                max_val = tails[j]
        span = ' '.join(tokens[potential_heads[i]: max_index+1]).replace(' ##', '') \
                                                                .replace(' - ', '-') \
                                                                .replace(' \' ', '\'') \
                                                                .replace(' . ', '.') \
                                                                .replace(' ¶ ', '¶') \
                                                                .replace(' ¡ ', '¡')
        spans.append(span)
    return spans