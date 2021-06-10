import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, ModuleList

from bert import BertPreTrainedModel, BertModel
from utils import RunningAverage, RunningEMA

from batch_loader import WHITESPACE_PLACEHOLDER

class Direct(BertPreTrainedModel):
    def __init__(self, config, predicates, args):
        super(Direct, self).__init__(config)
        self.layer_num = config.num_hidden_layers
        self.predicates = predicates
        self.args = args
        self.bert = BertModel(config)
        self.s_layer = nn.Linear(config.hidden_size, 2) # head, tail, bce
        self.o_layer = nn.Linear(config.hidden_size, 2) # head, tail, bce
        self.p_layer = nn.Linear(config.hidden_size, len(predicates)) # label, bce
        self.bce_loss = BCEWithLogitsLoss(reduction='none')
        self.avg_metrics = [RunningEMA(args.ema_decay) for _ in range(3)]
        self.apply(self.init_bert_weights)

    def forward(self, batch_data, task_id):
        if task_id in [0, 1]: # task is a, o
            token_ids, token_types, heads, tails = batch_data
            attention_mask = token_ids.gt(0)
            lengths = attention_mask.sum(-1)
            sequence_output, _ = self.bert(token_ids, token_types, attention_mask, output_all_encoded_layers=False)
            if task_id == 0: # task a
                head_logits, tail_logits = self.s_layer(sequence_output).split(1, dim=-1)
            else:
                head_logits, tail_logits = self.o_layer(sequence_output).split(1, dim=-1)
            head_logits = head_logits.squeeze(dim=-1)
            tail_logits = tail_logits.squeeze(dim=-1)
            a = self.bce_loss(head_logits, heads)
            b = self.bce_loss(tail_logits, tails)
            h_loss = ((a * attention_mask.float()).sum(-1) / lengths).mean()
            t_loss = ((b * attention_mask.float()).sum(-1) / lengths).mean()
            loss = 2*h_loss + t_loss
            self.avg_metrics[task_id].update(((a * attention_mask.float()).sum()+(b * attention_mask.float()).sum()).item())
        else: # task is p
            token_ids, token_types, label = batch_data
            attention_mask = token_ids.gt(0)
            _, pooled_output = self.bert(token_ids, token_types, attention_mask, output_all_encoded_layers=False)
            p_logits = self.p_layer(pooled_output)
            c = self.bce_loss(p_logits, label)
            loss = c.mean()
            self.avg_metrics[task_id].update(c.sum().item())
        return loss

    def get_res(self, token_ids, token_types, batch_tokens, task_type, te, tc):
        res = []
        attention_mask = token_ids.gt(0)
        lengths = attention_mask.sum(-1).cpu().tolist()
        sequence_output, pooled_output = self.bert(token_ids, token_types, attention_mask, False)
        if task_type in ['sub', 'obj']:
            extract_layer = self.s_layer if task_type == 'sub' else self.o_layer
            head_logits, tail_logits = extract_layer(sequence_output).split(1, dim=-1) # (batch_size, seq_len, 2)
            head_logits = (head_logits.squeeze(-1).sigmoid() * attention_mask.float()).cpu().tolist() # (batch_size, seq_len)
            tail_logits = (tail_logits.squeeze(-1).sigmoid() * attention_mask.float()).cpu().tolist() # (batch_size, seq_len)
            for length, heads, tails, tokens in zip(lengths, head_logits, tail_logits, batch_tokens):
                _, mentions = self._get_spans(length, tokens, heads, tails, te)
                res.append(mentions)
        else:
            labels = self.p_layer(pooled_output).sigmoid().cpu().tolist() # (batch_size, seq_len)
            for label in labels:
                pres = []
                for i in range(len(label)):
                    if label[i] > tc:
                        pre = self.predicates[i]
                        pres.append(pre)
                res.append(pres)
        return res

    def _get_spans(self, length, tokens, heads, tails, threshold=0.5):
        potential_heads = []
        for i in range(length):
            if heads[i] > threshold:
                potential_heads.append(i)
        potential_heads.append(length)
        spans = []
        mentions = []
        for i in range(len(potential_heads)-1):
            max_index, max_val = i, 0
            for j in range(potential_heads[i], potential_heads[i+1]):
                if tails[j] > max_val:
                    max_index = j
                    max_val = tails[j]
            mention = self._get_mention(tokens[potential_heads[i]: max_index+1])
            spans.append((potential_heads[i], max_index+1))
            mentions.append(mention)
        return spans, mentions

    def _get_mention(self, tokens):
        mention = ' '.join(tokens).replace(' ##', '') \
                                .replace('##', '')
        for special_token in ['-', '\'', '.', '¶', '¡', '‰']:
            mention = mention.replace(' '+special_token, special_token).replace(special_token+' ', special_token)
        mention = mention.split(' ')[0] # partial match
        return mention
