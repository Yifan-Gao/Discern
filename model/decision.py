import os
import shutil
import torch
import logging
import importlib
import itertools
import numpy as np
import json
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from argparse import Namespace
from model.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.init import xavier_uniform_
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']


class Module(nn.Module):

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.epoch = 0
        self.dropout = nn.Dropout(self.args.dropout)

        # Entailment Tracking
        # roberta_model_path = '/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base'
        roberta_model_path = args.pretrained_lm_path
        roberta_config = RobertaConfig.from_pretrained(roberta_model_path, cache_dir=None)
        self.roberta = RobertaModel.from_pretrained(roberta_model_path, cache_dir=None, config=roberta_config)
        encoder_layer = TransformerEncoderLayer(self.args.bert_hidden_size, 12, 4 * self.args.bert_hidden_size)
        encoder_norm = nn.LayerNorm(self.args.bert_hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layer, args.trans_layer, encoder_norm)
        self._reset_transformer_parameters()
        self.w_entail = nn.Linear(self.args.bert_hidden_size, 3, bias=True)

        # Logic Reasoning
        self.entail_emb = nn.Parameter(torch.rand(3, self.args.bert_hidden_size))
        nn.init.normal_(self.entail_emb)

        self.w_selfattn = nn.Linear(self.args.bert_hidden_size*2, 1, bias=True)
        self.w_output = nn.Linear(self.args.bert_hidden_size*2, 4, bias=True)

    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                xavier_uniform_(param)

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('model.{}'.format(name)).Module

    @classmethod
    def load(cls, fname, override_args=None):
        load = torch.load(fname, map_location=lambda storage, loc: storage)
        args = vars(load['args'])
        if override_args:
            args.update(override_args)
        args = Namespace(**args)
        model = cls.load_module(args.model)(args)
        model.load_state_dict(load['state'])
        return model

    def save(self, metrics, dsave, early_stop):
        files = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.pt') and f != 'best.pt']
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(files) > self.args.keep-1:
            for f in files[self.args.keep-1:]:
                os.remove(f)

        fsave = os.path.join(dsave, 'step{}-{}.pt'.format(metrics['global_step'], metrics[early_stop]))
        torch.save({
            'args': self.args,
            'state': self.state_dict(),  # comment to save space
            'metrics': metrics,
        }, fsave)
        fbest = os.path.join(dsave, 'best.pt')
        if os.path.isfile(fbest):
            os.remove(fbest)
        shutil.copy(fsave, fbest)

    def create_input_tensors(self, batch):
        feat = {
            k: torch.stack([e['entail'][k] for e in batch], dim=0).to(self.device) for k in ['input_ids', 'input_mask']
        }
        return feat

    def logic_op(self, input, input_mask):
        selfattn_unmask = self.w_selfattn(self.dropout(input))
        selfattn_unmask.masked_fill_(~input_mask, -float('inf'))
        selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        selfattn = torch.sum(selfattn_weight * input, dim=1)
        score = self.w_output(self.dropout(selfattn))
        return score

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['roberta_enc'] = roberta_enc = self.roberta(input_ids=out['input_ids'], attention_mask=out['input_mask'])[0]

        # transformer encoding
        tenc_input, tenc_mask = [], []
        rule_mask = []
        for idx, e in enumerate(batch):
            tenc_idx = torch.cat([e['entail']['rule_idx'], e['entail']['user_idx']], dim=-1).to(self.device)
            tenc_input.append(torch.index_select(roberta_enc[idx], 0, tenc_idx))
            tenc_mask.append(torch.tensor([False] * tenc_idx.shape[0], dtype=torch.uint8))
            rule_mask.append(torch.tensor([1] * e['entail']['rule_idx'].shape[0], dtype=torch.uint8))
        if self.args.trans_layer > 0:
            tenc_input_padded = torch.nn.utils.rnn.pad_sequence(tenc_input).to(self.device)  # [seqlen, N, dmodel]
            tenc_mask_padded = torch.nn.utils.rnn.pad_sequence(tenc_mask, batch_first=True, padding_value=True).to(self.device)
            tenc_out = self.transformer_encoder(tenc_input_padded, src_key_padding_mask=tenc_mask_padded)
            tenc_out_t = torch.transpose(tenc_out, 0, 1).contiguous()  # [bz * seqlen * dim]
        else:
            tenc_out_t = torch.nn.utils.rnn.pad_sequence(tenc_input, batch_first=True).to(self.device)

        # predict entailment score
        entail_score_mask = nn.utils.rnn.pad_sequence(rule_mask, batch_first=True).unsqueeze(-1).to(self.device)  # [bz * seqlen * 1]
        max_num_rule_in_batch = entail_score_mask.shape[1]
        out['entail_score_pred'] = self.w_entail(self.dropout(tenc_out_t[:, :max_num_rule_in_batch, :]))  # [bz * seqlen * 3]
        entail_state = torch.matmul(out['entail_score_pred'], self.entail_emb)

        cat_state = torch.cat([entail_state, tenc_out_t[:, :max_num_rule_in_batch, :]], dim=-1)
        out['clf_score'] = self.logic_op(cat_state, entail_score_mask)  # [bz, 4]
        return out

    def extract_preds(self, out, batch):
        preds = []
        clf = out['clf_score'].max(1)[1].tolist()
        for idx, ex in enumerate(batch):
            clf_i = clf[idx]
            a = DECISION_CLASSES[clf_i]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'pred_answer': a,
                'pred_answer_cls': clf_i,
            })
            if 'entailment_score_gold_ce' in ex['entail']:
                gold_entail = ex['entail']['entailment_score_gold_ce'].tolist()
                pred_entail = out['entail_score_pred'][idx, :, :].max(1)[1][:ex['entail']['entailment_score_gold_ce'].shape[0]].tolist()
                preds[-1]['pred_entail'] = pred_entail
                preds[-1]['gold_entail'] = gold_entail

        return preds

    def compute_loss(self, out, batch):
        gclf = torch.tensor([ex['logic']['answer_class'] for ex in batch], device=self.device, dtype=torch.long)
        gentail = torch.nn.utils.rnn.pad_sequence([ex['entail']['entailment_score_gold_ce'] for ex in batch], padding_value=-1, batch_first=True).to(self.device).view(-1)
        loss = {
            'clf': F.cross_entropy(out['clf_score'], gclf),
            'entail': F.cross_entropy(out['entail_score_pred'].view(-1, 3), gentail, ignore_index=-1) * self.args.loss_entail_weight,
        }
        return loss

    def compute_metrics(self, predictions, data):
        from sklearn.metrics import accuracy_score, confusion_matrix
        metrics = {}
        preds = [pred['pred_answer_cls'] for pred in predictions]
        golds = [gold['logic']['answer_class'] for gold in data]
        micro_accuracy = accuracy_score(golds, preds)
        metrics["0c_micro_accuracy"] = float("{0:.2f}".format(micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        conf_mat = confusion_matrix(golds, preds, labels=[0, 1, 2, 3])
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat.shape[0])])
        metrics["0b_macro_accuracy"] = float("{0:.2f}".format(macro_accuracy * 100))  # int(100 * macro_accuracy) / 100
        metrics["0a_combined"] = float("{0:.2f}".format(macro_accuracy * micro_accuracy * 100))
        metrics["0d_confmat"] = conf_mat.tolist()
        # entailment tracking
        entail_preds = [i for ex in predictions for i in ex['pred_entail']]
        entail_golds = [i for ex in predictions for i in ex['gold_entail']]
        entail_micro_accuracy = accuracy_score(entail_golds, entail_preds)
        metrics["1b_entail_micro_accuracy"] = float("{0:.2f}".format(entail_micro_accuracy * 100))  # int(100 * micro_accuracy) / 100
        entail_conf_mat = confusion_matrix(entail_golds, entail_preds, labels=[0, 1, 2])
        entail_conf_mat_norm = entail_conf_mat.astype('float') / entail_conf_mat.sum(axis=1)[:, np.newaxis]
        entail_macro_accuracy = np.mean([entail_conf_mat_norm[i][i] for i in range(entail_conf_mat.shape[0])])
        metrics["1a_entail_macro_accuracy"] = float("{0:.2f}".format(entail_macro_accuracy * 100))  # int(100 * macro_accuracy) / 100

        return metrics

    def run_pred(self, dev):
        preds = []
        self.eval()
        for i in trange(0, len(dev), self.args.dev_batch, desc='batch', disable=self.args.tqdm_bar):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            preds += self.extract_preds(out, batch)
        return preds

    def run_train(self, train, dev):
        if not os.path.isdir(self.args.dsave):
            os.makedirs(self.args.dsave)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args.dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)

        num_train_steps = int(len(train) / self.args.train_batch * self.args.epoch)
        num_warmup_steps = int(self.args.warmup * num_train_steps)

        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)  # PyTorch scheduler

        print('num_train', len(train))
        print('num_dev', len(dev))

        global_step = 0
        best_metrics = {self.args.early_stop: -float('inf')}
        for epoch in trange(self.args.epoch, desc='epoch',):
            self.epoch = epoch
            train = train[:]
            np.random.shuffle(train)

            train_stats = defaultdict(list)
            preds = []
            self.train()
            for i in trange(0, len(train), self.args.train_batch, desc='batch'):
                actual_train_batch = int(self.args.train_batch / self.args.gradient_accumulation_steps)
                batch_stats = defaultdict(list)
                batch = train[i: i + self.args.train_batch]

                for accu_i in range(0, len(batch), actual_train_batch):
                    actual_batch = batch[accu_i : accu_i + actual_train_batch]
                    out = self(actual_batch)
                    pred = self.extract_preds(out, actual_batch)
                    loss = self.compute_loss(out, actual_batch)

                    for k, v in loss.items():
                        loss[k] = v / self.args.gradient_accumulation_steps
                        batch_stats[k].append(v.item()/ self.args.gradient_accumulation_steps)
                    sum(loss.values()).backward()
                    preds += pred

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                for k in batch_stats.keys():
                    train_stats['loss_' + k].append(sum(batch_stats[k]))

                if global_step % self.args.eval_every_steps == 0:
                    dev_stats = defaultdict(list)
                    dev_preds = self.run_pred(dev)
                    dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
                    dev_metrics.update(self.compute_metrics(dev_preds, dev))
                    metrics = {'global_step': global_step}
                    metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
                    logger.critical(pformat(metrics))

                    if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                        logger.critical('Found new best! Saving to ' + self.args.dsave)
                        best_metrics = metrics
                        self.save(best_metrics, self.args.dsave, self.args.early_stop)
                        with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                            json.dump(dev_preds, f, indent=2)
                        with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                            json.dump(best_metrics, f, indent=2)

                    self.train()

            train_metrics = {k: sum(v) / len(v) for k, v in train_stats.items()}
            train_metrics.update(self.compute_metrics(preds, train))

            dev_stats = defaultdict(list)
            dev_preds = self.run_pred(dev)
            dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
            dev_metrics.update(self.compute_metrics(dev_preds, dev))
            metrics = {'global_step': global_step}
            metrics.update({'train_' + k: v for k, v in train_metrics.items()})
            metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
            logger.critical(pformat(metrics))

            if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                logger.critical('Found new best! Saving to ' + self.args.dsave)
                best_metrics = metrics
                self.save(best_metrics, self.args.dsave, self.args.early_stop)
                with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                    json.dump(dev_preds, f, indent=2)
                with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                    json.dump(best_metrics, f, indent=2)

        logger.critical('Best dev')
        logger.critical(pformat(best_metrics))
