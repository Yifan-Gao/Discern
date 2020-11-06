import os
import shutil
import torch
import logging
import importlib
import itertools
import collections
import numpy as np
import json
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from argparse import Namespace
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from tempfile import NamedTemporaryFile
from sklearn.metrics import accuracy_score, confusion_matrix
DECISION_THREE_CLASSES = ['yes', 'no', 'more']





def compute_metrics(preds, data):
    import evaluator
    with NamedTemporaryFile('w') as fp, NamedTemporaryFile('w') as fg:
        json.dump(preds, fp)
        fp.flush()
        json.dump([{'utterance_id': e['utterance_id'], 'answer': e['answer']} for e in data], fg)
        fg.flush()
        results = evaluator.evaluate(fg.name, fp.name, mode='follow_ups')
        # results['combined'] = results['macro_accuracy'] * results['bleu_4']
        return results


class Module(nn.Module):

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.epoch = 0
        self.dropout = nn.Dropout(self.args.dropout)

        # Entailment Tracking
        roberta_model_path = args.pretrained_lm_path
        roberta_config = RobertaConfig.from_pretrained(roberta_model_path, cache_dir=None)
        self.roberta = RobertaModel.from_pretrained(roberta_model_path, cache_dir=None, config=roberta_config)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path, cache_dir=None)

        self.top_k = args.span_top_k
        self.span_scorer = nn.Linear(self.args.bert_hidden_size, 2, bias=True)

    def roberta_decode(self, doc):
        decoded = self.tokenizer.decode(doc, clean_up_tokenization_spaces=False).strip('\n').strip()
        return decoded

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

        fsave = os.path.join(dsave, 'step{}.pt'.format(metrics['global_step']))
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
            k: torch.stack([e['entail'][k] for e in batch], dim=0).to(self.device) for k in ['input_ids', 'input_mask', 'span_pointer_mask']
        }
        return feat

    def get_top_k(self, probs, k, sent_start, sent_end):
        p = list(enumerate(probs.tolist()))[sent_start : sent_end]
        p.sort(key=lambda tup: tup[1], reverse=True)
        return p[:k]

    def mask_scores(self, scores, mask):
        invalid = 1 - mask
        scores -= invalid.unsqueeze(2).expand_as(scores).float().mul(1e20)
        return scores

    def extract_preds(self, out, batch):
        preds = []
        scores = out['span_scores']
        ystart, yend = scores.split(1, dim=-1)
        pstart = F.softmax(ystart.squeeze(-1), dim=1)
        pend = F.softmax(yend.squeeze(-1), dim=1)

        for idx, (pstart_i, pend_i, ex, pointer_mask_i) in enumerate(zip(pstart, pend, batch, out['span_pointer_mask'])):
            pred_i = {'utterance_id': ex['utterance_id']}
            # start inclusive, end exclusive
            sentence_start = ex['entail']['rule_idx'].to(self.device) + 1  # +1 to skip the cls token, inclusive
            sentence_end = torch.cat([ex['entail']['rule_idx'][1:].to(self.device), (pointer_mask_i == 1).nonzero()[-1] + 1])  # exclusive
            top_preds = []
            for jdx, (sent_s, sent_e) in enumerate(zip(sentence_start, sentence_end)):
                top_start = self.get_top_k(pstart_i, self.top_k, sent_s, sent_e)
                top_end = self.get_top_k(pend_i, self.top_k, sent_s, sent_e)
                for s, ps in top_start:
                    for e, pe in top_end:
                        if e >= s:
                            top_preds.append((jdx, s, e, ps * pe))
            top_preds = sorted(top_preds, key=lambda tup: tup[-1], reverse=True)[:self.top_k]
            top_answers = [(self.roberta_decode(ex['entail']['inp'][s:e + 1]), j, s, e, p) for j, s, e, p in top_preds]
            top_ans, top_j, top_s, top_e, top_p = top_answers[0]
            pred_i['answer_span_start'] = top_s
            pred_i['answer_span_end'] = top_e
            pred_i['answer'] = top_ans
            pred_i['answer_sent_jdx'] = top_j
            preds.append(pred_i)
        return preds

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['roberta_enc'] = roberta_enc = self.roberta(input_ids=out['input_ids'], attention_mask=out['input_mask'])[0]
        span_scores = self.span_scorer(self.dropout(roberta_enc))
        out['span_scores'] = self.mask_scores(span_scores, out['span_pointer_mask'])
        return out

    def compute_f1(self, span_gold, span_pred):
        gold_toks = list(range(span_gold['s'], span_gold['e']+1))
        pred_toks = list(range(span_pred['s'], span_pred['e']+1))
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return 0
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_metrics(self, preds, data):
        metrics = compute_metrics(preds, data)
        span_f1 = []
        span_preds = {ex['utterance_id']: {'s': ex['answer_span_start'], 'e': ex['answer_span_end']} for ex in preds}
        span_golds = {ex['utterance_id']: {'s': ex['entail']['answer_span_start'], 'e': ex['entail']['answer_span_end']} for ex in data}
        for id in span_golds.keys():
            span_pred = span_preds[id]
            span_gold = span_golds[id]
            span_f1.append(self.compute_f1(span_gold, span_pred))
        metrics['span_f1'] = float("{0:.2f}".format(sum(span_f1)/len(span_f1) * 100))
        metrics['0_combined'] = float("{0:.2f}".format(metrics['bleu_1'] * metrics['bleu_4']/100))

        # compute sentence selction accuracy
        sent_preds = [ex['answer_sent_jdx'] for ex in preds]
        sent_golds = [ex['entail']['selected_edu_idx'] for ex in data]
        micro_accuracy = accuracy_score(sent_golds, sent_preds)
        metrics['span_sent_acc'] = float("{0:.2f}".format(micro_accuracy*100))
        return metrics

    def compute_loss(self, out, batch):
        loss = {}
        scores = out['span_scores']
        ystart, yend = scores.split(1, dim=-1)
        gstart = torch.tensor([e['entail']['answer_span_start'] for e in batch], dtype=torch.long, device=self.device)
        loss['span_start'] = F.cross_entropy(ystart.squeeze(-1), gstart, ignore_index=-1)
        gend = torch.tensor([e['entail']['answer_span_end'] for e in batch], dtype=torch.long, device=self.device)
        loss['span_end'] = F.cross_entropy(yend.squeeze(-1), gend, ignore_index=-1)
        loss['span_start'] *= self.args.loss_span_weight
        loss['span_end'] *= self.args.loss_span_weight

        return loss

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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)  # PyTorch scheduler

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
