import torch
import random
import numpy as np
from argparse import ArgumentParser
from model.decision import Module
from pprint import pprint
import pathlib
import os
import json

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_batch', default=10, type=int, help='training batch size')
    parser.add_argument('--dev_batch', default=5, type=int, help='dev batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--keep', default=1, type=int, help='number of model saves to keep')
    parser.add_argument('--seed', default=3, type=int, help='random seed')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.35, type=float, help='dropout rate')
    parser.add_argument('--warmup', default=0.1, type=float, help='optimizer warmup')
    parser.add_argument('--loss_span_weight', default=1., type=float, help='span loss weight')
    parser.add_argument('--span_top_k', default=20, type=int, help='extract top k spans')
    parser.add_argument('--loss_entail_weight', default=1., type=float, help='entailment loss weight')
    parser.add_argument('--debug', action='store_true', help='debug flag to load less data')
    parser.add_argument('--dsave', default='saved_models/{}', help='save directory')
    parser.add_argument('--model', default='base_argpred', help='model to use')
    parser.add_argument('--pretrained_lm_path', default='./pretrained_models/roberta_base/', help='path/to/pretrained/lm')
    parser.add_argument('--early_stop', default='dev_macro_accuracy', help='early stopping metric')
    parser.add_argument('--bert_hidden_size', default=768, type=int, help='hidden size for the bert model')
    parser.add_argument('--data', default='sharc', help='directory for data')
    parser.add_argument('--data_type', default='bu', help='data type, b:base bert, l: large bert, u: uncased, c: cased')
    parser.add_argument('--prefix', default='debug', help='prefix for experiment name')
    parser.add_argument('--resume', default='', help='model .pt file')
    parser.add_argument('--test', action='store_true', help='only run evaluation')
    parser.add_argument('--tqdm_bar', default=True, help='disable tqdm progress bar')
    parser.add_argument('--bert_model', default='base', help='bert model')
    parser.add_argument('--trans_layer', default=2, type=int, help='num of layers for transformer encoder/decoder')
    parser.add_argument('--eval_every_steps', default=50, type=int, help='evaluate model every xxx steps')

    args = parser.parse_args()
    args.dsave = args.dsave.format(args.prefix)
    pathlib.Path(args.dsave).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device('cuda')

    if args.debug:
        limit = 27
        data = {k: torch.load('{}/proc_{}_{}.pt'.format(args.data, args.data_type, 'dev'))[:limit] for k in ['train', 'dev']}
    else:
        limit = None
        data = {k: torch.load('{}/proc_{}_{}.pt'.format(args.data, args.data_type, k))[:limit] for k in ['dev', 'train']}

    if args.resume:
        print('resuming model from ' + args.resume)
        model = Module.load(args.resume, {'model': args.model, 'pretrained_lm_path': args.pretrained_lm_path})
    else:
        print('instanting model')
        model = Module.load_module(args.model)(args, device)

    model.device = device
    model.to(model.device)

    if args.test:
        preds = model.run_pred(data['dev'])
        print('saving {}'.format(os.path.join(args.dsave, 'dev.preds.json')))
        with open(os.path.join(args.dsave, 'dev.preds.json'), 'wt') as f:
            json.dump(preds, f, indent=2)
        metrics = model.compute_metrics(preds, data['dev'])
        pprint(metrics)
    else:
        model.run_train(data['train'], data['dev'])
