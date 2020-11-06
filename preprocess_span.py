#!/usr/bin/env python
import os
import re
import torch
import string
import spacy
import json
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from pprint import pprint
import editdistance
from transformers import RobertaTokenizer


MATCH_IGNORE = {'do', 'did', 'does',
                'is', 'are', 'was', 'were', 'have', 'will', 'would',
                '?',}
PUNCT_WORDS = set(string.punctuation)
IGNORE_WORDS = MATCH_IGNORE | PUNCT_WORDS
MAX_LEN = 350
FILENAME = 'roberta_base'
FORCE=False
MODEL_FILE = './pretrained_models/roberta_base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_FILE, cache_dir=None)
DECISION_THREE_CLASSES = ['yes', 'no', 'more']
ENTAILMENT_CLASSES = ['yes', 'no', 'unknown']


def roberta_encode(doc):
    encoded = tokenizer.encode(doc.strip('\n').strip(), add_prefix_space=True, add_special_tokens=False)
    return encoded


def roberta_decode(doc):
    decoded = tokenizer.decode(doc, clean_up_tokenization_spaces=False).strip('\n').strip()
    return decoded


def filter_token(text):
    filtered_text = []
    for token_id in text:
        if roberta_decode(token_id).lower() not in MATCH_IGNORE:
            filtered_text.append(token_id)
    return roberta_decode(filtered_text)


def get_span(context, answer):
    answer = filter_token(answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_token(context[i:j+1])
            if '\n' in chunk or '*' in chunk:  # do not extract span across sentences/bullets
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j-i < best[1]-best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    if best:
        s, e = best
        while (not roberta_decode(context[s]).strip() or roberta_decode(context[s]) in PUNCT_WORDS) and s < e:
            s += 1
        while (not roberta_decode(context[e]).strip() or roberta_decode(context[e]) in PUNCT_WORDS) and s < e:
            e -= 1
        return s, e, best_score
    else:
        return -1, -1, best_score


def merge_edus(edus):
    # v2. merge edu with its beforehand one except
    # 1) this edu is not starting with 'if', 'and', 'or', 'to', 'unless', or
    # 2) its beforehand edu is end with ',', '.', ':'
    special_toks = ['if ', 'and ', 'or ', 'to ', 'unless ', 'but ', 'as ', 'except ']
    special_puncts = ['.', ':', ',',]
    spt_idx = []
    for idx, edu in enumerate(edus):
        if idx == 0:
            continue
        is_endwith = False
        for special_punct in special_puncts:
            if edus[idx-1].strip().endswith(special_punct):
                is_endwith = True
        is_startwith = False
        for special_tok in special_toks:
            if edu.startswith(special_tok):
                is_startwith = True
        if (not is_endwith) and (not is_startwith):
            spt_idx.append(idx)
    edus_spt = []
    for idx, edu in enumerate(edus):
        if idx not in spt_idx or idx == 0:
            edus_spt.append(edu)
        else:
            edus_spt[-1] += ' ' + edu
    return edus_spt


def _extract_edus(all_edus, title_tokenized, sentences_tokenized):
    # return a nested tokenized edus, with (start, end) index for each edu
    edus_span = []  # for all sentences
    edus_tokenized = []
    # add title
    if all_edus['title'].strip('\n').strip() != '':
        edus_tokenized.append([title_tokenized])
        edus_span.append([(0,len(title_tokenized)-1)])

    if all_edus['is_bullet']:
        for sentence_tokenized in sentences_tokenized:
            edus_tokenized.append([sentence_tokenized])
            edus_span.append([(0, len(sentence_tokenized) - 1)])
    else:
        edus_filtered = []
        for edus in all_edus['edus']:
            merged_edus = merge_edus(edus)
            edus_filtered.append(merged_edus)

        # print('debug')
        for idx_sentence in range(len(sentences_tokenized)):
            edus_span_i = []  # for i-th sentence
            edus_tokenized_i = []
            current_edus = edus_filtered[idx_sentence]
            current_sentence_tokenized = sentences_tokenized[idx_sentence]

            p_start, p_end = 0, 0
            for edu in current_edus:
                edu = edu.strip('\n').strip().replace(' ', '').lower()
                # handle exception case
                if ('``' in edu) and ('\'\'' in edu):
                    edu = edu.replace('``', '"').replace('\'\'', '"')
                for p_sent in range(p_start, len(current_sentence_tokenized)):
                    sent_span = roberta_decode(current_sentence_tokenized[p_start:p_sent+1]).replace(' ', '').lower()
                    if edu == sent_span:
                        p_end = p_sent
                        edus_span_i.append((p_start, p_end))  # [span_s,span_e]
                        edus_tokenized_i.append(current_sentence_tokenized[p_start:p_end + 1])
                        p_start = p_end + 1
                        break
            assert len(current_edus) == len(edus_tokenized_i) == len(edus_span_i)
            assert p_end == len(current_sentence_tokenized) - 1
            edus_span.append(edus_span_i)  # [sent_idx, ]
            edus_tokenized.append(edus_tokenized_i)
    assert len(edus_span) == len(edus_tokenized) == len(sentences_tokenized) + int(title_tokenized != None)

    return edus_tokenized, edus_span


def extract_edus(data_raw, all_edus):
    assert data_raw['snippet'] == all_edus['snippet']

    output = {}

    # 1. tokenize all sentences
    if all_edus['title'].strip('\n').strip() != '':
        title_tokenized = roberta_encode(all_edus['title'])
    else:
        title_tokenized = None
    sentences_tokenized = [roberta_encode(s) for s in all_edus['clauses']]
    output['q_t'] = {k: roberta_encode(k) for k in data_raw['questions']}
    output['scenario_t'] = {k: roberta_encode(k) for k in data_raw['scenarios']}
    output['initial_question_t'] = {k: roberta_encode(k) for k in data_raw['initial_questions']}
    output['snippet_t'] = roberta_encode(data_raw['snippet'])
    output['clause_t'] = [title_tokenized] + sentences_tokenized if all_edus['title'].strip('\n').strip() != '' else sentences_tokenized
    output['edu_t'], output['edu_span'] = _extract_edus(all_edus, title_tokenized, sentences_tokenized)

    # 2. map question to edu
    # iterate all sentences, select the one with minimum edit distance
    output['q2clause'] = {}
    output['clause2q'] = [[] for _ in output['clause_t']]
    output['q2edu'] = {}
    output['edu2q'] = [[] for _ in output['edu_t']]
    for idx, sent in enumerate(output['edu_t']):
        output['edu2q'][idx].extend([[] for _ in sent])
    for question, question_tokenized in output['q_t'].items():
        all_editdist = []
        for idx, clause in enumerate(output['clause_t']):
            start, end, editdist = get_span(clause, question_tokenized)  # [s,e] both inclusive
            all_editdist.append((idx, start, end, editdist))

        # take the minimum one
        clause_id, clause_start, clause_end, clause_dist = sorted(all_editdist, key=lambda x: x[-1])[0]
        output['q2clause'][question] = {
            'clause_id': clause_id,
            'clause_start': clause_start,  # [s,e] both inclusive
            'clause_end': clause_end,
            'clause_dist': clause_dist,
        }
        output['clause2q'][clause_id].append(question)

        # mapping to edus
        extract_span = set(range(output['q2clause'][question]['clause_start'],
                                 output['q2clause'][question]['clause_end'] + 1))
        output['q2edu'][question] = {
            'clause_id': output['q2clause'][question]['clause_id'],
            'edu_id': [],  # (id, overlap_toks)
        }

        for idx, span in enumerate(output['edu_span'][output['q2clause'][question]['clause_id']]):
            current_span = set(range(span[0], span[1] + 1))
            if extract_span.intersection(current_span):
                output['q2edu'][question]['edu_id'].append((idx, len(extract_span.intersection(current_span))))
                output['edu2q'][output['q2clause'][question]['clause_id']][idx].append(question)
        sorted_edu_id = sorted(output['q2edu'][question]['edu_id'], key=lambda x: x[-1], reverse=True)
        top_edu_id = sorted_edu_id[0][0]
        top_edu_span = output['edu_span'][output['q2clause'][question]['clause_id']][top_edu_id]
        top_edu_start = max(output['q2clause'][question]['clause_start'], top_edu_span[0])
        top_edu_end = min(output['q2clause'][question]['clause_end'], top_edu_span[1])
        output['q2edu'][question]['top_edu_id'] = top_edu_id
        output['q2edu'][question]['top_edu_start'] = top_edu_start
        output['q2edu'][question]['top_edu_end'] = top_edu_end  # [s,e] both inclusive
    return output


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


if __name__ == '__main__':
    sharc_path = './data/'
    with open(os.path.join(sharc_path, 'sharc_raw', 'negative_sample_utterance_ids',
                           'sharc_negative_question_utterance_ids.txt')) as f:
        negative_question_ids = f.read().splitlines()
    for split in ['dev', 'train']:
        fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
        with open(os.path.join(sharc_path, 'sharc_raw/json/{}_question_fixed.json'.format(fsplit))) as f:
            data_raw = json.load(f)
        ########################
        # construct tree mappings
        ########################
        ftree = os.path.join(sharc_path, 'trees_mapping_{}_{}.json'.format(FILENAME, split))
        with open(ftree) as f:
            mapping = json.load(f)

        ########################
        # construct samples
        ########################
        fproc = os.path.join(sharc_path, 'proc_span_{}_{}.pt'.format(FILENAME, split))
        data = []
        num_edu = []
        for ex in tqdm(data_raw):
            if ex['answer'].lower() in ['yes', 'no', 'irrelevant']:
                continue

            ex_answer = ex['answer'].lower()

            m = mapping[ex['tree_id']]

            # ######################
            # entailment tracking
            # ######################
            sep = tokenizer.sep_token_id
            cls = tokenizer.cls_token_id
            pad = tokenizer.pad_token_id

            # span extraction
            span_clause_id, span_clause_start, span_clause_end, span_clause_dist = m['q2clause'][ex['answer']].values()
            span_offset = 0
            span_pointer_mask = []

            # snippet
            inp = []
            rule_idx = []
            selected_edu_idx = -1
            for clause_id, clause in enumerate(m['clause_t']):
                if span_clause_id == clause_id:
                    span_offset = len(inp) + 1
                    selected_edu_idx = len(rule_idx)
                if len(inp) < MAX_LEN:
                    rule_idx.append(len(inp))
                    inp += [cls] + clause
                    span_pointer_mask += ([0] + [1] * len(clause))  # [0] for [CLS]
            inp += [sep]

            # add answer span
            span_se = (span_clause_start + span_offset, span_clause_end + span_offset)  # [,] both inclusive
            answer_span_test = inp[span_se[0]:span_se[1]+1]
            answer_span_test_2 = m['clause_t'][span_clause_id][span_clause_start:span_clause_end+1]
            assert answer_span_test == answer_span_test_2

            # user info (scenario, dialog history)
            user_idx = []
            question_tokenized = m['initial_question_t'][ex['question']]
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            question_idx = len(inp)
            inp += [cls] + question_tokenized + [sep]
            scenario_idx = -1
            if ex['scenario'] != '':
                scenario_tokenized = m['scenario_t'][ex['scenario']]
                if len(inp) < MAX_LEN: user_idx.append(len(inp))
                scenario_idx = len(inp)
                inp += [cls] + scenario_tokenized + [sep]
            for fqa in ex['history']:
                if len(inp) < MAX_LEN: user_idx.append(len(inp))
                fq, fa = fqa['follow_up_question'], fqa['follow_up_answer']
                fa = 'No' if 'no' in fa.lower() else 'Yes'  # fix possible typos like 'noe'
                inp += [cls] + roberta_encode('question') + m['q_t'][fq] + roberta_encode('answer') + roberta_encode(fa) + [sep]
            span_pointer_mask += [0] * (len(inp) - len(span_pointer_mask))

            # all
            input_mask = [1] * len(inp)
            if len(inp) > MAX_LEN:
                inp = inp[:MAX_LEN]
                input_mask = input_mask[:MAX_LEN]
                span_pointer_mask = span_pointer_mask[:MAX_LEN]
            while len(inp) < MAX_LEN:
                inp.append(pad)
                input_mask.append(0)
                span_pointer_mask.append(0)
            assert len(inp) == len(input_mask) == len(span_pointer_mask)

            ex['entail'] = {
                'inp': inp,
                'input_ids': torch.LongTensor(inp),
                'input_mask': torch.LongTensor(input_mask),
                'rule_idx': torch.LongTensor(rule_idx),
                'user_idx': torch.LongTensor(user_idx),
                'question_idx': question_idx,
                'scenario_idx': scenario_idx,
                'answer_span': span_se,
                'answer_span_start': span_se[0],
                'answer_span_end': span_se[1],
                'span_pointer_mask': torch.LongTensor(span_pointer_mask),
                'selected_edu_idx': selected_edu_idx,  # which edu contains the span
            }

            data.append(ex)
        print('{}: {}'.format(split, len(data)))
        preds = [{'utterance_id': e['utterance_id'],
                  'answer': roberta_decode(e['entail']['inp'][e['entail']['answer_span'][0]:e['entail']['answer_span'][1] + 1])} for e in data]
        pprint(compute_metrics(preds, data))
        torch.save(data, fproc)
