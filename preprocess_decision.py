#!/usr/bin/env python
import os
import torch
import string
import json
from tqdm import tqdm
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
MODEL_FILE = '/research/king3/ik_grp/yfgao/pretrain_models/huggingface/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(MODEL_FILE, cache_dir=None)
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']
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
                # handle exception case train 261
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


if __name__ == '__main__':
    sharc_path = './data'
    with open(os.path.join(sharc_path, 'sharc_raw', 'negative_sample_utterance_ids',
                           'sharc_negative_question_utterance_ids.txt')) as f:
        negative_question_ids = f.read().splitlines()
    for split in ['dev', 'train']:
        fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
        with open(os.path.join(sharc_path, 'sharc_raw/json/{}_question_fixed.json'.format(fsplit))) as f:
            data_raw = json.load(f)
        with open(os.path.join(sharc_path, '{}_snippet_parsed.json'.format(split))) as f:
            edu_segment = json.load(f)

        ########################
        # construct tree mappings
        ########################
        ftree = os.path.join(sharc_path, 'trees_mapping_{}_{}.json'.format(FILENAME, split))
        if not os.path.isfile(ftree) or FORCE:
            tasks = {}
            for ex in data_raw:
                if ex['tree_id'] in tasks:
                    task = tasks[ex['tree_id']]
                else:
                    task = tasks[ex['tree_id']] = {'snippet': ex['snippet'], 'questions': set(), 'scenarios': set(),
                                                   'initial_questions': set()}
                for h in ex['history'] + ex['evidence']:
                    task['questions'].add(h['follow_up_question'])
                if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
                    task['questions'].add(ex['answer'])
                if ex['scenario'] != '':
                    task['scenarios'].add(ex['scenario'])
                task['initial_questions'].add(ex['question'])
            keys = sorted(list(tasks.keys()))
            vals = [extract_edus(tasks[k], edu_segment[k]) for k in tqdm(keys)]
            mapping = {k: v for k, v in zip(keys, vals)}
            with open(ftree, 'wt') as f:
                json.dump(mapping, f, indent=2)
        else:
            with open(ftree) as f:
                mapping = json.load(f)

        ########################
        # construct samples
        ########################
        fproc = os.path.join(sharc_path, 'proc_decision_{}_{}.pt'.format(FILENAME, split))
        data = []
        for ex in tqdm(data_raw):

            m = mapping[ex['tree_id']]

            # ######################
            # entailment tracking
            # ######################
            sep = tokenizer.sep_token_id
            cls = tokenizer.cls_token_id
            pad = tokenizer.pad_token_id

            # snippet
            inp = []
            rule_idx, rule_idx_relevant_label = [], []  # here we record all rule idx, and question relevant idx
            for clause_id, edus in enumerate(m['edu_t']):
                for edu_id, edu in enumerate(edus):
                    if len(inp) < MAX_LEN:
                        rule_idx.append(len(inp))
                        if len(m['edu2q'][clause_id][edu_id]):
                            rule_idx_relevant_label.append(1)  # 1: relevant, 0: irrelevant
                        else:
                            rule_idx_relevant_label.append(0)
                    inp += [cls] + edu
            inp += [sep]

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

            # all
            input_mask = [1] * len(inp)

            assert len(inp) == len(input_mask)

            if len(inp) > MAX_LEN:
                inp = inp[:MAX_LEN]
                input_mask = input_mask[:MAX_LEN]
            while len(inp) < MAX_LEN:
                inp.append(pad)
                input_mask.append(0)
            assert len(inp) == len(input_mask)

            ex['entail'] = {
                'inp': inp,
                'input_ids': torch.LongTensor(inp),
                'input_mask': torch.LongTensor(input_mask),
                'rule_idx': torch.LongTensor(rule_idx),
                'user_idx': torch.LongTensor(user_idx),
                'question_idx': question_idx,
                'scenario_idx': scenario_idx,
                'rule_idx_relevant_label': torch.LongTensor(rule_idx_relevant_label),
            }

            fqs, fas = [], []
            for fqa in ex['evidence'] + ex['history']:
                fq, fa = fqa['follow_up_question'], fqa['follow_up_answer'].lower()
                fa = 'no' if 'no' in fa else 'yes'  # fix possible typos like 'noe'
                fqs.append(fq)
                fas.append(ENTAILMENT_CLASSES.index(fa))

            # CE loss
            entailment_score_gold_ce_loss = []
            for clause_id, edus2q in enumerate(m['edu2q']):
                for edu_id, edu2q in enumerate(edus2q):
                    sentence_entail_states = []
                    for edu2qj in edu2q:
                        edu2aj = fas[fqs.index(edu2qj)] if edu2qj in fqs else ENTAILMENT_CLASSES.index('unknown')
                        sentence_entail_states.append(edu2aj)
                    if len(sentence_entail_states) > 1:
                        if ENTAILMENT_CLASSES.index('yes') in sentence_entail_states:
                            entailment_score_gold_ce_loss.append(ENTAILMENT_CLASSES.index('yes'))
                        elif ENTAILMENT_CLASSES.index('no') in sentence_entail_states:
                            entailment_score_gold_ce_loss.append(ENTAILMENT_CLASSES.index('no'))
                        else:
                            entailment_score_gold_ce_loss.append(ENTAILMENT_CLASSES.index('unknown'))
                    elif len(sentence_entail_states) == 1:
                        entailment_score_gold_ce_loss.append(sentence_entail_states[0])
                    else:
                        entailment_score_gold_ce_loss.append(ENTAILMENT_CLASSES.index('unknown'))
            assert len(ex['entail']['rule_idx']) == len(entailment_score_gold_ce_loss)
            ex['entail']['entailment_score_gold_ce'] = torch.LongTensor(entailment_score_gold_ce_loss)

            ########################
            # logic reasoning
            ########################
            ex['logic'] = {}
            ex_answer = ex['answer'].lower()
            ex['logic']['answer_class'] = DECISION_CLASSES.index(ex_answer) if ex_answer in DECISION_CLASSES else DECISION_CLASSES.index('more')

            data.append(ex)
        torch.save(data, fproc)
