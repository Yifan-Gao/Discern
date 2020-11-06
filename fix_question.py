# fix annotation errors in sharc
import os
import json
import editdistance


def merge_question(questions):
    original_questions = sorted(list(questions))
    merged_questions = {q:q for q in original_questions}
    # if two question have edit distance smaller than some threshold, then they will be the same thing
    for i in range(len(original_questions)):
        for j in range(i, len(original_questions)):
            qi = original_questions[i]
            qj = original_questions[j]
            if 0 < editdistance.eval(qi, qj) <= 3:
                if len(qi) > len(qj):
                    merged_questions[qj] = qi
                else:
                    merged_questions[qi] = qj
    return merged_questions


sharc_path = './data'
for split in ['dev', 'train']:
    fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
    with open(os.path.join(sharc_path, 'sharc_raw/json/{}.json'.format(fsplit))) as f:
        data_raw = json.load(f)
    tasks = {}
    for ex in data_raw:
        for h in ex['evidence']:
            if 'followup_question' in h:
                h['follow_up_question'] = h['followup_question']
                h['follow_up_answer'] = h['followup_answer']
                del h['followup_question']
                del h['followup_answer']
        if ex['tree_id'] in tasks:
            task = tasks[ex['tree_id']]
        else:
            task = tasks[ex['tree_id']] = {'questions': set()}
        for h in ex['history'] + ex['evidence']:
            task['questions'].add(h['follow_up_question'])
        if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
            task['questions'].add(ex['answer'])
    keys = sorted(list(tasks.keys()))
    vals = [merge_question(tasks[k]['questions']) for k in keys]
    mapping = {k: v for k, v in zip(keys, vals)}
    fixed_count = 0
    for ex in data_raw:
        for h in ex['history'] + ex['evidence']:
            if h['follow_up_question'] not in mapping[ex['tree_id']].values():
                h['follow_up_question'] = mapping[ex['tree_id']][h['follow_up_question']]
                fixed_count += 1
        if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
            if ex['answer'] not in mapping[ex['tree_id']].values():
                ex['answer'] = mapping[ex['tree_id']][ex['answer']]
                fixed_count += 1
    print('{}: {} questions fixed'.format(split, fixed_count))

    with open(os.path.join(sharc_path, 'sharc_raw/json/{}_question_fixed.json'.format(fsplit)), 'wt') as f:
        json.dump(data_raw, f)
