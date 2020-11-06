import os
import json
import csv
import re
import nltk.data

sharc_path = '../data'


def parsing_snippet(snippet):
    title_matched = re.match(r'#.{2,}\n\n', snippet)
    if title_matched:
        title_span = title_matched.span()
        title = snippet[title_span[0]:title_span[1]].strip('\n').strip()
        context = snippet[title_span[1]:]
    else:
        title = ''
        context = snippet
    # check if exist bullets
    bullet_segmenter = '* '
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    is_bullet = False
    if bullet_segmenter in context:
        clauses = []
        is_bullet = True
        bullet_position = [m.start() for m in re.finditer('\* ', context)]
        if bullet_position[0] != 0:
            prompt = context[:bullet_position[0]].strip()
            if '. ' in prompt:
                prompt_sentences = sent_detector.tokenize(prompt)
                clauses.extend(prompt_sentences)
            else:
                clauses.append(prompt)
        for idx in range(len(bullet_position)):
            current_start_pos = bullet_position[idx]
            next_start_pos = bullet_position[idx + 1] if idx + 1 < len(bullet_position) else len(context) + 1
            clauses.append(context[current_start_pos:next_start_pos].strip('\n').strip())
    else:
        clauses = sent_detector.tokenize(context)
    return (title, clauses, is_bullet)


for split in ['dev', 'train']:
    with open(os.path.join(sharc_path, 'sharc_raw', 'json', 'sharc_{}.json'.format(split))) as f:
        data_raw = json.load(f)
    tasks = {}
    for ex in data_raw:
        for h in ex['evidence']:
            if 'followup_question' in h:
                h['follow_up_question'] = h['followup_question']
                h['follow_up_answer'] = h['followup_answer']
                del h['followup_question']
                del h['followup_answer']
        if ex['tree_id'] not in tasks:
            task = tasks[ex['tree_id']] = {'snippet': ex['snippet']}
    for id, v in tasks.items():
        title, clauses, is_bullet = parsing_snippet(v['snippet'])
        tasks[id]['title'] = title
        tasks[id]['clauses'] = clauses
        tasks[id]['is_bullet'] = is_bullet
    with open(os.path.join(sharc_path, '{}_snippet_parsed.json'.format(split)), 'wt') as f:
        json.dump(tasks, f, indent=2)


