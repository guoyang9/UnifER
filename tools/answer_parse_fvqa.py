import os
import sys
import json
import yaml
from tqdm import tqdm
sys.path.append(os.getcwd())

from param import args
import utils.utils as utils


def filter_answers(all_qs, min_occurence):
    """ Filtering answers whose frequency is less than min_occurence. """
    occurence = {}
    for entry in all_qs.values():
        answer = entry['answer']
        answer = utils.preprocess_answer(answer)
        if answer not in occurence:
            occurence[answer] = set()
        occurence[answer].add(entry['question_id'])
    for answer in list(occurence.keys()):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= {} times: {}'.format(
                                min_occurence, len(occurence)))
    return occurence


def create_ans2label(occurence, cache_root):
    """ Map answers to label. """
    label, label2ans, ans2label = 0, [], {}
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, 'ans2label.json')
    json.dump(ans2label, open(cache_file, 'w'))
    cache_file = os.path.join(cache_root, 'label2ans.json')
    json.dump(label2ans, open(cache_file, 'w'))
    return ans2label


def compute_target(all_qs, ans2label, cache_root):
    """ Augment ans_dset with soft score as label. """
    target = []
    for entry in tqdm(all_qs.values(),
                        desc='iter answers',
                        unit_scale=True,
                        total=len(all_qs)):
        answer = entry['answer']
        answer = utils.preprocess_answer(answer)

        labels, scores = [], []
        if answer in ans2label:
            labels.append(ans2label[answer])
            scores.append(1.0)
        target.append({
            'question_id': entry['question_id'],
            'labels': labels,
            'scores': scores,
        })

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, 'target.json')
    json.dump(target, open(cache_file, 'w'))


if __name__ == '__main__':
    with open(args.cfg) as stream:
        config = yaml.safe_load(stream)

    with open(config['path']['all_qs'], 'r') as fd:
        all_qs = json.load(fd)
    
    print("filtering answers less than minimum occurrence...")
    occurence = filter_answers(all_qs, config['run']['min_occurance'])
    print("create answers to integer labels...")
    ans2label = create_ans2label(occurence, config['path']['cache'])
    print("converting target for train and test answers...")
    compute_target(all_qs, ans2label, config['path']['cache'])
