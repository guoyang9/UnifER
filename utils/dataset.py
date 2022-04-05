import os
import json
import h5py
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizerFast
import utils.utils as utils


def _create_entry(config, split, question, answer):
    """ Create dataset entries for ok-vqa. 
    Args:
        config (dict): default config file
        split (str): 'train' or 'val, do not support 'test' for ok-vqa
        question (dict): raw question dict
        answer (dict): processed answer dict
    """
    utils.assert_eq(question['question_id'], answer['question_id'])
    utils.assert_eq(question['image_id'], answer['image_id'])

    img_path = utils.find_path(config['path']['img'], split, question['image_id'])
    entry = {
        'question': question['question'],
        'question_id': question['question_id'],
        'img_id': question['image_id'],
        'img_path': img_path,
        'answer': answer
        }
    return entry


class BaseDataset(Dataset):
    def __init__(self, split: str, config) -> None:
        super(BaseDataset, self).__init__()
        assert split in ['train', 'test']
        self.config = config
        self.entries = []

        # loading answer-label
        with open(os.path.join(
                config['path']['cache'], 'ans2label.json'), 'r') as fd:
            self.ans2label = json.load(fd)
        with open(os.path.join(
                config['path']['cache'], 'label2ans.json'), 'r') as fd:
            self.label2ans = json.load(fd)
        self.num_ans_candidates = len(self.ans2label)

        # for question tokenizer and image extractor
        self.max_q_len = config['train']['q_max_len']
        self.tokenizer = eval(config['transformer']['tokenizer']
            ).from_pretrained(config['transformer']['checkpoint_token'])

        self.extractor = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def load_image(self, image_id):
        """ Load one image feature. """
        if not hasattr(self, 'img_id2idx'):
            with open(self.config['path']['feat_ids'], 'r') as fd:
                if type(image_id) == int: # for ok-vqa dataset
                    self.img_id2idx = json.load(fd, object_hook=utils.json_keys2int)
                else:
                    self.img_id2idx = json.load(fd)
        image_id = self.img_id2idx[image_id]
        if not hasattr(self, 'image_feat'):
            self.image_feat = h5py.File(self.config['path']['img_feat'], 'r')
        features = self.image_feat['features'][image_id]
        spatials = self.image_feat['boxes'][image_id]
        return torch.from_numpy(features), torch.from_numpy(spatials)

    def __getitem__(self, index):
        entry = self.entries[index]
        
        # image id and path
        img_id = entry['img_id']
        img_path = entry['img_path']

        # question tokenize
        question_id = entry['question_id']
        question = entry['question']
        inputs_question = self.tokenizer(question, 
            return_tensors='pt',
            max_length=self.max_q_len, padding='max_length', truncation=True)
        inputs_question = {k: v[0] for k, v in inputs_question.items()}
      
        # answer tensorize
        answer = entry['answer']
        labels = torch.tensor(answer['labels'], dtype=torch.long)
        scores = torch.tensor(answer['scores'], dtype=torch.float32)
        target = torch.zeros(self.num_ans_candidates)

        if labels is not None:
            target.scatter_(0, labels, scores)
        input_dict = {
            'img': {
                'id': img_id,
                'path': img_path},
            'question': {
                'id': question_id,
                'words': question,
                'inputs': inputs_question,
                'length': len(question) if len(question) < self.max_q_len else self.max_q_len},
            'answer': {
                'target': target},
        }

        # for different image inputs
        if self.config['transformer']['img_embed']:
            features, spatials = self.load_image(img_id)
            input_dict['img']['feat'] = features
            input_dict['img']['pos'] = spatials
        else:
            features = self.extractor(Image.open(img_path).convert("RGB"))
            input_dict['img']['feat'] = features
       
        return input_dict

    def __len__(self):
        return len(self.entries)


class OKVQADataset(BaseDataset):
    def __init__(self, split: str, config) -> None:
        super(OKVQADataset, self).__init__(split, config)
        if split == 'test':
            split = 'val'  # different for ok-vqa since there are three split names

        # loading questions
        train = True if split == 'train' else False
        val = True if split == 'val' else False
        question_path = utils.path_for(config, train=train, val=val, question=True)
        with open(question_path, 'r') as fd:
            questions = json.load(fd)['questions']
        questions = sorted(questions, key=lambda x: x['question_id'])
        
        # loading answers
        with open(os.path.join(config['path']['cache'], 
                '{}_target.json'.format(split)), 'r') as fd:
            answers = json.load(fd)
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))

        # building dataset entry
        for question, answer in zip(questions, answers):
            self.entries.append(_create_entry(config, split, question, answer))


class FVQADataset(BaseDataset):
    def __init__(self, split: str, config, set_id=0) -> None:
        super(FVQADataset, self).__init__(split, config)
        assert set_id in [0, 1, 2, 3, 4]

        # loading instances and answers
        with open(config['path']['all_qs'], 'r') as fd:
            all_qs = list(json.load(fd).values())
            all_qs = sorted(all_qs, key=lambda x: x['question_id'])
        with open(os.path.join(config['path']['cache'], 
                'target.json'.format(split)), 'r') as fd:
            answers = json.load(fd)
            answers = sorted(answers, key=lambda x: x['question_id'])
        
        # leveraging image ids to select train instances
        img_ids = [line.split('\n')[0] for line in open(os.path.join(
                    config['path']['raw'], split + '_list_' + str(set_id) + '.txt'))]
        print("The number of {} instances for set_id {} is {}".format(
            split, set_id, len(img_ids)))

        for inst, answer in zip(all_qs, answers):
            img_id = inst['img_file']
            if img_id in img_ids:
                utils.assert_eq(inst['question_id'], answer['question_id'])
                img_path = os.path.join(config['path']['img'], img_id)

                if 'COCO' in img_id:
                    img_id = 'coco' + str(int(img_id.split('.')[0].split('_')[-1]))
                elif 'ILSVRC' in img_id:
                    img_id = 'ilsvrc' + str(int(img_id.split('.')[0].split('_')[-1]))

                entry = {
                    'question': inst['question'],
                    'question_id': inst['question_id'],
                    'img_id': img_id,
                    'img_path': img_path,
                    'answer': answer,
                }
                self.entries.append(entry)


class KnowledgeDataset(Dataset):
    def __init__(self, knowledge_list, max_len, tokenizer) -> None:
        """ for tokenizing knowledge with roberta, supporting both datasets.
        Args:
            knowledge_list (list): list of knowledge sentences
            max_len (int): maximum length of each knowledge sentence
            tokenizer (nn.Module): transformer tokenizer
        """
        super().__init__()
        self.knowledge_list = knowledge_list
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.knowledge_list)

    def __getitem__(self, idx):
        sent = self.knowledge_list[idx]
        inputs = self.tokenizer(sent, 
            max_length=self.max_len, padding='max_length', truncation=True)
        inputs = {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}
        return inputs
    