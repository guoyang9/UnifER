import os
import json
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()

import utils.utils as utils
from param import args
from utils.optim import PlainLoss
from utils.dataset import OKVQADataset, FVQADataset
from train import train, evaluate
from modules.model_build import KnowledgeNotNoise

from tensorboardX import SummaryWriter


def run_okvqa(args, config, 
        train_dset, test_dset, 
        model, optim, lr_scheduler,
        knowledge_full, knowledge_embed):
    """ Run okvqa training and testing. """
    test_score, best_test_score, start_epoch, best_epoch = 0.0, 0.0, 0, 0
    tracker = utils.Tracker()
    # writer = SummaryWriter()
    writer = None

    if args.test_only:
        args.resume = True
    if args.resume:
        logs = torch.load(args.name)
        print("loading logs from {}".format(args.name))
        model.load_state_dict(logs['model_state'])
        start_epoch = logs['epoch']
        best_epoch = logs['epoch']
        best_test_score = logs['best_test_score']    

    # --------------------------Run Model--------------------------- #
    test_loader = DataLoader(test_dset,
        args.batch_size, shuffle=False, num_workers=0)
    if args.test_only:
        test_score, results = evaluate(args.dataset,
            model, test_loader, 
            knowledge_full, knowledge_embed,
            write=True)
        best_test_score = test_score['acc']
    else:
        tb_count = 0
        train_loader = DataLoader(train_dset, 
            args.batch_size, pin_memory=True,
            shuffle=True, num_workers=config['run']['num_workers'])
        for epoch in range(start_epoch, config['run']['epochs']):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(args.dataset,
                model, optim, lr_scheduler,
                train_loader, 
                knowledge_full, knowledge_embed,
                tracker, tb_count, writer)

            print("testing after epoch {:03d}".format(epoch))
            model.train(False)
            test_score, results = evaluate(args.dataset,
                model, test_loader, 
                knowledge_full, knowledge_embed,
                write=False)
            test_score = test_score['acc']
            model.train(True)
            print("test score: {:.2f} \n".format(100 * test_score))

            if test_score > best_test_score:
                best_test_score = test_score
                best_epoch = epoch
                status = {
                    'epoch': epoch + 1,
                    'best_test_score': best_test_score,
                    'model_state': model.state_dict(),
                }
                torch.save(status, args.name)
                # with open('ok-vqa_results.json', 'w') as fd:
                #     json.dump(results, fd)
    print("best accuracy {:.2f} on epoch {:03d}".format(100 * best_test_score, best_epoch))


def run_fvqa(args, config, 
        train_dset, test_dset, 
        model, optim, lr_scheduler,
        knowledge_full, knowledge_embed):
    """ Run fvqa training and testing. """
    top1_accs, top3_accs = [], []
    for set_id in range(5):
        tracker = utils.Tracker()
        writer = SummaryWriter()
        best_top1, best_top3 = 0.0, 0.0

        # ------------------------Create Dataset------------------------ #
        tb_count = 0
        train_dset = eval(args.dataset + 'Dataset')('train', config, set_id)
        test_dset = eval(args.dataset + 'Dataset')('test', config, set_id)
        train_loader = DataLoader(train_dset, 
            args.batch_size, pin_memory=True,
            shuffle=True, num_workers=config['run']['num_workers'])
        test_loader = DataLoader(test_dset,
            args.batch_size, shuffle=False, num_workers=0)

        for epoch in range(config['run']['epochs']):
            print("training epoch {:03d}".format(epoch))
            tb_count = train(args.dataset,
                model, optim, lr_scheduler,
                train_loader, 
                knowledge_full, knowledge_embed,
                tracker, tb_count, writer)

            print("testing after epoch {:03d}".format(epoch))
            model.train(False)
            test_score = evaluate(args.dataset,
                model, test_loader, 
                knowledge_full, knowledge_embed,
                write=False)
            model.train(True)
            print("test score: top1 - {:.2f}, top3 - {:.2f} \n".format(
                100 * test_score['acc'], 100 * test_score['top3_acc']))

            if test_score['acc'] > best_top1:
                best_top1 = test_score['acc']
                best_top3 = test_score['top3_acc']
        top1_accs.append(best_top1)
        top3_accs.append(best_top3)
        print("best accuracy : top1 - {:.2f}, top3 - {:.2f} \n".format(
                100 * best_top1, 100 * best_top3))

        # ------------------------Reset Parameters------------------------ #
        loss_fn = model.loss_fn
        del(model)
        model = KnowledgeNotNoise(config, 
            loss_fn, test_dset.num_ans_candidates).cuda()
        if ',' in args.gpu:
            model = nn.DataParallel(model)

        # freeze some model parameters
        begin_layer_id = (12 - config['train']['num_fix_layer']) // 2 
        fixed_layer_ids = list(range(
            begin_layer_id, begin_layer_id + config['train']['num_fix_layer']))
        for name, param in model.named_parameters():
            if 'layer.' in name:
                if int(name.split('layer.')[-1].split('.')[0]) in fixed_layer_ids:
                    param.requires_grad = False

        # optimizer and lr scheduler
        optim = torch.optim.AdamW(model.parameters(), 
            lr=config['run']['lr'], 
            weight_decay=config['run']['weight_decay'])

        if config['run']['lr_scheduler']:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.75)
        else:
            lr_scheduler = None

    print("average accuracy : top1 - {:.2f}, top3 - {:.2f} \n".format(
                100 * np.mean(top1_accs), 100 * np.mean(top3_accs)))


def main():
    with open(args.cfg) as stream:
        config = yaml.safe_load(stream)
    print(config); print(args)

    torch.manual_seed(config['run']['seed'])
    torch.cuda.manual_seed(config['run']['seed'])
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if 'log' not in args.name:
        args.name = 'logs/' + args.name
    
    # ------------------------Model Definition---------------------- #
    if args.loss_fn == 'Plain':
        loss_fn = PlainLoss(config['train']['loss_type'])
    else:
        raise RuntimeError('not implement for {}'.format(args.loss_fn))
    
    train_dset = eval(args.dataset + 'Dataset')('train', config)
    test_dset = eval(args.dataset + 'Dataset')('test', config)
    model = KnowledgeNotNoise(config, loss_fn, test_dset.num_ans_candidates).cuda()
    if ',' in args.gpu:
        model = nn.DataParallel(model)

    # freeze some model parameters
    begin_layer_id = (12 - config['train']['num_fix_layer']) // 2 
    fixed_layer_ids = list(range(
        begin_layer_id, begin_layer_id + config['train']['num_fix_layer']))
    for name, param in model.named_parameters():
        if 'layer.' in name:
            if int(name.split('layer.')[-1].split('.')[0]) in fixed_layer_ids:
                param.requires_grad = False

    # optimizer and lr scheduler
    optim = torch.optim.AdamW(model.parameters(), 
        lr=config['run']['lr'], 
        weight_decay=config['run']['weight_decay'])

    if config['run']['lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.75)
    else:
        lr_scheduler = None

    # -----------------------Import Knowledge----------------------- #
    with open(config['path']['knowledge'], 'r') as fd:
        knowledge_full = json.load(fd)
    knowledge_embed = torch.load(config['path']['knowledge_embed']).cuda()

    # ------------------------Model Running------------------------- #
    eval('run_' + args.dataset.lower())(
        args, config, 
        train_dset, test_dset, 
        model, optim, lr_scheduler,
        knowledge_full, knowledge_embed)


if __name__ == '__main__':
    main()
    