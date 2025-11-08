import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from augmentloader import AugmentLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import train_func as tf
from loss import MaximalCodingRateReduction
import utils
from utils import save_model
import sys
from feature_visualisation import feature_similarity_analysis, extract_features
import logging
from Inf_quantity import calculate_information_quantification
import torchvision.transforms as transforms
import pandas as pd
import RL_evaluate
import random

print = logging.info

torch.cuda.empty_cache()
import matplotlib.pyplot as plt

def evaluate_and_save_metrics(epoch, model, test_loader, args):
    test_features, test_labels = extract_features(model, test_loader)
    test_accuracies = RL_evaluate.get_all_slefsup_acc(args, test_features, test_labels)
    print(f'Test: [{epoch}]  '
          f'NMI {test_accuracies[0]:.4f}  '
          f'ACC {test_accuracies[1]:.4f}  '
          f'ARI {test_accuracies[2]:.4f}')
    sys.stdout.flush()
    utils.save_accuracy(args.model_dir, epoch, *test_accuracies, filename='selfsup_test_accuracy.csv')

def parse_option():
    parser = argparse.ArgumentParser(description='Supervised Learning with Deep Neural Networks')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--arch', type=str, default='resnet18ctrl')
    parser.add_argument('--fd', type=int, default=64)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--epo', type=int, default=100)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--gam1', type=float, default=1)
    parser.add_argument('--gam2', type=float, default=1)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--corrupt', type=str, default="default")
    parser.add_argument('--aug', type=int, default=50)
    parser.add_argument('--lcr', type=float, default=0)
    parser.add_argument('--lcs', type=int, default=42)
    parser.add_argument('--trail', type=str, default='0')
    parser.add_argument('--transform', type=str, default='cifar')
    parser.add_argument('--save_dir', type=str, default='./selfsup_saved_models/CIFAR10/')
    parser.add_argument('--sampler', type=str, default='random')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--pretrain_epo', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--save_measure_freq', type=int, default=100)
    parser.add_argument('--save_model_freq', type=int, default=100)
    parser.add_argument('--save_fea_ana_freq', type=int, default=500)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--n_comp', type=int, default=30)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--aug_width', type=int, default=4)

    args = parser.parse_args()
    args.model_name = 'RL-eps:{}_selfsup_{}+{}_{}_epo_{}_bs_{}_opt_{}_lr_{}_mom_{}_wd_{}_gam1_{}_gam2_{}_lcr_{}_aug_{}_trail_{}'.format(
        args.eps, args.arch, args.fd, args.data, args.epo, args.bs, args.optimizer, args.lr, args.mom,
        args.wd, args.gam1, args.gam2, args.lcr, args.aug, args.trail)

    args.model_dir = os.path.join(args.save_dir, args.model_name)
    folder_path = os.path.join(args.model_dir, "checkpoints")
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if "curr_epoch.pth" in file_name:
                args.resume = folder_path + "/curr_epoch.pth"
                args.need_train = False
                break

    if len(args.resume):
        args.model_name = args.resume.split('/')[-3]

    utils.init_pipeline(args.model_dir)
    utils.save_params(args.model_dir, vars(args))

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    print(f"Model name: {args.model_name}")
    print(f"Arguments: {args}")
    return args

def set_loader(args):
    from augmentloader import AugmentLoader
    transforms = tf.load_transforms(args.transform)
    train_set = tf.load_trainset(args.data, path=args.data_dir)
    train_loader = AugmentLoader(train_set,
                                 transforms=transforms,
                                 sampler=args.sampler,
                                 batch_size=args.bs,
                                 num_aug=args.aug)

    test_transforms = tf.load_transforms('test')
    test_set = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)
    test_loader = DataLoader(test_set, batch_size=args.bs, drop_last=True, num_workers=args.n_workers, shuffle=False)
    return train_loader, train_set.num_classes, test_loader

def set_model(args):
    if args.pretrain_dir is not None:
        model, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
        utils.update_params(args.model_dir, args.pretrain_dir)
    else:
        model = tf.load_architectures(args.arch, args.fd, args.n_class)
    criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    utils.save_params(args.model_dir, vars(args))
    return model, criterion

def train(train_loader, model, criterion, optimiser, epoch, args):
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()

    for idx, data_tuple in enumerate(train_loader):
        if idx % 100 == 0:
            print(f'   The {idx + 1}-th mini batch of the dataset')

        batch_imgs, _, batch_labels = data_tuple
        data_time.update(time.time() - end)
        features = model(batch_imgs.cuda())
        total_loss_rough, loss_rough, loss_precise, total_loss_precise = criterion(features, batch_labels)
        losses.update(total_loss_rough.item(), args.bs)

        optimiser.zero_grad()
        total_loss_rough.backward()
        optimiser.step()

        batch_time.update(time.time() - end)
        end = time.time()

        utils.save_state(args.model_dir, epoch, idx, total_loss_rough.item(), *loss_rough, *loss_precise,
                         total_loss_precise.item())

def set_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    return optimizer

def main():
    utils.set_seed(42)
    args = parse_option()
    train_loader, num_classes, test_loader = set_loader(args)
    args.n_class = num_classes
    model, criterion = set_model(args)
    optimizer = set_optimizer(args, model)
    if args.optimizer == 'SGD':
        scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=1e-6)

    start_epoch = 1
    if len(args.resume):
        ckpt_state = torch.load(args.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {args.resume}!")

    for epoch in range(start_epoch, args.epo + 1):
        print(f'The {epoch}-th epoch')
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step(epoch)

        model.eval()
        if epoch % args.save_model_freq == 0:
            save_file = os.path.join(args.model_dir, 'checkpoints', f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, args, epoch, save_file)

        save_file = os.path.join(args.model_dir, 'checkpoints', 'curr_epoch.pth')
        save_model(model, optimizer, args, epoch, save_file)

    model.eval()
    evaluate_and_save_metrics(args.epo, model, test_loader, args)
    print("Training complete.\n\n\n")

if __name__ == '__main__':
    main()
