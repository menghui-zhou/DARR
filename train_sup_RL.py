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
from loss_RL import RoughLearning
import utils
from utils import save_model
import sys
from feature_visualisation import feature_similarity_analysis, extract_features
import logging
from Inf_quantity import calculate_information_quantification
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import pandas as pd
import RL_evaluate
import random

print = logging.info

torch.cuda.empty_cache()
import matplotlib.pyplot as plt

def evaluate_and_save_metrics(epoch, model, train_loader, test_loader, args):
    train_features, train_labels = extract_features(model, train_loader)
    (
        overall_coding_length,
        overall_spectral_entropy,
        overall_var_information,
        coding_length,
        spectral_entropy,
        var_information
    ) = calculate_information_quantification(train_features, train_labels, args)

    to_print = (
        f"Train: [Epoch {epoch}] "
        f"\tOverall Coding Length: {overall_coding_length:.3f} "
        f"\tOverall Spectral Entropy: {overall_spectral_entropy:.3f} "
        f"\tOverall Var Information: {overall_var_information:.3f}"
        f"\tCoding Length: {coding_length:.3f} "
        f"\tSpectral Entropy: {spectral_entropy:.3f} "
        f"\tVar Information: {var_information:.3f}"
    )
    print(to_print)
    sys.stdout.flush()

    utils.save_information_quantity(
        args.model_dir,
        epoch,
        overall_coding_length,
        overall_spectral_entropy,
        overall_var_information,
        coding_length,
        spectral_entropy,
        var_information
    )

    train_accuracies = RL_evaluate.get_all_acc(args, train_features, train_labels, train_features, train_labels)
    to_print = f'Train: [{epoch}]\t' \
               f'Train Accuracy: Linear_SVM {train_accuracies[0]:.4f}\t' \
               f'KNN {train_accuracies[1]:.4f}\t' \
               f'NCC {train_accuracies[2]:.4f}\t' \
               f'LogisticSR {train_accuracies[3]:.4f}'
    print(to_print)
    sys.stdout.flush()
    utils.save_accuracy(args.model_dir, epoch, *train_accuracies, filename='train_accuracy.csv')

    test_features, test_labels = extract_features(model, test_loader)
    test_accuracies = RL_evaluate.get_all_acc(args, train_features, train_labels, test_features, test_labels)
    to_print = f'Test: [{epoch}]\t' \
               f'Test Accuracy: Linear_SVM {test_accuracies[0]:.4f}\t' \
               f'KNN {test_accuracies[1]:.4f}\t' \
               f'NCC {test_accuracies[2]:.4f}\t' \
               f'LogisticSR {test_accuracies[3]:.4f}'
    print(to_print)
    sys.stdout.flush()
    utils.save_accuracy(args.model_dir, epoch, *test_accuracies, filename='test_accuracy.csv')

    if epoch % args.save_fea_ana_freq == 0:
        feature_similarity_analysis(train_features, train_labels, args.model_dir, epoch)

def parse_option():
    parser = argparse.ArgumentParser(description='Supervised Learning with Deep Neural modelworks')
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--arch', type=str, default='mnisttinynet')
    parser.add_argument('--fd', type=int, default=256)
    parser.add_argument('--data', type=str, default='MNIST')
    parser.add_argument('--epo', type=int, default=500)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--gam1', type=float, default=1.)
    parser.add_argument('--gam2', type=float, default=1.)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--corrupt', type=str, default="default")
    parser.add_argument('--lcr', type=float, default=0)
    parser.add_argument('--lcs', type=int, default=42)
    parser.add_argument('--trail', type=str, default='0')
    parser.add_argument('--transform', type=str, default='f_mnist')
    parser.add_argument('--save_dir', type=str, default='./saved_models/MNIST/')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--pretrain_epo', type=int, default=None)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--print_freq', type=int, default=60)
    parser.add_argument('--save_measure_freq', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=500)
    parser.add_argument('--save_fea_ana_freq', type=int, default=500)
    parser.add_argument('--rough_alpha', type=float, default=0.00)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--n_comp', type=int, default=30)
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()

    args.model_name = 'RL-eps:{}_sup_{}+{}_{}_epo_{}_bs_{}_lr_{}_mom_{}_wd_{}_gam1_{}_gam2_{}_lr_{}_trail_{}'.format(
        args.eps, args.arch, args.fd, args.data, args.epo, args.bs, args.lr, args.mom,
        args.wd, args.gam1, args.gam2, args.lcr, args.trail)

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
    transforms = tf.load_transforms(args.transform)
    train_set = tf.load_trainset(args.data, transforms, train=True, path=args.data_dir)
    train_set = tf.corrupt_labels(args.corrupt)(train_set, args.lcr, args.lcs)
    test_transforms = tf.load_transforms('test')
    test_set = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)

    train_loader = DataLoader(train_set, batch_size=args.bs, drop_last=True, num_workers=args.n_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, drop_last=True, num_workers=args.n_workers, shuffle=False)
    return train_loader, train_set.num_classes, test_loader

def set_model(args):
    if args.pretrain_dir is not None:
        model, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
        utils.update_params(args.model_dir, args.pretrain_dir)
    else:
        model = tf.load_architectures(args.arch, args.fd, args.n_class)
    criterion = RoughLearning(gam1=args.gam1, gam2=args.gam2, eps=args.eps, alpha=args.rough_alpha)
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
    sum_grad_norm = 0.0
    batch_count = 0
    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        batch_imgs, batch_labels = data_tuple
        data_time.update(time.time() - end)
        features = model(batch_imgs)
        total_loss_rough, loss_rough, loss_precise, total_loss_precise = criterion(features, batch_labels)
        losses.update(total_loss_rough.item(), args.bs)
        optimiser.zero_grad()
        total_loss_rough.backward()
        batch_grad_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                batch_grad_norm_sq += p.grad.data.norm().item() ** 2
        batch_grad_norm = batch_grad_norm_sq ** 0.5
        sum_grad_norm += batch_grad_norm
        batch_count += 1
        optimiser.step()
        batch_time.update(time.time() - end)
        end = time.time()
        current_lr = optimiser.param_groups[0]['lr']
        if (idx + 1) % args.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'lr {lr:.5f}\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                lr=current_lr,
                loss=losses
            )
            print(to_print)
            sys.stdout.flush()
        utils.save_state(args.model_dir, epoch, idx, total_loss_rough.item(), *loss_rough, *loss_precise,
                         total_loss_precise.item())
    avg_grad_norm = sum_grad_norm / batch_count if batch_count > 0 else 0.0
    return avg_grad_norm

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
        scheduler = lr_scheduler.MultiStepLR(optimizer, [200, 400], gamma=0.1)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epo, eta_min=1e-6)
    prev_params = None
    start_epoch = 1
    if len(args.resume):
        ckpt_state = torch.load(args.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {args.resume}!")
    for epoch in range(start_epoch, args.epo + 1):
        avg_grad_norm = train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step(epoch)
        current_params = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        if prev_params is None:
            diff = 0.0
        else:
            total_diff_sq = 0.0
            for name in current_params:
                d = current_params[name] - prev_params[name]
                total_diff_sq += torch.norm(d).item() ** 2
            diff = total_diff_sq ** 0.5
        prev_params = current_params
        utils.save_parameter_flow(args.model_dir, epoch, avg_grad_norm, diff)
        model.eval()
        if epoch % args.save_measure_freq == 0 or epoch == 1:
            evaluate_and_save_metrics(epoch, model, train_loader, test_loader, args)
        if epoch % args.save_model_freq == 0:
            save_file = os.path.join(args.model_dir, 'checkpoints',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)
        save_file = os.path.join(args.model_dir, 'checkpoints', 'curr_epoch.pth')
        save_model(model, optimizer, args, epoch, save_file)
    model.eval()
    evaluate_and_save_metrics(args.epo, model, train_loader, test_loader, args)
    from feature_importance import (visualize_smoothed_saliency_as_grad_sum,
                                    visualize_avg_saliency_as_grad_sum)
    visualize_avg_saliency_as_grad_sum(
        model=model,
        dataset=train_loader.dataset,
        num_classes=args.n_class,
        save_dir=os.path.join(args.model_dir, "Train_avg_saliency_maps_grad_sum")
    )
    visualize_avg_saliency_as_grad_sum(
        model=model,
        dataset=test_loader.dataset,
        num_classes=args.n_class,
        save_dir=os.path.join(args.model_dir, "Test_avg_saliency_maps_grad_sum")
    )
    print("Training complete.\n\n\n")

if __name__ == '__main__':
    main()
