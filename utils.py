import os
import logging
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import random
from collections import defaultdict

def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes."""
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def init_pipeline(model_dir, headers=None):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'plabels'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'parameter_change'), exist_ok=True)

    if headers is None:
        headers = ["epoch", "step", "total_loss_rough", "discrimn_loss_e", "compress_loss_e",
                   "discrimn_loss_t", "compress_loss_t", "total_loss_precise"]

    losses_csv = os.path.join(model_dir, 'losses.csv')
    if not os.path.exists(losses_csv):
        create_csv(model_dir, 'losses.csv', headers)

    headers_acc = ["epoch", 'LinearSVM', 'KNN', 'NCC', 'LogisticSR']
    train_csv = os.path.join(model_dir, 'train_accuracy.csv')
    if not os.path.exists(train_csv):
        create_csv(model_dir, 'train_accuracy.csv', headers_acc)
    test_csv = os.path.join(model_dir, 'test_accuracy.csv')
    if not os.path.exists(test_csv):
        create_csv(model_dir, 'test_accuracy.csv', headers_acc)

    headers_selfsup_acc = ["epoch", 'NMI', 'ACC', 'ARI']
    train_csv = os.path.join(model_dir, 'selfsup_train_accuracy.csv')
    if not os.path.exists(train_csv):
        create_csv(model_dir, 'selfsup_train_accuracy.csv', headers_selfsup_acc)
    test_csv = os.path.join(model_dir, 'selfsup_test_accuracy.csv')
    if not os.path.exists(test_csv):
        create_csv(model_dir, 'selfsup_test_accuracy.csv', headers_selfsup_acc)

    headers_parameter = ["epoch", "gradient_flow", "parameter_change_flow"]
    parameter_flow_csv = os.path.join(model_dir, 'parameter_flow.csv')
    if not os.path.exists(parameter_flow_csv):
        create_csv(model_dir, 'parameter_flow.csv', headers_parameter)

    headers_information = ["epoch", 'Overall_coding_length', 'Overall_spectral_entropy',
                           'Overall_var_information', 'coding_length',
                           'spectral_entropy', 'var_information']
    info_csv = os.path.join(model_dir, 'information_quantity.csv')
    if not os.path.exists(info_csv):
        create_csv(model_dir, 'information_quantity.csv', headers_information)

    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers):
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)

def load_params(model_dir):
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        f.write('\n' + ','.join(map(str, entries)))

def save_accuracy(model_dir, *entries, filename='train_accuracy.csv'):
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        f.write('\n' + ','.join(map(str, entries)))

def save_information_quantity(model_dir, *entries, filename='information_quantity.csv'):
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        f.write('\n' + ','.join(map(str, entries)))

def save_parameter_flow(model_dir, *entries, filename='parameter_flow.csv'):
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        f.write('\n' + ','.join(map(str, entries)))

def save_ckpt(model_dir, net, epoch):
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints',
                                              'model-epoch{}.pt'.format(epoch)))

def save_model(model, optimizer, args, epoch, save_file):
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def calculate_model_params_mean(model):
    total_sum = 0.0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_sum += param.data.mean().item() * param.numel()
            total_params += param.numel()
    overall_mean = total_sum / total_params
    return overall_mean

def save_labels(model_dir, labels, epoch):
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)

def compute_accuracy(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def clustering_accuracy(labels_true, labels_pred):
    unique_labels_true = np.unique(labels_true)
    unique_labels_pred = np.unique(labels_pred)
    contingency_matrix = np.zeros((len(unique_labels_true), len(unique_labels_pred)), dtype=int)
    for i, true_label in enumerate(unique_labels_true):
        for j, pred_label in enumerate(unique_labels_pred):
            contingency_matrix[i, j] = np.sum((labels_true == true_label) & (labels_pred == pred_label))
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    accuracy = contingency_matrix[row_ind, col_ind].sum() / len(labels_true)
    return accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
