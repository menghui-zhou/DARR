import os
import math
import random
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC, LinearSVC
import umap
from torchvision import transforms
from collections import defaultdict

def visualize_avg_saliency_as_classification(model, dataset, num_classes, save_dir, max_samples_per_class=500):
    """Compute average saliency maps for each class using up to max_samples_per_class samples per class."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    saliency_sums = defaultdict(lambda: None)
    class_counts = defaultdict(int)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)
        if class_counts[label] >= max_samples_per_class:
            continue
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_score = output[0, pred_class]
        model.zero_grad()
        pred_score.backward()
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        if saliency_sums[label] is None:
            saliency_sums[label] = saliency
        else:
            saliency_sums[label] += saliency
        class_counts[label] += 1
        if len(class_counts) == num_classes and all(v >= max_samples_per_class for v in class_counts.values()):
            break
    for class_label in saliency_sums:
        avg_saliency = saliency_sums[class_label] / class_counts[class_label]
        avg_saliency = avg_saliency - avg_saliency.min()
        avg_saliency = avg_saliency / (avg_saliency.max() + 1e-8)
        plt.figure(figsize=(4, 4))
        plt.imshow(avg_saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'avg_saliency_class_{class_label}.png'))
        plt.close()
        print(f"Saved average saliency map for class {class_label} ({class_counts[class_label]} samples).")

def visualize_saliency_as_classification(model, dataset, num_classes, save_dir):
    """For each class, find the first sample and plot saliency map."""
    model.eval()
    class_to_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break
    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")
    os.makedirs(save_dir, exist_ok=True)
    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_score = output[0, pred_class]
        model.zero_grad()
        pred_score.backward()
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()
        print(f"Saved original and saliency for class {class_label}.")

def visualize_saliency_as_sum_grad(model, dataset, num_classes, save_dir):
    """For each class, find the first sample and plot saliency map (embedding output)."""
    model.eval()
    class_to_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break
    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")
    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        embedding = model(input_tensor)
        scalar_score = embedding.sum()
        model.zero_grad()
        scalar_score.backward()
        saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.figure(figsize=(8, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()
        print(f"Saved saliency for class {class_label}.")

def visualize_saliency_as_grad_sum(model, dataset, num_classes, save_dir):
    """For each class, find the first sample and plot saliency map by summing gradients per output dimension."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    class_to_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break
    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")
    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        embedding = model(input_tensor)
        saliency_accum = torch.zeros_like(input_tensor)
        for dim in range(embedding.shape[1]):
            scalar_output = embedding[0, dim]
            model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            scalar_output.backward(retain_graph=True)
            saliency_accum += input_tensor.grad.data.abs()
        saliency, _ = torch.max(saliency_accum, dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'saliency_class_{class_label}.png'))
        plt.close()
        print(f"Saved original and saliency for class {class_label}.")

def visualize_avg_saliency_as_grad_sum(model, dataset, num_classes, save_dir, max_samples_per_class=500):
    """For each class, compute average saliency map from up to max_samples_per_class samples (sum of gradients)."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    saliency_sums = defaultdict(lambda: None)
    class_counts = defaultdict(int)
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        label = int(label)
        if class_counts[label] >= max_samples_per_class:
            continue
        input_tensor = image.unsqueeze(0).cuda()
        input_tensor.requires_grad = True
        embedding = model(input_tensor)
        saliency_accum = torch.zeros_like(input_tensor)
        for dim in range(embedding.shape[1]):
            scalar_output = embedding[0, dim]
            model.zero_grad()
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            scalar_output.backward(retain_graph=True)
            saliency_accum += input_tensor.grad.data.abs()
        saliency, _ = torch.max(saliency_accum, dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        if saliency_sums[label] is None:
            saliency_sums[label] = saliency
        else:
            saliency_sums[label] += saliency
        class_counts[label] += 1
        if len(class_counts) == num_classes and all(c >= max_samples_per_class for c in class_counts.values()):
            break
    for class_label in saliency_sums:
        avg_saliency = saliency_sums[class_label] / class_counts[class_label]
        avg_saliency = avg_saliency - avg_saliency.min()
        avg_saliency = avg_saliency / (avg_saliency.max() + 1e-8)
        plt.figure(figsize=(4, 4))
        plt.imshow(avg_saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'avg_gradsum_top{class_counts[class_label]}_class_{class_label}.png'))
        plt.close()
        print(f"Saved average grad-sum saliency for class {class_label} using {class_counts[class_label]} samples.")

def compute_smoothgrad_saliency(model, input_tensor, mode='classification', n_samples=50, noise_std=0.1):
    """Compute SmoothGrad saliency map."""
    model.eval()
    saliency_accum = torch.zeros_like(input_tensor)
    for _ in range(n_samples):
        noise = torch.normal(mean=0, std=noise_std, size=input_tensor.shape).cuda()
        noisy_input = (input_tensor + noise).detach()
        noisy_input.requires_grad = True
        if mode == 'classification':
            output = model(noisy_input)
            pred_class = output.argmax(dim=1).item()
            scalar = output[0, pred_class]
            model.zero_grad()
            scalar.backward()
            saliency = noisy_input.grad.abs()
        elif mode == 'embedding_sum':
            embedding = model(noisy_input)
            scalar = embedding.sum()
            model.zero_grad()
            scalar.backward()
            saliency = noisy_input.grad.abs()
        elif mode == 'embedding_separate':
            embedding = model(noisy_input)
            saliency = torch.zeros_like(noisy_input)
            for dim in range(embedding.shape[1]):
                scalar = embedding[0, dim]
                model.zero_grad()
                if noisy_input.grad is not None:
                    noisy_input.grad.zero_()
                scalar.backward(retain_graph=True)
                saliency += noisy_input.grad.abs()
        else:
            raise ValueError(f"Unknown mode: {mode}")
        saliency_accum += saliency
    saliency_mean = saliency_accum / n_samples
    saliency_final, _ = torch.max(saliency_mean, dim=1)
    return saliency_final.squeeze().cpu().numpy()

def visualize_smoothed_saliency_as_grad_sum(model, dataset, num_classes, save_dir):
    """SmoothGrad: For each class, compute saliency by summing gradients (embedding_separate mode)."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    class_to_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break
    if len(class_to_index) < num_classes:
        print(f"Warning: dataset only has {len(class_to_index)} classes!")
    for class_label, idx in class_to_index.items():
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        saliency = compute_smoothgrad_saliency(
            model, input_tensor, mode='embedding_separate', n_samples=50, noise_std=0.1
        )
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.figure(figsize=(4, 4))
        plt.imshow(img_np)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'original_class_{class_label}.png'))
        plt.close()
        plt.figure(figsize=(4, 4))
        plt.imshow(saliency, cmap='hot')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'smooth_saliency_gradsum_class_{class_label}.png'))
        plt.close()
        print(f"Saved original and SmoothGrad (Grad Sum) saliency for class {class_label}.")

def visualize_smoothed_saliency_all_classes_one_figure(model, dataset, num_classes, save_dir):
    """SmoothGrad: embedding_separate mode, all classes in one figure."""
    model.eval()
    class_to_index = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = int(label)
        if label not in class_to_index:
            class_to_index[label] = idx
        if len(class_to_index) == num_classes:
            break
    fig, axes = plt.subplots(nrows=num_classes, ncols=1, figsize=(4, num_classes * 3))
    for row_idx, class_label in enumerate(sorted(class_to_index.keys())):
        idx = class_to_index[class_label]
        image, label = dataset[idx]
        input_tensor = image.unsqueeze(0).cuda()
        saliency = compute_smoothgrad_saliency(
            model, input_tensor, mode='embedding_separate', n_samples=50, noise_std=0.1
        )
        ax = axes[row_idx] if num_classes > 1 else axes
        ax.imshow(saliency, cmap='hot')
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'smooth_all_classes.png'))
    plt.close()
    print(f"Saved combined SmoothGrad saliency figure to {save_dir}")
