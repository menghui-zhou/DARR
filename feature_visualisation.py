import os
import math
import random
import csv
from collections import defaultdict

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

def extract_features(model, loader, IsCE=False):
    features_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for idx, data_tuple in enumerate(loader):
            images, labels = data_tuple
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            if IsCE:
                features = model.module.extract_feature(images)
            else:
                features = model(images)
            features_list.append(features)
            labels_list.append(labels)
    features_tensor = torch.cat(features_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    return features_tensor.detach().cpu(), labels_tensor.detach().cpu()

def group_features_by_class(features, labels):
    class_features = defaultdict(list)
    for i, label in enumerate(labels):
        class_features[label.item()].append(features[i])
    for label in class_features:
        class_features[label] = torch.stack(class_features[label])
    class_features_dict = {}
    sorted_labels = sorted(class_features)
    for label in sorted_labels:
        class_features_dict[label] = class_features[label]
    feature_labels = []
    for label in sorted_labels:
        feature_labels.extend([label] * class_features_dict[label].size(0))
    all_sorted_features = torch.cat(list(class_features_dict.values()), dim=0)
    return all_sorted_features, feature_labels, class_features_dict

def compute_euclidean_distance_matrix(features):
    diff = features.unsqueeze(1) - features.unsqueeze(0)
    distance_matrix = torch.norm(diff, p=2, dim=2)
    return distance_matrix

def compute_manhattan_distance_matrix(features):
    diff = features.unsqueeze(1) - features.unsqueeze(0)
    distance_matrix = torch.norm(diff, p=1, dim=2)
    return distance_matrix

def compute_cosine_similarity_matrix(features):
    normalised_features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(normalised_features, normalised_features.t())
    similarity_matrix.mul(1000).round_().div_(1000)
    return similarity_matrix

def plot_similarity_heatmap(sorted_class_features, labels, save_folder, epoch):
    if epoch == 0:
        epoch = 'end'
    else:
        epoch = str(epoch)
    total_number_samples = len(labels)
    if total_number_samples > 5000:
        sorted_class_features = sorted_class_features[::10, :]
        labels = labels[::10]
    cosine_similarity_matrix = compute_cosine_similarity_matrix(sorted_class_features)
    cosine_similarity_np = cosine_similarity_matrix.cpu().numpy()
    abs_cosine_similarity_np = np.abs(cosine_similarity_np)
    tick_positions = []
    tick_labels = []
    current_label = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            mid_point = (start_idx + i - 1) // 2
            tick_positions.append(mid_point)
            tick_labels.append(current_label)
            current_label = labels[i]
            start_idx = i
    mid_point = (start_idx + len(labels) - 1) // 2
    tick_positions.append(mid_point)
    tick_labels.append(current_label)
    file_path = os.path.join(save_folder, 'figures', 'Feature_Sim')
    os.makedirs(file_path, exist_ok=True)
    def save_heatmap(matrix, metric_name):
        plt.figure(figsize=(6, 5))
        ax = sns.heatmap(
            matrix,
            cmap='Reds',
            annot=False,
            xticklabels=False,
            yticklabels=False,
        )
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1.2)
        fig = ax.get_figure()
        cax = None
        for axes in fig.axes:
            if axes != ax:
                cax = axes
                break
        if cax is not None:
            cax.tick_params(labelsize=20)
        file_path_and_name = os.path.join(file_path, f'{metric_name}_{epoch}.png')
        plt.savefig(file_path_and_name, dpi=300)
        plt.close()
    save_heatmap(cosine_similarity_np, 'cosine_similarity')
    save_heatmap(abs_cosine_similarity_np, 'cosine_similarity_abs')
    np.save(os.path.join(file_path, f'cosine_similarity_{epoch}.npy'), cosine_similarity_np)
    np.save(os.path.join(file_path, f'cosine_similarity_abs_{epoch}.npy'), abs_cosine_similarity_np)

def compute_pca_singular_values(features):
    pca = PCA()
    pca.fit(features)
    return pca.singular_values_

from matplotlib.ticker import MaxNLocator
def plot_pca_singular_values(all_class_features, classwise_feature_dict, save_folder, epoch):
    epoch = str(epoch) if epoch != 0 else 'end'
    all_features_singular_values = compute_pca_singular_values(all_class_features)
    num_components = all_class_features.shape[1]
    overall_color = '#FFB6C1'
    point_outline_color = '#DC143C'
    point_outline_size = 4
    plt.figure(figsize=(6, 5))
    plt.tick_params(labelsize=18)
    x = list(range(1, num_components + 1))
    y = all_features_singular_values[:num_components]
    plt.plot(x, y, '-o', color=overall_color, markersize=4)
    plt.plot(x, y, linestyle='None', marker='o', color=point_outline_color, markersize=point_outline_size)
    plt.xlabel('Components', fontsize=20)
    plt.ylabel('Singular Values', fontsize=20)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    pca_figures_path = os.path.join(save_folder, 'figures', 'PCA')
    os.makedirs(pca_figures_path, exist_ok=True)
    plt.savefig(os.path.join(pca_figures_path, f'PCA_overall_{epoch}.png'), dpi=300)
    plt.close()
    np.savetxt(
        os.path.join(pca_figures_path, f'PCA_overall_singular_values_{epoch}.txt'),
        all_features_singular_values,
        fmt='%.6f'
    )
    class_singular_values_dict = {
        class_label: compute_pca_singular_values(class_features)
        for class_label, class_features in classwise_feature_dict.items()
    }
    class_color = '#6495ED'
    in_point_outline_color = '#00008B'
    in_point_outline_size = 4
    plt.figure(figsize=(6, 5))
    plt.tick_params(labelsize=18)
    for class_label, singular_values in class_singular_values_dict.items():
        x_c = list(range(1, len(singular_values) + 1))
        y_c = singular_values
        plt.plot(x_c, y_c, '-o', color=class_color, alpha=0.7, markersize=4)
        plt.plot(x_c, y_c, linestyle='None', marker='o', color=in_point_outline_color, markersize=in_point_outline_size)
    plt.xlabel('Components', fontsize=20)
    plt.ylabel('Singular Values', fontsize=20)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(pca_figures_path, f'PCA_individual_{epoch}.png'), dpi=300)
    plt.close()
    combined_sv_csv = os.path.join(pca_figures_path, f'PCA_all_class_singular_values_{epoch}.csv')
    with open(combined_sv_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for class_label, singular_values in class_singular_values_dict.items():
            sv_list = [f"{sv:.6f}" for sv in singular_values]
            writer.writerow([class_label] + sv_list)
    subset_count = min(100, len(all_features_singular_values))
    subset_overall = all_features_singular_values[:subset_count]
    plt.figure(figsize=(6, 5))
    plt.tick_params(labelsize=18)
    x_s = list(range(1, subset_count + 1))
    y_s = subset_overall
    plt.plot(x_s, y_s, '-o', color=overall_color, markersize=4)
    plt.plot(x_s, y_s, linestyle='None', marker='o', color=point_outline_color, markersize=point_outline_size)
    plt.xlabel('Components', fontsize=20)
    plt.ylabel('Singular Values', fontsize=20)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(pca_figures_path, f'PCA_overall_subset_{epoch}.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.tick_params(labelsize=18)
    for class_label, singular_values in class_singular_values_dict.items():
        subset_vals = singular_values[:subset_count]
        x_cs = list(range(1, len(subset_vals) + 1))
        y_cs = subset_vals
        plt.plot(x_cs, y_cs, '-o', color=class_color, alpha=0.7, markersize=4)
        plt.plot(x_cs, y_cs, linestyle='None', marker='o', color=in_point_outline_color, markersize=in_point_outline_size)
    plt.xlabel('Components', fontsize=20)
    plt.ylabel('Singular Values', fontsize=20)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(pca_figures_path, f'PCA_individual_subset_{epoch}.png'), dpi=300)
    plt.close()
    
def pca_analysis_and_plot(all_features_tensor, class_features_dict, save_folder, epoch):
    if isinstance(all_features_tensor, torch.Tensor):
        all_features_tensor = all_features_tensor.cpu().numpy()
    for class_label in class_features_dict:
        if isinstance(class_features_dict[class_label], torch.Tensor):
            class_features_dict[class_label] = class_features_dict[class_label].cpu().numpy()
    plot_pca_singular_values(all_features_tensor, class_features_dict, save_folder, epoch)

def plot_tSNE(features, labels, save_folder, epoch):
    labels = np.array(labels)
    features_np = features.cpu().numpy()
    data_folder = os.path.join(save_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)
    np.save(os.path.join(data_folder, f'features_{epoch}.npy'), features_np)
    np.save(os.path.join(data_folder, f'labels_{epoch}.npy'), labels)
    num_classes = len(np.unique(labels))
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features_np)
    cmap = cm.get_cmap('Reds', num_classes)
    boundaries = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap=cmap, norm=norm, s=10
    )
    plt.xticks([])
    plt.yticks([])
    out_dir = os.path.join(save_folder, 'figures', 'tSNE')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f't-SNE_visualisation_{epoch}.png'), dpi=300)
    plt.close()

def plot_UMAP(features, labels, save_folder, epoch):
    labels = np.array(labels)
    features_np = features.cpu().numpy()
    data_folder = os.path.join(save_folder, 'data')
    os.makedirs(data_folder, exist_ok=True)
    np.save(os.path.join(data_folder, f'features_{epoch}.npy'), features_np)
    np.save(os.path.join(data_folder, f'labels_{epoch}.npy'), labels)
    num_classes = len(np.unique(labels))
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(features_np)
    cmap = cm.get_cmap('Reds', num_classes)
    boundaries = np.arange(num_classes + 1) - 0.5
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=labels, cmap=cmap, norm=norm, s=10
    )
    cbar = plt.colorbar(
        sc, boundaries=boundaries, ticks=range(num_classes),
        spacing='proportional', drawedges=False
    )
    cbar.outline.set_visible(False)
    if hasattr(cbar, 'solids'):
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_linewidth(0)
    cbar.ax.tick_params(axis='y', which='both', length=0)
    plt.xticks([])
    plt.yticks([])
    out_dir = os.path.join(save_folder, 'figures', 'UMAP')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'umap_visualisation_{epoch}.png'), dpi=300)
    plt.close()

def feature_similarity_analysis(features_tensor, labels_tensor, save_folder, epoch=0):
    if epoch == 0:
        epoch = 'end'
    else:
        epoch = str(epoch)
    sorted_class_features, sorted_labels, class_features_dict = group_features_by_class(features_tensor, labels_tensor)
    plot_similarity_heatmap(sorted_class_features, sorted_labels, save_folder, epoch)
    pca_analysis_and_plot(sorted_class_features, class_features_dict, save_folder, epoch)
