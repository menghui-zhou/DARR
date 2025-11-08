import os
import logging
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import random
from collections import defaultdict

def preprocess_for_matrix_entropy(matrix):
    min_value = np.min(matrix)
    if min_value < 0:
        matrix = matrix - min_value
    return matrix

def calculate_matrix_entropy(matrix):
    matrix = matrix.cpu().numpy()
    matrix = preprocess_for_matrix_entropy(matrix)
    flattened_matrix = matrix.flatten()
    total_sum = np.sum(flattened_matrix)
    if total_sum == 0:
        raise ValueError("The sum of matrix element is zero, can not normalize")
    probability_distribution = flattened_matrix / total_sum
    non_zero_probs = probability_distribution[probability_distribution > 0]
    entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
    return entropy

def calculate_variance_information(feature_matrix):
    feature_variances = torch.var(feature_matrix, dim=0, unbiased=True)
    variance_information = torch.sum(feature_variances).item()
    return variance_information

def calculate_optimal_coding_length(Z, args):
    epsilon = 1
    m, p = Z.shape
    scalar = p / (m * epsilon)
    if Z.shape[0] > Z.shape[1]:
        Z = Z.T
    I = torch.eye(Z.shape[0])
    logdet = torch.logdet(I + scalar * Z.matmul(Z.T)) / 2.0
    return logdet / torch.log(torch.tensor(2.0))

def calculate_spectral_entropy(matrix):
    matrix = matrix.cpu().numpy()
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_values = singular_values / np.sum(singular_values)
    non_zero_singular_values = singular_values[singular_values > 0]
    spectral_entropy = -np.sum(non_zero_singular_values * np.log(non_zero_singular_values))
    return spectral_entropy

def calculate_mutual_coherence(matrix):
    matrix = matrix.cpu().numpy()
    matrix = matrix.T
    column_norm = np.linalg.norm(matrix, axis=0, keepdims=True)
    normalized_columns = matrix / column_norm
    coherence_matrix = np.abs(np.dot(normalized_columns.T, normalized_columns))
    np.fill_diagonal(coherence_matrix, 0)
    mutual_coherence = np.max(coherence_matrix)
    return mutual_coherence

def calculate_kernel_entropy(matrix, sigma=0.1):
    matrix = matrix.cpu().numpy()
    n = matrix.shape[0]
    kernel_sum = 0
    p_values = np.zeros(n)
    for i in range(n):
        pairwise_distances = np.linalg.norm(matrix - matrix[i], axis=1)
        kernel_values = np.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))
        p_values[i] = np.sum(kernel_values)
        kernel_sum += p_values[i]
    p_values /= kernel_sum
    non_zero_p = p_values[p_values > 0]
    kernel_entropy = -np.sum(non_zero_p * np.log(non_zero_p))
    return kernel_entropy

def calculate_information_fractal_dimension(matrix, epsilon_values=None):
    if epsilon_values is None:
        epsilon_values = [0.1, 0.05, 0.01, 0.005]
    matrix = matrix.cpu().numpy()
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    entropies = []
    for epsilon in epsilon_values:
        grid_size = int(1 / epsilon)
        sub_matrices = [matrix[i:i + grid_size, j:j + grid_size]
                        for i in range(0, matrix.shape[0], grid_size)
                        for j in range(0, matrix.shape[1], grid_size) if
                        i + grid_size <= matrix.shape[0] and j + grid_size <= matrix.shape[1]]
        p_values = [np.sum(sub_mat > 0) / np.size(matrix) for sub_mat in sub_matrices]
        p_values = np.array(p_values)
        p_values = p_values[p_values > 0]
        entropy = -np.sum(p_values * np.log(p_values))
        entropies.append(entropy)
    log_eps = np.log(1 / np.array(epsilon_values))
    slope, _ = np.polyfit(log_eps, entropies, 1)
    return slope

def calculate_information_quantification(features, labels, args):
    class_features = defaultdict(list)
    for i, label in enumerate(labels):
        class_features[label.item()].append(features[i])
    for label in class_features:
        class_features[label] = torch.stack(class_features[label])
    class_features_dict = {}
    sorted_labels = sorted(class_features)
    for label in sorted_labels:
        class_features_dict[label] = class_features[label]
    list_optimal_coding_length = []
    list_spectral_entropies = []
    list_var_information = []
    overall_coding_length = calculate_optimal_coding_length(features, args)
    overall_spectral_entropy = calculate_spectral_entropy(features)
    overall_variance_information = calculate_variance_information(features)
    for label in sorted_labels:
        coding_length = calculate_optimal_coding_length(class_features_dict[label], args)
        list_optimal_coding_length.append(coding_length)
        spectral_entropy = calculate_spectral_entropy(class_features_dict[label])
        list_spectral_entropies.append(spectral_entropy)
        variance_information = calculate_variance_information(class_features_dict[label])
        list_var_information.append(variance_information)
    average_optimal_coding_length = np.mean(list_optimal_coding_length)
    average_spectral_entropy = np.mean(list_spectral_entropies)
    average_variance_information = np.mean(list_var_information)
    return (overall_coding_length.item(), overall_spectral_entropy, overall_variance_information,
            average_optimal_coding_length, average_spectral_entropy, average_variance_information)
