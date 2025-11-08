#!/bin/bash

# CIFAR-10 experiments with different eps values
for lcr in 0; do
    echo "Running experiments with lcr=$lcr"

    eps_list=(0.5 3.33 12.81)

    for eps in "${eps_list[@]}"; do
        echo "RL: lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 128 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar10' \
            --save_dir './saved_models/CIFAR10/' \
            --print_freq 50 \
            --transform 'default'
    done
done

echo "All CIFAR-10 experiments complete."



# CIFAR-100 coarse experiments with different eps values
for lcr in 0; do
    echo "Running experiments with lcr=$lcr"

    eps_list=(0.5 4.47 38.18)

    for eps in "${eps_list[@]}"; do
        echo "RL: lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 512 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar100coarse' \
            --save_dir './saved_models/CIFAR100Coarse/' \
            --print_freq 50 \
            --transform 'default'
    done
done

echo "All CIFAR-100 coarse experiments complete."




# CIFAR-100 experiments with different eps values
for lcr in 0; do
    echo "Running experiments with lcr=$lcr"

    eps_list=(0.5 10 34.13)

    for eps in "${eps_list[@]}"; do
        echo "RL: lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.1 \
            --fd 1024 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'resnet18' \
            --data 'cifar100' \
            --save_dir './saved_models/CIFAR100/' \
            --print_freq 50 \
            --transform 'default'
    done
done

echo "All CIFAR-100 experiments complete."



# MNIST experiments with different eps values
for lcr in 0; do
    echo "Running experiments with lcr=$lcr"

    eps_list=(0.5 3.33 6.40)

    for eps in "${eps_list[@]}"; do
        echo "RL: lcr=$lcr, eps=$eps"
        python train_sup_RL.py \
            --eps $eps \
            --epo 500 \
            --lcr $lcr \
            --lr 0.01 \
            --fd 64 \
            --trail '0' \
            --save_fea_ana_freq 500 \
            --arch 'mnistnet' \
            --data 'mnist' \
            --save_dir './saved_models/MNIST/' \
            --print_freq 50 \
            --transform 'test'
    done
done

echo "All MNIST experiments complete."

