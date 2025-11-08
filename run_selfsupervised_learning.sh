#!/bin/bash


for eps in 0.5 3.33 12.81; do
  python train_selfsup.py --eps "$eps" \
                          --arch resnet18ctrl \
                          --fd 128 \
                          --save_dir ./selfsup_saved_models/CIFAR10/ \
                          --transform cifar \
                          --data cifar100 \
                          --aug 50 \
                          --gam1 20 \
                          --gam2 0.05
done




for eps in 0.5 4.47 38.18; do
  python train_selfsup.py --eps "$eps" \
                          --arch resnet18ctrl \
                          --fd 512 \
                          --save_dir ./selfsup_saved_models/CIFAR100coarse/ \
                          --transform cifar \
                          --data cifar100coarse \
                          --aug 50 \
                          --gam1 20 \
                          --gam2 0.05
done




for eps in 0.5 10 34.13; do
  python train_selfsup.py --eps "$eps" \
                          --arch resnet18ctrl \
                          --fd 1024 \
                          --save_dir ./selfsup_saved_models/CIFAR100/ \
                          --transform cifar \
                          --data cifar100 \
                          --aug 50 \
                          --gam1 20 \
                          --gam2 0.05
done


