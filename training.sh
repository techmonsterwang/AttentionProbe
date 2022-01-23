#!/bin/bash
# original dataset: cifar, wild dataset: imagenet
root_cifar10='/database/cifar10/'
root_cifar100='/database/cifar100/'
root_wild='/database/imagenet/train/'
selected_cifar10='/selected/cifar10/'
selected_cifar100='/selected/cifar100/'
teacher_cifar10='/output/cifar10/teacher/checkpoint.pth'
teacher_cifar100='/output/cifar100/teacher/checkpoint.pth'
output_student_cifar10='/output/cifar10/'
output_student_cifar100='/output/cifar100/'
# CIFAR10
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-train.py --dataset cifar10 --data_cifar $root_cifar10 --data_imagenet $root_wild --num_select 650000 --teacher_dir $teacher_cifar10 --selected_file $selected_cifar10 --output_dir $output_student_cifar10 --nb_classes 10 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist 
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-train.py --dataset cifar100 --data_cifar $root_cifar100 --data_imagenet $root_wild --num_select 650000 --teacher_dir $teacher_cifar100 --selected_file $selected_cifar100 --output_dir $output_student_cifar100 --nb_classes 100 --lr_S 8.5e-4 --attnprobe_sel --attnprobe_dist 


# original dataset: imagenet, wild dataset: flicker1m
root_imagenet='/database/imagenet/'
root_wild='/database/flicker1m/train/'
selected_imagenet='/selected/imagenet/'
teacher_imagenet='/output/imagenet/teacher/deit_base_patch16_224-b5f2ef4d.pth'
output_student_imagenet='/output/imagenet/'
# imagenet
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-imagenet.py --dataset imagenet --data_cifar $root_imagenet --data_imagenet $root_wild --num_select 1000000 --teacher deit_base_patch16_224 --teacher_dir $teacher_imagenet --selected_file $selected_imagenet --output_dir $output_student_imagenet --nb_classes 1000 --pos_num 129 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist


# original dataset: tinyimagenet, wild dataset: flicker1m
root_tinyimagenet='/database/tiny-imagenet/'
root_wild='/database/flicker1m/'
selected_tinyimagenet='/selected/tinyimagenet/'
teacher_tinyimagenet='/output/tinyimagenet/teacher/checkpoint.pth'
output_student_tinyimagenet='/output/tinyimagenet/'
# tinyimagenet
CUDA_VISIBLE_DEVICES=0 python DFND_DeiT-imagenet.py --dataset tinyimagenet --data_cifar $root_tinyimagenet --data_imagenet $root_wild --num_select 800000 --teacher deit_small_patch16_224 --teacher_dir $teacher_tinyimagenet --selected_file $selected_tinyimagenet --output_dir $output_student_tinyimagenet --nb_classes 200 --pos_num 50 --lr_S 7.5e-4 --attnprobe_sel --attnprobe_dist 
