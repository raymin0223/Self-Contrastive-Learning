#!/bin/bash

seed_list="2022"
dataset="cifar100"
methods="SelfCon"
model="resnet18"
selfcon_arch="resnet"
selfcon_size="fc"
selfcon_pos="[False,True,False]"
bsz="1024"
lr="0.5"
label="True"
multiview="False"

                          
for seed in $seed_list
do
    for data in $dataset
    do
        for method in $methods
        do
            for arch in $selfcon_arch
            do
                for sz in $selfcon_size
                do
                    for pos in $selfcon_pos
                    do
                        python main_represent.py --exp_name "${arch}_${sz}_${pos}" \
                          --method $method \
                          --dataset $data \
                          --seed $seed \
                          --model $model \
                          --selfcon_pos $pos \
                          --selfcon_arch $arch \
                          --selfcon_size $sz \
                          --batch_size $bsz \
                          --learning_rate $lr \
                          --temp 0.1 \
                          --cosine 

                        python main_linear.py --batch_size 512 \
                          --dataset $data \
                          --model $model \
                          --selfcon_pos $pos \
                          --selfcon_arch $arch \
                          --selfcon_size $sz \
                          --learning_rate 5 \
                          --ckpt ./save/representation/${method}/${data}_models/${method}_${data}_${model}_lr_${lr}_multiview_${multiview}_label_${label}_decay_0.0001_bsz_${bsz}_temp_0.1_seed_${seed}_cosine_warm_${arch}_${sz}_${pos}/last.pth
                    done
                done
            done
        done
    done
done
