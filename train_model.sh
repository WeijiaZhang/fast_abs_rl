#!/usr/bin/env bash

# path: [path/to/abstractor/model]
# w2v: [path/to/word2vec/word2vec.128d.226k.bin]
function train_abstractor() {
    path=$1
    w2v=$2
    gpu_idx=$3
    echo "train_abstractor"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train_abstractor.py \
            --path=$1 \
            --w2v=$2 \
            > logs/train_abstractor.log &
}

# path: [path/to/abstractor/model]
# w2v: [path/to/word2vec/word2vec.128d.226k.bin]
function train_extractor_ml() {
    path=$1
    w2v=$2
    gpu_idx=$3
    echo "train_extractor_ml"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train_extractor_ml.py \
            --path=${path} \
            --w2v=${w2v} \
            > logs/train_extractor_ml.log &

}

# path: [path/to/abstractor/model]
# abs_dir: [path/to/abstractor/model]
# ext_dir: [path/to/extractor/model]
function train_full_rl() {
    path=$1
    abs_dir=$2
    ext_dir=$3
    gpu_idx=$4
    echo "train_full_rl"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python train_full_rl.py \
            --path=${path} \
            --abs_dir=${abs_dir} \
            --ext_dir=${ext_dir} \
            > logs/train_full_rl.log &
}

order=$1
path_g=$2
gpu_idx_g=$3
if [[ ${order} == "abs" ]]; then
    w2v_g=$4
    train_abstractor ${path_g}  ${w2v_g}  ${gpu_idx_g}
elif [[ ${order} == "ext" ]]; then
    w2v_g=$4
    train_extractor_ml ${path_g}  ${w2v_g}  ${gpu_idx_g}
elif [[ ${order} == "full" ]]; then
    abs_dir_g=$4
    ext_dir_g=$5
    train_full_rl ${path_g} ${abs_dir_g} ${ext_dir_g} ${gpu_idx_g}
else
        echo "Unknown Order (${order}) !!!"
fi
