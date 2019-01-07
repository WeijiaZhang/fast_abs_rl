#!/usr/bin/env bash


# path: [path/to/save/decoded/files]
# model: [path/to/pretrained]
# beam: [beam_size]
# mode: [--test/--val]
function decode_full_model() {
    path=$1
    model_dir=$2
    beam=$3
    gpu_idx=$4
    echo "decode_full_model"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python decode_full_model.py --test \
                --path=${path} \
                --model_dir=${model_dir} \
                --beam=${beam} \
                > logs/decode_full_model_acl.log &
}

order=$1
path_g=$2
model_dir_g=$3
beam_g=$4
gpu_idx_g=$5
if [[ ${order} == "full" ]]; then
    decode_full_model ${path_g} ${model_dir_g} ${beam_g} ${gpu_idx_g}
else
        echo "Unknown Order (${order}) !!!"
fi