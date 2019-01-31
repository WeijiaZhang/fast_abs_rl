#!/usr/bin/env bash


# path: [path/to/save/decoded/files]
# model: [path/to/pretrained]
# beam: [beam_size]
# mode: [--test/--val]
function decode_full_model() {
    path=$1
    model_dir=$2
    mode=$3
    beam=$4
    gpu_idx=$5
    echo "decode_full_model"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python decode_full_model.py --${mode} \
                --path=${path} \
                --model_dir=${model_dir} \
                --beam=${beam} \
                > "logs/decode_full_model_first_run_${mode}.log" &
}

function decode_pretrain_model() {
    path=$1
    abs_dir=$2
    ext_dir=$3
    mode=$4
    beam=$5
    gpu_idx=$6
    echo "decode_pretrain_model"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python decode_pretrain_model.py --${mode} \
                --path=${path} \
                --abs_dir=${abs_dir} \
                --ext_dir=${ext_dir} \
                --beam=${beam} \
                > "logs/decode_pretrain_model_first_run_${mode}.log" &
}

function decode_full_plus_analysis() {
    path=$1
    model_dir=$2
    method=$3
    mode=$4
    beam=$5
    gpu_idx=$6
    echo "decode_full_plus_analysis"
    CUDA_VISIBLE_DEVICES=${gpu_idx} nohup python decode_full_plus_analysis.py --${mode} \
                --method=${method} \
                --path=${path} \
                --model_dir=${model_dir} \
                --beam=${beam} \
                > "logs/decode_full_plus_analysis_${method}_${mode}.log" &
}

order=$1
path_g=$2
mode_g=$3
beam_g=$4
gpu_idx_g=$5
if [[ ${order} == "full" ]]; then
    model_dir_g=$6
    decode_full_model ${path_g} ${model_dir_g} ${mode_g} ${beam_g} ${gpu_idx_g}
elif [[ ${order} == "analysis" ]]; then
    method_g=$6
    model_dir_g=$7
    decode_full_plus_analysis ${path_g} ${model_dir_g} ${method_g} ${mode_g} ${beam_g} ${gpu_idx_g}
elif [[ ${order} == "pretrain" ]]; then
    abs_dir_g=$6
    ext_dir_g=$7
    decode_pretrain_model ${path_g} ${abs_dir_g} ${ext_dir_g} ${mode_g} ${beam_g} ${gpu_idx_g}
else
    echo "Unknown Order (${order}) !!!"
fi