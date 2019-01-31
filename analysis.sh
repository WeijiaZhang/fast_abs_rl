#!/usr/bin/env bash

SPLIT_ALL=("val" "test")
# THRE_ALL=("0.5" "0.6" "0.7" "0.8" "0.9")
# THRE_ALL=("ext_pred" "ext_gt")
THRE_ALL=("ext" "ext_ff" "ext_rl")

function switch_rewrite() {
	echo "switch_rewrite"
	for split_ in ${SPLIT_ALL[@]}
	do
	for thre in ${THRE_ALL[@]}
		do
			nohup python data_analysis.py -func switch_rewrite \
				-in ./output/data_analysis/first_run \
				-out ./output/data_analysis/first_run_switch \
				-suf json \
				-spl ${split_} \
				-thre ${thre} \
				> "logs/analysis/first_run_switch_${thre}_${split_}.log" &
		done
	done
}

function merge_to_one_file() {
	echo "merge_to_one_file"
	for split_ in ${SPLIT_ALL[@]}
	do
		for thre in ${THRE_ALL[@]}
		do
			nohup python data_analysis.py -func merge_to_one_file \
				-in "./output/data_analysis/first_run_${thre}/${split_}" \
				-out "./output/data_analysis/first_run_ext_merge/first_run_${thre}_${split_}.txt" \
				-suf json \
				> "logs/analysis/first_run_merge_${thre}_${split_}.log" &
		done
	done
}

function eval_rouge() {
	echo "eval_rouge"
	for split_ in ${SPLIT_ALL[@]}
	do
		for thre in ${THRE_ALL[@]}
		do
			nohup python my_test_rouge.py -s "./output/data_analysis/first_run_ext_merge/first_run_${thre}_${split_}.txt" \
				-t "./output/${split_}_refs.txt" \
				-spl ${split_} \
				-thre ${thre} \
				> "logs/analysis/first_run_rouge_${thre}_${split_}.log" &
		done
	done
}


order=$1
if [[ ${order} == "switch" ]]; then
    switch_rewrite
elif [[ ${order} == "merge" ]]; then
    merge_to_one_file
elif [[ ${order} == "eval" ]]; then
    eval_rouge
else
    echo "Unknown Order (${order}) !!!"
fi
