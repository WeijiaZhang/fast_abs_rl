#!/usr/bin/env bash

function switch_rewrite() {
	split_=$1
	thre_all=("0.5" "0.6" "0.7" "0.8" "0.9")
	echo "switch_rewrite"
	for thre in ${thre_all[@]}
	do
		nohup python data_analysis.py -in ./output/data_analysis/first_run \
			-out ./output/data_analysis/first_run_switch \
			-suf json \
			-spl ${split_} \
			-thre ${thre} \
			> "logs/analysis/first_run_switch_${thre}_${split_}.log" &
	done
}

function merge_to_one_file() {
	split_all=("val" "test")
	# thre_all=("0.5" "0.6" "0.7" "0.8" "0.9")
	thre_all=("dec")
	echo "merge_to_one_file"
	for split_ in ${split_all[@]}
	do
		for thre in ${thre_all[@]}
		do
			nohup python data_analysis.py -in "./output/data_analysis/first_run_switch_${thre}/${split_}" \
				-out "./output/data_analysis/first_run_switch_merge/first_run_switch_${thre}_${split_}.txt" \
				-suf json \
				> "logs/analysis/first_run_merge_${thre}_${split_}.log" &
		done
	done
}

function eval_rouge() {
	split_all=("val" "test")
	# thre_all=("0.5" "0.6" "0.7" "0.8" "0.9")
	thre_all=("dec")
	echo "eval_rouge"
	for split_ in ${split_all[@]}
	do
		for thre in ${thre_all[@]}
		do
			nohup python my_test_rouge.py -s "./output/data_analysis/first_run_switch_merge/first_run_switch_${thre}_${split_}.txt" \
				-t "./output/${split_}_refs.txt" \
				-spl ${split_} \
				-thre ${thre} \
				> "logs/analysis/first_run_rouge_switch_${thre}_${split_}.log" &
		done
	done
}


order=$1
if [[ ${order} == "switch" ]]; then
    split_g=$2
    switch_rewrite ${split_g}
elif [[ ${order} == "merge" ]]; then
    merge_to_one_file
elif [[ ${order} == "eval" ]]; then
    eval_rouge
else
    echo "Unknown Order (${order}) !!!"
fi
