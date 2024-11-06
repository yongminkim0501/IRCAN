gpu_id=$1

# test origin model
enhance_strength=1
enhance_cn_num=1

model_path=meta-llama/Llama-2-7b-hf
model_name=llama2-7b

echo "Evaluating origin $model_name with Memo..."
python src/3_enhance_and_evaluate.py \
    --model_path $model_path \
    --data_path data/MemoTrap/memo-trap_classification_test.jsonl \
    --model_name $model_name \
    --dataset_name Memo \
    --cn_dir results/cn \
    --output_dir eval_results/test_results/Memo/outputs \
    --metric_dir eval_results/test_results/Memo/metrics \
    --gpu_id $gpu_id \
    --max_seq_length 128 \
    --enhance_cn_num ${enhance_cn_num} \
    --enhance_strength ${enhance_strength}


# test CAD
echo "Evaluating enhanced $model_name with Memo, where enhance_cn_num=${enhance_cn_num} enhance_strength=${enhance_strength}"
python src/3_enhance_and_evaluate.py \
    --model_path $model_path \
    --data_path data/MemoTrap/memo-trap_classification_test.jsonl \
    --model_name $model_name \
    --dataset_name Memo \
    --cn_dir results/cn \
    --output_dir eval_results/test_results/Memo/outputs \
    --metric_dir eval_results/test_results/Memo/metrics \
    --gpu_id $gpu_id \
    --max_seq_length 128 \
    --enhance_cn_num ${enhance_cn_num} \
    --enhance_strength ${enhance_strength} \
    --use_cad_decoding