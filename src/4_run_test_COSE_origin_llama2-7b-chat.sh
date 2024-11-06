gpu_id=$1

# test origin model
enhance_strength=1
enhance_cn_num=1

model_path=meta-llama/Llama-2-7b-chat-hf
model_name=llama2-7b-chat

echo "Evaluating origin $model_name with COSE..."
python src/3_enhance_and_evaluate.py \
    --model_path $model_path \
    --data_path data/KRE/COSE/cose_test.json \
    --model_name $model_name \
    --dataset_name COSE \
    --cn_dir results/cn \
    --output_dir eval_results/test_results/COSE/outputs \
    --metric_dir eval_results/test_results/COSE/metrics \
    --gpu_id $gpu_id \
    --max_seq_length 512 \
    --enhance_cn_num ${enhance_cn_num} \
    --enhance_strength ${enhance_strength}


# test CAD
echo "Evaluating origin $model_name with COSE..."
python src/3_enhance_and_evaluate.py \
    --model_path $model_path \
    --data_path data/KRE/COSE/cose_test.json \
    --model_name $model_name \
    --dataset_name COSE \
    --cn_dir results/cn \
    --output_dir eval_results/test_results/COSE/outputs \
    --metric_dir eval_results/test_results/COSE/metrics \
    --gpu_id $gpu_id \
    --max_seq_length 512 \
    --enhance_cn_num ${enhance_cn_num} \
    --enhance_strength ${enhance_strength} \
    --use_cad_decoding