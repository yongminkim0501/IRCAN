gpu_id=$1

model_path=meta-llama/Llama-2-7b-chat-hf
model_name=llama2-7b-chat

# Grid search for model with enhanced neurons
echo "$model_path $model_name"
for enhance_cn_num in {1..16}
do
    for enhance_strength in {2..20}
    do
        echo "Evaluating enhanced $model_name with COSE, where enhance_cn_num=${enhance_cn_num}, enhance_strength=${enhance_strength}"
        python src/3_enhance_and_evaluate.py \
            --model_path $model_path \
            --data_path data/KRE/COSE/cose_dev.jsonl \
            --model_name $model_name \
            --dataset_name COSE \
            --cn_dir results/cn \
            --output_dir eval_results/dev_results/COSE/outputs \
            --metric_dir eval_results/dev_results/COSE/metrics \
            --gpu_id $gpu_id \
            --max_seq_length 512 \
            --enhance_cn_num ${enhance_cn_num} \
            --enhance_strength ${enhance_strength}
    done
done


# Grid search for CAD model with enhanced neurons
echo "$model_path $model_name"
for enhance_cn_num in {1..16}
do
    for enhance_strength in {2..20}
    do
        echo "Evaluating enhanced $model_name with COSE, where enhance_cn_num=${enhance_cn_num}, enhance_strength=${enhance_strength}"
        python src/3_enhance_and_evaluate.py \
            --model_path $model_path \
            --data_path data/KRE/COSE/cose_dev.jsonl \
            --model_name $model_name \
            --dataset_name COSE \
            --cn_dir results/cn \
            --output_dir eval_results/dev_results/COSE/outputs \
            --metric_dir eval_results/dev_results/COSE/metrics \
            --gpu_id $gpu_id \
            --max_seq_length 512 \
            --enhance_cn_num ${enhance_cn_num} \
            --enhance_strength ${enhance_strength} \
            --use_cad_decoding
    done
done