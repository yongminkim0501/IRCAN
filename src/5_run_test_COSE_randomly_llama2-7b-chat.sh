gpu_id=$1

model_path=meta-llama/Llama-2-7b-chat-hf
model_name=llama2-7b-chat


# Erasing the detected context-aware neurons
enhance_cn_num=change_this_to_the_optimal_parameter_from_validation
enhance_strength=0

echo "Evaluating enhanced $model_name with COSE, where enhance_cn_num=${enhance_cn_num} enhance_strength=${enhance_strength}"
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


# Enhancing randomly selected neurons
enhance_cn_num=change_this_to_the_optimal_parameter_from_validation
enhance_strength=change_this_to_the_optimal_parameter_from_validation

for i in {1..10}; do
    random_seed=$RANDOM
    let "random_seed %= 100"
    echo "Evaluating enhanced $model_name with COSE, enhance_cn_num=${enhance_cn_num} enhance_strength=${enhance_strength} random_seed=${random_seed}"
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
        --do_random_kn \
        --seed ${random_seed}
done


# Erasing randomly selected neurons
enhance_cn_num=change_this_to_the_optimal_parameter_from_validation
enhance_strength=0

for i in {1..10}; do
    random_seed=$RANDOM
    let "random_seed %= 100"
    echo "Evaluating enhanced $model_name with COSE, enhance_cn_num=${enhance_cn_num} enhance_strength=${enhance_strength} random_seed=${random_seed}"
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
        --do_random_kn \
        --seed ${random_seed}
done