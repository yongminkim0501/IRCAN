gpu_id=$1

model_path=meta-llama/Llama-2-7b-chat-hf
model_name=llama2-7b-chat


# test IRCAN
enhance_cn_num=change_this_to_the_optimal_parameter_from_validation
enhance_strength=change_this_to_the_optimal_parameter_from_validation

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


# # test IRCAN-CAD
# enhance_cn_num=change_this_to_the_optimal_parameter_from_validation
# enhance_strength=change_this_to_the_optimal_parameter_from_validation

# echo "$model_path $model_name"
# for enhance_cn_num in ${enhance_cn_num_array[@]}; do
#     for enhance_strength in ${enhance_strength_array[@]}; do
#         echo "Evaluating enhanced $model_name with COSE, where enhance_cn_num=${enhance_cn_num} enhance_strength=${enhance_strength}"
#         python src/3_enhance_and_evaluate.py \
#             --model_path $model_path \
#             --data_path data/KRE/COSE/cose_test.json \
#             --cn_dir results/cn \
#             --output_dir eval_results/test_results/COSE/outputs \
#             --metric_dir eval_results/test_results/COSE/metrics \
#             --dataset_name COSE \
#             --model_name $model_name \
#             --gpu_id $gpu_id \
#             --max_seq_length 512 \
#             --enhance_cn_num ${enhance_cn_num} \
#             --enhance_strength ${enhance_strength} \
#             --use_cad_decoding
#     done
# done