import os
from tqdm import tqdm
import json
import numpy as np
import argparse
from collections import Counter


threshold_ratio = 0.2


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])  # '5@1094'


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def stat(cn_bag_list, pos_type, type):
    ave_len = 0
    for cn_bag in cn_bag_list:
        ave_len += len(cn_bag)
    ave_len /= len(cn_bag_list)
    print(f'{type}\'s {pos_type} has on average {ave_len} imp pos. ')
    

def analysis_context_file(filename, metric='all_attr_gold'):
    print(f'===========> parsing important position in {os.path.join(args.result_dir, filename)}')
    rlts_bag = []
    with open(os.path.join(args.result_dir, filename), 'r') as fr:
        for idx, line in enumerate(tqdm(fr.readlines())):
            try:
                example = json.loads(line)
                rlts_bag.append(example)
            except Exception as e:
                print(f"Exception {e} happend in line {idx}")

    print(f"Total examples: {len(rlts_bag)}")
    
    cn_bag_list = []
    for rlt in rlts_bag:
        metric_triplets = rlt[metric]
        metric_triplets.sort(key=lambda x: x[2], reverse=True)
        cn_bag = metric_triplets[:20]
        cn_bag_list.append(cn_bag)
    return cn_bag_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir",
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")
    parser.add_argument("--cn_dir",
                        type=str,
                        required=True,
                        help="The directory where context neurons will be stored.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        required=True,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_name",
                        type=str,
                        default=None,
                        required=True,
                        help="Model name as output prefix to indentify each running of experiment.")

    args = parser.parse_args()

    if not os.path.exists(args.cn_dir):
        os.makedirs(args.cn_dir)
        
    filename = f"{args.dataset_name}-{args.model_name}-context.rlt.jsonl"
    cn_bag_list= analysis_context_file(filename)

    type = filename.split('.')[0].split('-')[-1]
    stat(cn_bag_list, 'cn_bag', type)
    output_cn_path = os.path.join(args.cn_dir, f'{args.dataset_name}-{args.model_name}-cn_bag-{type}.json')
    with open(output_cn_path, 'w') as fw:
        json.dump(cn_bag_list, fw, indent=2)
    print(f"save cns in {output_cn_path}")
