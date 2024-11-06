import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F


# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def scaled_input(minimum_weight, maximum_weights, batch_size, num_batch):

    num_points = batch_size * num_batch
    step = (maximum_weights - minimum_weight) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(minimum_weight, step * i) for i in range(1, num_points + 1)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)
    max_ig = ig.max() # maximum attribution score
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def get_context_attr(idx, prompt_without_context, prompt_with_context, example, answer_obj, args, model, tokenizer, device):
    # Perform different processing on different models to obtain correct tokenization
    if args.model_name == 'llama2-7b-chat':
        tokens = tokenizer.tokenize(answer_obj)
        gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])
    elif args.model_name == 'llama3-8b-instruct':
        tokens = tokenizer.tokenize(f" {answer_obj}")
        gold_label_id = tokenizer.convert_tokens_to_ids(tokens[0])
    else:
        gold_label_id = tokenizer.convert_tokens_to_ids(answer_obj)
     
    wo_tokenized_inputs = tokenizer(prompt_without_context, return_tensors="pt")
    wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
    wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)
    
    w_tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
    w_input_ids = w_tokenized_inputs["input_ids"].to(device)
    w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)

    # record results
    res_dict = {
        'id': idx,
        'wo_all_ffn_weights': [],
        'w_all_ffn_weights': [],
        'all_attr_gold': [],
    }

    for tgt_layer in range(model.model.config.num_hidden_layers):
        wo_ffn_weights_dict = dict()
        def wo_forward_hook_fn(module, inp, outp):
            wo_ffn_weights_dict['input'] = inp[0]  # inp's type is Tuple

        w_ffn_weights_dict = dict()
        def w_forward_hook_fn(module, inp, outp):
            w_ffn_weights_dict['input'] = inp[0]
        # ========================== get weights when there is no context in the prompt =========================
        wo_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(wo_forward_hook_fn)
        with torch.no_grad():
            wo_outputs = model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
        wo_ffn_weights = wo_ffn_weights_dict['input'].detach()
        wo_ffn_weights = wo_ffn_weights[:, -1, :]
        wo_logits = wo_outputs.logits[:, -1, :]
        wo_hook.remove()
        
        # =========================== get weights when there is context in the prompt ============================
        w_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_hook(w_forward_hook_fn)
        with torch.no_grad():
            w_outputs = model(input_ids=w_input_ids, attention_mask=w_attention_mask)
        w_ffn_weights = w_ffn_weights_dict['input'].detach()
        w_ffn_weights = w_ffn_weights[:, -1, :]
        w_logits = w_outputs.logits[:, -1, :]
        w_hook.remove()
        
        wo_ffn_weights.requires_grad_(True)
        w_ffn_weights.requires_grad_(True)
        scaled_weights, weights_step = scaled_input(wo_ffn_weights, w_ffn_weights, args.batch_size, args.num_batch)
        scaled_weights.requires_grad_(True)

        # integrated grad at the gold label for each layer
        ig_gold = None
        for batch_idx in range(args.num_batch):       
            grad = None
            all_grads = None
            for i in range(0, args.batch_size, args.batch_size_per_inference):
                batch_weights = scaled_weights[i: i + args.batch_size_per_inference] # (batch_size_per_inference, ffn_size)
                batch_w_weights = w_ffn_weights.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, ffn_size)
                batched_w_input_ids = w_input_ids.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)
                batched_w_attention_mask = w_attention_mask.repeat(args.batch_size_per_inference, 1) # (batch_size_per_inference, seq_len)

                def i_forward_hook_change_fn(module, inp):
                    inp0 = inp[0].detach()
                    inp0[:, -1, :] = batch_weights
                    inp = tuple([inp0])
                    return inp
                change_hook = model.model.layers[tgt_layer].mlp.down_proj.register_forward_pre_hook(i_forward_hook_change_fn)
                
                outputs = model(input_ids=batched_w_input_ids, attention_mask=batched_w_attention_mask)  # (batch, n_vocab), (batch, ffn_size)
                # compute final grad at a layer at the last position
                tgt_logits = outputs.logits[:, -1, :] # (batch_size_per_inference, n_vocab)
                tgt_probs = F.softmax(tgt_logits, dim=1) # (batch_size_per_inference, n_vocab)
                grads_i = torch.autograd.grad(torch.unbind(tgt_probs[:, gold_label_id]), w_ffn_weights, retain_graph=True) # grads_i[0].shape: (1, ffn_size)
                del tgt_probs

                change_hook.remove()
                
                all_grads = grads_i[0] if all_grads is None else torch.cat((all_grads, grads_i[0]), dim=0)
            grad = all_grads.sum(dim=0)
            ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)
            
        ig_gold = ig_gold * weights_step  # (ffn_size)
        res_dict['wo_all_ffn_weights'].append(wo_ffn_weights.squeeze().tolist())
        res_dict['w_all_ffn_weights'].append(w_ffn_weights.squeeze().tolist())
        res_dict['all_attr_gold'].append(ig_gold.tolist())
        
    return res_dict


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file. ")
    parser.add_argument("--model_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Path to local pretrained model or model identifier from huggingface.co/models or local.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the results will be written.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name as output prefix to indentify each running of experiment.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpu_id",
                        type=str,
                        default='0',
                        help="available gpu id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # parameters about integrated grad
    parser.add_argument("--batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=1,
                        type=int,
                        help="Num batch of an example.")
    parser.add_argument("--batch_size_per_inference",
                        default=2,
                        type=int,
                        choices=[1,2,4,5,10,20],
                        help="The batch size for each inference, you can choose an appropriate value from 1, 2, 4, 5, 10, 20 according to the model size and CUDA memory.")


    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # TODO: To implement multi gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))


    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    output_prefix = f"{args.dataset_name}-{args.model_name}-context"
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # prepare dataset
    with open(args.data_path, "r", encoding="utf-8") as fin:
        lines = fin.read()
        data = json.loads(lines)


    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, device_map=device)
    
    if "llama" in args.model_name:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    

    tic = time.perf_counter()
    count = 0
    print(f"Start process, dataset size: {len(data)}")
    with jsonlines.open(os.path.join(args.output_dir, output_prefix + '.rlt.jsonl'), 'w') as fw:
        for idx, example in enumerate(tqdm(data)):
            answer_obj = example["candidate"]
            choices = example["choices"]
            query = example["question"]
            context = example["negative_context"]
            choice_str = ""
            for choice_idx, choice in enumerate(choices):
                if choice_idx != len(choices)-1:
                    choice_str += chr(ord('A')+choice_idx) + ". " + choice.strip() + " "
                else:
                    choice_str += chr(ord('A')+choice_idx) + ". " + choice.strip() + "\n"
            
            if args.model_name == 'llama2-7b-chat':
                prompt_without_context = f"Choose the correct option to answer the following question: \n{query}\n{choice_str}Answer: "
                prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{query}\n{choice_str}Answer: "
            if args.model_name == 'llama2-13b-chat':
                prompt_without_context = f"Choose the correct option to answer the following question: \n{query}\n{choice_str}Answer: \n"
                prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{query}\n{choice_str}Answer: \n"
            if args.model_name == 'llama3-8b-instruct':
                prompt_without_context = f"Choose the correct option to answer the following question: \n{query}\n{choice_str}The correct answer is "
                prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{query}\n{choice_str}The correct answer is "
            if args.model_name == 'gemma-2b-it':
                prompt_without_context = f"Choose the correct option to answer the following question: \n{query}\n{choice_str}Answer: \nThe correct option is \n**"
                prompt_with_context = f"Choose the correct option to answer the following question: \n{context}\n{query}\n{choice_str}Answer: \nThe correct option is \n**"
    
            if idx == 0:
                print(prompt_with_context)
                prompt_dict = {
                    "prompt_without_context" : prompt_without_context,
                    "prompt_with_context" : prompt_with_context,
                }
                json.dump(prompt_dict, open(os.path.join(args.output_dir, output_prefix + '.prompt.json'), 'w'), indent=2)
            
            res_dict = get_context_attr(idx, prompt_without_context, prompt_with_context, example, answer_obj, args, model, tokenizer, device)
            res_dict["all_attr_gold"] = convert_to_triplet_ig(res_dict["all_attr_gold"])
            
            fw.write(res_dict)
            count += 1
    
    print(f"Saved in {os.path.join(args.output_dir, output_prefix + '.rlt.jsonl')}")

    toc = time.perf_counter()
    time_str = f"***** Costing time: {toc - tic:0.4f} seconds *****"
    print(time_str)
    json.dump(time_str, open(os.path.join(args.output_dir, output_prefix + '.time.json'), 'w'))


if __name__ == "__main__":
    main()