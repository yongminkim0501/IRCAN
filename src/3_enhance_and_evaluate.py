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
from typing import Union, List, Optional
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from utils.enhance_model import enhance_neurons


# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_prompts(example, args):
    """Obtain prompts of an example"""
    if args.dataset_name == "Memo":
        prompt_with_context = example["prompt"]
        prompt_without_context = prompt_with_context.split(": ")[-1]

    if args.dataset_name == "COSE" or args.dataset_name == "ECARE":
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

    prompts_info = {
        "prompt_without_context": prompt_without_context,
        "prompt_with_context": prompt_with_context,
    }
    return prompts_info


def get_information(prompt, example, args, tokenizer):
    """Obtain information of an example"""
    if args.dataset_name == "Memo":
        classes = example["classes"].strip('[').strip(']').split(',')
        classes = [answer.strip().strip("\'").strip(".").strip() for answer in classes]
        answer_index = example["answer_index"]
        trap_index = 1 if example["answer_index"]==0 else 0 
        answer_obj = classes[answer_index]
        trap_obj= classes[trap_index]
        objs = {
            "answer_obj":answer_obj,
            "trap_obj":trap_obj,
        }
        tokens = tokenizer.tokenize(prompt)
        # All templates are simple, almost no one will exceed the length limit.
        if len(tokens) > args.max_seq_length - 2:
            return None
        tokens_info = {
            "tokens": tokens,
            "answer_obj": objs["answer_obj"],
            "trap_obj": objs["trap_obj"],
            "pred_obj": None
        }
    if args.dataset_name == "COSE" or args.dataset_name == "ECARE":
        tokens = tokenizer.tokenize(prompt)
        answer_obj = example["candidate"]
        trap_obj = example["answer"]
        objs = {
            "answer_obj": answer_obj,
            "trap_obj": trap_obj,
        }
        tokens_info = {
            "tokens": tokens,
            "answer_obj": objs["answer_obj"],
            "trap_obj": objs["trap_obj"],
            "pred_obj": None
        }
    return tokens_info


def completion_task_generation(model, tokenizer, args, prompt_with_context, prompt_without_context, device):
    tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)

    if not args.use_cad_decoding: 
        # In model.generate()，if num_beams=1 and do_sample=False, do greedy decoding.
        generate_ids = model.generate(input_ids=input_ids, 
                                        num_beams = 1,
                                        do_sample=False, 
                                        max_new_tokens=10)
        full_prediction = tokenizer.decode(generate_ids[0], skip_special_tokens=True)[len(prompt_with_context):]
        pred_label = full_prediction.strip().split('.')[0]
    else: # use CAD decoding, adapted from https://github.com/hongshi97/CAD
        wo_tokenized_inputs = tokenizer(prompt_without_context, return_tensors="pt")
        wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
        wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)

        max_length = 20
        cur_len = 0
        cad_alpha = 0.5
        
        batch_size = len(wo_input_ids)
        unfinished_sents = input_ids.new(batch_size).fill_(1) # [[1]]
        sent_lengths = input_ids.new(batch_size).fill_(max_length) # [[20]]
        generated_tokens = [[] for _ in range(1)] # [[]]

        while cur_len < max_length:
            wo_outputs = model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
            wo_logits = wo_outputs.logits[:, -1, :]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :] # (batch, n_vocab)
            
            logits = (1 + cad_alpha) * logits - cad_alpha * wo_logits
            # Predict next token according to greedy decoding strategy                        
            next_token = torch.argmax(logits, dim=-1)
            # Handle EOS token and padding
            if tokenizer.eos_token_id is not None and tokenizer.pad_token_id is not None:
                tokens_to_add = next_token * unfinished_sents + (tokenizer.pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token

            # Update input_ids and attention masks for the next forward pass
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, unfinished_sents.unsqueeze(-1)], dim=-1)

            wo_input_ids = torch.cat([wo_input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            wo_attention_mask = torch.cat([wo_attention_mask, unfinished_sents.unsqueeze(-1)], dim=-1)

            cur_len += 1

            # Update generated tokens and check for completion
            for i, token in enumerate(tokens_to_add.tolist()):
                if unfinished_sents[i] == 1:
                    generated_tokens[i].append(token)

            # Check for sentences that are finished
            if tokenizer.eos_token_id is not None:
                eos_in_sents = tokens_to_add == tokenizer.eos_token_id
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                unfinished_sents.mul_((~eos_in_sents).long())

            # Break if all sentences are finished : stop when there is a EOS token in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break
        full_prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        pred_label = full_prediction.strip().split('.')[0]

    return full_prediction, pred_label


def mcq_task_generation(model, tokenizer, args, choices, prompt_with_context, prompt_without_context, device):
    tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)

    if not args.use_cad_decoding:    
        w_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        w_logits = w_outputs.logits[:, -1, :]
        
        final_logits = w_logits
    else: # use CAD decoding
        cad_alpha = 0.5
        
        w_tokenized_inputs = tokenizer(prompt_with_context, return_tensors="pt")
        w_input_ids = w_tokenized_inputs["input_ids"].to(device)
        w_attention_mask = w_tokenized_inputs["attention_mask"].to(device)
        w_outputs = model(input_ids=w_input_ids, attention_mask=w_attention_mask)
        w_logits = w_outputs.logits[:, -1, :]

        wo_tokenized_inputs = tokenizer(prompt_without_context, return_tensors="pt")
        wo_input_ids = wo_tokenized_inputs["input_ids"].to(device)
        wo_attention_mask = wo_tokenized_inputs["attention_mask"].to(device)
        wo_outputs = model(input_ids=wo_input_ids, attention_mask=wo_attention_mask)
        wo_logits = wo_outputs.logits[:, -1, :]

        final_logits = (1 + cad_alpha) * w_logits - cad_alpha * wo_logits

    final_prob = F.softmax(final_logits, dim=-1)
    pred_token_id_from_all = torch.argmax(final_prob, dim=-1).item()
    if args.model_name == 'llama3-8b-instruct':
        pred_label_from_all = tokenizer.decode(pred_token_id_from_all).strip()
    else:
        pred_label_from_all = tokenizer.convert_ids_to_tokens(pred_token_id_from_all)
    
    logits_of_options = []

    for choice_idx, choice in enumerate(choices):
        option = chr(ord('A')+choice_idx)
        if args.model_name == 'llama2-7b-chat':
            tokens = tokenizer.tokenize(option)
            option_id = tokenizer.convert_tokens_to_ids(tokens[0])
        elif args.model_name == 'llama3-8b-instruct':
            tokens = tokenizer.tokenize(" "+option)
            option_id = tokenizer.convert_tokens_to_ids(tokens[0])
        else:
            option_id = tokenizer.convert_tokens_to_ids(option)
        logits_of_options.append(final_logits[0][option_id])
    prob = F.softmax(torch.tensor(logits_of_options), dim=0)
    pred_idx_from_options = torch.argmax(prob, dim=-1)
    pred_label_from_options = chr(ord('A')+pred_idx_from_options)
    
    pred_label = pred_label_from_options
    
    return pred_label


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the MLM task. ")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--metric_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the implementation metrics will be written.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name as output prefix to indentify each running of experiment.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="Model name as output prefix to indentify each running of experiment.")
    parser.add_argument("--cn_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where important positions are stored.")

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
                        help="available gpu_id id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--enhance_cn_num",
                        default=20,
                        type=int,
                        help="The number of context-aware neurons to be erased")
    parser.add_argument("--enhance_strength",
                        type=float,
                        required=True,
                        help="The factor by which the weights of the context neurons need to be enhanced.")
    
    parser.add_argument("--do_random_cn",
                        default=False,
                        action='store_true',
                        help="Set this flag if you want to randomly select context-aware neurons to operate on")
    parser.add_argument("--use_cad_decoding",
                        default=False,
                        action='store_true',
                        help="Set this flag if you want to use CAD decoding.")
    parser.add_argument("--use_instruction_prompt",
                        default=False,
                        action='store_true',
                        help="Set this flag if you want to use instruction prompt.")
    parser.add_argument("--instruction_prompt_type",
                        type=str)
    
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpu_id) == 1:
        device = torch.device("cuda:%s" % args.gpu_id)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metric_dir, exist_ok=True)

    # prepare eval set
    if args.dataset_name == "Memo":
        with open(args.data_path, "r", encoding="utf-8") as fin:
            data = []
            for json_line in fin:
                json_data = json.loads(json_line)
                data.append(json_data)
    if args.dataset_name == "COSE" or args.dataset_name == "ECARE":
        with open(args.data_path, "r", encoding="utf-8") as fin:
            lines = fin.read()
            data = json.loads(lines)

    # ========================== load model and tokenizer ===============================
    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_cache=True,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    if "llama" in args.model_name:
        print("删除了add_special_tokens")
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))
    
    model.eval()

    # logger.info("***** CUDA.empty_cache() *****")
    # torch.cuda.empty_cache()

    print(f'evaluating {args.model_name} with {args.dataset_name}...')
    if args.use_cad_decoding:
        print('Using CAD decoding...')
    
    # calculate and enhance context neurons
    if args.enhance_strength != 1.0:
        print(f'Eval enhanced model...')
        model, cns = enhance_neurons(args=args, model=model)

    total = 0
    correct = 0
    trap_correct = 0
    predictions = []

    for idx, example in enumerate(tqdm(data)):
        prompts_info = get_prompts(example, args)
        prompt_without_context, prompt_with_context = prompts_info["prompt_without_context"], prompts_info["prompt_with_context"]
        
        if idx == 0:
            print(prompt_with_context)
            print('-'*50)
        
        tokens_info = get_information(prompt_with_context, example, args, tokenizer)
        if tokens_info == None:
            print(f"skip example with id: {idx}")
            continue

        if args.dataset_name == "Memo":
            full_prediction, pred_label = completion_task_generation(model, tokenizer, args, prompt_with_context, prompt_without_context, device)
            prediction = {
                'full_prediction': full_prediction,
                'pred_label': pred_label,
                'answer_obj': tokens_info['answer_obj'],
                'trap_obj': tokens_info['trap_obj'],
                'prompt': prompt_with_context,
            }
            predictions.append(prediction)
            total += 1
            if pred_label == tokens_info["answer_obj"]:
                correct += 1
            if pred_label == tokens_info["trap_obj"]:
                trap_correct += 1

        if args.dataset_name == "COSE" or args.dataset_name == "ECARE":
            pred_label = mcq_task_generation(model, tokenizer, args, example["choices"], prompt_with_context, prompt_without_context, device)
            prediction = {
                'prediction': pred_label,
                'answer_obj': tokens_info['answer_obj'],
                'trap_obj': tokens_info['trap_obj'],
                'prompt': prompt_with_context,
            }
            predictions.append(prediction)
            total += 1
            if pred_label == tokens_info["answer_obj"]:
                correct += 1
            if pred_label == tokens_info["trap_obj"]:
                trap_correct += 1
        
    acc = correct / total
    trap_acc = trap_correct / total

    # TODO: del
    if not args.do_random_cn:
        with open(os.path.join(args.output_dir, f'{args.dataset_name}-{args.model_name}-{args.enhance_cn_num}-{args.enhance_strength}-prediction.json'), 'w') as fout:
            json.dump(predictions, fout, indent=2)

    if args.enhance_strength != 1.0:
        print(f'======================================== {args.dataset_name} {args.model_name} ===========================================')
        print(f'dataset size: {total}')
        print(f'enhanced cn num: {args.enhance_cn_num}')
        print(f"enhance strength: {args.enhance_strength}")
        print(f'accuracy: {acc:.4}')
        print(f'trap accuracy: {trap_acc:.4}')

        if args.do_random_cn:
            file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-random-all-metrics.json")
            with open(file_out_path, 'a') as fout:
                metrics_json ={
                    "enhanced cn num": args.enhance_cn_num,
                    "enhance strength": args.enhance_strength,
                    "random seed": args.seed,
                    "acc": round(acc * 100, 2),
                    "trap acc": round(trap_acc * 100, 2),
                    'changed cns': cns,
                }
                fout.write(json.dumps(metrics_json) + "\n")
        else:
            if args.use_cad_decoding:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-CAD-all-metrics.json")
            elif args.use_instruction_prompt:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-{args.instruction_prompt_type}-all-metrics.json")
            else:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-all-metrics.json")
            
            with open(file_out_path, 'a') as fout:
                metrics_json ={
                    "enhanced cn num": args.enhance_cn_num,
                    "enhance strength": args.enhance_strength,
                    "acc": round(acc * 100, 2),
                    "trap acc": round(trap_acc * 100, 2),
                    'changed cns': cns,
                }
                fout.write(json.dumps(metrics_json) + "\n")
        print(f"Saved metric results in {file_out_path}.")
    else:
        print(f'======================================== {args.dataset_name} {args.model_name} ===========================================')
        print(f'dataset size: {total}')
        print(f'enhanced cn num: {args.enhance_cn_num}')
        print(f"enhance strength: {args.enhance_strength}")
        print(f'accuracy: {acc:.4}')
        print(f'trap accuracy: {trap_acc:.4}')

        if args.do_random_cn:
            file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-random-all-metrics.json")
            with open(file_out_path, 'a') as fout:
                metrics_json = {
                    "enhanced cn num": args.enhance_cn_num,
                    "enhance strength": args.enhance_strength,
                    "random_seed": args.seed,
                    "correct num": correct,
                    "acc": round(acc * 100, 2),
                    "trap acc": round(trap_acc * 100, 2),
                }
                fout.write(json.dumps(metrics_json) + "\n")
        else:
            if args.use_cad_decoding:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-CAD-all-metrics.json")
            elif args.use_instruction_prompt:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-{args.instruction_prompt_type}-all-metrics.json")
            else:
                file_out_path = os.path.join(args.metric_dir, f"{args.dataset_name}-{args.model_name}-all-metrics.json")
            
            with open(file_out_path, 'a') as fout:
                metrics_json ={
                    "enhanced cn num": args.enhance_cn_num,
                    "enhance strength": args.enhance_strength,
                    "correct num": correct,
                    "acc": round(acc * 100, 2),
                    "trap acc": round(trap_acc * 100, 2),
                }
                fout.write(json.dumps(metrics_json) + "\n")
        print(f"Saved metric results in {file_out_path}.")
        

if __name__ == "__main__":
    main()