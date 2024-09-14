# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from __future__ import absolute_import

import datetime
import pickle
import time

from utils import output_weights, WeightOutputer, delete_token
from thop import profile
import heapq
import re
import os
import sys
import bleu
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5EncoderModel, T5Config)
from prune_dietcode import delete_with_algorithm_of_dietcode
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset,Dataset
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer), 'codet5': (T5Config, T5EncoderModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
low_rated_tokens = []


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def load_leancode_weights():
    with open('./leancode_weights/code2nl_codebert.pkl', 'rb') as handle:
        all_weight_dicts = pickle.load(handle)
    return  all_weight_dicts['code2nl_codetbert']


def read_examples(filename,stragety='None'):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if stragety=='slimcode':
                code, nl = line.split('<CODESPLIT>')
            else:
                js = json.loads(line)
                if 'idx' not in js:
                    js['idx'] = idx
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
    return examples

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask

class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return feature.source_ids, feature.target_ids, feature.example_id
class InputFeatures_test(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,

                 ):
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.example_id = example_id



def convert_examples_to_features(examples, tokenizer, args,weights_dicts,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        if args.prune_strategy=='leancode':
            source_tokens = tokenizer.tokenize(example.source)
            source_tokens = delete_token(source_tokens, args.max_source_length - 2, weights_dicts)
        elif args.prune_strategy=='dietcode' or args.prune_strategy=='leancode_d':
            example.source = delete_with_algorithm_of_dietcode(example.source, args.max_source_length,
                                                 args.prune_strategy,'code2nl')

            source_tokens = tokenizer.tokenize(example.source)
            source_tokens = source_tokens[:args.max_source_length - 2]
        else:
            source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length


        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id]*padding_length
        target_mask += [0]*padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )

    return features

def camel_case_split(str):
        RE_WORDS = re.compile(r'''
            [A-Z]+(?=[A-Z][a-z]) |
            [A-Z]?[a-z]+ |
            [A-Z]+ |
            \d+ |
            [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
            ''', re.VERBOSE)
        return RE_WORDS.findall(str)

def caculate_tokens(code):
        tokens = code.split(' ')
        if tokens[0] == '@':
            if tokens[2] == '(':
                start = tokens.index(')')
                tokens = tokens[start + 1:]
            else:
                tokens = tokens[2:]
        current_token = []
        for i in range(len(tokens)):
            token = tokens[i]
            token = camel_case_split(token)
            for t in token:
                current_token.append(t)
        tokens = current_token
        return len(tokens)

def convert_examples_to_features_test(examples, tokenizer, args,weights_dicts,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        if args.prune_strategy=='leancode':
            source_tokens = tokenizer.tokenize(example.source)
            max_source_length = int(len(source_tokens) * args.ratio)
            source_tokens = delete_token(source_tokens, max_source_length, weights_dicts)[:510]
        elif args.prune_strategy=='dietcode' or args.prune_strategy=='leancode_d':
            target_len = int(caculate_tokens(example.source) * args.ratio)
            example.source = delete_with_algorithm_of_dietcode(example.source, target_len,
                                                 args.prune_strategy,'code2nl')

            source_tokens = tokenizer.tokenize(example.source)
            source_tokens = source_tokens[:510]
        else:
            source_tokens = tokenizer.tokenize(example.source)[:510]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.idx))

                logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))

                logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))

        features.append(
            InputFeatures_test(
                example_index,
                source_ids,
                target_ids,
            )
        )
    dataset = CustomDataset(features)

    return dataset


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch, tokenizer):
    source_ids = [torch.tensor(item[0], dtype=torch.long) for item in batch]
    target_ids = [torch.tensor(item[1], dtype=torch.long) for item in batch]

    # Padding within batch
    padded_source_ids = pad_sequence(source_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_target_ids = pad_sequence(target_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Create attention masks: 1 for actual tokens, 0 for padding tokens
    attention_masks = (padded_source_ids != tokenizer.pad_token_id).long()

    example_ids = [item[2] for item in batch]  # Keeping track of example_ids if needed

    return padded_source_ids, padded_target_ids, attention_masks, example_ids


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    # Other parameters
    parser.add_argument("--prune_strategy", default="None", type=str,choices=['leancode', 'dietcode', 'leancode_d','None','slimcode'],
                        help="prune strategy ,  leancode_d means leancode + dietcode's removal")
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--lang', default='java', type=str, help='training or testing program language')
    parser.add_argument("--gen_weight", action='store_true', help="Whether to output attentions after training.")
    parser.add_argument('--num_train_epochs', type=int, default=8,
                        help="train epochs")
    parser.add_argument("--ratio",  default=-1, type=float,
                        help="prune ratio")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # budild model
    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        # model=torch.load(args.load_model_path)
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.prune_strategy=='leancode':
        weight_dicts = load_leancode_weights()
    else:
        weight_dicts = {}


    if args.do_train:
        train_examples = read_examples(args.train_filename)
        train_features = convert_examples_to_features(train_examples, tokenizer, args,weight_dicts, stage='train')
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        example_ids = []
        for f in train_features:
            example_ids.append(f.example_id)
        all_index = torch.tensor(example_ids, dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask,all_index)


        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size//args.gradient_accumulation_steps)


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)

        train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch",
                                disable=args.local_rank not in [-1, 0])

        model.train()
        dev_dataset = {}
        nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss = 0, 0, 0, 0, 0, 1e6

        eval_flag = True

        for idx, _ in enumerate(train_iterator):
            if args.gen_weight:
                wo = WeightOutputer()
                wo.set_output_file_dir('./code2nl/CodeBERT/weights/epoch_'+str(idx))

            for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):

                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask, target_ids, target_mask,indexs = batch
                outputs=model(source_ids=source_ids, source_mask=source_mask,
                               target_ids=target_ids, target_mask=target_mask)
                loss=outputs[0]
                if args.gen_weight:
                    tokens = []
                    attentions = outputs[3]
                    # size if attentions :layer_num ,shape of per layer attention : (batch_size,target_len,source_len)
                    attentions_new = []
                    attentions_tensor = torch.stack(attentions, dim=0)
                    attentions_tensor_permuted = attentions_tensor.permute(1, 0, 2, 3)
                    # mask the attention of <pad> in source,mask the attention of <pad> in target and remove <pad> in the source token
                    # size of attentions_new :batch_size,shape of per batch attention : (layer_num,target_len,source_len)
                    for i, batch_attention in enumerate(attentions_tensor_permuted):
                        batch_mask = source_mask[i]
                        tar_mask = target_mask[i]
                        n = torch.sum(batch_mask).item()
                        n_tar = torch.sum(tar_mask).item()
                        batch_attention = batch_attention[:, :n_tar, :n]
                        attentions_new.append(batch_attention)
                        token = source_ids[i]
                        tokens.append(tokenizer.convert_ids_to_tokens(token[:n]))
                    output_weights(attentions_new, tokens, wo, indexs)


                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                train_loss = round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1), 4)
                # bar.set_description("loss {}".format(train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    eval_flag = True

            if args.do_eval and eval_flag:
                # Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                eval_flag = False
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,weight_dicts, stage='dev')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
                    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)
                    dev_dataset['dev_loss'] = eval_examples, eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                # Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0, 0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask, target_ids, target_mask = batch
                    with torch.no_grad():
                        outputs = model(source_ids=source_ids, source_mask=source_mask,
                                        target_ids=target_ids, target_mask=target_mask)
                        loss=outputs[1]
                        num=outputs[2]
                    eval_loss += loss.sum().item()
                    tokens_num += num.sum().item()
                # Pring loss of dev dataset
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss), 5),
                        'global_step': global_step+1,
                        'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)

                # save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                if eval_loss < best_loss:
                    logger.info("  Best ppl:%s", round(np.exp(eval_loss), 5))
                    logger.info("  "+"*"*20)
                    best_loss = eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                # Calculate bleu
                if 'dev_bleu' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_bleu']
                else:
                    eval_examples = read_examples(args.dev_filename)
                    eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args,weight_dicts, stage='test')
                    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
                    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
                    eval_data = TensorDataset(all_source_ids, all_source_mask)
                    dev_dataset['dev_bleu'] = eval_examples, eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                p = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, source_mask = batch
                    with torch.no_grad():
                        preds = model(source_ids=source_ids, source_mask=source_mask)
                        for pred in preds:
                            t = pred[0].cpu().numpy()
                            t = list(t)
                            if 0 in t:
                                t = t[:t.index(0)]
                            text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                            p.append(text)
                model.train()
                predictions = []
                with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                    for ref, gold in zip(p, eval_examples):
                        predictions.append(str(gold.idx)+'\t'+ref)
                        f.write(str(gold.idx)+'\t'+ref+'\n')
                        f1.write(str(gold.idx)+'\t'+gold.target+'\n')

                (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(args.output_dir, "dev.gold"))
                dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
                logger.info("  "+"*"*20)
                if dev_bleu > best_bleu:
                    logger.info("  Best bleu:%s", dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu = dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)



    if args.do_test:
        files = []
        if args.test_filename is not None:
            files.append(args.test_filename)
        for idx_, file in enumerate(files):
            logger.info("Test file: {}".format(file))
            eval_examples = read_examples(file,args.prune_strategy)
            dataset = convert_examples_to_features_test(eval_examples, tokenizer, args, weight_dicts,stage='test')
            sorted_indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i][0]))
            sorted_dataset = torch.utils.data.Subset(dataset, sorted_indices)

            batch_size = args.eval_batch_size

            eval_dataloader = DataLoader(sorted_dataset, batch_size=batch_size, sampler=SequentialSampler(sorted_dataset),
                                    collate_fn=lambda batch: collate_fn(batch, tokenizer))
            # Calculate bleu

            model.eval()
            p = []
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                # batch = tuple(t.to(device) for t in batch[:2])
                source_ids, target_ids ,source_mask,example_ids = batch
                source_ids = source_ids.to(device)
                source_mask = source_mask.to(device)
                with torch.no_grad():
                    preds=model(source_ids=source_ids,source_mask=source_mask)
                    for pred in preds:
                        t = pred[0].cpu().numpy()
                        t = list(t)
                        if 0 in t:
                            t = t[:t.index(0)]
                        text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                        p.append(text)
            model.train()
            pres=[]
            predictions = sorted(zip(p, sorted_indices), key=lambda x: x[1])
            with open(os.path.join(args.output_dir, "test_{}.output".format(str(idx_))), 'w') as f, \
                    open(os.path.join(args.output_dir, "test_{}.gold".format(str(idx_))), 'w') as f1:
                for ref, idx in predictions:
                    original_example = eval_examples[idx]
                    pres.append(str(original_example.idx) + '\t' + ref + '\n')
                    f.write(str(original_example.idx) + '\t' + ref + '\n')
                    f1.write(str(original_example.idx) + '\t' + original_example.target + '\n')

            (goldMap, predictionMap) = bleu.computeMaps(pres,
                                                        os.path.join(args.output_dir, "test_{}.gold".format(idx_)))
            dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            with open(os.path.join(args.output_dir, "result.txt"),'w')as w:
                w.write("  %s : %s " % ("bleu-4", str(dev_bleu)))

            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  "+"*"*20)



if __name__ == "__main__":
    main()
