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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import copy
import heapq
import pickle
import sys
from prune_dietcode import delete_with_algorithm_of_dietcode ,merge_statements_from_tokens
import numpy as np
import random
import re
import csv
import logging
import os
from io import open
from sklearn.metrics import f1_score
from prune_dietcode import (is_if_statement,is_case_statement,
                                            is_try_statement,is_break_statement,
                                            is_catch_statement,is_expression,is_finally_statement,
                                            is_return_statement,is_synchronized_statement,
                                            is_method_declaration_statement,is_getter,
                                            is_function_caller,is_setter,
                                            is_logger,is_annotation,is_throw_statement,
                                            is_continue_statement,is_while_statement,
                                            is_for_statement,is_variable_declaration_statement,is_switch_statement)
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)
outputFileIndex = 1
low_rated_tokens = []
origin_code_length = 0.0
pruned_code_length = 0.0



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.idx=idx



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if (set_type == 'test'):
                # label = self.get_labels()[0]
                label = line[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples

class WeightOutputer():
    def __init__(self):
        self.outputFileDir = ''

    def set_output_file_dir(self, fileDir):
        self.outputFileDir = fileDir


def delete_token(source_tokens, max_source_len, weight_dict):
    source_tokens_new = []
    min_heap = []
    i = 0
    source_tokens=['<s>'] + source_tokens+['</s>']
    statements, _ = merge_statements_from_tokens(source_tokens)
    last_index = _[-1][1]
    if last_index < len(source_tokens)-2:
        last = source_tokens[last_index+1:len(source_tokens)-2]
        statements.append(last)
    for statement in statements:
        statement_str = ' '.join(statement).replace('Ġ', '')
        catgory = get_category(statement_str)
        weight_dic = weight_dict[catgory]
        for token in statement:
            source_tokens_new.append(token)
            token_weight = weight_dic.get(token, {})
            if token_weight:
                weight = token_weight
            else:
                weight = 1
            heapq.heappush(min_heap, (weight, i, token))
            i += 1
    assert len(source_tokens_new) == len(min_heap)
    delete_token_indexs = []
    len_tokens = len(source_tokens_new)
    while True:
        if len_tokens <= max_source_len:
            break
        else:
            _, token_idx, token = heapq.heappop(min_heap)
            delete_token_indexs.append(token_idx)
            len_tokens -= 1
    source_tokens_ne = [source_tokens_new[i] for i in range(len(source_tokens_new)) if i not in delete_token_indexs]
    source_tokens = source_tokens_ne
    return source_tokens

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
            tokens = tokens[start+1:]
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


def convert_examples_to_features_test(examples, label_list, ratio,
                                 tokenizer, output_mode,weight_dict,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, prune_strategy='None'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)[:50]

        tokens_b = None
        if prune_strategy == 'leancode':
            tokens_b = tokenizer.tokenize(example.text_b)
            target_len = int(len(tokens_b) * ratio)
            max_len = 512 - 3 - len(tokens_a)
            tokens_b = delete_token(tokens_b, target_len, weight_dict)[:max_len]
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
            max_len = 512 - 3 - len(tokens_a)
            tokens_b = tokens_b[:max_len]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create attention mask
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=attention_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          idx=ex_index))
    return features




def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,weight_dict,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, prune_strategy='None'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)[:50]

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            tokens_a, tokens_b = _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3, weight_dict,
                                                    prune_strategy)

        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          idx=ex_index))
    return features


def assimilate_code_string_and_integer(code, string_mask=" string ", number_mask="10"):
    quotation_index_list = []
    for i in range(0, len(code)):
        if code[i] == "\"":
            quotation_index_list.append(i)
    for i in range(len(quotation_index_list) - 1, 0, -2):
        code = code[:quotation_index_list[i-1] + 1] + string_mask + code[quotation_index_list[i]:]

    tokens = code.split(" ")
    for i in range(0, len(tokens)):
        if is_number(tokens[i]):
            tokens[i] = number_mask
    code = " ".join(tokens)

    return code

def merge_statements(code):
    statements = []
    tokens = code.split(' ')
    start = 0
    try:
        end_function_def = tokens.index('{')
        statements.append(tokens[:end_function_def+1])
        start = end_function_def+1
    except ValueError:
        pass
    index = start
    in_brace = 0
    endline_keyword = [';', '{']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['(']:
            in_brace += 1
        elif current_token in [')']:
            in_brace -= 1
        if current_token == ';' and in_brace > 0:
            index += 1
            continue
        if current_token in endline_keyword:
            statements.append(tokens[start:index+1])
            start = index + 1
            index += 1
    if start < len(tokens):
        statements.append(tokens[start:])
    return statements


def output_weights(attentions, tokens, wo,indexs,starts,ends):
    # size of attentions :batch_size,shape of per batch attention : (layer_num,num_heads,1,source_len)
    for i,batch_attention in enumerate(attentions):
        # token=['<s>']+tokens[i]
        index = indexs[i].item()
        batch_attention_avg = batch_attention.sum(dim=(1, 2)) / (batch_attention.size(1) * batch_attention.size(2))
        # we use the last layer of cls attention
        output_layer_attention(11, batch_attention_avg[11], wo.outputFileDir, index)
        output_tokens(wo.outputFileDir, tokens[i], index)
        statements, tokenIndexList = merge_statements_from_tokens(tokens[i])
        output_statementIndex(tokenIndexList,index,wo.outputFileDir)

def output_statementIndex(tokenIndexList,outputFileIndex,output_file_dir):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir+'/statementIndex', 'a+') as f:
        f.write(str(outputFileIndex) + "<SPLIT>" + str(tokenIndexList)+'\n')


def output_layer_attention(layer, attentions, output_file_dir,outputFileIndex):
    if not os.path.exists(output_file_dir+'/layer_'+str(layer)):
        os.makedirs(output_file_dir+'/layer_'+str(layer))
    with open(output_file_dir+'/layer_'+str(layer)+'/weights_all', 'a+') as f:
        f.write(str(outputFileIndex) + "<SPLIT>" + str(attentions.tolist())+'\n')


def output_tokens(output_file_dir, tokens,outputFileIndex):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir+'/token', 'a+') as f:
        f.write(str(outputFileIndex) + "<SPLIT>" + str(tokens)+'\n')


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def load_leancode_weights(model_type):
    if model_type=='codet5':
        with open('./leancode_weights/codesearch_codet5.pkl', 'rb') as handle:
            weight_dicts = pickle.load(handle)['codesearch_codet5']
    else:
        with open('./leancode_weights/codesearch_codebert.pkl', 'rb') as handle:
            weight_dicts = pickle.load(handle)['codesearch_codetbert']
    return  weight_dicts



def get_category(statement):
    if is_try_statement(statement):
        return 'try'
    elif is_catch_statement(statement):
        return 'catch'
    elif is_finally_statement(statement):
        return 'finally'
    elif is_break_statement(statement):
        return 'break'
    elif is_continue_statement(statement):
        return 'continue'
    elif is_return_statement(statement):
        return 'return'
    elif is_throw_statement(statement):
        return 'throw'
    elif is_annotation(statement):
        return 'annotation'
    elif is_while_statement(statement):
        return 'while'
    elif is_for_statement(statement):
        return 'for'
    elif is_if_statement(statement):
        return 'if'
    elif is_switch_statement(statement):
        return 'switch'
    elif is_expression(statement):
        return 'expression'
    elif is_synchronized_statement(statement):
        return 'synchronized'
    elif is_case_statement(statement):
        return 'case'
    elif is_method_declaration_statement(statement):
        return 'method'
    elif is_variable_declaration_statement(statement):
        return 'variable'
    elif is_logger(statement):
        return 'logger'
    elif is_setter(statement):
        return 'setter'
    elif is_getter(statement):
        return 'getter'
    elif is_function_caller(statement):
        return 'function'
    else:
        return 'none'


def _truncate_seq_pair(tokens_a, tokens_b, max_length,weight_dict,prune_strategy):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if len(tokens_a)+len(tokens_b)<=max_length:
        return tokens_a, tokens_b
    len_token_b = len(tokens_b)
    # len_token_b = len(tokens_b_new)
    while True:
        total_length = len(tokens_a) + len_token_b
        if total_length <= max_length:
            break
        if len(tokens_a) > len_token_b:
            tokens_a.pop()
        else:
            tokens_b.pop()
            len_token_b -= 1
    return tokens_a,tokens_b


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}
