import heapq
import json
import os
import re

import torch
from prune_dietcode import merge_statements_from_tokens ,delete_with_algorithm_of_dietcode



from prune_dietcode import (is_if_statement,is_case_statement,
                                            is_try_statement,is_break_statement,
                                            is_catch_statement,is_expression,is_finally_statement,
                                            is_return_statement,is_synchronized_statement,
                                            is_method_declaration_statement,is_getter,
                                            is_function_caller,is_setter,
                                            is_logger,is_annotation,is_throw_statement,
                                            is_continue_statement,is_while_statement,
                                            is_for_statement,is_variable_declaration_statement,is_switch_statement)
class WeightOutputer():
    def __init__(self):
        self.outputFileDir = ''

    def set_output_file_dir(self, fileDir):
        self.outputFileDir = fileDir

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

def delete_token(source_tokens, max_source_len, weight_dict,logger,example_index):
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

def output_weights(attentions, tokens, wo,indexs):
    # size of attentions :batch_size,shape of per batch attention : (layer_num,num_heads,target_len,source_len)
    for i,batch_attention in enumerate(attentions):
        index = indexs[i].item()
        # (layer_num,num_heads,target_len,source_len)
        batch_attention = batch_attention.sum(dim=1) / batch_attention.size(1)
        batch_attention=batch_attention.permute(0,2,1)
        # (layer_num,source_len,target_len)
        batch_attention,_=torch.max(batch_attention,dim=-1)
        # (layer_num,source_len)
        output_layer_attention(11, batch_attention[11], wo.outputFileDir, index)
        output_tokens(wo.outputFileDir, tokens[i], index)
        statements, tokenIndexList = merge_statements_from_tokens(tokens[i])
        last_index = tokenIndexList[-1][1]
        if last_index < len(tokens[i])-2:
            last = tokens[i][last_index:]
            statements.append(last)
            tokenIndexList.append([last_index+1,len(tokens[i])-2])
        output_statementIndex(tokenIndexList,index,wo.outputFileDir)


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

def output_statementIndex(tokenIndexList,outputFileIndex,output_file_dir):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    with open(output_file_dir+'/statementIndex', 'a+') as f:
        f.write(str(outputFileIndex) + "<SPLIT>" + str(tokenIndexList)+'\n')

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

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

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage,weights_dicts = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    # source_tokens = tokenizer.tokenize(source_str)
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )

def convert_examples_to_features_test(item):
    example, example_index, tokenizer, args, stage,weights_dicts,logger = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    # source_tokens = tokenizer.tokenize(source_str)
    if args.prune_strategy=='leancode':
        source_tokens = tokenizer.tokenize(source_str)
        max_source_length=int(len(source_tokens)*args.ratio)
        source_tokens = delete_token(source_tokens, max_source_length, weights_dicts, logger, example_index)[:510]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    elif args.prune_strategy=='dietcode' or args.prune_strategy=='leancode_d':
        target_len = int(caculate_tokens(source_str) * args.ratio)
        source = delete_with_algorithm_of_dietcode(source_str, target_len,args.prune_strategy,'code2nl')
        source_tokens = tokenizer.tokenize(source)
        source_tokens = source_tokens[:510]
        source_tokens = [tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    else:
        source_tokens = tokenizer.tokenize(source_str)[:510]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        # url=example.url
        url=""
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)

def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)

class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num,args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num,args):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num,args):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num,args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if args.prune_strategy=='slimcode':
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
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num,args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num,args):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
