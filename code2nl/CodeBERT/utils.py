import os
import heapq
import torch
import sys
from prune_dietcode import (is_if_statement,is_case_statement,
                                            is_try_statement,is_break_statement,
                                            is_catch_statement,is_expression,is_finally_statement,
                                            is_return_statement,is_synchronized_statement,
                                            is_method_declaration_statement,is_getter,
                                            is_function_caller,is_setter,
                                            is_logger,is_annotation,is_throw_statement,
                                            is_continue_statement,is_while_statement,
                                            is_for_statement,is_variable_declaration_statement,is_switch_statement)

from prune_dietcode import merge_statements_from_tokens

class WeightOutputer():
    def __init__(self):
        self.outputFileDir = ''

    def set_output_file_dir(self, fileDir):
        self.outputFileDir = fileDir


def get_category(statement):
    # c_path='weights_new/roberta/train/category_weight'
    # if not os.path.exists(c_path):
    #     os.makedirs(c_path)
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

def delete_token(source_tokens, max_source_len, weight_dict):
    if len(source_tokens)<max_source_len:
        return source_tokens
    source_tokens_new = []
    min_heap = []
    i = 0
    statements, _ = merge_statements_from_tokens(['null'] + source_tokens)
    last_index = _[-1][1]
    if last_index < len(source_tokens):
        last = source_tokens[last_index:]
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
    # size of attentions :batch_size,shape of per batch attention : (layer_num,target_len,source_len)
    for i,batch_attention in enumerate(attentions):
        index=indexs[i].item()
        batch_attention=batch_attention.permute(0,2,1)
        batch_attention,_=torch.max(batch_attention,dim=-1)
        output_layer_attention(11, batch_attention[11], wo.outputFileDir, index)
        output_tokens(wo.outputFileDir, tokens[i], index)
        statements, tokenIndexList = merge_statements_from_tokens(tokens[i])
        output_statementIndex(tokenIndexList,index,wo.outputFileDir)


def output_layer_attention(layer, attentions, output_file_dir,outputFileIndex):
    if not os.path.exists(output_file_dir+'/layer_'+str(layer)):
        os.makedirs(output_file_dir+'/layer_'+str(layer))
    with open(output_file_dir+'/layer_'+str(layer)+'/weights_all', 'a+') as f:
        f.write(str(outputFileIndex) + "<SPLIT>" + str(attentions.tolist())+'\n')
    # return layer_attention

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