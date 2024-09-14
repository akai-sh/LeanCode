import os
import numpy as np
import pickle


def is_logger(statement):
    if statement.startswith('log ') or statement.startswith('Logger ') or ' Log ' in statement \
            or '. print ' in statement or ' . println ' in statement or 'LOG ' in statement \
            or statement.startswith('logger ') or statement.startswith('debug .'):
        return True
    return False

def is_getter(statement):
    if '. get ' in statement:
        return True
    return False

def is_setter(statement):
    if '. set' in statement and ')' in statement:
        return True
    return False

def is_if_statement(statement):
    if statement.startswith('if (') or statement.startswith('else if ') or statement.startswith('else '):
        return True
    return False

def is_while_statement(statement):
    if statement.startswith('while'):
        return True
    return False

def is_synchronized_statement(statement):
    if statement.startswith('synchronized '):
        return True
    return False

def is_for_statement(statement):
    if statement.startswith('for ('):
        return True
    return False

def is_throw_statement(statement):
    if statement.startswith('throw'):
        return True
    return False

def is_method_declaration_statement(statement):
    if statement.startswith('public ') or statement.startswith('protected ') or statement.startswith('private ') \
            or statement.startswith('@ Over ride') or statement.startswith('@ Bench mark') \
            or statement.startswith('@ Gener ated ') or statement.startswith('@ Test') \
            or statement.endswith(') {') or ("throws " in statement and 'Exception' in statement):
        return True
    return False

def is_switch_statement(statement):
    if statement.startswith('switch'):
        return True
    return False

def is_return_statement(statement):
    if statement.startswith('return'):
        return True
    return False

def is_variable_declaration_statement(statement):
    if statement.startswith('String ') or statement.startswith('int ') or statement.startswith('float ') \
            or statement.startswith('boolean') or statement.startswith('long') or statement.startswith('List') \
            or statement.startswith('Array ') or 'new ' in statement or statement.startswith('Collection') \
            or statement.startswith('final ') or "= true" in statement or "= null" in statement \
            or statement.startswith('Object ') or "= \" string \"" in statement or statement.startswith('Map <') \
            or statement.startswith('Class <') \
            and '=' in statement:
        return True
    return False

def is_reassign_statement(statement):
    if ")" not in statement and "=" in statement or " = (" in statement:
        return True
    return False

def is_try_statement(statement):
    if statement.startswith('try'):
        return True
    return False

def is_catch_statement(statement):
    if statement.startswith('catch'):
        return True
    return False

def is_finally_statement(statement):
    if statement.startswith('finally'):
        return True
    return False

def is_break_statement(statement):
    if statement.startswith('break'):
        return True
    return False

def is_case_statement(statement):
    if statement.startswith('case'):
        return True
    return False

def is_continue_statement(statement):
    if statement.startswith('continue'):
        return True
    return False

def is_expression(statement):
    if " * = " in statement or "++ ;" in statement or "/\ = " in statement \
            or "+=" in statement or "-- ;" in statement or " / = " in statement:
        return True
    return False

def is_function_caller(statement):
    if statement.endswith(') ;') and '(' in statement:
        return True
    return False

def is_annotation(statement):
    if statement.startswith('//') or statement.endswith('< p >') or statement.startswith('*/') \
            or statement.startswith('/*'):
        return True
    return False

def get_catagory_and_write(path,statement,line,s_token):
    global function_caller_num, method_declaration_statement_num, case_statement_num, try_statement_num, catch_statement_num, finally_statement_num, break_statement_num, continue_statement_num, return_statement_num, throw_statement_num, annotation_num, annotation_num, while_statement_num, for_statement_num, if_statement_num, switch_statement_num, expression_num, synchronized_statement_num, variable_declaration_statement_num, logger_num, setter_num, getter_num, none_num, split_num
    if is_try_statement(statement):
        try_statement_num += len(s_token)
        c_path = path+'/category_weight/try'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_catch_statement(statement):
        catch_statement_num += len(s_token)
        c_path = path+'/category_weight/catch'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_finally_statement(statement):
        finally_statement_num += len(s_token)
        c_path = path+'/category_weight/finally'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_break_statement(statement):
        break_statement_num += len(s_token)
        c_path = path+'/category_weight/break'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_continue_statement(statement):
        continue_statement_num += len(s_token)
        c_path = path+'/category_weight/continue'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_return_statement(statement):
        return_statement_num += len(s_token)
        c_path = path+'/category_weight/return'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_throw_statement(statement):
        throw_statement_num += len(s_token)
        c_path = path+'/category_weight/throw'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_annotation(statement):
        annotation_num += len(s_token)
        c_path = path+'/category_weight/annotation'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_while_statement(statement):
        while_statement_num += len(s_token)
        c_path = path+'/category_weight/while'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_for_statement(statement):
        for_statement_num += len(s_token)
        c_path = path+'/category_weight/for'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_if_statement(statement):
        if_statement_num += len(s_token)
        c_path = path+'/category_weight/if'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_switch_statement(statement):
        switch_statement_num += len(s_token)
        c_path = path+'/category_weight/switch'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_expression(statement):
        expression_num += len(s_token)
        c_path = path+'/category_weight/expression'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_synchronized_statement(statement):
        synchronized_statement_num += len(s_token)
        c_path = path+'/category_weight/synchronized'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_case_statement(statement):
        case_statement_num += len(s_token)
        c_path = path+'/category_weight/case'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_method_declaration_statement(statement):
        method_declaration_statement_num += len(s_token)
        c_path = path+'/category_weight/method'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_variable_declaration_statement(statement):
        variable_declaration_statement_num += len(s_token)
        c_path = path+'/category_weight/variable'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_logger(statement):
        logger_num += len(s_token)
        c_path = path+'/category_weight/logger'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_setter(statement):
        setter_num += len(s_token)
        c_path = path+'/category_weight/setter'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_getter(statement):
        getter_num += len(s_token)
        c_path = path+'/category_weight/getter'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    elif is_function_caller(statement):
        function_caller_num += len(s_token)
        c_path = path+'/category_weight/function'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)
    else:
        none_num += len(s_token)
        c_path = path+'/category_weight/none'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        with open(c_path + '/statements_and_tokens', 'a+') as f:
            f.write(line)

def categrize_statement(source_path,layer):
    global try_statement_num
    global catch_statement_num
    global finally_statement_num
    global break_statement_num
    global continue_statement_num
    global return_statement_num
    global throw_statement_num
    global annotation_num
    global expression_num
    global while_statement_num
    global for_statement_num
    global if_statement_num
    global switch_statement_num
    global synchronized_statement_num
    global case_statement_num
    global method_declaration_statement_num
    global variable_declaration_statement_num
    global reassign_num
    global function_caller_num
    global setter_num
    global getter_num
    global logger_num
    global split_num
    global none_num

    try_statement_num=0
    catch_statement_num = 0
    finally_statement_num = 0
    break_statement_num = 0
    continue_statement_num = 0
    return_statement_num = 0
    throw_statement_num = 0
    annotation_num = 0
    expression_num = 0
    while_statement_num = 0
    for_statement_num = 0
    if_statement_num = 0
    switch_statement_num = 0
    synchronized_statement_num = 0
    case_statement_num = 0
    method_declaration_statement_num = 0
    variable_declaration_statement_num = 0
    reassign_num = 0
    function_caller_num = 0
    setter_num = 0
    getter_num = 0
    logger_num = 0
    split_num = 0
    none_num = 0

    token_path = source_path+'/token'
    weight_path = source_path+layer+'/weights_all'
    index_path = source_path+'/statementIndex'
    tokens = []
    weights = []
    statementIndexs = []
    with open(token_path, 'r') as w:
        tokens = w.readlines()
    with open(weight_path, 'r') as w:
        weights = w.readlines()
    with open(index_path, 'r') as w:
        statementIndexs = w.readlines()
    for i in range(len(tokens)):
        token = eval(tokens[i].split('<SPLIT>')[1].strip())
        weight = eval(weights[i].split('<SPLIT>')[1].strip())
        statementIndex = eval(statementIndexs[i].split('<SPLIT>')[1].strip())
        for statement_range in statementIndex:
            statement_start = statement_range[0]
            statement_end = statement_range[1]
            statement_token = token[statement_start:statement_end + 1]
            statement_weight = weight[statement_start:statement_end + 1]
            statement_str = ' '.join(statement_token).replace('Ġ', '')
            line = str(statement_token) + '<SPLIT>' + str(statement_weight)+'\n'
            get_catagory_and_write(source_path,statement_str, line,statement_token)
    with open(source_path+'/category_nums' , 'a+') as f:
        f.write('try_statement_num :  '+str(try_statement_num)+'\n')
        f.write('catch_statement_num :  ' + str(catch_statement_num) + '\n')
        f.write('finally_statement_num :  ' + str(finally_statement_num) + '\n')
        f.write('break_statement_num :  ' + str(break_statement_num) + '\n')
        f.write('continue_statement_num :  ' + str(continue_statement_num) + '\n')
        f.write('return_statement_num :  ' + str(return_statement_num) + '\n')
        f.write('throw_statement_num :  ' + str(throw_statement_num) + '\n')
        f.write('annotation_num :  ' + str(annotation_num) + '\n')
        f.write('expression_num :  ' + str(expression_num) + '\n')
        f.write(' while_statement_num :  ' + str( while_statement_num) + '\n')
        f.write('for_statement_num :  ' + str(for_statement_num) + '\n')
        f.write('if_statement_num :  ' + str(if_statement_num) + '\n')
        f.write('switch_statement_num :  ' + str(switch_statement_num) + '\n')
        f.write('synchronized_statement_num :  ' + str(synchronized_statement_num) + '\n')
        f.write('case_statement_num :  ' + str(case_statement_num) + '\n')
        f.write('method_declaration_statement_num :  ' + str(method_declaration_statement_num) + '\n')
        f.write('variable_declaration_statement_num :  ' + str(variable_declaration_statement_num) + '\n')
        f.write('reassign_num :  ' + str(reassign_num) + '\n')
        f.write('function_caller_num :  ' + str(function_caller_num) + '\n')
        f.write('setter_num :  ' + str(setter_num) + '\n')
        f.write('getter_num :  ' + str(getter_num) + '\n')
        f.write('logger_num :  ' + str(logger_num) + '\n')
        f.write('split_num :  ' + str(split_num) + '\n')
        f.write('none_num :  ' + str(none_num) + '\n')

def gen_weight(source_path):
    directory_path = source_path+'/category_weight'
    filename_to_open = 'statements_and_tokens'
    for entry in os.scandir(directory_path):
        tokenMap = {}
        file_path = os.path.join(entry.path, filename_to_open)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tokens = eval(line.split('<SPLIT>')[0].strip())
                weights = eval(line.split('<SPLIT>')[1].strip())
                for i, token in enumerate(tokens):
                    weight = weights[i]
                    if token not in tokenMap.keys():
                        tokenMap[token] = []
                        tokenMap[token].append(weight)
                    else:
                        tokenMap[token].append(weight)
        write_path=os.path.join(entry.path, 'token_weight')
        with open(write_path , 'a+') as f:
            for token in tokenMap:
                weight_list=tokenMap[token]
                avg_weight=sum(weight_list) / len(weight_list)
                variance = np.var(weight_list)
                f.write(str(token) + "<SPLIT>" + str(avg_weight) + "<SPLIT>" + str(variance)+ '\n')

def load_weights(path):
    weight_dicts={}
    filename_to_open = 'token_weight'
    for entry in os.scandir(path):
        weights_dict={}
        file_path = os.path.join(entry.path, filename_to_open)
        with open(file_path, 'r') as r:
            lines = r.readlines()
            for line in lines:
                line_dict = line.split('<SPLIT>')
                word = line_dict[0]
                weight = float(line_dict[1].strip())
                weights_dict[word] = weight
        weight_dicts[entry.name]=weights_dict
    return weight_dicts

if __name__ == '__main__':
    # paths=['./codesearch/weights/roberta','./codesearch/weights/codet5','./code2nl/CodeBERT/weights','./code2nl/CodeT5/weights']
    # keys = ['codesearch_codebert', 'codesearch_codet5', 'code2nl_codebert', 'code2nl_codet5']
    # layers=['/layer_11','/layer_11''/layer_5','/layer_11']

    paths=['./code2nl/CodeBERT/weights/epoch_4']
    # paths=['./code2nl/CodeT5/weights_16/epoch_23']
    keys = ['code2nl_codetbert']
    layers=['/layer_11']

    filename_to_open = 'token_weight'
    all_weight_dicts = {}
    for i, path in enumerate(paths):
        categrize_statement(path,layers[i])
        gen_weight(path)
        weights_dic = load_weights(path+'/category_weight')
        all_weight_dicts[keys[i]] = weights_dic

    # save_path = "./utils/leancode_weights.pkl"
    save_path = "./utils/code2nl_codebert.pkl"

    # save
    with open(save_path, 'wb') as handle:
        pickle.dump(all_weight_dicts, handle, protocol=pickle.HIGHEST_PROTOCOL)
