import copy
import os
import re


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

def camel_case_split(str):
    RE_WORDS = re.compile(r'''
        [A-Z]+(?=[A-Z][a-z]) |
        [A-Z]?[a-z]+ |
        [A-Z]+ |
        \d+ |
        [^\u4e00-\u9fa5^a-z^A-Z^0-9]+
        ''', re.VERBOSE)
    return RE_WORDS.findall(str)

def merge_statements(code):
    statements = []
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
    start = 0
    # try:
        # end_function_def = tokens.index('{')
        # statements.append(tokens[:end_function_def+1])
        # start = end_function_def+1
    # except ValueError:
        # pass
    index = start
    in_brace = 0
    endline_keyword = [';', '{', '}']
    while index < len(tokens):
        current_token = tokens[index]
        if current_token in ['(']:
            in_brace += 1
        elif current_token in [')']:
            in_brace -= 1
        if current_token in endline_keyword and in_brace > 0:
            index += 1
            continue
        if current_token in endline_keyword:
            statements.append(tokens[start:index+1])
            start = index + 1
            index += 1
            continue
        index += 1
    if start < len(tokens):
        statements.append(tokens[start:])
    return statements


def get_statement_classification(statement,statement_classification_map):
    if is_try_statement(statement):
        return 'try', statement_classification_map['try']
    elif is_catch_statement(statement):
        return 'catch', statement_classification_map['catch']
    elif is_finally_statement(statement):
        return 'finally', statement_classification_map['finally']
    elif is_break_statement(statement):
        return 'break', statement_classification_map['break']
    elif is_continue_statement(statement):
        return 'continue', statement_classification_map['continue']
    elif is_return_statement(statement):
        return 'return', statement_classification_map['return']
    elif is_throw_statement(statement):
        return 'throw', statement_classification_map['throw']
    elif is_annotation(statement):
        return 'annotation', statement_classification_map['annotation']
    elif is_while_statement(statement):
        return 'while', statement_classification_map['while']
    elif is_for_statement(statement):
        return 'for', statement_classification_map['for']
    elif is_if_statement(statement):
        return 'if', statement_classification_map['if']
    elif is_switch_statement(statement):
        return 'switch', statement_classification_map['switch']
    elif is_expression(statement):
        return 'expression', statement_classification_map['expression']
    elif is_synchronized_statement(statement):
        return 'synchronized', statement_classification_map['syncronized']
    elif is_case_statement(statement):
        return 'case', statement_classification_map['case']
    elif is_method_declaration_statement(statement):
        return 'method', statement_classification_map['method']
    elif is_variable_declaration_statement(statement):
        return 'variable', statement_classification_map['variable']
    elif is_logger(statement):
        return 'logger', statement_classification_map['log']
    elif is_setter(statement):
        return 'setter', statement_classification_map['setter']
    elif is_getter(statement):
        return 'getter', statement_classification_map['getter']
    elif is_function_caller(statement):
        return 'function', statement_classification_map['function']
    return 'None', 0.0001

def delete_with_algorithm_of_dietcode(code,target_len ,strategy,task):
    result = ''
    reduction = Code_Reduction(code, strategy, target_len,task)
    result = reduction.prune()
    return result

#reference statement scores, you can change it to the results generated by yourself

dietcode_statement_classification_map = {
    'try': 0.0029647741585358297,
    'catch': 0.0025092298911411127,
    'finally': 0.003843427080920313,
    'break': 0.002504047667805474,
    'continue': 0.0025862206572769947,
    'return': 0.003415540177420338,
    'throw': 0.002409465431368352,
    'annotation': 0.0028472381356659383,
    'while': 0.002679541985162062,
    'for': 0.002537917195113055,
    'if': 0.0025404393423889915,
    'switch': 0.0025462886222332426,
    'expression': 0.0032153782437548553,
    'syncronized': 0.0023616513586135323,
    'case': 0.002325461992369871,
    'method': 0.004119854399696806,
    'variable': 0.0024165516139185456,
    'log': 0.002416770362746685,
    'setter': 0.0026460245897558608,
    'getter': 0.0025480630285627617,
    'function': 0.0027256693629142824,
}

encoder_decoder_statement_classification_map={'method':0.06226000905192169,
'syncronized':0.047540884080184254,
'try':0.045563047743308,
'finally':0.04543945751409665,
'switch':0.04110764038659783,
'return':0.03648761209133361,
'if':0.036468428845559286,
'for':0.035486707953482445,
'while':0.03495013814390084,
'catch':0.03338125490784059,
'function':0.03309703521337963,
'setter':0.031417832369926166,
'throw':0.03120454573275256,
'variable':0.030332775402629228,
'getter':0.030263106211701904,
'log':0.02799851633142405,
'break':0.027977263975670408,
'expression':0.026444517888675002,
'case':0.023813584551912344,
'annotation':0.017865848563101992,
'continue':0.014506434519103707}

dietcode_lowest_ranked_token = []


encoder_decoder_lowest_ranked_token = []

def get_token_attention():
    with open('./utils/low_rated_word_dietcode', 'r') as f:
        for token in f.readlines():
            dietcode_lowest_ranked_token.append(token.replace('\n', ''))
    with open('./utils/low_rated_word_encoder_decoder', 'r') as f:
        for token in f.readlines():
            encoder_decoder_lowest_ranked_token.append(token.replace('\n', ''))


get_token_attention()

def merge_statements_from_tokens(tokens):
    tokenIndexList = []
    start = 1
    end = 1
    result = []
    is_for_statement = False
    for i in range(1, len(tokens)):
        if tokens[i] == '</s>' and (i == len(tokens) - 1 or tokens[i + 1] == '<s>'):
            break
        if tokens[i] == '</s>':
            start = i + 1
            continue
        if is_for_statement:
            if '{' in tokens[i]:
                is_for_statement = False
                end = i
                result.append(tokens[start:end + 1])
                tokenIndexList.append([start, end])
                start = end + 1
            continue
        try:
            if tokens[i] == 'Ġfor' and tokens[i + 1] == 'Ġ(':
                is_for_statement = True
                start = i
                continue
        except:
            pass
        try:
            if (tokens[i] == 'Ġ>' and tokens[i - 1] == 'p' and tokens[i - 2] == 'Ġ<') \
                    or ';' in tokens[i] or '{' in tokens[i] or '}' in tokens[i]:
                end = i
                result.append(tokens[start:end + 1])
                tokenIndexList.append([start, end])
                start = end + 1
        except:
            pass
    return result, tokenIndexList


class Code_Reduction():  # self.statement_attention: statement categories' attention. Form as:
    #      [{category: 'if statement', content: 'statement content', attention: 0.01, length: 10}]
    # self.token_attention: token attention. Form as {'a': 0.01, 'b': 0.02}
    def __init__(self, code, sterategy, targetLength,task):
        self.code = code
        self.sterategy = sterategy
        self.targetLength = targetLength
        self.task = task
        self.result = []
        self.generate_statements()


    def generate_statements(self):
        statements = None
        self.statements = []
        if self.sterategy == 'dietcode':
            statements = merge_statements(self.code)
            self.statements = []
            for statement in statements:
                category, attention = get_statement_classification(' '.join(statement),dietcode_statement_classification_map)
                current_statement = {'category': category, 'content': statement,
                                     'length': len(statement), 'attention': attention}
                self.statements.append(current_statement)
        elif self.sterategy == 'leancode_d':
            statements = merge_statements(self.code)
            self.statements = []
            for statement in statements:
                category, attention = get_statement_classification(' '.join(statement),encoder_decoder_statement_classification_map)
                current_statement = {'category': category, 'content': statement,
                                     'length': len(statement), 'attention': attention}
                self.statements.append(current_statement)

    def prune_lowest_ranked_token(self, statements, prune_num,lowest_ranked_token):
        result = []
        # check pruning items
        candidate = []
        for statement in statements:
            for token in statement:
                if token in lowest_ranked_token:
                    attention_pos = lowest_ranked_token.index(token)
                    if len(candidate) <= prune_num:
                        candidate.append(attention_pos)
                    elif attention_pos < max(candidate) and attention_pos not in candidate:
                        candidate.remove(max(candidate))
                        candidate.append(attention_pos)
        # prune phase
        pruned_num = 0
        need_check = True
        candidate = [lowest_ranked_token[x] for x in candidate]
        for statement in statements:
            if not need_check:
                result.append(statement)
                continue
            current_statement = []
            for token in statement:
                if token in candidate and need_check:
                    pruned_num += 1
                    if pruned_num >= prune_num:
                        need_check = False
                else:
                    current_statement.append(token)
                continue
            result.append(current_statement)
        return result

    def zero_one_backpack(self):
        # after the 0-1 backpack problem solution get the chosen statements we prefer to reduce tokens insteam of add
        # tokens so we choose to increase the target length by the max length of all the statements so that the solution
        # will at least consist of more than one statement than the solution of the previous target length.
        max_length = 0
        for statement in self.statements:
            if statement['length'] > max_length:
                max_length = statement['length']
        max_length += self.targetLength
        dp = [[{'attention': 0.0, 'statements': []}
               for i in range(max_length + 1)] for j in range(len(self.statements) + 1)]
        for i in range(1, len(self.statements) + 1):
            for j in range(1, max_length + 1):
                current_map = {'attention': dp[i-1][j]['attention'],
                               'statements': copy.deepcopy(dp[i-1][j]['statements'])}
                dp[i][j] = current_map
                if j >= self.statements[i-1]['length']:
                    if dp[i][j]['attention'] < \
                            dp[i-1][j-self.statements[i-1]['length']]['attention'] + self.statements[i-1]['attention']:
                        dp[i][j]['attention'] = dp[i-1][j-self.statements[i-1]
                                                        ['length']]['attention'] + self.statements[i-1]['attention']
                        dp[i][j]['statements'] = copy.deepcopy(
                            dp[i-1][j-self.statements[i-1]['length']]['statements'])
                        dp[i][j]['statements'].append(i-1)
        return dp[-1][-1]['attention'], dp[-1][-1]['statements']

    def prune(self, **kwargs):
        # adding statments: greedy
        total_attention, chosen_statements = self.zero_one_backpack()
        current_length = 0
        result = []
        for statement_index in chosen_statements:
            current_length += self.statements[statement_index]['length']
            result.append(self.statements[statement_index]['content'])
        pruned_token_num = current_length - self.targetLength
        if pruned_token_num > 0:
            if self.sterategy=='dietcode':
                result = self.prune_lowest_ranked_token(result, pruned_token_num,dietcode_lowest_ranked_token)
            elif self.sterategy=='leancode_d':
                result = self.prune_lowest_ranked_token(result, pruned_token_num,encoder_decoder_lowest_ranked_token)
        result_=[]
        for x in result:
            for y in x:
                result_.append(y)
        return ' '.join(result_[:self.targetLength])