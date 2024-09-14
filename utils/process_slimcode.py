import json
from transformers import RobertaTokenizer

def remove_special_tokens(token_list, special_char='Ġ'):
    return [token.replace(special_char, '') for token in token_list]

tokenizer_codet5 = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer_codebert =RobertaTokenizer.from_pretrained('microsoft/codebert-base')
# code search
with open('./../slimcode/data/codesearch/test.txt','r')as r\
        ,open('./../slimcode/data/codesearch/test_with_tokens.txt','w')as w:
    lines=r.readlines()
    for line in lines:
        label,url,name,nl,code=line.strip().split('<CODESPLIT>')
        # code,nl=line.strip().split('<CODESPLIT>')
        words = code.split()
        word_to_subtoken_map = {}
        for idx, word in enumerate(words):
            word_tokens = tokenizer_codet5.tokenize(word)
            word_tokens = remove_special_tokens(word_tokens)
            # key = f"{word}_{idx}"  # 生成唯一key
            word_to_subtoken_map[idx] = word_tokens
        all_tokens=tokenizer_codet5.tokenize(code)
        all_tokens = remove_special_tokens(all_tokens)
        # 使用json.dumps来确保双引号
        all_tokens_json = json.dumps(all_tokens)
        word_to_subtoken_map_json = json.dumps(word_to_subtoken_map)
        new_line=code+'<CODESPLIT>'+nl+'<CODESPLIT>'+label+'<CODESPLIT>'+str(all_tokens_json)+'<CODESPLIT>'+str(word_to_subtoken_map_json)+'\n'
        w.write(new_line)

with open('./../data/codesearch/slimcode/test.txt','r')as r\
        ,open('./../slimcode/data/codesearch/test_with_tokens_codebert.txt','w')as w:
    lines=r.readlines()
    for line in lines:
        label,url,name,nl,code=line.strip().split('<CODESPLIT>')
        # code,nl=line.strip().split('<CODESPLIT>')
        words = code.split()
        word_to_subtoken_map = {}
        for idx, word in enumerate(words):
            word_tokens = tokenizer_codet5.tokenize(word)
            word_tokens = remove_special_tokens(word_tokens)
            # key = f"{word}_{idx}"  # 生成唯一key
            word_to_subtoken_map[idx] = word_tokens
        all_tokens=tokenizer_codet5.tokenize(code)
        all_tokens = remove_special_tokens(all_tokens)
        # 使用json.dumps来确保双引号
        all_tokens_json = json.dumps(all_tokens)
        word_to_subtoken_map_json = json.dumps(word_to_subtoken_map)
        new_line=code+'<CODESPLIT>'+nl+'<CODESPLIT>'+label+'<CODESPLIT>'+str(all_tokens_json)+'<CODESPLIT>'+str(word_to_subtoken_map_json)+'\n'
        w.write(new_line)

with open('./../data/code2nl/java/slimcode/test.txt','r')as r\
        ,open('./../slimcode/data/code2nl/test_with_tokens.txt','w')as w:
    lines=r.readlines()
    for line in lines:
        code,nl=line.strip().split('<CODESPLIT>')
        words = code.split()
        word_to_subtoken_map = {}
        for idx, word in enumerate(words):
            word_tokens = tokenizer_codet5.tokenize(word)
            word_tokens = remove_special_tokens(word_tokens)
            # key = f"{word}_{idx}"  # 生成唯一key
            word_to_subtoken_map[idx] = word_tokens
        all_tokens=tokenizer_codet5.tokenize(code)
        all_tokens = remove_special_tokens(all_tokens)
        # 使用json.dumps来确保双引号
        all_tokens_json = json.dumps(all_tokens)
        word_to_subtoken_map_json = json.dumps(word_to_subtoken_map)
        new_line=code+'<CODESPLIT>'+nl+'<CODESPLIT>'+str(all_tokens_json)+'<CODESPLIT>'+str(word_to_subtoken_map_json)+'\n'
        w.write(new_line)

with open('./../data/code2nl/java/slimcode/test.txt','r')as r\
        ,open('./../slimcode/data/code2nl/test_with_tokens_codebert.txt','w')as w:
    lines=r.readlines()
    for line in lines:
        code,nl=line.strip().split('<CODESPLIT>')
        words = code.split()
        word_to_subtoken_map = {}
        for idx, word in enumerate(words):
            word_tokens = tokenizer_codebert.tokenize(word)
            word_tokens = remove_special_tokens(word_tokens)
            # key = f"{word}_{idx}"  # 生成唯一key
            word_to_subtoken_map[idx] = word_tokens
        all_tokens=tokenizer_codebert.tokenize(code)
        all_tokens = remove_special_tokens(all_tokens)
        # 使用json.dumps来确保双引号
        all_tokens_json = json.dumps(all_tokens)
        word_to_subtoken_map_json = json.dumps(word_to_subtoken_map)
        new_line=code+'<CODESPLIT>'+nl+'<CODESPLIT>'+str(all_tokens_json)+'<CODESPLIT>'+str(word_to_subtoken_map_json)+'\n'
        w.write(new_line)