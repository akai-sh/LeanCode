import os

import numpy as np
def output_weight(output_path,tokenMap):
    f = open(output_path + "/" + 'weights_of_every_token', "w")
    for key in tokenMap:
        f.write(key + '<SPLIT>' + str(tokenMap[key]) + '\n')
    f.close()

def read_tokens(path):
    tokens={}
    with open(path,'r') as r:
        lines=r.readlines()
        for line in lines:
            line_dict=line.split('<SPLIT>')
            idx=line_dict[0]
            token=eval(line_dict[1])
            tokens[idx]=token
    return tokens

def read_weights(path):
    weights={}
    with open(path,'r') as r:
        lines=r.readlines()
        for line in lines:
            line_dict=line.split('<SPLIT>')
            idx=line_dict[0]
            weight=eval(line_dict[1])
            weights[idx]=weight
    return weights


def init_tokenMap(tokens,tokenMap):
    for token in tokens:
        if token not in tokenMap.keys():
            tokenMap[token] = []


def update_tokenMap(attentions, tokens,tokenMap):
    init_tokenMap( tokens,tokenMap)
    for i in range(0, len(attentions)):
        tokenMap[tokens[i]].append(attentions[i])

def caculate_var(path,w_path):
    # for every token , caculate its glabal average attention score and variance
    # write into file named global_avg_and_var
    # in which every line contrains token<SPLIT>global_avg_weight<SPLIT>variance
    w_lines = []
    with open(path,'r')as r:
        lines=r.readlines()
        for line in lines:
            token=line.split('<SPLIT>')[0]
            data=eval(line.split('<SPLIT>')[1].strip())
            # utils = [w for sublist in data.values() for w in sublist]
            variance = np.var(data)
            w_line=token+'<SPLIT>'+str(np.mean(data))+'<SPLIT>'+str(variance)+'\n'
            w_lines.append(w_line)
    with open(w_path,'w')as w:
        for line in w_lines:
            w.write(line)

def get_tokens_with_highest_lowest_var(path,w_path):
    # get the tokens with highest or lowest variance
    # which occurance in the dataset more than 100 times
    w_lines = []
    with open(path,'r')as r:
        lines=r.readlines()
        for line in lines:
            token=line.split('<SPLIT>')[0]
            data=eval(line.split('<SPLIT>')[1].strip())
            if len(data)>=100:
                # utils = [w for sublist in data.values() for w in sublist]
                variance = np.var(data)
                w_line=token+'<SPLIT>'+str(len(data))+'<SPLIT>'+str(variance)+'<SPLIT>'+str(data)+'\n'
                w_lines.append(w_line)
    sorted_lines = sorted(w_lines, key=lambda x: float(x.split('<SPLIT>')[2].strip()), reverse=True)
    max_20=sorted_lines[:20]
    min_20=sorted_lines[-20:][::-1]
    with open(w_path+'var_max','w')as w:
        for line in max_20:
            w.write(line)
    with open(w_path+'var_min','w')as w:
        for line in min_20:
            w.write(line)

def get_dic(path):
    # get global attention scores(global average score) dic
    dic={}
    with open(path,'r')as r:
        lines=r.readlines()
        for line in lines:
            token,weight,var=line.split('<SPLIT>')
            dic[token.replace('Ġ','')]=(float(weight),float(var.strip()))
    return dic

def get_statistics(path,dic):
    # get Statistics for a category of CLS/Encoder-decoder attention scores
    # (Max/Min:the maximum/minimum of CLS/Encoder-decoder attention scores in this category
    # Global/Global_variance:the average/variance of the global attention scores of tokens for this category
    # Category-local/Local_variance:the averages/variance of category-local attention scores.)
    global wrong
    glo_weight=[]
    glo_var=[]
    weights=[]
    vars=[]
    with open(path,'r')as r:
        lines=r.readlines()
        for line in lines:
            # the weight here is the category-local attention
            # and var is the variance of this token's utils appear in this category
            token,weight,var=line.split('<SPLIT>')
            weights.append(float(weight))
            vars.append(float(var.strip()))
            try:
                glo_weight.append(dic[token.replace('Ġ','')][0])
                glo_var.append(dic[token.replace('Ġ', '')][1])
            except:
                pass
    max_weight=max(weights)
    min_weight=min(weights)
    avg_weight=np.mean(weights)
    avg_var=np.mean(vars)
    avg_glo_weight=np.mean(glo_weight)
    avg_glo_var=np.mean(glo_var)
    return max_weight,min_weight,avg_weight,avg_var,avg_glo_weight,avg_glo_var


if __name__ == '__main__':
    dirs=['./codesearch/weights/roberta','./code2nl/CodeBRT/weights']
    layers=['/layer_11','/layer_5']
    for i ,dir in enumerate(dirs):
        # Step1:analyse the attention utils of all the code snieept
        # for every token ,acquire the attention scores of all its appearance in the training dataset
        # write in a file named 'weights_of_every_token' format of every line :token<SPLIT>[weight1,weight2,...,weightn]
        layer=layers[i]
        tokenMap = {}
        all_tokens = read_tokens(dir + '/token')
        all_weights = read_weights(dir + layer + '/weights_all')
        assert len(all_tokens) == len(all_weights)
        for i in range(len(all_weights)):
            tokens = all_tokens[str(i)]
            attentions = all_weights[str(i)]
            assert len(tokens) == len(attentions)
            update_tokenMap(attentions, tokens, tokenMap)
        output_weight(dir + layer, tokenMap)
        # Step2:
        # for every token , caculate its glabal average attention score and variance
        # write into file named global_avg_and_var
        # in which every line contrains token<SPLIT>global_avg_weight<SPLIT>variance
        caculate_var(dir+ layer + '/weights_of_every_token', dir+ layer + '/global_avg_and_var')
        # Step3:
        # get the tokens with highest or lowest variance
        # which occurance in the dataset more than 100 times
        get_tokens_with_highest_lowest_var(dir+ layer + '/global_avg_and_var', dir+ layer)
        # Step4: code for Table1 and Table2
        # get Statistics of CLS/ Encoder-decoder attention scores
        # Max/Min:the maximum/minimum of CLS/ Encoder-decoder attention scores in each category
        # Global/Global_variance:the average/variance of the global attention scores of tokens for eachcategory;
        # Category-local/Local_variance:the averages/variance of category-local attention scores
        filename_to_open = 'token_weight'
        dic = get_dic(dir+layer+'/global_avg_and_var')
        with open(dir+layer + '/statistics', 'a+') as w:
            for entry in os.scandir(dir + '/category_weight'):
                file_path = os.path.join(entry.path, filename_to_open)
                max_w, min_w, avg_w, avg_v, g_avg_w, g_avg_v = get_statistics(file_path, dic)
                w_line = entry.name + '<SPLIT>' + str('{:.5f}'.format(max_w * 100)) + '<SPLIT>' + str(
                    '{:.5f}'.format(min_w * 100)) + '<SPLIT>' + str('{:.5f}'.format(avg_w * 100)) + '<SPLIT>' + str(
                    '{:.5f}'.format(avg_v * 10000)) + '<SPLIT>' + str('{:.5f}'.format(g_avg_w * 100)) + '<SPLIT>' + str(
                    '{:.5f}'.format(g_avg_v * 10000)) + '\n'
                w.write(w_line)



