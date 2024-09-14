import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Figure 5: The variance of the top and bottom 10 tokens of CodeBert for code search and summarization.

# 1.Top 10 tokens with highest variance in encoder-decoder attention
with open('./var_max_en', 'r') as r:
    lines = r.readlines()
    for i, line in enumerate(lines[:10]):
        token, num, var, data = line.split('<SPLIT>')
        token = token.replace('Ġ', '')
        data = eval(data.strip())
        df_ = pd.DataFrame({'值': data, '组': [token] * len(data)})
        if i == 0:
            df = pd.concat([df_])
        else:
            df = pd.concat([df, df_])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='组', y='值', data=df)
    sns.despine()
    plt.xlabel('Token', fontsize=17)  # Adding x-axis label
    plt.ylabel('Encoder-decoder Attention', fontsize=17)
    plt.xticks(fontsize=17)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Top 10 tokens with highest variance in encoder-decoder attention.pdf', format='pdf')

# 2.Bottom 10 tokens with lowest variance in CLS attention
with open('./var_min_cls', 'r') as r:
    lines = r.readlines()
    for i, line in enumerate(lines[:10]):
        token, num, var, data = line.split('<SPLIT>')
        token = token.replace('Ġ', '')
        data = eval(data.strip())
        df_ = pd.DataFrame({'值': data, '组': [token] * len(data)})
        if i == 0:
            df = pd.concat([df_])
        else:
            df = pd.concat([df, df_])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='组', y='值', data=df)
    sns.despine()
    plt.xlabel('Token', fontsize=17)  # Adding x-axis label
    plt.ylabel('Encoder-decoder Attention', fontsize=17)
    plt.xticks(fontsize=17)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Bottom 10 tokens with lowest variance in CLS attention.pdf', format='pdf')

# 3.Top 10 tokens with highest variance in CLS attention
with open('./var_max_cls', 'r') as r:
    lines = r.readlines()
    for i, line in enumerate(lines[:10]):
        token, num, var, data = line.split('<SPLIT>')
        token = token.replace('Ġ', '')
        data = eval(data.strip())
        df_ = pd.DataFrame({'值': data, '组': [token] * len(data)})
        if i == 0:
            df = pd.concat([df_])
        else:
            df = pd.concat([df, df_])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='组', y='值', data=df)
    sns.despine()
    plt.xlabel('Token', fontsize=17)  # Adding x-axis label
    plt.ylabel('Encoder-decoder Attention', fontsize=17)
    plt.xticks(fontsize=17)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Top 10 tokens with highest variance in CLS attention.pdf', format='pdf')

# 4.Bottom 10 tokens with lowest variance in encoder-decoder attention
with open('./var_min_en', 'r') as r:
    lines = r.readlines()
    for i, line in enumerate(lines[:10]):
        token, num, var, data = line.split('<SPLIT>')
        token = token.replace('Ġ', '')
        data = eval(data.strip())
        df_ = pd.DataFrame({'值': data, '组': [token] * len(data)})
        if i == 0:
            df = pd.concat([df_])
        else:
            df = pd.concat([df, df_])
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='组', y='值', data=df)
    sns.despine()
    plt.xlabel('Token', fontsize=17)  # Adding x-axis label
    plt.ylabel('Encoder-decoder Attention', fontsize=17)
    plt.xticks(fontsize=17)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Bottom 10 tokens with lowest variance in encoder-decoder attention.pdf', format='pdf')








