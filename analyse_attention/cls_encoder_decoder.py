import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
# Figure 4: ‘CLS’ and encoder-decoder attentions on the input of the example of bubble sort.

ma=[]
tokens=[]
with open('./attention_scores','r')as r:
    lines=r.readlines()
    for line in lines:
        ma.append(eval(line.split('<SPLIT>')[1].strip()))
with open('./token','r')as r:
    line=r.readlines()[0]
    tokens=eval(line.split('<SPLIT>')[1].strip())
# shape of attention_matrix : 200x200 (self attention) (contains the cls attention)


padding_index = tokens[1:].index('<s>') + 1
tokens_without_padding = tokens[:padding_index]
attention_matrix_without_padding = [row[:padding_index] for row in ma[:padding_index]]
tokens_new = []
attention_matrix_new = []

for index, token in enumerate(tokens_without_padding):
    token_cleaned = token.replace('Ġ', '')
    if token_cleaned and token_cleaned != ' ':
        tokens_new.append(token_cleaned)
        attention_matrix_new.append([row[index] for row in attention_matrix_without_padding])

attention_matrix_new = list(map(list, zip(*attention_matrix_new)))
attention_matrix_without_padding=attention_matrix_new

separator_index = tokens_new.index('</s>')
description_tokens = tokens_new[:separator_index]
tokens_cls = tokens_new[separator_index + 1:]

description_to_code_attention = [
    row[separator_index + 1:] for row in attention_matrix_without_padding[:separator_index]
]
cls_attention=[
    row for row in attention_matrix_without_padding[:1]
]
cls_to_code_attention = [
    row[separator_index + 1:] for row in attention_matrix_without_padding[:1]
]
cls_to_describe_attention=  [
    row[:separator_index] for row in attention_matrix_without_padding[:1]
]

weight_cross=[ 0.030509009957313538, 0.27568450570106506, 0.09202372282743454, 0.07329730689525604, 0.023065324872732162, 0.024194981902837753, 0.014893531799316406, 0.0008815412293188274, 0.0020640133880078793, 0.008061666041612625, 0.004436918534338474, 0.01849263533949852, 0.007955043576657772, 0.004070699214935303, 0.0012783127604052424, 0.0006491352687589824, 0.0013718127738684416, 0.0006753314519301057, 0.0007730414508841932, 0.0024972562678158283, 0.0021290823351591825, 0.036508459597826004, 0.010310453362762928, 0.0018013067310675979, 0.0003284827107563615, 0.0004432021523825824, 0.0015177035238593817, 0.06836757063865662, 0.0005731540732085705, 0.0013341134181246161, 0.0011626677587628365, 0.0011021229438483715, 0.001221867511048913, 0.03185426443815231, 0.008120223879814148, 0.002114947885274887, 0.010079992935061455, 0.0029478571377694607, 0.0007359916344285011, 0.0034352410584688187, 0.014212551526725292, 0.00987257994711399, 0.0019271228229627013, 0.00039229891262948513, 0.0010406807996332645, 0.0016307076439261436, 0.0015071736415848136, 0.0050208233296871185, 0.011318644508719444, 0.0033124820329248905, 0.0006638695485889912, 0.0030927720945328474, 0.00946046318858862, 0.0029728091321885586, 0.0004762086900882423, 0.001391328638419509, 0.0017248447984457016, 0.0015555372228845954, 0.004669420421123505, 0.009579085744917393, 0.0014329776167869568, 0.00032884819665923715, 0.0015392638742923737, 0.0023977719247341156, 0.001418412895873189, 0.0066956388764083385, 0.009824621491134167, 0.001508761546574533, 0.0005030931788496673, 0.0015186530072242022, 0.0733971819281578]
code_token_cros=[ 'def', 'Ġbubble', 'Sort', 'Ġ(', 'Ġarr', 'Ġ)', 'Ġ:', 'Ġn', 'Ġ=', 'Ġlen', 'Ġ(', 'Ġarr', 'Ġ)', 'Ġfor', 'Ġi', 'Ġin', 'Ġrange', 'Ġ(', 'Ġn', 'Ġ-', 'Ġ1', 'Ġ)', 'Ġ:', 'Ġfor', 'Ġj', 'Ġin', 'Ġrange', 'Ġ(', 'Ġn', 'Ġ-', 'Ġ1', 'Ġ-', 'Ġi', 'Ġ)', 'Ġ:', 'Ġif', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ]', 'Ġ>', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ+', 'Ġ1', 'Ġ]', 'Ġ:', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ]', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ+', 'Ġ1', 'Ġ]', 'Ġ=', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ+', 'Ġ1', 'Ġ]', 'Ġ,', 'Ġarr', 'Ġ[', 'Ġj', 'Ġ]', '</s>']

norm_cls = Normalize(vmin=np.percentile(attention_matrix_without_padding, 25), vmax=np.percentile(attention_matrix_without_padding, 85))
norm_cross = Normalize(vmin=np.percentile(weight_cross, 15), vmax=np.percentile(weight_cross, 85))

code_token_cross = [code.replace('Ġ', '') for code in code_token_cros]

cmap = 'Reds'
cmap2 = 'Blues'
fig = plt.figure(figsize=(30, 9))

# CLS-to-code attention
ax1 = fig.add_subplot(3, 1, 1)
im1 = ax1.imshow(cls_to_code_attention, cmap=cmap, norm=norm_cls)
ax1.set_aspect(2.2)

# ax1.set_xticks(np.arange(0, len(tokens_cls), 2))
# ax1.set_xticklabels(tokens_cls[0::2], rotation=0, ha='center', fontsize=14, va='bottom', y=-0.25)  # 调整 y 偏移
#
# ax1.set_xticks(np.arange(1, len(tokens_cls), 2), minor=True)
# ax1.set_xticklabels(tokens_cls[1::2], minor=True, rotation=0, ha='center', fontsize=14, va='bottom', y=-0.5)  # 调整 y 偏移

ax1.set_xticks(np.arange(len(tokens_cls)))
ax1.set_xticklabels(tokens_cls, rotation=45, ha='right', fontsize=18, verticalalignment='top')

ax1.set_yticks(np.arange(len(description_tokens[:1])))
ax1.set_yticklabels(description_tokens[:1], fontsize=18)
ax1.set_title('CLS-to-code attention (a)', fontsize=25)

# CLS-to-description attention
ax2 = fig.add_subplot(3, 1, 2)
im2 = ax2.imshow(cls_to_describe_attention, cmap=cmap, norm=norm_cls)
ax2.set_aspect(1.2)

# ax2.set_xticks(np.arange(0, len(description_tokens), 2))
# ax2.set_xticklabels(description_tokens[0::2], rotation=0, ha='center', fontsize=14, va='bottom', y=-0.25)  # 调整 y 偏移
#
# ax2.set_xticks(np.arange(1, len(description_tokens), 2), minor=True)
# ax2.set_xticklabels(description_tokens[1::2], minor=True, rotation=0, ha='center', fontsize=14, va='bottom', y=-0.5)  # 调整 y 偏移

ax2.set_xticks(np.arange(len(description_tokens)))
ax2.set_xticklabels(description_tokens, rotation=40, ha='right', fontsize=18)

ax2.set_yticks(np.arange(len(description_tokens[:1])))
ax2.set_yticklabels(description_tokens[:1], fontsize=18)
ax2.set_title('CLS-to-description attention (b)', fontsize=25)

# Encoder-decoder attention for generating bubble token
ax3 = fig.add_subplot(3, 1, 3)
weight_2d = np.array(weight_cross)[np.newaxis, :]
im3 = ax3.imshow(weight_2d, cmap=cmap2, norm=norm_cross)
ax3.set_aspect(2.2)

ax3.set_xticks(np.arange(len(code_token_cross)))
ax3.set_xticklabels(code_token_cross, rotation=45, ha='right', fontsize=18, verticalalignment='top')

# ax3.set_xticks(np.arange(0, len(code_token_cross), 2))
# ax3.set_xticklabels(code_token_cross[0::2], rotation=0, ha='center', fontsize=14, va='bottom', y=-0.25)  # 调整 y 偏移
#
# ax3.set_xticks(np.arange(1, len(code_token_cross), 2), minor=True)
# ax3.set_xticklabels(code_token_cross[1::2], minor=True, rotation=0, ha='center', fontsize=14, va='bottom', y=-0.5)  # 调整 y 偏移

ax3.set_yticks([0])
ax3.set_yticklabels(['bubble'], fontsize=18)
ax3.set_title('Encoder-decoder attention for generating bubble token (c)', fontsize=25)

cbar_ax_right = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar_right = fig.colorbar(im1, cax=cbar_ax_right)
cbar_right.ax.tick_params(labelsize=14)

cbar_ax_left = fig.add_axes([0.02, 0.15, 0.02, 0.7])
cbar_left = fig.colorbar(im3, cax=cbar_ax_left)
cbar_left.ax.tick_params(labelsize=14)

plt.subplots_adjust(hspace=0.6, left=0.1, right=0.91)
# plt.show()
plt.savefig('heatmap_cls-en-de.pdf', format='pdf')
