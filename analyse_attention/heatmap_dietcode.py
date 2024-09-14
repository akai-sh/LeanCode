import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# figure6-a heatmap of Self-attention scores used by DietCode
text = [
    ['def', 'Ġbubble', 'Sort', '(', 'arr', '):'],
    ['Ġn', 'Ġ=', 'Ġlen', '(', 'arr', ')'],
    ['Ġfor', 'Ġi', 'Ġin', 'Ġrange', '(', 'n', 'Ġ-', 'Ġ1', '):'],
    ['Ġfor', 'Ġj', 'Ġin', 'Ġrange', '(', 'n', 'Ġ-', 'Ġ1', 'Ġ-', 'Ġi', '):'],
    [ 'Ġif', 'Ġarr', '[', 'j', ']', 'Ġ>', 'Ġarr', '[', 'j', 'Ġ+', 'Ġ1', ']:'],
    [ 'Ġarr', '[', 'j', '],', 'Ġarr', '[', 'j', 'Ġ+', 'Ġ1', ']', 'Ġ=', 'Ġarr', '[', 'j', 'Ġ+', 'Ġ1', '],'],
    ['Ġarr', '[', 'j', ']']
]

sum_len=sum(len(row) for row in text)
max_length = max(len(row) for row in text)

text_padded_list = [row + [''] * (max_length - len(row)) for row in text]
text_padded_np = np.array(text_padded_list)


# self attention values
values=[0.010802875272929668, 0.005576903000473976, 0.007220533210784197, 0.009333671070635319, 0.004940148442983627, 0.010889705270528793, 0.004858548287302256, 0.0044057006016373634, 0.003878258168697357, 0.003967720549553633, 0.004133904352784157, 0.00599263422191143, 0.006758876610547304, 0.005196806509047747, 0.004609977826476097, 0.0041518015787005424, 0.005144626833498478, 0.004515495616942644, 0.004750418476760387, 0.0027575085405260324, 0.008025869727134705, 0.004884077236056328, 0.004344541113823652, 0.004232617560774088, 0.0037970186676830053, 0.006809974554926157, 0.003923951182514429, 0.004091339651495218, 0.0031443543266505003, 0.004864844493567944, 0.0043549430556595325, 0.007557119242846966, 0.005554207134991884, 0.0036523176822811365, 0.004763451404869556, 0.002807542448863387, 0.0038361253682523966, 0.007208491675555706, 0.0031439634039998055, 0.005193482618778944, 0.0028458621818572283, 0.004293190315365791, 0.0031068865209817886, 0.00789548922330141, 0.003916734363883734, 0.0051392377354204655, 0.002630856353789568, 0.008501756936311722, 0.0032123730052262545, 0.004581917077302933, 0.002559886546805501, 0.004125684965401888, 0.003485896857455373, 0.00403300765901804, 0.00977752823382616, 0.003294259775429964, 0.008648610673844814, 0.002893261145800352, 0.005172452889382839, 0.0034235569182783365, 0.011833596043288708, 0.004067128524184227, 0.004846702329814434, 0.003206337569281459, 0.006992546375840902]
values_log_transformed = np.log(np.array(values) + 1e-6)
padded_values_log_transformed = np.full(text_padded_np.shape, np.nan)  # 先填充为nan
value_index = 0
for i, row in enumerate(text_padded_np):
    for j, _ in enumerate(row):
        if text_padded_np[i, j]:
            padded_values_log_transformed[i, j] = values_log_transformed[value_index]
            value_index += 1

plt.figure(figsize=(14, 8))
ax = sns.heatmap(padded_values_log_transformed, cmap='Blues', cbar=True, linewidths=0.5, linecolor='lightgrey')

for i in range(text_padded_np.shape[0]):
    for j in range(text_padded_np.shape[1]):
        if text_padded_np[i, j]:
            ax.text(j + 0.5, i + 0.5, text_padded_np[i, j].replace('Ġ',''),
                    ha="center", va="center", color="black")

plt.title('Self-attention scores used by DietCode')
plt.show()
# plt.savefig('Heatmap_self.pdf', format='pdf')