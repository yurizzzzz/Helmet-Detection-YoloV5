import os
from tqdm import tqdm

train_list = [
    file
    for file in os.listdir('C:/Users/dell/Desktop/datasets/val/Annotations')
    if file.endswith('.xml')
]
train_list = tqdm(train_list)
j = 0

for i in train_list:
    os.rename(
        'C:/Users/dell/Desktop/datasets/val/Annotations/' + i,
        'C:/Users/dell/Desktop/datasets/val/Annotations/' +
        '{:0>4s}'.format(str(j)) + '.xml')
    j += 1

# for i in range(100):
#     s = str(i + 1)
#     s = s.zfill(6)
#     os.rename('C:/Users/dell/Desktop/HR/DIV2K_' + s + '.png', 'C:/Users/dell/Desktop/HR/' + str(i+1) + '.jpg')
