import os
from tqdm import tqdm

for i in range(1083):
    filename = '{:0>4s}'.format(str(i))
    size = os.path.getsize('C:/Users/dell/Desktop/VOC2021/labels/' + filename +
                           '.txt')
    if size == 0:
        os.remove('C:/Users/dell/Desktop/VOC2021/labels/' + filename + '.txt')
        os.remove('C:/Users/dell/Desktop/VOC2021/JPEGImages/' + filename + '.jpg')
        os.remove('C:/Users/dell/Desktop/VOC2021/Annotations/' + filename + '.xml')
