import os
import random
 
trainval_percent = 1.0
train_percent = 0.8
xmlfilepath = 'C:\\Users\\dell\\Desktop\\VOC\\Annotations'
txtsavepath = 'C:\\Users\\dell\\Desktop\\VOC\\ImageSets\\Main'
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
ftrainval = open('C:\\Users\\dell\\Desktop\\VOC\\ImageSets\\Main\\trainval.txt', 'w')
ftest = open('C:\\Users\\dell\\Desktop\\VOC\\ImageSets\\Main\\test.txt', 'w')
ftrain = open('C:\\Users\\dell\\Desktop\\VOC\\ImageSets\\Main\\train.txt', 'w')
fval = open('C:\\Users\\dell\\Desktop\\VOC\\ImageSets\\Main\\val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()