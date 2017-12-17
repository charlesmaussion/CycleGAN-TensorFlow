import os
import random

seed = 490
random.seed(seed)

testProportion = 0.25

folderA = './data/selectedLines'
folderB = './data/times'

for folderName in [folderA, folderB]:
    for subFolder in ['train', 'test']:
        path = os.path.join(folderName, subFolder)

        if not os.path.isdir(path):
            os.makedirs(path)

    fileNameList = list(filter(lambda x: not os.path.isdir(os.path.join(folderName, x)), os.listdir(folderName)))
    random.shuffle(fileNameList)
    n = len(fileNameList)

    for i, fileName in enumerate(fileNameList):
        destinationFolder = 'train' if i < int(testProportion * n) else 'test'
        os.system('mv {} {}'.format(os.path.join(folderName, fileName), os.path.join(folderName, destinationFolder)))
