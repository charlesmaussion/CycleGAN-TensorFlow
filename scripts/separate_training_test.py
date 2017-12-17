import os
import random
from tqdm import tqdm

seed = 490
random.seed(seed)

testProportion = 0.25

folderA = './data/handWritten'
folderB = './data/fontTyped'

for folderName in tqdm([folderA, folderB]):
    for subFolder in ['train', 'test']:
        path = os.path.join(folderName, subFolder)

        if not os.path.isdir(path):
            os.makedirs(path)

    fileNameList = list(filter(lambda x: not os.path.isdir(os.path.join(folderName, x)), os.listdir(folderName)))
    random.shuffle(fileNameList)
    n = len(fileNameList)

    for i, fileName in tqdm(enumerate(fileNameList)):
        destinationFolder = 'test' if i < int(testProportion * n) else 'train'
        os.system('mv {} {}'.format(os.path.join(folderName, fileName), os.path.join(folderName, destinationFolder)))
