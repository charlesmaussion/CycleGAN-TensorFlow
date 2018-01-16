import os
import random
from tqdm import tqdm

def parseSelectedGroundTruth():
    with open('./scripts/selected_ground_truth.txt') as f:
        parsedValue =  {}
        data = f.readlines()

        for n, line in enumerate(data, 1):
                chunks = line.split('\|/')
                fileName = chunks[0]

                words = chunks[-1]
                words = words.replace('\n', '')

                parsedValue[fileName] = words

        f.close()

    return parsedValue

if __name__ == '__main__':
    seed = 490
    random.seed(seed)

    testProportion = 0.25

    folderA = './data/handWritten'
    folderB = './data/fontTyped'

    selectedGroundTruth = parseSelectedGroundTruth()
    print(selectedGroundTruth)

    with open('./scripts/train_ground_truth.txt', 'w') as f:
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

                if destinationFolder == 'train':
                    fileNumber = fileName.split('.')[0]
                    if len(fileNumber) > 0:
                        f.write('{}\|/{}\n'.format(fileNumber, selectedGroundTruth[fileNumber]))
