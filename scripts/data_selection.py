import os
from PIL import Image
from tqdm import tqdm

def parseGroundTruth():
    with open('./scripts/ground_truth.txt') as f:
        parsedValue =  {}
        data = f.readlines()

        for n, line in enumerate(data, 1):
                chunks = line.split(' ')
                fileName = chunks[0]

                words = chunks[-1].split('|')
                words[-1] = words[-1].replace('\n', '')

                parsedValue[fileName] = ' '.join(words)

        f.close()

    return parsedValue

def selectLines(groundTruth, rootPath, distPath):
    selectWidth = 1800
    selectHeight = 100
    widthDelta = 40
    heightDelta = 10

    targetWidth = 448
    targetHeight = 24

    currentNumberSamples = 0
    targetNumberSamples = 1000

    widths = []
    heights = []

    with open('./scripts/selected_ground_truth.txt', 'w') as f:
        with tqdm(total = targetNumberSamples) as pbar:
            for folderName in os.listdir(rootPath):
                folderPath = os.path.join(rootPath, folderName)

                if os.path.isdir(folderPath):
                    for subFolderName in os.listdir(folderPath):
                        subFolderPath = os.path.join(folderPath, subFolderName)

                        if os.path.isdir(subFolderPath):
                            for fileName in os.listdir(subFolderPath):
                                img = Image.open(os.path.join(folderPath, subFolderName, fileName))
                                width, height = img.size
                                widths.append(width)
                                heights.append(height)

                                eligibleWidth = (selectWidth - widthDelta) <= width and width <= (selectWidth + widthDelta)
                                eligibleHeight = (selectHeight - widthDelta) <= height and height <= (selectHeight + heightDelta)

                                if eligibleWidth and eligibleHeight:
                                    outFilename = str(currentNumberSamples) + '.jpeg'
                                    img = img.resize(size=(targetWidth, targetHeight))
                                    img.save(os.path.join(distPath, outFilename))

                                    groudTruthFilename = fileName.split('.')[0]
                                    f.write('{}\|/{}\n'.format(str(currentNumberSamples), groundTruth[groudTruthFilename]))

                                    currentNumberSamples += 1
                                    pbar.update(1)

                                if currentNumberSamples > targetNumberSamples:
                                    return widths, heights

        # pbar.close()

    return widths, heights

if __name__ == '__main__':
    rootPath = './data/IAMDatabase'
    distPath = './data/handWritten'

    if os.path.exists(distPath):
        os.system('rm -r ' + distPath + '/')

    os.makedirs(distPath)

    groundTruth = parseGroundTruth()
    widths, heights = selectLines(groundTruth, rootPath, distPath)
    print('\nAverages:\n{}\n{}'.format(str(sum(widths) / len(widths)), str(sum(heights) / len(heights))))
