import os
from PIL import Image
from tqdm import tqdm

rootPath = './data/IAMDatabase'
distPath = './data/handWritten'

if os.path.exists(distPath):
    os.system('rm ' + distPath + '/*')
else:
    os.makedirs(distPath)

def selectLines(rootPath, distPath):
    selectWidth = 1800
    selectHeight = 100
    widthDelta = 40
    heightDelta = 10

    targetWidth = 128
    targetHeight = 128

    currentNumberSamples = 0
    targetNumberSamples = 1000

    widths = []
    heights = []

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
                                img = img.resize(size=(targetWidth, targetHeight))
                                img.save(os.path.join(distPath, str(currentNumberSamples) + '.jpeg'))
                                currentNumberSamples += 1
                                pbar.update(1)

                            if currentNumberSamples > targetNumberSamples:
                                return widths, heights

        # pbar.close()

    return widths, heights

widths, heights = selectLines(rootPath, distPath)
print('\nAverages:\n{}\n{}'.format(str(sum(widths) / len(widths)), str(sum(heights) / len(heights))))
