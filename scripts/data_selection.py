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
    acc = 0
    targetWidth = 1800
    targetHeight = 100
    widthDelta = 40
    heightDelta = 10
    targetNumberSamples = 1000

    widths = []
    heights = []

    for folderName in tqdm(os.listdir(rootPath)):
        folderPath = os.path.join(rootPath, folderName)

        if os.path.isdir(folderPath):
            for subFolderName in tqdm(os.listdir(folderPath)):
                subFolderPath = os.path.join(folderPath, subFolderName)

                if os.path.isdir(subFolderPath):
                    for fileName in os.listdir(subFolderPath):
                        img = Image.open(os.path.join(folderPath, subFolderName, fileName))
                        width, height = img.size
                        widths.append(width)
                        heights.append(height)

                        eligibleWidth = (targetWidth - widthDelta) <= width and width <= (targetWidth + widthDelta)
                        eligibleHeight = (targetHeight - widthDelta) <= height and height <= (targetHeight + heightDelta)
                        if eligibleWidth and eligibleHeight:
                            img = img.resize(size=(targetWidth, targetHeight))
                            img.save(os.path.join(distPath, str(acc) + '.jpeg'))
                            acc += 1

                        if acc > targetNumberSamples:
                            return widths, heights

    return widths, heights

widths, heights = selectLines(rootPath, distPath)

print('\nAverages:')
print(sum(widths) / len(widths))
print(sum(heights) / len(heights))
