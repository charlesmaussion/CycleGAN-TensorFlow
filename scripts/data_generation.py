from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from tqdm import tqdm
import os

distPath = './data/fontTyped'

def create_image(distPath, text, fileName, i):
    finalImage = Image.open('./scripts/background.png')
    font = ImageFont.truetype('./scripts/Times_New_Roman_Normal.ttf', 40)
    draw = ImageDraw.Draw(finalImage)
    draw.text((0, 0), text, (0, 0, 0), font=font)

    path = '{}/{}.jpeg'.format(distPath, fileName)
    finalImage.save(path)

def generateFontTypedImages(distPath):
    currentNumberSamples = 0
    targetNumberSamples = 1000

    if not os.path.exists(distPath):
        os.makedirs(distPath)

    with open('./scripts/Metamorphosis.txt', 'r') as f:
        with tqdm(total = targetNumberSamples) as pbar:
            for line in f:
                a = line.split(' ')
                string = ''

                while len(string) < 53 and len(a) > 0:
                    string = string + ' ' + a.pop(0)

                    if len(string) > 49 and len(string) < 53:
                        currentNumberSamples += 1
                        pbar.update(1)
                        create_image(distPath, string, str(currentNumberSamples), currentNumberSamples)

                    if currentNumberSamples > targetNumberSamples:
                        return

            pbar.close()

generateFontTypedImages(distPath)
