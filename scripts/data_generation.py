from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from tqdm import tqdm
import os

def create_image(text, success):
    img = Image.open('./scripts/background.png')

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./scripts/Times_New_Roman_Normal.ttf', 80)
    draw.text((0, 0), text, (0, 0, 0), font=font)
    string = '{}/{}.jpeg'.format(distPath, str(success))
    img.save(string)

success = 0
distPath = './data/fontTyped'
f = open('./scripts/Metamorphosis.txt', 'r')

if not os.path.exists(distPath):
    os.makedirs(distPath)

for line in f:
    a = line.split(' ')
    string = ''
    while len(string) < 53 and len(a) > 0:
        string = string + ' ' + a.pop(0)
        if len(string) > 49 and len(string) < 53:
            success += 1
            create_image(string, success)

            if success % 100 == 0:
                print('{} images created'.format(str(success)))

    if success > 1000:
        break
