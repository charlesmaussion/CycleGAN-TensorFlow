from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def create_image(text, success):
    img = Image.open('../data/times/background.png')

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('./Times_New_Roman_Normal.ttf', 80)
    draw.text((0, 0), text, (0, 0, 0), font=font)
    string = '../data/times/image' + str(success) + '.png'
    img.save(string)

success = 0

f = open('text.txt','r')
for line in f:
    a = line.split(' ')
    string = ''
    while len(string) < 53 and len(a) > 0:
        string = string + ' ' + a.pop(0)
        if len(string) > 49 and len(string) < 53:
            success += 1
            create_image(string, success)
            if success % 100 == 0:
                print('%i images created'%(success))
    if success > 1000:
        break
