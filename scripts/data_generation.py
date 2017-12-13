from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

img = Image.open('../data/times/background.png')

draw = ImageDraw.Draw(img)
font = ImageFont.truetype('./the_brooklyn/The Brooklyn.otf', 16)
draw.text((0, 0), 'Sample Text', (0, 0, 0), font=font)

img.save('sample-out.jpg')
