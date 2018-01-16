from PIL import Image
import pyocr
import pyocr.builders

def image_to_text(data, width, height):
    image = Image.new('RGB', (width, height), 'white')
    image.putdata(data, 1, 0)
    image.save('./test.jpg')

image = Image.open('./data/fontTyped/test/6.jpeg')
(width, height) = image.size
data = image.getdata()

image_to_text(data, width, height)
