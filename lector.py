import tensorflow as tf
from PIL import Image
import sys

import pyocr
import pyocr.builders


def image_to_text(imagePath=None, data=None):
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print('No OCR tool found')
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    # print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    # print("Available languages: %s" % ', '.join(langs))
    lang = langs[0]
    # print("Will use lang '%s'" % (lang))

    if imagePath:
        image = Image.open(imagePath)
    else:
        [batch_size, width, height, channels] = data.get_shape().as_list()
        image = Image.new('RGB', (width, height), 'white')
        image.putdata(tf.squeeze(data, [0]).eval())

    txt = tool.image_to_string(
        image,
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )

    # txt is a Python string
    return txt


# print(image_to_text(imagePath='./data/fontTyped/train/1.jpeg'))
# print(image_to_text(imagePath='./data/fontTyped/train/20.jpeg'))
