from PIL import Image
import pyocr
import pyocr.builders
import tensorflow as tf

def image_to_text(data, width, height):
    image = Image.new('RGB', (width, height), 'white')
    image.putdata(data, 1, 0)
    image.save('./test.jpg')

def parseArray(array):
    output = []
    for row in array:
        for col in row:
            output.append(tuple(map(lambda x: int(128 * x + 128), col)))

    return output

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./data/fontTyped/test/6.jpeg"))

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_jpeg(image_file)
    image = tf.image.transpose_image(image)

    # Start a new session to show example output.
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image_tensor, shape = sess.run([image, tf.shape(image)])

        imagep = Image.open("./data/fontTyped/test/6.jpeg")
        print(imagep.size)
        imageArray = image.eval()
        print(len(imageArray))
        # parsedArray = parseArray(imageArray)
        data = imagep.getdata()
        imagep.putdata(data)
        data1 = [data[i] for i in range(448*24)]
        subList = [data1[c:c+24] for c in range(0, len(data1), 24)]
        print(subList)
        data2 = []
        for i in range(448):
            for j in range(24):
                data2.append(imagep.getpixel((i,j)))
        print(data1 == data2)

        imagep.save('./output.jpg')

        coord.request_stop()
        coord.join(threads)

# with tf.Session() as sess:
#     image = tf.image.decode_jpeg('./data/fontTyped/test/6.jpeg', channels=3)
#     image = tf.image.transpose_image(image)
#
#     shape = tf.shape(image)
#     # (width, height, channels) = shape.eval()
#
#     imagep = Image.new('RGB', (448, 24), 'white')
#     imageArray = image.eval()
#     parsedArray = parseArray(imageArray)
#
#     imagep.putdata(parsedArray)
#
#     imagep.save('./output.jpg')
