import tensorflow as tf
import ops
import utils


class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_length=64, image_height=64):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_length = image_length
        self.image_height = image_height

    def __call__(self, input):
        """
        Args:
            input: batch_size x width x height x 3
        Returns:
            output: same size as input
        """
        with tf.variable_scope(self.name):
            # conv layers

            # Filters dim: (7, 7, 3); Output dim: (w, h, 64)
            c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                                 reuse=self.reuse, name='c7s1_32')
            # Filters dim: (3, 3, 64); Output dim: (w/2, h/2, 128)
            d64 = ops.dk(c7s1_32, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d64')
            # Filters dim: (3, 3, 128); Output dim: (w/4, h/4, 256)
            d128 = ops.dk(d64, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='d128')

            # Filters dim: (3, 3, 256); Output dim: (w/4, h/4, 256)
            res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)

            # fractional-strided convolution
            # Filters dim: (3, 3, 256); Output dim: (w/2, h/2, 128)
            u64 = ops.uk(res_output, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u64')
            # Filters dim: (3, 3, 128); Output dim: (w, h, 64)
            u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u32', output_length=self.image_length, output_height=self.image_height)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            # Filters dim: (3, 3, 64); Output dim: (w/2, h/2, 3)
            output = ops.c7s1_k(u32, 3, norm=None,
                    activation='tanh', reuse=self.reuse, name='output')

        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def sample(self, input):
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.transpose_image(tf.squeeze(image, [0]))
        image = tf.image.encode_jpeg(image)
        return image
