import tensorflow as tf
import ops
import utils


class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_length=900, image_height=25):
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
            c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                                 reuse=self.reuse, name='c7s1_32')                                                      # (?, w, h, 64)
            d64 = ops.dk(c7s1_32, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d64')                                                                  # (?, w/2, h/2, 128)
            d128 = ops.dk(d64, 4 * self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='d128')                                                                # (?, w/4, h/4, 256)

            res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)                                                  # (?, w/4, h/4, 256)

            # fractional-strided convolution
            u64 = ops.uk(res_output, 2 * self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u64')                                                                  # (?, w/2, h/2, 128)
            u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u32', output_length=self.image_length, output_height=self.image_height)# (?, w, h, 64)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            output = ops.c7s1_k(u32, 3, norm=None,
                    activation='tanh', reuse=self.reuse, name='output')                     # (?, w, h, 3)
        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

    def sample(self, input):
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
