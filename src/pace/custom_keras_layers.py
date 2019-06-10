from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Concatenate
import numpy as np
import tensorflow as tf

# craft: note this file was originally called WeaveAndRSD.py

class WeaveConcatenate(Concatenate):

    # note need to go back and correctly handle axis and the other stuff here i'm not considering
    def __init__(self, axis=-1, **kwargs):
        super(WeaveConcatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

    def _merge_function(self, inputs):
        # here we are assuming equal length inputs to merge
        # so we can split inputs in half
        # for now ignore axis argument.
        # doing the weave with reshape, transpose, and another reshape.

        # https://stackoverflow.com/questions/44952886/tensorflow-merge-two-2-d-tensors-according-to-even-and-odd-indices
        # pretty sure this is doing weave correctly but should check with some real tensor examples.
        a = inputs[0]
        b = inputs[1]
        w = tf.reshape(
            tf.concat([a[..., tf.newaxis], b[..., tf.newaxis]], axis=-1),
            [tf.shape(a)[0], -1])
        return w


def Weaveconcatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `WeaveConcatenate` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    return WeaveConcatenate(axis=axis, **kwargs)(inputs)


class RSDLayer(Dense):
    # note: 'units' is output_length. using keras terminology
    def __init__(self, units, input_num_blocks, **kwargs):
        self.units = units
        self.input_num_blocks = input_num_blocks
        super(RSDLayer, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # input_dim = input_shape[-1]

        input_length = input_shape[1]

        self.input_block_size = input_length // self.input_num_blocks
        # checks for correct arguments to integer division..

        # step down:
        self.output_num_blocks = self.input_num_blocks - 1

        output_block_size = self.units // self.output_num_blocks

        self.kernel = self.add_weight(shape=(self.input_block_size * 2, output_block_size, self.output_num_blocks),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_length})
        self.built = True

    def call(self, x):
        # do the pairwise dense tensor dot products and concatenate
        # y = []
        seqPos = 0
        seqLength = self.input_block_size * 2
        for i in range(self.output_num_blocks):
            temp = K.dot(x[:, seqPos:seqPos + seqLength], self.kernel[:, :, i])
            if i == 0:
                y = temp
            else:
                y = K.concatenate([y, temp], axis=1)
            seqPos = seqPos + self.input_block_size

        output = y
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

# class definition finished
