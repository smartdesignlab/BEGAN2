import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def Generator(z, hidden_num, output_num, repeat_num,reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = upscale(x, 2)

        out = slim.conv2d(x, 1, 3, 1, activation_fn=None)  # 3->1로 변경(channel)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def Discriminator(x, input_channel, z_num, repeat_num, hidden_num):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)

        
        for idx in range(repeat_num):
            channel_num = repeat_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = upscale(x, 2)
        x = slim.dropout(x, 0.7)     
        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape
    
def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def reshape(x, h, w, c):
    x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size):
  
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale):
    _, h, w, _ =  x.shape
    return resize_nearest_neighbor(x,  (h.value*scale, w.value*scale))
