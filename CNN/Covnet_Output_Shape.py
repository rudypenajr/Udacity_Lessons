import tensorflow as tf

########################
########################
input = tf.placeholder(tf.float32, (None, 32, 32, 3)) # input
# (height, width, input_depth, output_depth)
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # filters
filter_bias = tf.Variable(tf.zeros(20))
strides = [1,2,2,1]
padding = 'VALID'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
# expected output shape of conv would be [1,13,13,20]

########################
# Quiz: Pooling Mechanics
########################
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)
# output shape of pool will be [1, 2, 2, 5], even if `padding` is changed to `SAME`


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(conv)
    print(output)
