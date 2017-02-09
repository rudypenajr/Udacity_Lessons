import tensorflow as tf

# Output Depth
k_output = 64

# Image Properities
image_width = 10
image_height = 10
color_channels = 3

# Convolution Filter
filter_size_width = 5
filter_Size_height = 5

# Input/Image
input = tf.placeholder(tf.float32, shape=[None, image_width, image_height, color_channels])

# Weight/Bias
weight = tf.placeholder(
    tf.truncated_normal([filter_size_width, filter_Size_height, color_channels, k_output])
)
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
# Note:
#   - `weight` is equal to filter
#   - TF uses a stride for each `input` dimension, [batch, input_height, input_width, input_channels]
#     Generally always set the stride for `batch` and `input_channels` to 1
conv_layer = tf.nn.conv2d(input, weight, strides=[1,2,2,1], padding='SAME')

# Add Bias
# Note: `tf.nn.bias_add` adds a 1-d bias to the last dimension in a matrix
conv_layer = tf.nn.bias_add(conv_layer, bias)

# Apply Activation Function
conv_layer = tf.nn.relu(conv_layer)