# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)
#
# import tensorflow as tf
#
# #Parameters
# learning_rate = 0.001
# training_epochs = 20
# batch_size = 128
# display_step = 1
#
# n_input = 784 # MNIST data input (img shape: 28*28)
# n_classes = 10 # MNIST total classes (0-9 digits)
#
# # variable n_hidden_layer determines the size of the hidden layer in the NN
# # also known as the width of a layer
# n_hidden_layer = 256 # layer number of features
#
# # Store layers weight & bias
# weights = {
#     'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
#     'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
# }
#
# biases = {
#     'hidden_layer': tf.Variable(tf.random_normal(n_hidden_layer)),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
#
# # tf Graph Input
# x = tf.placeholder("float", [None, 28, 28, 1])
# y = tf.placeholder("float", [None, n_classes])
#
# # reshapes the 28px by 28px matrices in x into vectors of 784px by 1px.
# x_flat = tf.reshape(x, [-1, n_input])
#
# # Hidden layer with RELU Activation
# layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
# layer_1 = tf.nn.relu(layer_1)
#
# # Output layer with linear activation
# logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])
#
# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# # Launch the Graph
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Training Cycle
#     for epoch in range(training_epochs):
#         total_batch = int(mnist.train.num_examples/batch_size)
#
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_x, batch_y = mnist.train.next_batch(batch_size)
#             # Run optimization op (backprop) and cost op (to get loss value
#             output = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
#             print(output)