"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

# import numpy as np
# from miniflow import *

"""
Lesson 9: Cost
# Answer: 23.4166666667
# Not Getting the Right Answer
"""
# y, a = Input(), Input()
# cost = MSE(y, a)

# y_ = np.array([1, 2, 3])
# a_ = np.array([4.5, 5, 10])

# feed_dict = {y: y_, a: a_}
# graph = topological_sort(feed_dict)
# forward pass
# forward_pass(graph)

"""
Expected output

23.4166666667
"""
# print(cost.value)


"""
Lesson 8: Sigmoid Function
# Not Getting the Right Answer
"""
# X, W, b = Input(), Input(), Input()
#
# f = Linear(X, W, b)
# g = Sigmoid(f)
#
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
#
# feed_dict = {X: X_, W: W_, b: b_}
#
# graph = topological_sort(feed_dict)
# output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
# print(output)


"""
Lesson 6: Learning and Loss
"""
# inputs, weights, bias = Input(), Input(), Input()
#
# f = Linear(inputs, weights, bias)
#
# feed_dict = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
# print("output: ", output)


"""
Lesson 4 & 5: Forward Propagation

"""
# x, y = Input(), Input()
# x, y, z = Input(), Input(), Input()

# f = Add(x, y)
# f = Add(x, y, z)

# feed_dict = {x: 10, y: 5}
# feed_dict = {x: 4, y: 5, z: 10}

# sorted_nodes = topological_sort(feed_dict)
# output = forward_pass(f, sorted_nodes)
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)


# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
# print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
# print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
