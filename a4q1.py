#!/usr/bin/python

import tensorflow as tf

###############################################################################
# Create the tensor x of the value 
#   [[[2, -2, 4], [0, 1, 2]], [[1, 2, 3], [7, 1, 9]]] 
# and y as a tensor of ones with the same shape as x.
# Return a Boolean tensor that yields Trues with the same shape as x and y if 
# x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################
def p1():
    x = tf.constant([[[2, -2, 4], [0, 1, 2]], [[1, 2, 3], [7, 1, 9]]])
    y = tf.ones_like(x)
    z = tf.equal(x, y)
    return z

################################################################################
# Creates one variable 'x' of the value [3., -4.] and a placeholder 'y' of the 
# same shape as 'x'. Given a scalar z returns a triple containing 
#   x
#   y 
#   and a TensorFlow object that returns x + y if z > 0, and x - y otherwise. 
# Hint: Look up tf.cond().
################################################################################
def p2(z):
    x = tf.Variable([3., -4.])
    y = tf.placeholder(tf.float32,shape= (1,2)) #x.get_shape())
    result = tf.cond(tf.greater(z,0), lambda: tf.add(x,y), lambda: tf.subtract(x,y))
    obj = [x,y,result]
    return obj

###############################################################################
# Given 2d tensors x, y, and z, returns a tensor object for  x' * y^-1 + z 
# where x' is the transpose of x and y^-1 is the inverse of y. The dimensions 
# of tensors will be compatible.
# Hint: See "Matrix Math Functions" in TensorFlow documentation.
###############################################################################
def p3(x, y, z):
    y_inv = tf.matrix_inverse(y)
    x_trans = tf.transpose(x)
    mul = tf.matmul(x_trans, y_inv)
    result = tf.add(mul,z)
    return result

###############################################################################
# Given a TensorFlow object that describes a convex function and a TensorFlow 
# session, return the the minimum value of the function found by gradient 
# decent. Use 0.01 learning rate and 10000 steps. 
###############################################################################
def p4(objective_function, sess):
    learning_rate = .01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(objective_function)
    for _ in xrange(10000):
        sess.run(train)
    return sess.run(objective_function)




