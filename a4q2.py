
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn


tr,te  = [],[]
with open('a4q2_train.txt', 'r') as f:
	lines = f.read().splitlines()
for line in lines:
	line = map(float, line.split())
        tr.append(line)
train = np.array(tr)

trX1 = train[:, [0]]
trX2 = train[:, [1]]
trY = train[:, [2]]

with open('a4q2_test.txt', 'r') as g:
	lines = g.read().splitlines()
for line in lines:
	line = map(float, line.split())
	te.append(line)
test = np.array(te)

teX1 = test[:, [0]]
teX2 = test[:, [1]]
teY = test[:,[2]]

features = [tf.contrib.layers.real_valued_column("x1"), tf.contrib.layers.real_valued_column("x2")]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
input_fn = tf.contrib.learn.io.numpy_input_fn({"x1":trX1, "x2":trX2}, trY, batch_size =4, num_epochs = 1000)
testing_in = tf.contrib.learn.io.numpy_input_fn({"x1": teX1, "x2":teX2}, teY, batch_size=4, num_epochs = 1000)
print(estimator.fit(input_fn=input_fn, steps=7000))
variable = estimator.get_variable_names()
for var in variable:
	print(var)
	print(estimator.get_variable_value(var))
print(estimator.evaluate(input_fn=testing_in))



