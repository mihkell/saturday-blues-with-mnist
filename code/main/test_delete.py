import numpy as np
import tensorflow as tf

into = tf.placeholder(tf.float32, (None, 6))
layer = tf.nn.relu(into)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

x = [[1., 1., 2., 1., -1., 2.]]
result = sess.run(layer, {into: x})
print(result.shape)
print(result)
