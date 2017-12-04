import numpy as np
import tensorflow as tf

def init_weights(qtd):
    return tf.Variable(tf.random_normal(qtd, stddev=0.01))

# Model parameters
W = init_weights([2, 2])
b = tf.Variable([.5], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
model = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.nn.sigmoid(model) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)

# training data
x_train = [[0,1],[1,2],[2,3],[3,4]]
y_train = [[0],[1],[2],[3]]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(10000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})

print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
print(sess.run(model, {x:x_train, y:y_train}))