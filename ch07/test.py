import tensorflow as tf
import numpy as np

x_data=np.random.rand(100)
y_data=x_data*4+3

w=tf.Variable(0.)
b=tf.Variable(0.)
y=x_data*w+b

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.2)

train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([w,b]),sess.run(loss))
    
