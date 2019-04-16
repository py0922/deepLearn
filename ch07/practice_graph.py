#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:37:07 2019

@author: Emma
"""

import tensorflow as tf
import numpy as np

x_data=np.random.rand(100)
y_data=2*x_data+3

w=tf.Variable(0.)
b=tf.Variable(0.)
y=w*x_data+b


loss=tf.reduce_mean(tf.square(y_data-y))
optimizer=tf.train.GradientDescentOptimizer(0.2)

train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([w,b]),sess.run(loss))
