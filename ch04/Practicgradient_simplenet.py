#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:31:34 2019

@author: Emma
"""

import os,sys
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient



class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)
    
    def predict(self,x):
        return np.dot(x,self.W)
    
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        
        return loss
        
        
        
net=simpleNet()
t = np.array([0, 0, 1])
x=np.array([0.6,0.9])


def f(W):
    return net.loss(x,t)




if __name__=='__main__':
    net=simpleNet()
    print(net.W)
    p=net.predict(x)
    print(p)
    t = np.array([0, 0, 1])
    print(net.loss(x,t))
    dW = numerical_gradient(f, net.W)
    print(dW)