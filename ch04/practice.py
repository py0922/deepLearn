#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:28:26 2019

@author: Emma
"""
import numpy as np
import os,sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist 
import matplotlib.pylab as plt

def cross_entropy_error(y,t): 
    if y.ndim==1:
        t=t.reshape(1,t.size())
        y=y.reshape(1,y.size())
    batch_size=y.shape[0]
    delta=1e-7
    return -np.sum(t*np.log(y+delta))/batch_size

def numerical_diff(f,x,i):
    delta=1e-4
    x2=x.copy()
    x1=x.copy()
    x2[i]=x2[i]+delta   
    x1[i]=x1[i]-delta
    
    return (f(x2)-f(x1))/(2*delta)

def numerical_gradient(f,x):
    grad=np.zeros_like(x)
    for i in range(x.size):
        grad[i]=numerical_diff(f,x,i)
    return grad

def gradient_descent(f,init_x,lr=0.1,step_num=100):
    x=init_x
    
    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad
    return x
    

def function_1(x):
    return 0.01*x**2+0.1*x

def function_2(x):
    return np.sum(x**2)



if __name__=="__main__":
#    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
#    train_size=x_train.shape[0]
#    batch_size=10
#    mask_batch=np.random.choice(train_size,batch_size)
#    x_batch=x_train[mask_batch]
#    t_batch=t_train[mask_batch]
    
#    x=np.arange(-20,10,0.1)
#    y=function_1(x)
#    plt.xlabel("x")
#    plt.ylabel("f(x)")
#    plt.plot(x,y)
#    plt.show()
#    
    print(numerical_diff(function_2,np.array([4.5,3.6]),0))
    print(numerical_diff(function_2,np.array([4.5,3.6]),1))
    print(numerical_gradient(function_2,np.array([4.5,3.6])))
    print(gradient_descent(function_2,np.array([-3.0,4.0])))
