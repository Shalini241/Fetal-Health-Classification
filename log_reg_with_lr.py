import numpy as np
import pandas as pd 

def logisticRegression(X, y,theta,num_iter):
    
    #Sigmoid function
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
    
    #Cost function
    def costFunc(theta, X, y, lr = 0.001):
        h = sigmoid(X.dot(theta))
        r = (lr/(2 * len(y))) * np.sum(theta**2)
        return (1 / len(y)) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + r
    
    #Gradient descent function
    def gradientFunc(theta, X, y, lr = 0.001):
        m, n = X.shape
        theta = theta.reshape((n, 1))
        y = y.reshape((m, 1))
        h = sigmoid(X.dot(theta))
        r = lr * theta /m
        return ((1 / m) * X.T.dot(h - y)) + r
    
    #Finding best theta
    for i in range(num_iter):
        lineq = np.dot(X, theta)
        h = sigmoid(lineq)
        #Calculating cost function of each class
        cost = costFunc(theta, X,y) 
        cost = cost.sum(axis = 0)
        #Applying gradient descent to find new theta
        delta = gradientFunc(theta,X,y) 
        theta = theta - delta    
    return theta