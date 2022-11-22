import numpy as np

def logisticRegression(X, y,theta,num_iter):
    
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

#Sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Cost function
def costFunc(theta, X, y):
    h = sigmoid(X.dot(theta))
    cost0 = y.T.dot(np.log(h))
    cost1 = (1-y).T.dot(np.log(1-h))
    cost = -((cost1 + cost0))/len(y)
    return cost

#Gradient descent function
def gradientFunc(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    return ((1 / m) * X.T.dot(h - y))