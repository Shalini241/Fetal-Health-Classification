import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from log_reg import logisticRegression, sigmoid

#KFold cross validation
def train_model(X_train,y_train, X_test, y_test, _label_dict):
        #OneVsRest
    i,k,n= 0,3,21 #No of classes and features
    missclass =0
    all_theta = np.zeros((k, n))
    for hazelnut in np.unique(y_train):
        np_y_train = np.array(y_train == hazelnut, dtype = int)
        best_theta = logisticRegression(X_train, np_y_train, np.zeros((n,1)),10000)
        all_theta[i] = best_theta.T
        i += 1   
    #Predictions
    prediction = sigmoid(X_test.dot(all_theta.T))
    prediction = prediction.tolist()
    pred = list()
    act = list()
    missclass =0
    for _i,i in enumerate(prediction):
        pred.append(_label_dict[ i.index(max(i))+1 ])
        if _label_dict[ i.index(max(i)) +1] !=  _label_dict[y_test[_i]] :
            missclass += 1
        act.append(_label_dict[y_test[_i]]) 
    score = round(accuracy_score(pred, act)*100,2)

    return pred, prediction, score, missclass

def getOutput(pred, act):
    output=list()
    for i in range(len(pred)):
        output.append([pred[i],act[i], 'Matched' if pred[i] == act[i] else 'Unmatched'])
        Result = pd.DataFrame(output, columns=["Predicted Values", "Actual Value", "Matched/Unmatched"])
    Result.to_csv('output.csv', header=True, index=False)
