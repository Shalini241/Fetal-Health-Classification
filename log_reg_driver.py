from preprocessing import data_preprocessing
import numpy as np
import pandas as pd
from train_model import train_model
from sklearn.model_selection import train_test_split
from plot_roc_curve import plot_roc_driver

#preprocess the data: 
# 1. Check for null values
# 2.Discretized the continuous features
data_file = data_preprocessing()

# data_file = 'fetal_health_discretized.csv'

data_frame = pd.read_csv(data_file)

all_features = data_frame.drop(["status"],axis=1) 
target_feature = data_frame["status"]

y = target_feature.values.astype(int)
X = all_features.values.astype(int)

label_dict = {'Normal':1, 'Suspect':2, 'Pathological':3}
_label_dict = {1 :'Normal' , 2 :'Suspect' , 3 :'Pathological'} 

score_list = list()
for fold in range(10):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
    pred, prediction, score, missclass = train_model(X_train, y_train, X_test, y_test, _label_dict)
    score_list.append(score)
    print("The score for Logistic Regression for fold",fold+1,"is: ",score_list[fold] ,'%', " No of misclassfied",missclass)
    

print("The overall score for Logistic Regression is: ", round(sum(score_list)/len(score_list),2),'%')

# plot_roc_driver(y_test, pred, prediction, label_dict)

