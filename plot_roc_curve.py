import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

#Multiclass roc estimation referenced from scikit learn
def roc_estimation(y_test, y_pred , y_pred_proba, average="macro",num_class=3):  
    y_test = list(y_test)
    y_pred = list(y_pred)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    # print(y_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # print(num_class)
    for i in range(num_class):
        # print(i)
        y__ = [y[i] for y in y_pred_proba ]
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y__)
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc_score(y_test, y_pred, average=average) , (fpr,tpr,roc_auc)

#Plotting ROC curve
def plotting_roc(fpr, tpr, roc, label_dict, num_classes = 3):
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc["macro"] = auc(fpr["macro"], tpr["macro"])
    for i in range(num_classes):
        plt.plot(fpr[i],tpr[i],label='ROC = for %s %s' % (list(label_dict.keys())[i] , roc[i] * 100))
    plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc["macro"]),
         color='navy', linestyle=':', linewidth=4)
    plt.legend(loc="top right")
    plt.show()

def plot_roc_driver(y_test, pred, prediction, label_dict):
    roc, add_roc = roc_estimation(y_test,pred,prediction,average="macro")
    fpr,tpr,roc_auc = add_roc
    plotting_roc(fpr,tpr,roc_auc, label_dict)