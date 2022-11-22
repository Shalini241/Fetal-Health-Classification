import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def confusion_metric(pred, act):
    sns.heatmap(confusion_matrix(pred,act),annot=True,fmt='3.0f',cmap="summer")
    plt.title('Our model Confusion Matrix', y=1.05, size=15)