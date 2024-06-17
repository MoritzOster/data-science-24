import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

def plot_confusion_matrix(model, x_test, y_test):
    plt.clf()
    y_pred = model.predict(x_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.savefig('heatmap.png')
    
def plot_roc(model, x_test, y_test):
    plt.clf()
    y_pred = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('roc.png')
    
