from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    
    return np.sum(np.diagonal(cm)) / np.sum(cm)

def precision(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    
    average = 0
    
    for c in range(cm.shape[0]):
        average += cm[c, c] / np.sum(cm[:, c])
        
    return average / cm.shape[0]
                            
def recall(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    
    average = 0
    
    for l in range(cm.shape[0]):
        average += cm[l, l] / np.sum(cm[l, :])
        
    return average / cm.shape[0]

# F1 = 2 * (precision * recall) / (precision + recall)

def f1_measure(y, y_pred):
    _precision = precision(y, y_pred)
    _recall = recall(y, y_pred)
    
    return 2 * (_precision * _recall) / (_precision + _recall)