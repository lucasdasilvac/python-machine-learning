def accuracy(y_pred, y_test):
    return sum(y_pred == y_test) / y_test.shape[0]