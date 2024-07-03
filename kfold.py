import pandas as pd

def k_fold(k, i, X, y):
    '''将X, y分成k折，获取第i折用于验证(X与y有相同的样本数)'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_vaild, y_valid = X[fold_size * (i - 1) : fold_size * i].values, y[fold_size * (i - 1) : fold_size * i].values
    X_train, y_train = pd.concat([X[:fold_size * (i - 1)], X[fold_size * i:]]).values, pd.concat([y[:fold_size * (i - 1)], y[fold_size * i:]]).values
    return X_train, y_train, X_vaild, y_valid