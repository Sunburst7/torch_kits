import pandas as pd

def k_fold_data(k, i, X, y):
    '''将X, y分成k折，获取第i折用于验证(X与y有相同的样本数)'''
    assert k > 1
    fold_size = X.shape[0] // k
    X_vaild, y_valid = X[fold_size * (i - 1) : fold_size * i].values, y[fold_size * (i - 1) : fold_size * i].values
    X_train, y_train = pd.concat([X[:fold_size * (i - 1)], X[fold_size * i:]]).values, pd.concat([y[:fold_size * (i - 1)], y[fold_size * i:]]).values
    return X_train, y_train, X_vaild, y_valid

def k_fold_single_data(k, i, arr):
    '''
    '''
    assert i < k
    fold_size = len(arr) // k
    arr_valid = arr[fold_size*i : fold_size*(i + 1)]
    if i == 0:
        arr_train = arr[fold_size*(i + 1):]
    else:
        arr_train = arr[:fold_size*i] + arr[fold_size*(i + 1):]
    return arr_train, arr_valid