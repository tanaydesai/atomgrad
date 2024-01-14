import numpy as np

def binary_cross_entropy(ypred, y):
    neg_output = 1-ypred
    tot =  (-(y * ypred.log().reshape(ypred.shape[0]) + (1-y)*neg_output.log().reshape(ypred.shape[0]))).sum()
    return tot

def cat_cross_entropy(ypred, y):
  logs = ypred.log()
  loss = y * logs
  tot = -1.0 * loss.sum()
  return tot

def binary_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred.data).round()
    correct_results = np.sum(y_pred.reshape(y_pred.shape[0]) == y_true.data)
    acc = correct_results / y_true.shape[0]
    acc = np.round(acc * 100)
    return acc

def softmax(data):
    exp = data.exp()
    tot = exp.sum().data
    out = exp / tot
    return out