def binary_cross_entropy(ypred, y):
    logs = ypred.log().reshape(ypred.shape[0])
    loss = y * logs
    tot = -1.0 * loss.sum()
    return tot


def cat_cross_entropy(ypred, y):
  logs = ypred.log()
  loss = y * logs
  tot = -1.0 * loss.sum()
  return tot