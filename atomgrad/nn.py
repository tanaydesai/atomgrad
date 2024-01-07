from atomgrad.atom import Atom
import time
import numpy as np

class Layer:
  def __init__(self, inputs, units, activation=None):
    self.w =  Atom.uniform(0, 1, (inputs, units))
    self.b = Atom.uniform(0, 1, units)
    self.inputs = inputs
    self.activation = activation

  def __call__(self, x):
    y = x @ self.w + self.b
    out = y.tanh() if self.activation == "tanh" else y.relu() if self.activation == "relu" else y.sigmoid() if self.activation == "sigmoid" else y
    return out

  def parameters(self,):
    return [self.w, self.b]

class AtomNet:
  def __init__(self, layers):
    self.layers = layers

  def fit(self, x, y, optimizer,metric_loss, epochs=5, batch_size=2):
    num_batches = int(np.ceil(x.shape[0] / batch_size))
    
    for epoch in range(epochs):
      losses = []
      s = time.monotonic()
      for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_X = x[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        ypred = self.train(batch_X)
        loss = metric_loss(ypred, batch_y)
        losses.append(float(loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      e = time.monotonic()
      t = e - s
      loss_epoch = sum(losses) / len(losses)
      if epoch % 10 == 0 or epoch==(epochs-1):
          print(f"epoch: {epoch} |", f"loss: {loss_epoch:.2f} |", f"time: {t:.2f} sec.")

  def train(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

  def params(self):
    return [p for layer in self.layers for p in layer.parameters()]