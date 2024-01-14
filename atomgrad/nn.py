from atomgrad.atom import Atom
from atomgrad.metrics import softmax
import numpy as np

class Layer:
  def __init__(self, inputs, units, activation=None):
    self.w =  Atom.uniform(-1, 1, (units, inputs))
    self.b = Atom.uniform(-1, 1, (1, units))
    self.inputs = inputs
    self.activation = activation

  def __call__(self, x):
    y =  x @ self.w.T
    y = y + self.b
    out = y.tanh() if self.activation == "tanh" else y.relu() if self.activation == "relu" else y.sigmoid() if self.activation == "sigmoid" else softmax(y) if self.activation == "softmax" else y
    return out

  def parameters(self,):
    return [self.w, self.b]

class AtomNet:
  def __init__(self, layers):
    self.layers = layers

  def fit(self, x, y, optimizer, loss_func, accuracy=None, epochs=5):
      for epoch in range(epochs):
        ypred = self(x)
        loss = loss_func(ypred, y)
        acc = accuracy(ypred, y) if accuracy else 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
          print(f"epoch: {epoch} | loss: {loss.data} | accuracy: {acc}%")

  def __call__(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

  def predict(self, x):
    for layer in self.layers:
        x = layer(x)
    return x

  def params(self):
    return [p for layer in self.layers for p in layer.parameters()]