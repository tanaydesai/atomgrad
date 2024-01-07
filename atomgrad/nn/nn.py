from atomgrad.atom import Atom

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

  def fit(self,x):
    for layer in self.layers:
      x = layer(x)
    return x

  def params(self):
    return [p for layer in self.layers for p in layer.parameters()]