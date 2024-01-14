# atomgrad

![pic](pic.jpeg)

Atomgrad is a simple autograd engine that aims to be between [micrograd](https://github.com/karpathy/micrograd/) and [tinygrad](https://github.com/tinygrad/tinygrad) that performs autodiff on vector-valued and scalar-valued tensors (atoms) coupled with a neural network api library.

## Features

- Supports Pytorch-like vector-valued and scalar-valued tensors.
- Supports basic unary ops, binary ops, reduce ops and movement ops i.e (activn funcs, `sum`, `exp`, `reshape`, `randint`, `uniform`, etc).
- Supports activation functions such as `relu`, `sigmoid`, `tanh`, etc.
- Supports `binary_cross_entropy` & `binary_accuracy`.
- Supports Graph Visualizations.

## Installation

You can install atomgrad using pip:

```bash
pip install atomgrad==0.3.1
```

## Usage

Here is a simple example of using atomgrad to compute the gradient of a function:

```python
from atomgrad.atom import Atom
from atomgrad.graph import draw_dot


# create two tensors with gradients enabled
x = Atom(2.0, requires_grad=True)
y = Atom(3.0, requires_grad=True)

# define a function
z = x * y + x ** 2

# compute the backward pass
z.backward()

# print the gradients
print(x.grad) # 7.0
print(y.grad) # 2.0

draw_dot(z)
```
![pic](graph.png)

Here is a simple example of using atomgrad to train a one 16-node hidden layer neural network for binary classification.

```python
import numpy as np
from atomgrad.atom import Atom
from atomgrad.nn import AtomNet, Layer
from atomgrad.optim import SGD
from atomgrad.metrics import binary_cross_entropy, binary_accuracy

# create a model
model = AtomNet(
  Layer(2, 16),
  Layer(16, 16),
  Layer(16, 1)
)
# create an optimizer
optim = SGD(model.parameters(), lr=0.01)

# load some data
x = [[2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
  [0.0, 4.0, 0.5],
  [3.0, -1.0, 0.5]]
y = [1, 1, 0, 1, 0, 1]

x = Atom(x)
y = Atom(y)

model.fit(x, y, optim, binary_cross_entropy, binary_accuracy, epochs=100)

#output
'''
...
epoch: 30 | loss: 0.14601783454418182  | accuracy: 100.0%
epoch: 35 | loss: 0.11600304394960403  | accuracy: 100.0%
epoch: 40 | loss: 0.09604986757040024  | accuracy: 100.0%
epoch: 45 | loss: 0.0816292017698288  | accuracy: 100.0%
'''
```

## Demos

An example of simple autodiff and four binary classifiers including `make_moons` dataset and MNIST digits dataset is in the `examples/demos.ipynb` notebook.

Note: Although `atom.nn` includes `softmax` activation and `cat_cross_entropy`, model results are quite dissapointing and are likely due to some bug (plz lmk if you find it!). As a result the `AtomNet` model is best suited for binary classification neural net tasks.