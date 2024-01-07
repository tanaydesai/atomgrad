from atomgrad.atom import Atom
import numpy as np

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self) :
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad