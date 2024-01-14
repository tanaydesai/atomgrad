import math
import numpy as np
import random
from typing import List, Union, Tuple, Optional ,Callable, NewType

class Atom:
  def __init__(self, data: Union[np.ndarray, List, int, float], _children: Tuple=(), _op:str = "", dtype = np.float32, requires_grad:bool=True, label =""):
    self.data = np.asarray(data).astype(dtype)
    if requires_grad:
        self.grad = np.zeros_like(data).astype(dtype)
    else:
        self.grad = None
    self._prev = set(_children)
    self.requires_grad = requires_grad
    self._op = _op
    self.label = label
    self._backward = lambda: None

  # ---------------------------------------------------- Calls ----------------------------------------------------

  def __repr__(self):
    return f"Atom(data={self.data}, requires_grad={self.requires_grad}, dtype={self.dtype})"

  def __getitem__(self, n: int):
    out = Atom(self.data[n], (self,), "slice", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad[n] += out.grad
    out._backward = _backward

    return out

  # ---------------------------------------------------- BinaryOps ----------------------------------------------------

  def __add__(self, other: Union["Atom", np.ndarray, List, int, float]):
    other = other if isinstance(other, Atom) else Atom.broadcast(self, other, dtype=self.data.dtype, requires_grad=self.requires_grad)
    out = Atom(self.data + other.data, (self, other), "+", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += np.sum(out.grad, axis=0) if self.data.shape != out.data.shape else out.grad
      other.grad += np.sum(out.grad, axis=0) if other.data.shape != out.data.shape else out.grad
    out._backward = _backward

    return out

  def __mul__(self, other: Union["Atom", np.ndarray, List, int, float]):
    other = other if isinstance(other, Atom) else Atom.broadcast(self, other, dtype=self.data.dtype, requires_grad=self.requires_grad)
    out = Atom(self.data * other.data, (self, other), "*", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += np.sum(other.data * out.grad, axis=0) if self.data.shape != out.data.shape else (other.data * out.grad)
      other.grad += np.sum(self.data * out.grad, axis=0) if other.data.shape != out.data.shape else (self.data * out.grad)
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Atom(self.data**other, (self,), f'**{other}', dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out._backward = _backward

    return out

  def __matmul__(self, other: Union["Atom", np.ndarray, List, int, float]):
    other = other if isinstance(other, Atom) else Atom(other, dtype=self.data.dtype, requires_grad=self.requires_grad)
    out = Atom(self.data @ other.data, (self, other),"@", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += out.grad @ other.data.T
      other.grad += self.data.T @ out.grad
    out._backward = _backward

    return out

  def __truediv__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return self * other**-1

  def __neg__(self):
    return self * -1

  def __radd__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return self + other

  def __sub__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return self + (-other)

  def __rsub__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return other + (-self)

  def __rmul__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return self * other

  def __rtruediv__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return other * self**-1

  def __rmatmul__(self, other: Union["Atom", np.ndarray, List, int, float]):
    return self @ other

  # ---------------------------------------------------- Methods ----------------------------------------------------

  @staticmethod
  def broadcast(reference: "Atom", target: Union["Atom", int, float, np.ndarray], **kwargs):
      if isinstance(target, Atom):
          if np.broadcast(reference.data, target.data).shape == np.broadcast(reference.data, reference.data).shape:
              return Atom(np.broadcast_to(target.data, reference.shape), _op="broadcasted", **kwargs)
          else:
              dim1 = np.broadcast(reference.data, target.data).shape
              dim2 = np.broadcast(reference.data, reference.data).shape
              raise ValueError(f"Tensors are not broadcastable. Info ref-target {dim1}, info ref-ref {dim2}")
      else:
          if np.broadcast(reference.data, target).shape == np.broadcast(reference.data, reference.data).shape:
              return Atom(np.broadcast_to(target, reference.shape), _op="broadcasted", **kwargs)
          else:
              dim1 = np.broadcast(reference.data, target).shape
              dim2 = np.broadcast(reference.data, reference.data).shape
              raise ValueError(f"Atom are not broadcastable. Info ref-target {dim1}, info ref-ref {dim2}")

  @staticmethod
  def randint(low: Union[int, float], high: Union[int, float], shape, **kwargs): return Atom(np.random.randint(low, high, shape), _op="randint", **kwargs)
  
  @staticmethod
  def zeros(shape: Tuple[int, int], **kwargs): return Atom(np.zeros(shape), _op="zeros", **kwargs)

  @staticmethod
  def zeros_like(atom: "Atom", **kwargs): return Atom(np.zeros_like(atom.data), _op="zeros_like", **kwargs)

  @staticmethod
  def full(shape: Tuple[int, int], fill_value: Union[float, int], **kwargs): return Atom(np.full(shape,fill_value), _op="full", **kwargs)

  @staticmethod
  def ones(shape: Tuple[int, int], **kwargs): return Atom(np.ones(shape), _op="ones", **kwargs)

  @staticmethod
  def ones_like(atom: "Atom", **kwargs): return Atom(np.ones_like(atom.data), _op="ones like", **kwargs)

  @staticmethod
  def uniform(low: Union[int, float], high: Union[int, float], shape, **kwargs): return Atom(np.random.uniform(low, high, shape), _op="uniform", **kwargs)

  # ---------------------------------------------------- UnaryOps ----------------------------------------------------

  def sum(self):
    out = Atom(self.data.sum(), (self,),"sum", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += np.ones_like(self.data) * out.grad
    out._backward = _backward

    return out

  def exp(self):
    x = np.where(self.data > 88, 88, self.data)
    out = Atom(np.exp(x), (self,), "exp", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out

  def log(self):
    epsilon = 1e-20
    if np.any(self.data < 0):
        raise ValueError(f"can't log: {self.data}")
    x = np.where(self.data == 0, self.data+epsilon, self.data)
    out = Atom(np.log(x), (self,), "log", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
      self.grad = (x**(-1)) * out.grad
    out._backward = _backward

    return out

  # ---------------------------------------------------- Activation Functions ----------------------------------------------------

  def tanh(self):
    x = self.data
    t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
    out = Atom(t, (self,), "tanh", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

  def relu(self):
    out = Atom(np.maximum(0, self.data), (self,), "relu", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def sigmoid(self):
    x = self.data
    t = (1 + np.exp(-x))**-1
    out = Atom(t, (self,), "sigmoid", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad += t*(1-t) * out.grad
    out._backward = _backward

    return out
  # ---------------------------------------------------- Properties ----------------------------------------------------

  @property
  def shape(self): return self.data.shape

  @property
  def dtype(self): return self.data.dtype

  # ---------------------------------------------------- MovementOps ----------------------------------------------------

  @property
  def T(self): return self.transpose()

  def reshape(self, shape: Tuple[int, int]):
    out = Atom(self.data.reshape(shape), (self,), "reshape", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad += out.grad.reshape(self.shape)
    out._backward = _backward

    return out

  def transpose(self, ax1: int = 1, ax2: int = 0):
    out = Atom(self.data.transpose(ax1, ax2), (self,) ,"T", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad += out.grad.transpose(ax1, ax2)
    out._backward = _backward

    return out

  def flatten(self):
    out = Atom(self.data.flatten(), (self,), "flatten", dtype=self.data.dtype, requires_grad=self.requires_grad)

    def _backward():
        self.grad = out.grad.reshape(self.shape)
    out._backward = _backward

    return out

  # ---------------------------------------------------- Backpropogation ----------------------------------------------------

  def backward(self):
    if self.shape != () and self.shape != (1,) and self.shape != (1,1):
      raise ValueError("Backward can only be called on a scalar atom.")
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1
    for v in reversed(topo):
        v._backward()