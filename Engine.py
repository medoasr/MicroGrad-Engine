import  math
import numpy as np
import  matplotlib.pyplot as plt

class Value:
    def __init__(self,data,_children=(),_op=''):
        self.data=data
        self.grad=0.0 #maintains dlastoutnode/d_instantofvalueclass
        self._backward=lambda:None #automate the backward operations on local nodes "chain rule computations"
        self._prev=set(_children)
        self._op=_op

    def __repr__(self):
        return f"Value(data={self.data}, Grad={self.grad})"
    def __add__(self, other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return  out
    def __radd__(self, other):
        return self+other
    def __mul__(self, other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return  out
    def __rmul__(self, other):
        return self * other
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad+=(1-out.data**2)*out.grad
        out._backward=_backward
        return out

    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out
    def __neg__(self):
        return self*-1
    def __sub__(self, other):
        return  self+(-other)
    def __truediv__(self, other):
        return  self*other**-1
    def pow(self,other):
        assert  isinstance(other,(int,float))
        out=Value(self.data**other,(self,),'Pow')
        def _backward():
            self.grad+= (other*self.data**(other-1))*out.grad
        out._backward=_backward
        return  out
    def backward(self):
        topo=[]
        visited=set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                topo.append(v)
        topo_sort(self)
        self.grad=1
        for node in reversed (topo):
            node._backward()


