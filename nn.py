import random
from Engine import Value

class Module: # to match pytorch implementation
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0
    def parameters(self):
            return []
class Neuron(Module):
    def __init__(self,nin): #nin number of inputs
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))

    def __call__(self, x):
        #w*x+b
        act=sum ((wi+xi for wi ,xi in zip(self.w,x)),self.b)
        return act.tanh()
    def parameters(self):
        return self.w+[self.b]

class Layer(Module):
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs=[n(x) for n in self.neurons]
        return  outs
    def parameters(self):
        params=[]
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return  params
class MLP(Module):

    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1]) for i in range (len(nouts))]


    def __call__(self, x):
        for layer in self.layers:
            x= layer(x)
        return  x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# x=[2,3,-1]
# n=MLP(3,[4,4,1])
# print(n(x))