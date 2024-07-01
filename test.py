import classes as cls
import torch

n = cls.Neuron(5)
data = []
for i in range(10):
    a = torch.randn(5)
    data.append(a)
    n.forward(a, 0)


