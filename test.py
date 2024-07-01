import classes as cls
import torch

n = cls.Neuron(5)
dataset = []
for i in range(30):
    data = torch.randn(5)
    dataset.append(data)
    n.forward(data)

print(n.age_list)
print(n.state)


