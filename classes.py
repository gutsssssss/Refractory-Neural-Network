import pickle
import math
import cv2
import numpy as np
import arguments as arg
import math_functions as mf
from matplotlib import pyplot as plt
import glob
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')


class Dataset:
    def __init__(self, address):
        # define the size and classes number
        self.img_height = arg.img_height
        self.img_width = arg.img_width
        self.num_classes = arg.num_classes
        self.address = address
        # create lists to restore img and labels
        self.x = []  # img
        self.y = []  # label

    def loadData(self):
        # get img and label
        for folder in glob.glob(self.address):
            label = folder[-1]
            label = int(label)
            for img_path in glob.glob(folder + '/*.png'):
                img = plt.imread(img_path)
                img = cv2.resize(img, (self.img_height, self.img_width))
                self.x.append(img)
                self.y.append(label)
        # list to numpy
        self.x = np.array(self.x).reshape(len(self.x), -1)
        self.y = one_hot_encode(np.array(self.y), self.num_classes)
        return DataLoader(self.toTorchDataset(), batch_size=arg.batch_size, shuffle=True)

    def toTorchDataset(self):
        x = torch.tensor(self.x)
        y = torch.tensor(self.y)
        return TensorDataset(x, y)


class Neuron:
    def __init__(self, num_inputs: int):
        self.weights = torch.randn(num_inputs)
        self.bias = torch.randn(1)
        self.activation_functions = activation_functions['step']
        self.gate_value = torch.tensor([1, 1])
        self.a_threshold = 1
        self.a_value = 1
        self.g = 0
        self.age = 0
        self.age_list = []
        self.state = []

        # timers
        self.timer_a = 0
        self.timer_rf = 4
        self.timer_rs = 0

        # t
        self.t_a = 5
        self.t_rf = 3

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # calculate weighted inputs
        weighted_inputs = torch.matmul(self.weights, inputs) + self.bias

        # gate 1: control the input in, all or zero
        self.g += torch.mul(self.gate_value[0], weighted_inputs)

        a = self.activation_functions(self.g, threshold=self.a_threshold, value=self.a_value)

        self.update(a)

        # gate 2: control the accumulation(g), drop or keep
        self.g = torch.mul(self.gate_value[1], self.g)

        return a

    def update(self, a):
        # different gate values corresponding to different neuron phases
        rest = torch.tensor([1, 0])
        active = torch.tensor([0, 1])
        refractory = torch.tensor([0, 0])

        rest_condition = a <= 0 and self.timer_rf >= self.t_rf

        if rest_condition:
            self.gate_value = rest
            self.timer_a = 0
            self.timer_rs += 1
            self.state.append("rest")

        refractory_condition = self.timer_a >= self.t_a and self.timer_rf < self.t_rf

        if refractory_condition:
            self.gate_value = refractory
            self.timer_rf += 1
            self.state.append("refractory")

        active_condition = a > 0 and self.timer_a < self.t_a

        if active_condition:
            self.gate_value = active
            self.timer_a += 1
            self.timer_rf = 0
            self.timer_rs = 0
            self.state.append("active")

        self.age += 1

        reset_condition = self.timer_rs == 1

        if reset_condition:
            self.age = 1

        self.age_list.append(self.age)

# Define the one-hot encoding function
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def plot_loss_curve(model):
    # Plot the loss curve
    losses = model.losses
    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()


def plot_output(output: list[torch.Tensor]):
    length = math.ceil(math.sqrt(len(output))) + 1
    for i in range(len(output)):
        plt.subplot(length, length, i + 1)
        plt.hist(output[i].flatten(), facecolor='g')
        plt.title('layer ' + i.__str__())
        plt.xlim([-100, 100])
        plt.yticks([])
    plt.show()


# save model
def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# 加载模型参数
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculateAccuracy(pred_label, true_label):
    acc = 0
    for i in range(len(pred_label)):
        if pred_label[i] == true_label[i]:
            acc += 1
    return acc / len(pred_label)


activation_functions = {
    'linear': mf.linear,
    'sigmoid': mf.sigmoid,
    'relu': mf.relu,
    'tanh': mf.tanh,
    'softmax': mf.softmax,
    'step': mf.step
}

activation_derivatives = {
    'linear': mf.linear_derivative,
    'sigmoid': mf.sigmoid_derivative,
    'relu': mf.relu_derivative,
    'tanh': mf.tanh_derivative,
    'softmax': mf.softmax_cross_entropy_derivative
}
