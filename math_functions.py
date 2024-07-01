import numpy as np
import torch


# 线性函数
def linear(x):
    return x


def linear_derivative(x):
    return 1


# ReLU函数及其导数
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x = np.array(x)  # 将列表转换为NumPy数组
    return torch.Tensor(np.where(x > 0, 1, 0))


# tanh函数及其导数
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# sigmoid函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# step函数
def step(x, threshold=5, value=1):
    if x > threshold:
        return value
    else:
        return 0


# softmax函数及其导数
def softmax(x):
    x_max, _ = torch.max(x, dim=1, keepdim=True)
    e_x = torch.exp(x - x_max)
    return torch.divide(e_x, torch.sum(e_x, dim=1, keepdim=True))


def softmax_cross_entropy_derivative(x, y_true):
    s = softmax(x)
    return s - y_true


def cross_entropy(y_pred, y_true):
    return -torch.sum(torch.mul(y_true, np.log(y_pred + 1e-25)))


def derivative_cross_entropy(y_pred, y_true):
    return -y_true / (y_pred + 1e-20)


# lr_arr
def lrArray(epochs, initial_lr, decay_rate):
    lr = np.zeros(epochs)
    for epoch in range(epochs):
        lr[epoch] = initial_lr * np.exp(-decay_rate * epoch)
    return lr


def normalize_input(input_tensor: torch.Tensor) -> torch.Tensor:
    # 计算输入张量的均值和标准差
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)

    # 归一化输入张量
    normalized_tensor = (input_tensor - mean) / (std + 1e-25)

    return normalized_tensor
