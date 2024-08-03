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
import tkinter as tk
from tkinter import Canvas, Button, Label, Entry, Radiobutton, StringVar
from math import cos, sin, radians

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
    def __init__(self, num_inputs: int, t_a, t_rf, bias):
        self.weights = torch.rand(num_inputs)
        self.bias = torch.tensor([bias])
        self.activation_functions = activation_functions['step']
        self.gate_value = torch.tensor([1, 1, 0])
        self.a_threshold = torch.tensor([1])
        self.a_value = torch.tensor([1])
        self.g = 0
        self.output = torch.tensor([0])
        self.output_record = [torch.tensor(0)]

        # t
        self.t_a = 4
        self.t_rf = 2

        # timers
        self.timer_a = 0
        self.timer_rf = self.t_rf + 1
        self.timer_rs = 0

        # age & state
        self.age = - 1
        self.age_list = [self.age]
        self.state = "rest"
        self.state_record = ["rest"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convert inputs to Float
        inputs = inputs.float()

        # calculate weighted inputs
        weighted_inputs = torch.matmul(self.weights, inputs) + self.bias

        # gate 1: control the input in, all or zero
        self.g += torch.mul(self.gate_value[0], weighted_inputs)

        a = self.activation_functions(self.g, threshold=self.a_threshold, value=self.a_value)

        # gate 3: control the output out, all or zero
        self.output = torch.mul(self.gate_value[2], a)
        self.output_record.append(self.output)

        self.update(a)

        # gate 2: control the accumulation(g), drop or keep
        self.g = torch.mul(self.gate_value[1], self.g)

        return self.output

    def update(self, a):
        # different gate values corresponding to different neuron phases
        rest = torch.tensor([1, 0, 0])
        active = torch.tensor([0, 1, 1])
        refractory = torch.tensor([0, 0, 0])

        self.age += 1

        rest_condition = a < self.a_value and self.timer_rf >= self.t_rf

        if rest_condition:
            self.gate_value = rest
            self.timer_a = 0
            self.timer_rs += 1
            self.state = "rest"
            self.state_record.append("rest")

        refractory_condition = self.timer_a >= self.t_a and self.timer_rf < self.t_rf

        if refractory_condition:
            self.gate_value = refractory
            self.timer_rf += 1
            self.state = "refractory"
            self.state_record.append("refractory")

        active_condition = a == self.a_value and self.timer_a < self.t_a

        reset_condition = a == self.a_value and self.timer_a == 0

        if reset_condition:
            self.age = 0

        if active_condition:
            self.gate_value = active
            self.timer_a += 1
            self.timer_rf = 0
            self.timer_rs = 0
            self.state = "active"
            self.state_record.append("active")

        self.age_list.append(self.age)

    def get_output(self):
        return self.output

    def get_state(self):
        return self.state


class Loop:
    def __init__(self, t_a, t_rf, bias, number):
        self.neuron_num = number
        self.neurons = [Neuron(1, t_a, t_rf, bias) for _ in range(self.neuron_num)]
        self.outputs = [[torch.tensor([0]) for _ in range(self.neuron_num)]]
        self.states = [['rest' for _ in range(self.neuron_num)]]

    def forward(self, external_input: torch.Tensor, input_type, time_steps):
        for t in range(time_steps):
            # spike input
            if input_type == 'one_time' and t == 0:
                ex_signal = external_input

            # constant input
            elif input_type == 'constant':
                ex_signal = external_input

            # periodic input
            elif input_type == 'periodic' and t % external_input == 0:
                ex_signal = external_input

            else:
                ex_signal = torch.tensor([0])

            input_signal = [self.neurons[i - 1].get_output() for i in range(self.neuron_num)]
            input_signal[0] = input_signal[0].float() + ex_signal

            # update neurons
            [self.neurons[i].forward(input_signal[i]) for i in range(self.neuron_num)]

            # record output and state
            current_states = [self.neurons[j].get_state() for j in range(self.neuron_num)]
            current_outputs = [self.neurons[j].get_output() for j in range(self.neuron_num)]

            self.states.append(current_states)
            self.outputs.append(current_outputs)

        return self.states, self.outputs


class CircleDiagram:
    def __init__(self, root):
        self.root = root
        self.running = False

        # All labels and entries using grid layout
        self.time_steps_label = Label(root, text="Time Steps:")
        self.time_steps_label.grid(row=0, column=0, sticky=tk.W)
        self.time_steps_entry = Entry(root)
        self.time_steps_entry.grid(row=0, column=1)
        self.time_steps_entry.insert(0, "50")
        self.time_steps_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.input_type_label = Label(root, text="Input Type:")
        self.input_type_label.grid(row=1, column=0, sticky=tk.W)
        self.input_type_var = StringVar(value="constant")
        self.one_time_radio = Radiobutton(root, text="One  Time", variable=self.input_type_var, value="one_time",
                                          command=self.pause_on_entry_change)
        self.one_time_radio.grid(row=1, column=1, sticky=tk.W)
        self.constant_radio = Radiobutton(root, text="Constant", variable=self.input_type_var, value="constant",
                                          command=self.pause_on_entry_change)
        self.constant_radio.grid(row=1, column=2, sticky=tk.W)
        self.periodic_radio = Radiobutton(root, text="Periodic", variable=self.input_type_var, value="periodic",
                                          command=self.pause_on_entry_change)
        self.periodic_radio.grid(row=1, column=3, sticky=tk.W)

        self.external_input_label = Label(root, text="External Input:")
        self.external_input_label.grid(row=2, column=0, sticky=tk.W)
        self.external_input_entry = Entry(root)
        self.external_input_entry.grid(row=2, column=1)
        self.external_input_entry.insert(0, "10")
        self.external_input_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.t_a_label = Label(root, text="τa:")
        self.t_a_label.grid(row=3, column=0, sticky=tk.W)
        self.t_a_entry = Entry(root)
        self.t_a_entry.grid(row=3, column=1)
        self.t_a_entry.insert(0, "4")
        self.t_a_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.t_r_label = Label(root, text="τrf:")
        self.t_r_label.grid(row=4, column=0, sticky=tk.W)
        self.t_r_entry = Entry(root)
        self.t_r_entry.grid(row=4, column=1)
        self.t_r_entry.insert(0, "2")
        self.t_r_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.b_label = Label(root, text="bias:")
        self.b_label.grid(row=5, column=0, sticky=tk.W)
        self.b_entry = Entry(root)
        self.b_entry.grid(row=5, column=1)
        self.b_entry.insert(0, "0.8")
        self.b_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.n_label = Label(root, text="neuron number:")
        self.n_label.grid(row=6, column=0, sticky=tk.W)
        self.n_entry = Entry(root)
        self.n_entry.grid(row=6, column=1)
        self.n_entry.insert(0, "6")
        self.n_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.s_label = Label(root, text="speed:")
        self.s_label.grid(row=7, column=0, sticky=tk.W)
        self.s_entry = Entry(root)
        self.s_entry.grid(row=7, column=1)
        self.s_entry.insert(0, "500")
        self.s_entry.bind("<KeyRelease>", self.pause_on_entry_change)

        self.current_time_step = 0
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=8, column=0, columnspan=4)
        self.label = Label(root, text=f"Time Step: {self.current_time_step + 1}")
        self.label.grid(row=9, column=0, columnspan=4)
        self.stop_button = Button(root, text="Stop", command=self.stop)
        self.stop_button.grid(row=10, column=0, columnspan=4)
        self.continue_button = Button(root, text="Continue", command=self.continue_)
        self.continue_button.grid(row=11, column=0, columnspan=4)
        self.restart_button = Button(root, text="Restart", command=self.restart)
        self.restart_button.grid(row=12, column=0, columnspan=4)

        self.loop = Loop(int(self.t_a_entry.get()), int(self.t_r_entry.get()), float(self.b_entry.get()), int(self.n_entry.get()))
        self.draw_circles()
        self.auto_step()

    def draw_circles(self):
        self.canvas.delete("all")
        time_steps = int(self.time_steps_entry.get())
        input_type = self.input_type_var.get()
        external_input = int(self.external_input_entry.get())
        states, outputs = self.loop.forward(torch.tensor([external_input]), input_type, time_steps)
        sts = states[self.current_time_step]
        ops = outputs[self.current_time_step]
        num_circles = len(sts)
        angle_step = 360 / num_circles
        radius = 150
        center_x, center_y = 200, 200

        for i, st in enumerate(sts):
            angle = 90 - i * angle_step
            x = center_x + radius * cos(radians(angle))
            y = center_y - radius * sin(radians(angle))
            self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=statesToColors(st))
            self.canvas.create_text(x, y - 30, text=st)

        for i, op in enumerate(ops):
            angle1 = 90 - i * angle_step
            angle2 = 90 - (i + 1) % num_circles * angle_step
            x1 = center_x + radius * cos(radians(angle1))
            y1 = center_y - radius * sin(radians(angle1))
            x2 = center_x + radius * cos(radians(angle2))
            y2 = center_y - radius * sin(radians(angle2))
            self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill=outputsToColors(op))

    def next_step(self):
        time_steps = int(self.time_steps_entry.get())
        if self.current_time_step < time_steps - 1:
            self.current_time_step += 1
            self.label.config(text=f"Time Step: {self.current_time_step + 1}")
            self.draw_circles()
        else:
            self.stop()

    def auto_step(self):
        if self.running:
            self.next_step()
        self.root.after(int(self.s_entry.get()), self.auto_step)

    def stop(self):
        self.running = False

    def continue_(self):
        self.running = True

    def restart(self):
        self.current_time_step = 0
        self.label.config(text=f"Time Step: {self.current_time_step + 1}")
        # Update the loop with new input values
        time_steps = int(self.time_steps_entry.get())
        input_type = self.input_type_var.get()
        external_input = int(self.external_input_entry.get())
        self.loop = Loop(int(self.t_a_entry.get()), int(self.t_r_entry.get()), float(self.b_entry.get()), int(self.n_entry.get()))  # Reinitialize the loop
        self.loop.forward(torch.tensor([external_input]), input_type, time_steps)  # Update loop with new parameters
        self.running = True
        self.draw_circles()

    def pause_on_entry_change(self, event=None):
        self.stop()


def drawLoop():
    root = tk.Tk()
    app = CircleDiagram(root)
    root.mainloop()


def statesToColors(st):
    return 'yellow' if st == 'rest' else 'red' if st == 'active' else 'orange'


def outputsToColors(op):
    return 'red' if op == torch.tensor([1]) else 'black'


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
        plt.subplot(length, length, i + 1)  # l行l列
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


# 创建一个字典来存储激活函数
activation_functions = {
    'linear': mf.linear,
    'sigmoid': mf.sigmoid,
    'relu': mf.relu,
    'tanh': mf.tanh,
    'softmax': mf.softmax,
    'step': mf.step
}

# 创建一个字典来存储激活函数的导数
activation_derivatives = {
    'linear': mf.linear_derivative,
    'sigmoid': mf.sigmoid_derivative,
    'relu': mf.relu_derivative,
    'tanh': mf.tanh_derivative,
    'softmax': mf.softmax_cross_entropy_derivative
}
