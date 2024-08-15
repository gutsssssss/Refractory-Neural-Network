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
    def __init__(self, num_inputs: int, t_a, t_rf, bias, weight_type="all one"):
        if weight_type == "all one":
            self.weights = torch.ones(num_inputs)
        else:
            self.weights = torch.rand(num_inputs)
        self.bias = torch.tensor([bias])
        self.activation_functions = activation_functions['step']
        self.gate_value = torch.tensor([1, 1, 0])
        self.a_threshold = torch.tensor([1])
        self.a_value = self.a_threshold.data
        self.g = 0
        self.output = torch.tensor([0])
        self.output_record = [torch.tensor(0)]

        # t
        self.t_a = t_a
        self.t_rf = t_rf

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

    def get_weight(self):
        return round(self.weights.item(), 3)


class Loop:
    def __init__(self, t_a, t_rf, bias, number, weight_type):
        self.neuron_num = number
        self.neurons = [Neuron(1, t_a, t_rf, bias, weight_type) for _ in range(self.neuron_num)]
        self.outputs = [[torch.tensor([0]) for _ in range(self.neuron_num)]]
        self.states = [['rest' for _ in range(self.neuron_num)]]

    def forward(self, external_input: torch.Tensor, input_duration, input_type, time_steps, period=1):
        for t in range(time_steps):
            # spike input
            if input_type == 'one_time' and t < input_duration:
                ex_signal = external_input

            # constant input
            elif input_type == 'constant':
                ex_signal = external_input

            # periodic input
            elif input_type == 'periodic' and t % period < input_duration:
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

    def get_weights(self):
        return [neuron.get_weight() for neuron in self.neurons]


class CircleDiagram:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.auto_step_id = None
        self.neuron_number = 6
        self.time_steps = 1000

        # All labels and entries using grid layout
        self.input_type_label = Label(root, text="Input Type:")
        self.input_type_label.grid(row=0, column=0, sticky=tk.W)
        self.input_type_var = StringVar(value="constant")
        self.one_time_radio = Radiobutton(root, text="OneTime", variable=self.input_type_var, value="one_time",
                                          command=self.pause_on_entry_change)
        self.one_time_radio.grid(row=0, column=1, sticky=tk.W)
        self.constant_radio = Radiobutton(root, text="Constant", variable=self.input_type_var, value="constant",
                                          command=self.pause_on_entry_change)
        self.constant_radio.grid(row=0, column=2, sticky=tk.W)
        self.periodic_radio = Radiobutton(root, text="Periodic", variable=self.input_type_var, value="periodic",
                                          command=self.pause_on_entry_change)
        self.periodic_radio.grid(row=0, column=3, sticky=tk.W)

        self.input_duration_label, self.input_duration_entry = self.create_input("Input Duration:", 1, 0, "1", "step",
                                                                                 self.pause_on_entry_change)
        self.input_period_label, self.input_period_entry = self.create_input("Input Period:", 2, 0, "1", "step",
                                                                             self.pause_on_entry_change)
        self.external_input_label, self.external_input_entry = self.create_input("Input Value:", 3, 0, "10", "",
                                                                                 self.pause_on_entry_change)
        self.t_a_label, self.t_a_entry = self.create_input("τa:", 4, 0, "4", "step", self.pause_on_entry_change)
        self.t_r_label, self.t_r_entry = self.create_input("τrf:", 5, 0, "2", "step",self.pause_on_entry_change)
        self.wt_label, self.wt_entry = self.create_input("Weight Type:", 6, 0, "random", "random or all one",
                                                         self.pause_on_entry_change)
        self.b_label, self.b_entry = self.create_input("Bias:", 7, 0, "0.8", "", self.pause_on_entry_change)
        self.n_label, self.n_entry = self.create_input("Neuron Number:", 8, 0, str(self.neuron_number), "",
                                                       self.pause_on_entry_change)
        self.s_label, self.s_entry = self.create_input("Speed:", 9, 0, "500", "ms/step", self.pause_on_entry_change)

        self.current_time_step = 0
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=10, column=0, columnspan=4)
        self.label = Label(root, text=f"Time Step: {self.current_time_step + 1}")
        self.label.grid(row=11, column=0, columnspan=4)
        self.stop_button = Button(root, text="Stop", command=self.stop)
        self.stop_button.grid(row=12, column=0, columnspan=4)
        self.go_button = Button(root, text="Go", command=self.go_)
        self.go_button.grid(row=13, column=0, columnspan=4)
        self.restart_button = Button(root, text="Restart", command=self.restart)
        self.restart_button.grid(row=14, column=0, columnspan=4)
        self.exit_button = Button(root, text="Exit", command=self.exit_program)
        self.exit_button.grid(row=15, column=0, columnspan=4)

        self.loop = Loop(int(self.t_a_entry.get()), int(self.t_r_entry.get()), float(self.b_entry.get()),
                         int(self.n_entry.get()), str(self.wt_entry.get()))
        self.draw_circles()
        self.auto_step()

    def draw_circles(self):
        self.canvas.delete("all")
        input_type = self.input_type_var.get()
        input_duration = int(self.input_duration_entry.get())
        external_input = int(self.external_input_entry.get())
        time_steps = self.time_steps
        states, outputs = self.loop.forward(torch.tensor([external_input]), input_duration, input_type, time_steps)
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

        for i, op in enumerate(ops):
            angle = 90 - i * angle_step
            entry_angle = angle - 180/num_circles
            x = center_x + 1 * radius * cos(radians(entry_angle))
            y = center_y - 1 * radius * sin(radians(entry_angle))

            weight_var = self.loop.get_weights()[i]
            weight_label = Label(self.root, text=weight_var)
            self.canvas.create_window(x, y, window=weight_label)

    def create_input(self, label_text, row, column, default_value, unit, command):
        label = Label(self.root, text=label_text)
        label.grid(row=row, column=column, sticky=tk.W)
        entry = Entry(self.root)
        entry.grid(row=row, column=column + 1)
        entry.insert(0, default_value)
        entry.bind("<KeyRelease>", command)
        unit = Label(self.root, text=unit)
        unit.grid(row=row, column=column + 2, sticky=tk.W)
        return label, entry

    def next_step(self):
        time_steps = self.time_steps
        if self.current_time_step < time_steps - 1:
            self.current_time_step += 1
            self.label.config(text=f"Time Step: {self.current_time_step + 1}")
            self.draw_circles()
        else:
            self.stop()

    def auto_step(self):
        if self.running:
            self.next_step()
        self.auto_step_id = self.root.after(int(self.s_entry.get()), self.auto_step)

    def stop(self):
        self.running = False

    def go_(self):
        self.running = True

    def restart(self):
        self.current_time_step = 0
        self.label.config(text=f"Time Step: {self.current_time_step + 1}")
        # Update the loop with new input values
        time_steps = self.time_steps
        input_duration = int(self.input_duration_entry.get())
        input_type = self.input_type_var.get()
        external_input = int(self.external_input_entry.get())
        self.loop = Loop(int(self.t_a_entry.get()), int(self.t_r_entry.get()), float(self.b_entry.get()),
                         int(self.n_entry.get()), str(self.wt_entry.get()))  # Reinitialize the loop
        self.loop.forward(torch.tensor([external_input]), input_duration, input_type, time_steps)
        self.running = False
        self.draw_circles()

    def exit_program(self):
        self.root.after_cancel(self.auto_step_id)
        self.running = False
        self.root.destroy()

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


# load model
def loadModel(filename):
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
