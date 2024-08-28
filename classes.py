import pickle
import math
import cv2
import numpy as np
# import arguments as arg
import math_functions as mf
from matplotlib import pyplot as plt
import glob
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib
import tkinter as tk
from tkinter import ttk # ttk is used for more modern widgets
from tkinter import Canvas, Button, Label, Entry, Radiobutton, StringVar
from math import cos, sin, radians
from save_load_functions import save_inputs, load_inputs, load_most_recent_inputs
from save_canvas_image import save_canvas

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
    def __init__(self, num_inputs: int, t, bias, weight_type, w, weight_boundary, forgetting_rate):
        t = t.split(',')
        t = [int(a) for a in t]
        wb = weight_boundary.split(',')
        wb = [float(b) for b in wb]
        if weight_type == "random":
            self.weights = torch.tensor(wb[0]) + torch.mul(torch.rand(num_inputs), wb[1] - wb[0])
        else:
            self.weights = torch.tensor([w for _ in range(num_inputs)])

        self.bias = torch.tensor([bias])
        self.forgetting_rate = forgetting_rate
        self.activation_functions = activation_functions['step']
        self.gate_value = torch.tensor([1, 1, 0])
        self.a_threshold = torch.tensor([1])
        # self.a_value = self.a_threshold.data  # relu
        self.a_value = torch.tensor([1])  # step
        self.g = torch.zeros(1)
        self.output = torch.tensor([0])
        self.output_record = [torch.tensor(0)]

        # t
        self.t_a = t[0]
        self.t_rf = t[1]

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
        inputs = inputs.float().unsqueeze(0)

        # calculate weighted inputs
        weighted_inputs = torch.matmul(self.weights, inputs)

        # gate 1: control the input in, all or zero
        self.g += torch.mul(self.gate_value[0], weighted_inputs)

        a = self.activation_functions(self.g + self.bias, threshold=self.a_threshold, value=self.a_value)

        # gate 3: control the output out, all or zero
        self.output = torch.mul(self.gate_value[2], a)
        self.output_record.append(self.output)

        self.update(a)

        # gate 2: control the accumulation(g), drop or keep
        self.g = torch.mul(self.gate_value[1], self.g)

        return self.output

    def update(self, a):
        # different gate values corresponding to different neuron phases
        rest = torch.tensor([1, 1 - self.forgetting_rate, 0])
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
        # return round(self.weights.item(), 3)
        return self.weights

    def get_g(self):
        return self.g

    def init_g(self):
        self.g = 0


class Loop:
    def __init__(self, t, bias, number, weight_type, w, weight_list, weight_boundary, forgetting_rate):
        self.neuron_num = number
        wl = weight_list.split(',')
        wl = [float(a) for a in wl]
        if weight_type == "custom":
            self.neurons = [
                Neuron(1, t, bias, weight_type, wl[i], weight_boundary, forgetting_rate) for i in range(self.neuron_num)]
        else:
            self.neurons = [
                Neuron(1, t, bias, weight_type, w, weight_boundary, forgetting_rate) for _ in range(self.neuron_num)]
        self.outputs = [[torch.tensor([0]) for _ in range(self.neuron_num)]]
        self.states = [['rest' for _ in range(self.neuron_num)]]
        self.weights = []
        self.g = [[torch.tensor([0]) for _ in range(self.neuron_num)]]

    def forward(self, external_input, input_duration, input_type, time_steps, period=1):
        external_input = external_input.split(',')
        external_input = [float(a) for a in external_input]
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
                ex_signal = torch.zeros(self.neuron_num)

            input_signal = [self.neurons[i - 1].get_output() + ex_signal[i] for i in range(self.neuron_num)]

            # update neurons
            [self.neurons[i].forward(input_signal[i]) for i in range(self.neuron_num)]

            # record output and state
            current_states = [self.neurons[j].get_state() for j in range(self.neuron_num)]
            current_outputs = [self.neurons[j].get_output() for j in range(self.neuron_num)]
            current_weights = [self.neurons[j].get_weight() for j in range(self.neuron_num)]
            current_g = [self.neurons[j].get_g() for j in range(self.neuron_num)]

            self.states.append(current_states)
            self.outputs.append(current_outputs)
            self.weights.append(current_weights)
            self.g.append(current_g)

        return self.states, self.outputs, self.weights, self.g

    def init_g(self):
        [neuron.init_g() for neuron in self.neurons]


def state_to_number(st):
    if st == "active":
        return 1
    elif st == "refractory":
        return -1
    elif st == "rest":
        return 0
    return 0


class CircleDiagram:
    def __init__(self, root):
        self.root = root
        self.screen_ratio = 0.7
        self.running = False
        self.auto_step_id = None
        self.neuron_number = 6
        self.time_steps = 1000
        self.current_time_step = 0
        # Create total_width attribute
        self.total_width = 0  # Initialize total_width
        
        # Initial Neuron Setting
        self.neuron_column = 0
        self.create_label("Weight Type:", 0, self.neuron_column)
        self.weight_type_var = StringVar(value="custom")
        self.create_radio_button("Random", self.weight_type_var, "random", 1, self.neuron_column)
        self.create_radio_button("All the same", self.weight_type_var, "all the same", 2, self.neuron_column)
        self.create_radio_button("Custom", self.weight_type_var, "custom", 3, self.neuron_column)

        self.setting_column = 1
        self.create_label("Neuron Setting:", 0, self.setting_column)
        self.wb_label, self.wb_entry = self.create_input("boundary:", 1, self.setting_column, "0.5,0.8")
        self.w_label, self.w_entry = self.create_input("value:", 2, self.setting_column, "0.6")
        self.wv_label, self.wv_entry = self.create_input("value list:", 3, self.setting_column, "0.4,0.6,0.7,0.5,0.6,0.9")
        self.b_label, self.b_entry = self.create_input("Bias:", 4, self.setting_column, "0.2")
        self.t_label, self.t_entry = self.create_input("τa, τrf:", 5, self.setting_column, "4,2")
        self.f_label, self.f_entry = self.create_input("Forgetting Rate:", 6, self.setting_column, "0.3")
        self.n_label, self.n_entry = self.create_input("Neuron Number:", 7, self.setting_column, str(self.neuron_number))

        # Input Setting
        self.input_column = 3
        self.create_label("Input Type:", 0, self.input_column)
        self.input_type_var = StringVar(value="constant")
        self.create_radio_button("OneTime", self.input_type_var, "one_time", 1, self.input_column)
        self.create_radio_button("Constant", self.input_type_var, "constant", 2, self.input_column)
        self.create_radio_button("Periodic", self.input_type_var, "periodic", 3, self.input_column)
        self.input_duration_label, self.input_duration_entry = self.create_input("Input Duration:", 4, self.input_column, "5")
        self.input_period_label, self.input_period_entry = self.create_input("Input Period:", 5, self.input_column, "1")
        self.external_input_label, self.external_input_entry = self.create_input("Input Value:", 6, self.input_column, "1,0,0,0,0,0")

        # Save and Load
        self.entries = {}
        self.vars = {}

        # Initial Neuron Setting
        self.neuron_column = 0
        self.create_label("Weight Type:", 0, self.neuron_column)
        self.vars['weight_type'] = StringVar(value="custom")
        self.create_radio_button("Random", self.vars['weight_type'], "random", 1, self.neuron_column)
        self.create_radio_button("All the same", self.vars['weight_type'], "all the same", 2, self.neuron_column)
        self.create_radio_button("Custom", self.vars['weight_type'], "custom", 3, self.neuron_column)

        self.setting_column = 1
        self.create_label("Neuron Setting:", 0, self.setting_column)
        self.wb_label, self.entries['boundary'] = self.create_input("boundary:", 1, self.setting_column, "0.5,0.8")
        self.w_label, self.entries['weight'] = self.create_input("value:", 2, self.setting_column, "0.6")
        self.wv_label, self.entries['value_list'] = self.create_input("value list:", 3, self.setting_column, "0.4,0.6,0.7,0.5,0.6,0.9")
        self.b_label, self.entries['bias'] = self.create_input("Bias:", 4, self.setting_column, "0.2")
        self.t_label, self.entries['tau'] = self.create_input("τa, τrf:", 5, self.setting_column, "4,2")
        self.f_label, self.entries['forgetting_rate'] = self.create_input("Forgetting Rate:", 6, self.setting_column, "0.3")
        self.n_label, self.entries['neuron_number'] = self.create_input("Neuron Number:", 7, self.setting_column, str(self.neuron_number))

        # Input Setting
        self.input_column = 3
        self.create_label("Input Type:", 0, self.input_column)
        self.vars['input_type'] = StringVar(value="constant")
        self.create_radio_button("OneTime", self.vars['input_type'], "one_time", 1, self.input_column)
        self.create_radio_button("Constant", self.vars['input_type'], "constant", 2, self.input_column)
        self.create_radio_button("Periodic", self.vars['input_type'], "periodic", 3, self.input_column)
        self.input_duration_label, self.entries['input_duration'] = self.create_input("Input Duration:", 4, self.input_column, "5")
        self.input_period_label, self.entries['input_period'] = self.create_input("Input Period:", 5, self.input_column, "1")
        self.external_input_label, self.entries['external_input'] = self.create_input("Input Value:", 6, self.input_column, "1,0,0,0,0,0")

        # Add Save and Load to the menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save as CSV", command=lambda: save_inputs(self.entries, self.vars))
        filemenu.add_command(label="Save as JPG", command=lambda: save_canvas(self.right_canvas, self.total_width))
        filemenu.add_command(label="Save Both (CSV + JPG)", command=lambda: (save_inputs(self.entries, self.vars), save_canvas(self.right_canvas, self.total_width)))
        filemenu.add_command(label="Load", command=lambda: load_inputs(self.entries, self.vars))
        filemenu.add_command(label="Load Most Recent", command=lambda: load_most_recent_inputs(self.entries, self.vars))
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)

        # Visualization Setting
        self.visualization_column = 0
        self.s_label, self.s_entry = self.create_input("Playback Speed:", 9, self.visualization_column, "500")
        self.label = Label(root, text=f"Time Step: {self.current_time_step + 1}")
        self.label.grid(row=10, column=self.visualization_column, sticky=tk.W)
        self.create_button("Stop", self.stop, 11, self.visualization_column)
        self.create_button("Go", self.go_, 12, self.visualization_column)
        self.create_button("Restart", self.restart, 13, self.visualization_column)
        self.create_button("Exit", self.exit_program, 14, self.visualization_column)

        # Canvas for drawing loop
        self.canvas = Canvas(root, width=int(520 * self.screen_ratio), height=int(400 * self.screen_ratio))
        self.canvas.grid(row=8, column=0, columnspan=2)

        # Canvas for drawing state history
        # Scrollable right canvas for drawing state history
        self.right_canvas_frame = tk.Frame(root)
        self.right_canvas_frame.grid(row=8, column=2, columnspan=2, padx=10, pady=10)

        self.right_canvas = Canvas(self.right_canvas_frame, width=int(500 * self.screen_ratio), height=(430 * self.screen_ratio))
        self.scrollbar = tk.Scrollbar(self.right_canvas_frame, orient="horizontal", command=self.right_canvas.xview)
        self.right_canvas.configure(xscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="bottom", fill="x")
        self.right_canvas.pack(side="left", fill="both", expand=True)

        # self.right_canvas = Canvas(root, width=int(450 * self.screen_ratio), height=(430 * self.screen_ratio))
        # self.right_canvas.grid(row = 8, column = 2, columnspan=2, padx=10, pady=10 )
        # Canvas for drawing state history scatter plot
        self.scatter_canvases = []
        for i in range(self.neuron_number):
            canvas = Canvas(root, width=int(250 * self.screen_ratio), height=int(250 * self.screen_ratio))
            canvas.grid(row=8 + i // 3, column=i % 3 + 4, padx=10, pady=10)
            self.scatter_canvases.append(canvas)

        self.state_history = [[] for _ in range(self.neuron_number)]
        self.time_steps_history = [[] for _ in range(self.neuron_number)]

        self.loop = Loop(str(self.t_entry.get()), float(self.b_entry.get()), int(self.n_entry.get()),
                         str(self.weight_type_var.get()), float(self.w_entry.get()), str(self.wv_entry.get()),
                         str(self.wb_entry.get()), float(self.f_entry.get()))
        self.draw_circles()
        self.auto_step() 


    def draw_circles(self):
        self.canvas.delete("all")
        input_type = self.input_type_var.get()
        input_duration = int(self.input_duration_entry.get())
        external_input = str(self.external_input_entry.get())
        time_steps = self.time_steps
        states, outputs, weights, g = self.loop.forward(external_input, input_duration, input_type,
                                                        time_steps)
        sts = states[self.current_time_step]
        ops = outputs[self.current_time_step]
        wts = weights[self.current_time_step]
        gs = g[self.current_time_step]
        num_circles = len(sts)
        angle_step = 360 / num_circles
        radius = int(150 * self.screen_ratio)
        center_x, center_y = int(200 * self.screen_ratio), int(200 * self.screen_ratio)
    

        # drawing
        for i, st in enumerate(sts):
            angle = 90 - i * angle_step
            x = center_x + radius * cos(radians(angle))
            y = center_y - radius * sin(radians(angle))
            self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=statesToColors(st))
            # self.canvas.create_text(x, y - 30, text=st)
            self.state_history[i].append(st)
            self.time_steps_history[i].append(self.current_time_step)
            # if len(self.state_history[i]) > 10:
            #     self.state_history[i].pop(0)
            #     self.time_steps_history[i].pop(0)

        self.draw_state_history()

        for i, wt in enumerate(wts):
            angle = 90 - i * angle_step
            entry_angle = angle + 180 / num_circles
            x_w = center_x + 1 * radius * cos(radians(entry_angle))
            y_w = center_y - 1 * radius * sin(radians(entry_angle))

            weight_label = Label(self.root, text=round(wt.item(), 3))
            self.canvas.create_window(x_w, y_w, window=weight_label)

        for i, op in enumerate(ops):
            angle1 = 90 - i * angle_step
            angle2 = 90 - (i + 1) % num_circles * angle_step
            x1 = center_x + radius * cos(radians(angle1))
            y1 = center_y - radius * sin(radians(angle1))
            x2 = center_x + radius * cos(radians(angle2))
            y2 = center_y - radius * sin(radians(angle2))
            self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill=outputsToColors(op))

        for i, g in enumerate(gs):
            angle = 90 - i * angle_step
            x_g = center_x + 0.62 * radius * cos(radians(angle))
            y_g = center_y - 0.62 * radius * sin(radians(angle))

            g_label = Label(self.root, text=round(g.item(), 3))
            self.canvas.create_window(x_g, y_g, window=g_label)

        # Drawing the state annotations closer to the main diagram
        annotation_radius = 10  # Radius for the annotation circles
        annotation_positions = [(center_x + 180, center_y - 100),  # Adjusted positions for no overlap
                                (center_x + 180, center_y),
                                (center_x + 180, center_y + 100)]
        colors = ['red', 'orange', 'yellow']
        labels = ['active stage', 'refractory stage', 'rest stage']

        for i, (color, label) in enumerate(zip(colors, labels)):
            x, y = annotation_positions[i]
            self.canvas.create_oval(x - annotation_radius, y - annotation_radius, 
                                    x + annotation_radius, y + annotation_radius, 
                                    fill=color)
            self.canvas.create_text(x, y + 30, text=label, font=("Arial", 12))



    def create_label(self, text, row, column):
        label = Label(self.root, text=text)
        label.grid(row=row, column=column, sticky=tk.W)

    def create_radio_button(self, text, variable, value, row, column):
        radio_button = Radiobutton(self.root, text=text, variable=variable, value=value,
                                   command=self.pause_on_entry_change)
        radio_button.grid(row=row, column=column, sticky=tk.W)

    def create_input(self, label_text, row, column, default_value, sticky=tk.W):
        label = Label(self.root, text=label_text)
        label.grid(row=row, column=column, sticky=sticky)
        entry = Entry(self.root)
        entry.grid(row=row, column=column + 1)
        entry.insert(0, default_value)
        entry.bind("<KeyRelease>", self.pause_on_entry_change)
        return label, entry

    def create_button(self, text, command, row, column, sticky=tk.W):
        button = Button(self.root, text=text, command=command)
        button.grid(row=row, column=column, sticky=sticky)

    def draw_state_history(self):
        # self.right_canvas.delete("all")
        # for canvas in self.scatter_canvases:
        #     canvas.delete("all")
        circle_radius = int(15 * self.screen_ratio)
        spacing = int(30 * self.screen_ratio)
        x_offset = 0  # Start from 0 to ensure the first step is visible
        total_width = len(self.time_steps_history[0]) * (circle_radius * 2 + spacing) + circle_radius + spacing - x_offset
        self.right_canvas.config(scrollregion=(0, 0, total_width, 500))
        self.total_width = total_width

        for i, history in enumerate(self.state_history):
            for j, state in enumerate(history):
                x = j * (circle_radius * 2 + spacing) + circle_radius + spacing
                y = i * (circle_radius * 2 + spacing) + circle_radius + spacing + int(50 * self.screen_ratio)
                self.right_canvas.create_oval(x - circle_radius, y - circle_radius, x + circle_radius,
                                              y + circle_radius, fill=statesToColors(state))

        for j in range(len(self.time_steps_history[0])):
            x = j * (circle_radius * 2 + spacing) + circle_radius + spacing
            self.right_canvas.create_text(x, 20, text=f"{self.time_steps_history[0][j]}", font=("Arial",12))


        # 绘制每个神经元的折线图
        for i, history in enumerate(self.state_history):
            canvas = self.scatter_canvases[i]
            prev_x, prev_y = None, None
            for j, state in enumerate(history):
                x = j * int(10 * self.screen_ratio) + int(50 * self.screen_ratio)
                y = int(100 * self.screen_ratio) - int(state_to_number(state) * 30 * self.screen_ratio)
                if prev_x is not None and prev_y is not None:
                    canvas.create_line(prev_x, prev_y, x, y, fill="blue")
                prev_x, prev_y = x, y

            # 绘制坐标轴
            canvas.create_line(int(50 * self.screen_ratio), int(100 * self.screen_ratio), int(200 * self.screen_ratio), int(100 * self.screen_ratio), arrow=tk.LAST)  # 横轴
            canvas.create_line(int(50 * self.screen_ratio), int(20 * self.screen_ratio), int(50 * self.screen_ratio), int(180 * self.screen_ratio), arrow=tk.LAST)  # 纵轴
            canvas.create_text(int(200 * self.screen_ratio), int(110 * self.screen_ratio), text="Time Step")
            canvas.create_text(int(30 * self.screen_ratio), int(20 * self.screen_ratio), text="State")
            canvas.create_text(int(125 * self.screen_ratio), int(10 * self.screen_ratio), font=('Arial','10'), text=f"Neuron {i + 1}")

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
        external_input = str(self.external_input_entry.get())
        self.neuron_number = int(self.n_entry.get())  # 更新神经元数量
        self.loop = Loop(str(self.t_entry.get()), float(self.b_entry.get()), int(self.n_entry.get()),
                         str(self.weight_type_var.get()), float(self.w_entry.get()), str(self.wv_entry.get()),
                         str(self.wb_entry.get()), float(self.f_entry.get()))
        self.loop.forward(external_input, input_duration, input_type, time_steps)
        self.running = False
        self.state_history = [[] for _ in range(self.neuron_number)]
        self.time_steps_history = [[] for _ in range(self.neuron_number)]
        self.draw_circles()
        self.draw_state_history()

    def exit_program(self):
        self.root.after_cancel(self.auto_step_id)
        self.running = False
        self.root.destroy()

    def pause_on_entry_change(self):
        self.stop()

def drawLoop():
    root = tk.Tk()
    CircleDiagram(root)
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