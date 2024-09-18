#!/usr/bin/env python3
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

# read data from file
x_up = []
y_up = []
x_down = []
y_down = []

with open('demo.data', 'r') as f:
    for line in f:
        data = line.strip().split()
        x_up.append(float(data[0]))
        y_up.append(float(data[1]))
        flag = float(data[3])
        if flag > 0.5:
            x_down.append(float(data[0]))
            y_down.append(float(data[1]))
        else:
            x_up.append(float(data[0]))
            y_up.append(float(data[1]))

x_array = np.array(x_up)
y_array = np.array(y_up)
x_array_down = np.array(x_down)
y_array_down = np.array(y_down)

# plot data
plt.scatter(x_array, y_array, color='blue')
plt.scatter(x_array_down, y_array_down, color='red')

plt.show()
