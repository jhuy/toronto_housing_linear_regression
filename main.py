import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_interations_of_regression = 300

# intercept
theta0 = 2
# slope
theta1 = 0
# learning rate
alpha = 0.25

# Reading Housing Data File
file = pd.read_csv('MLS.csv', delimiter=',')  
 
comp_index = []
comp_benchmark_price = []

for input in range(0,len(file.iloc[:,0])):
    if 'Toronto' in file.iloc[input,0]:
        comp_index.append(file.iloc[:,1][input]/100)
        comp_benchmark_price.append(file.iloc[:,2][input])

data = []
for index in range (0, len(comp_index)):
    data.append([comp_index[index], comp_benchmark_price[index]])

num_rand_ints = len(comp_index)

df = pd.DataFrame (data, columns=['x','y'])
df.plot(kind='scatter', x='x', y='y')

# Gradient Descent Calculation 
def new_theta_function (alpha, theta0, theta1, x, y):
    new_theta0 = float(theta0 - (alpha/num_rand_ints) * sum_theta_function_x(theta0, theta1, x, y))
    new_theta1 = float(theta1 - (alpha/num_rand_ints) * sum_theta_function_y(theta0, theta1, x, y))
    cost = (1/(2*num_rand_ints)) * sum_y_coordinates(theta0, theta1, x, y)
    print ([new_theta0, new_theta1, cost])
    return [new_theta0, new_theta1, cost]

#sum((theta0 + theta1*x -y)*x)
def sum_theta_function_x (theta0, theta1, x, y):
    total = 0
    for index in range(0, num_rand_ints):
        total += float((theta0*x.iloc[index] + theta1 - y.iloc[index])*x.iloc[index])
    return total

#sum(theta0 + theta1*x - y)
def sum_theta_function_y (theta0, theta1, x, y):
    total = 0
    for index in range(0, num_rand_ints):
        total += float(theta0*x.iloc[index] + theta1 - y.iloc[index])
    return total

def sum_y_coordinates (theta0, theta1, x, y):
    total = 0
    for index in range (0, num_rand_ints):
        total += (theta0*x.iloc[index] + theta1 - y.iloc[index])**2
    return total

# Main
# Lowest Cost Tracker
[lowest_theta0, lowest_theta1, lowest_cost] = [0, 0, 0]

print ("Computing theta0 and theta1 " + str(num_interations_of_regression) + " times, please wait.")
for index in range(num_interations_of_regression):
    [theta0, theta1, cost] = new_theta_function(alpha, theta0, theta1, df.loc[:, 'x'], df.loc[:, 'y'])
    if index == 0:
        lowest_cost = cost
    if index > 0:
        if lowest_cost > cost:
            lowest_cost = cost
            lowest_theta0 = theta0
            lowest_theta1 = theta1

print("The best theta parameters based on the lowest cost of: " + str(lowest_cost) + " is theta0: " + str(lowest_theta0) + " and theta1: " + str(lowest_theta1))
