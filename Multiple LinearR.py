import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("multiple_linear_regression_dataset.csv")
x = data[["age","experience"]].to_numpy()
y = data["income"].to_numpy()

y = y/1000

x_train = x[:15]
y_train = y[:15]
x_test = x[15:]
y_test = y[15:]


def model(x,w,b):
    return x.dot(w) + b
def costFunction(x,y,w,b,m):
    yhat = model(x,w,b)
    return (1/(2*m)) * np.sum((yhat - y)**2)
def gradientDescent(x,y,w,b,alpha,iterations,m):
    costList = []

    for i in range(iterations):
        yhat = model(x,w,b)
        error = yhat - y
        dw = (1/m) * (np.sum(error[:,None] * x, axis=0))
        db = (1/m) * np.sum(error)

        cost = costFunction(x,y,w,b,m)
        costList.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i} : Cost = {cost}")

        w -= alpha * dw
        b -= alpha * db
    return w,b,costList

m = len(x_train)
alpha = 0.001
iterations = 100000
w = np.array([0.0 , 0.0])
b = 0

w,b,CostHistory = gradientDescent(x_train,y_train,w,b,alpha, iterations, m)
print("---------------")
print(f"Final w = {w}")
print(f"Final bias = {b}")
print(f"Final cost = {CostHistory[-1]}")

plt.scatter(x[:, 0],y,c="red",label = "Actual")
plt.scatter(x[:, 0],model(x,w,b), c="green", label="Predicted")
plt.legend()
plt.show()
