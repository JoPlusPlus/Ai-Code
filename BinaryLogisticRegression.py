import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


X,Y = datasets.load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
w = np.zeros(30)
b = 0
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def costFunction(x,y,w,b,LAMBDA):
    m = len(x)
    f_wb = sigmoid(x.dot(w)+b)
    return (1/m) * np.sum( (-y * np.log(f_wb)) - ( (1 - y) * np.log(1 - f_wb)) ) + (LAMBDA/(2*m))*np.sum(w**2)


def fit(x,y,w,b,alpha,LAMBDA,iter):
    m = len(x)
    for i in range(iter):
        dw = (1/m) * np.dot((sigmoid(x.dot(w) + b)-y),x) + (LAMBDA/m)*w
        db = (1/m) * np.sum((sigmoid(x.dot(w) + b)-y))

        w = w - alpha * dw
        b = b - alpha * db
    return w,b
def predict(x,w,b):
    z = sigmoid(x.dot(w) + b)
    z = [0 if i < 0.5 else 1 for i in z]
    return z

w,b = fit(x_train,y_train,w,b,0.01,1,iter=1000)

x_train_pred = predict(x_train,w,b)

acc = accuracy_score(y_train,x_train_pred)
print(f"{acc*100}%")

#data
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],c="red",label="Loser")
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],c="green",label="winner")
plt.legend()

#Decision boundary
xvals = np.array([x_train[:,0].min(), x_train[:,0].max()])
yvals = - (w[0]/w[1]*xvals) - (b/w[1])
plt.plot(xvals,yvals,c="black",label="Decision Boundary")


plt.show()
