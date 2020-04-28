import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def initializeWeights(n):
    w = np.zeros((1,n))
    b = 0
    return w,b

def sigmoid(x):
    value = 1/(1+np.exp(-x))
    return value

def getData(test_size):
    file = open('data_banknote_authentication.txt')
    
    X = []
    y = []
    
    for line in file:
        temp = line.split(',')
        X.append([float(x) for x in temp[0:4]])
        y.append(float(temp[4]))
        
    X = pd.DataFrame(X)
    X = (X-X.mean())/X.std()
    y = pd.DataFrame(y)
    
    return train_test_split(X, y, test_size=test_size,random_state=0)

def getPrediction(X,y,w,b,lam):
    result = sigmoid(np.dot(w,X.T)+b)
    m = X.shape[0]
    y_t = np.array(y).T
    error = (-1/m)*(np.sum(y_t*np.log(result) + (1-y_t)*np.log(1-result)) + \
    lam*np.sum(abs(w)))
    #print(error)
    dw = (1/m)*(np.dot(X.T, (result-y_t).T)).T
    db = (1/m)*np.sum(result-y_t)
    
    return result,error,dw,db


X_train, X_test, y_train, y_test = getData(0.2)
w,b = initializeWeights(4)
alpha = 1

error_plot_x = []
error_plot_y = []

for i in range(1,2001):
    result,error,dw,db = getPrediction(X_train,y_train,w,b,1)
    w = w - alpha*dw
    b = b - alpha*db
    
    if(i%5==0):
        error_plot_x.append(i)
        error_plot_y.append(error)
        
plt.plot(error_plot_x,error_plot_y)
plt.xlabel('Epochs') 
plt.ylabel('Error')
plt.title('Logistic Regression with L1 Regularization(alpha=1)')    

y_pred = sigmoid(np.dot(w,X_test.T)+b)
y_test = np.array(y_test).T


confusion = [[0,0],[0,0]]

for i in range(y_pred.shape[1]):
    if y_pred[0][i]>=0.5:
        y_pred[0][i]=1
    else:
        y_pred[0][i]=0
    
    if(y_test[0][i]==0 and y_pred[0][i]==0):
        confusion[0][0]+=1
    elif(y_test[0][i]==0 and y_pred[0][i]==1):
        confusion[0][1]+=1
    elif(y_test[0][i]==1 and y_pred[0][i]==0):
        confusion[1][0]+=1
    elif(y_test[0][i]==1 and y_pred[0][i]==1):
        confusion[1][1]+=1

print(confusion)
