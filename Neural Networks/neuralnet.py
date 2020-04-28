import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

def relu(x):
    relu = np.maximum(0,x)
    return relu

def sigmoid_backward(dA, Z):
    S=sigmoid(Z)
    dS=S*(1-S)
    return dA*dS

def relu_backward(dA, Z):
    dZ=np.array(dA,copy = True)
    dZ[Z<=0]=0
    return dZ

data = pd.read_csv('housepricedata.csv')

X = data.iloc[:,0:10]
X = (X - X.mean())/X.std()
y = data.iloc[:,-1]


X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

network_layers = [
    {"layer_size": 10, "activation": "none"}, 
    {"layer_size": 20, "activation": "relu"},
    {"layer_size": 8, "activation": "relu"},
    {"layer_size": 12, "activation": "relu"},
    {"layer_size": 1, "activation": "sigmoid"}
]

def initialize_parameters(network_layers):
    parameters = {}
    num_layers = len(network_layers)
    
    np.random.seed(0)
    
    for i in range(1,num_layers):
        parameters['W' + str(i)] = \
        np.random.randn(network_layers[i]["layer_size"], \
                        network_layers[i-1]["layer_size"])*0.25
        
        parameters['b' + str(i)] = \
        np.zeros((network_layers[i]["layer_size"], 1))
    
    return parameters

def forward_prop_helper(A_prev,W,b,activation):
    if activation=='sigmoid':
        Z = np.dot(W,A_prev)+b
        A = sigmoid(Z)
        return Z,A
    elif activation=='relu':
        Z = np.dot(W,A_prev)+b
        A = relu(Z)
        return Z,A

def forward_prop(X,parameters,network_layers):
    forward_cache = {}
    A = X
    num_layers = len(network_layers)
    
    forward_cache['A0']=A
    
    for l in range(1,num_layers):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        activation = network_layers[l]['activation']
        Z,A=forward_prop_helper(A_prev,W,b,activation)
        forward_cache['Z'+str(l)]=Z
        forward_cache['A'+str(l)]=A
    
    AL = A
    return AL,forward_cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(1 - Y, np.log(1 - AL))
    cost = - np.sum(logprobs) / m
    #cost = np.squeeze(cost)   
    return cost

#parameters = initialize_parameters(network_layers)
#AL,forward_cache=forward_prop(X.T,parameters,network_layers)
#cost = compute_cost(AL,y.reshape(1,-1))

def backward_prop(AL,y,parameters,forward_cache,network_layers):
    grads = {}
    num_layers=len(network_layers)
    y=y.reshape(AL.shape)
    
    dAL=-(y*np.subtract(y, AL) - (1-y)*np.subtract(1 - y, 1 - AL))
    
    dA_prev=dAL
    
    for l in reversed(range(1, num_layers)):
        dA_curr=dA_prev
        act=network_layers[l]["activation"]
        #print(l,dA_curr.shape,act)
        W_curr=parameters['W'+str(l)]
        Z_curr=forward_cache['Z'+str(l)]
        A_prev=forward_cache['A'+str(l-1)]
        
        dA_prev,dW_curr,db_curr=\
        backward_prop_helper(dA_curr,Z_curr,A_prev,W_curr,act)
        
        grads['dW'+str(l)]=dW_curr
        grads['db'+str(l)]=db_curr
        
    return grads

def backward_prop_helper(dA,Z,A_prev,W,act):
    #print(dA.shape,Z.shape)
    if act=="relu":
        dZ=relu_backward(dA, Z)
        dA_prev,dW,db=backward(dZ, A_prev, W)
    elif act=="sigmoid":
        dZ=sigmoid_backward(dA, Z)
        dA_prev,dW,db=backward(dZ, A_prev, W)

    return dA_prev, dW, db

def backward(dZ,A_prev,W):
    m = A_prev.shape[1]
    
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T, dZ)
    
    #print(dW.shape)
    
    return dA_prev, dW, db

#grads = backward_prop(AL,y,parameters,network_layers)

def update_params(parameters,grads,alpha):
    for l in range(1,len(parameters)//2):
        parameters['W'+str(l)]-=alpha*grads['dW'+str(l)]
        parameters['b'+str(l)]-=alpha*grads['db'+str(l)]
    
    return parameters

np.random.seed(1)
costs=[]
parameters=initialize_parameters(network_layers)

num_iterations=5000
learn_rate=0.1

for i in range(0,num_iterations):
    AL,forward_cache=forward_prop(X.T,parameters,network_layers)
    
    cost=compute_cost(AL,y.reshape(1,-1))
    costs.append(cost)
    
    grads=backward_prop(AL,y,parameters,forward_cache,network_layers)
    parameters=update_params(parameters,grads,learn_rate)


#Plot cost against number of iterations
x_data = [i for i in range(len(costs))]
y_data = [costs[i] for i in range(len(costs))]
plt.plot(x_data,y_data)

#Get predictions
pred,forward_cache=forward_prop(X_test.T,parameters,network_layers)
actual=list(y_test.T)
pred=list(pred[0])
for i in range(0,len(pred)):
    if(pred[i]>=0.5):
        pred[i]=1
    else:
        pred[i]=0

predictions=[]
for i in range(len(pred)):
    predictions.append([actual[i],pred[i]])

#DataFrame to store actual and predicted values
predictions=pd.DataFrame(predictions,columns=['actual','predicted'])

