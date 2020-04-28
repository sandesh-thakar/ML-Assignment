import pandas as pd
import numpy as np

def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

def relu(x):
    relu = np.maximum(0,x)
    return relu

data = pd.read_csv('housepricedata.csv')
X = data.iloc[:,0:10]
y = data.iloc[:,-1]

network_layers = [
    {"layer_size": 10, "activation": "none"}, 
    {"layer_size": 5, "activation": "relu"},
    {"layer_size": 5, "activation": "relu"},
    {"layer_size": 4, "activation": "relu"},
    {"layer_size": 1, "activation": "sigmoid"}
]

def initialize_parameters(network_layers):
    parameters = {}
    num_layers = len(network_layers)
    
    np.random.seed(0)
    
    for i in range(1,num_layers):
        parameters['W' + str(i)] = \
        np.random.randn(network_layers[i]["layer_size"], \
                        network_layers[i-1]["layer_size"]) * 0.01
        
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
    
    for l in range(1,num_layers):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        activation = network_layers[l]['activation']
        Z,A=forward_prop_helper(A_prev,W,b,activation)
        forward_cache['Z'+str(l)]=Z
        forward_cache['A'+str(l-1)]=A
    
    AL = A
    return AL,forward_cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(1 - Y, np.log(1 - AL))
    cost = - np.sum(logprobs) / m
    #cost = np.squeeze(cost)   
    return cost

parameters = initialize_parameters(network_layers)
AL,forward_cache=forward_prop(X.T,parameters,network_layers)
cost = compute_cost(AL,y.reshape(1,-1))