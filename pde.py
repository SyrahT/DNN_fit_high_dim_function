import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt   
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import platform
import shutil 


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, R_variable['astddev'])
        m.bias.data.normal_(0, R_variable['bstddev'])
class Act_op(nn.Module):
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        return x ** 50
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.block = nn.Sequential()
        for i in range(len(R_variable['full_net'])-2):
            self.block.add_module('linear'+str(i), nn.Linear(R_variable['full_net'][i],R_variable['full_net'][i+1]))
            if R_variable['ActFuc']==1:
                self.block.add_module('tanh'+str(i), nn.Tanh())
            elif R_variable['ActFuc']==3:
                self.block.add_module('sin'+str(i), nn.sin())
            elif R_variable['ActFuc']==0:
                self.block.add_module('relu'+str(i), nn.ReLU())
            elif R_variable['ActFuc']==4:
                self.block.add_module('**50'+str(i), Act_op())
            elif R_variable['ActFuc']==5:
                self.block.add_module('sigmoid'+str(i), nn.sigmoid())
        i = len(R_variable['full_net'])-2
        self.block.add_module('linear'+str(i), nn.Linear(R_variable['full_net'][i],R_variable['full_net'][i+1]))
    def forward(self, x):
        out = self.block(x)
        return out

R_variable={}

R_variable['input_dim'] = 1
R_variable['output_dim'] = 1
R_variable['ActFuc'] = 1  ###  0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate
R_variable['hidden_units'] = [100,100,100]
R_variable['full_net'] = [R_variable['input_dim']] + R_variable['hidden_units'] + [R_variable['output_dim']]

R_variable['learning_rate'] = 1e-5
R_variable['learning_rateDecay'] = 2e-6
R_variable['belta'] = 1e-3

R_variable['astddev'] = 2/np.square(R_variable['input_dim']+R_variable['output_dim'])
R_variable['bstddev'] = 2/(R_variable['input_dim']+R_variable['output_dim'])
R_variable['beta'] = 0.01

plot_epoch = 500
R_variable['train_size'] = 1000; 
R_variable['batch_size'] = R_variable['train_size']
R_variable['test_size'] = R_variable['train_size'] 
R_variable['x_start'] = -np.pi/2
R_variable['x_end'] = np.pi/2

if R_variable['input_dim']==1:
    R_variable['test_inputs'] =np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                      endpoint=True),[R_variable['test_size'],1])
    R_variable['train_inputs']=np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                      endpoint=True),[R_variable['train_size'],1])

else:
    R_variable['test_inputs']=np.random.rand(R_variable['test_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['train_inputs']=np.random.rand(R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']

R_variable['test_inputs_b'] = R_variable['test_inputs'].copy()
R_variable['train_inputs_b'] = R_variable['train_inputs'].copy()

mask = np.random.choice(R_variable['input_dim']*2, R_variable['test_size'], replace=True)
for i in range(R_variable['test_size']):
    if mask[i]%2==0:
        R_variable['test_inputs_b'][i][mask[i]//2] = R_variable['x_start']
    else:
        R_variable['test_inputs_b'][i][mask[i]//2] = R_variable['x_end']
mask = np.random.choice(R_variable['input_dim']*2, R_variable['train_size'], replace=True)
for i in range(R_variable['train_size']):
    if mask[i]%2==0:
        R_variable['train_inputs_b'][i][mask[i]//2] = R_variable['x_start']
    else:
        R_variable['train_inputs_b'][i][mask[i]//2] = R_variable['x_end']

def get_y(xs):
    tmp = 0
    for ii in range(R_variable['input_dim']):
        tmp += np.cos(4*xs[:,ii:ii+1]) 
    return tmp

def get_g(xs):
    tmp = 0
    for ii in range(R_variable['input_dim']):
        tmp += 16*np.cos(4*xs[:,ii:ii+1]) 
    return tmp

def get_f(xs):
    return get_y(xs)


def get_y_t(xs):
    return torch.cos(4*xs) 

def get_g_t(xs):
    return 16*torch.cos(4*xs) 

def get_f_t(xs):
    return get_y_t(xs)

# delta u = -g
# u on edge = f

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, dx, x_, x_b): 
        return torch.mean(torch.pow(torch.tensor(dx), 2)- get_f_t(x) * y) 
        + R_variable['beta'] * torch.mean(torch.pow(x_-get_g_t(x_b), 2))


train_inputs=R_variable['train_inputs']
test_inputs=R_variable['test_inputs']
train_inputs_b = R_variable['train_inputs_b']
test_inputs_b = R_variable['test_inputs_b']

R_variable['y_true_train']=get_f(train_inputs)
R_variable['y_true_test']= get_f(test_inputs)
R_variable['y_true_train_b'] = get_g(train_inputs_b)
R_variable['y_true_test_b'] = get_g(test_inputs_b)

t0=time.time()
net_ = Network()
net_.apply(weights_init)
print(net_)
criterion = Loss()
optimizer = torch.optim.SGD(net_.parameters(), lr=R_variable['learning_rate'])

for epoh in range(10000):
    optimizer = torch.optim.SGD(net_.parameters(), lr=R_variable['learning_rate'])

    X = torch.tensor(train_inputs, requires_grad=True, dtype= torch.float32)
    X_b = torch.tensor(train_inputs_b, requires_grad=True, dtype= torch.float32)
    y_train = net_(X)
    y_train_b = net_(X_b)
    dx = torch.autograd.grad(outputs=y_train,inputs=X, grad_outputs=torch.ones_like(X), retain_graph=True, create_graph=True,only_inputs=True)[0]
    loss = criterion(x = X, y=y_train, dx = dx, x_=y_train_b, x_b = X_b)

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step() 

    if epoh%500==0:
        print(loss)

    #R_variable['learning_rate']=R_variable['learning_rate']*(1-R_variable['learning_rateDecay'])



