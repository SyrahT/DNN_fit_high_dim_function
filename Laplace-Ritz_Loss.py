#Author: SyrahT/ Kejie Tang
#Email: tangkj00@sjtu.edu.cn

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

Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]

def mkdir(fn):# Create a directory
    if not os.path.isdir(fn):
        os.mkdir(fn)
def save_fig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):# Save the figure
    if isax==1:
        pltm.rc('xtick',labelsize=18)
        pltm.rc('ytick',labelsize=18)
        ax.set_position(pos, which='both')
    fnm = '%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp!=0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic==1:
        pltm.show() 
    elif isShowPic==-1:
        return
    else:
        pltm.close()

def weights_init(m):# Initialization weight
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, R_variable['astddev'])
        m.bias.data.normal_(0, R_variable['bstddev'])
class Act_op(nn.Module):# Custom activation function
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        return x ** 50# or F.relu(x) * F.relu(1-x)
class Network(nn.Module):# DNN 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
    def __init__(self):
        super(Network, self).__init__()
        self.block = nn.Sequential()
        for i in range(len(R_variable['full_net'])-2):
            self.block.add_module('linear'+str(i), nn.Linear(R_variable['full_net'][i],R_variable['full_net'][i+1]))
            if R_variable['ActFuc']==0:
                self.block.add_module('relu'+str(i), nn.ReLU())
            elif R_variable['ActFuc']==1:
                self.block.add_module('tanh'+str(i), nn.Tanh())
            elif R_variable['ActFuc']==2:
                self.block.add_module('sin'+str(i), nn.sin())
            elif R_variable['ActFuc']==3:
                self.block.add_module('**50'+str(i), Act_op())
            elif R_variable['ActFuc']==4:
                self.block.add_module('sigmoid'+str(i), nn.sigmoid())
        i = len(R_variable['full_net'])-2
        self.block.add_module('linear'+str(i), nn.Linear(R_variable['full_net'][i],R_variable['full_net'][i+1]))
    def forward(self, x):
        out = self.block(x)
        return out

class Model():
    def __init__(self):

        y_train = net_(torch.FloatTensor(train_inputs).to(device))
        loss_train = float(distance(y_train, torch.FloatTensor(R_variable['y_true_train'])).cpu())
        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        loss_test = float(distance(y_test, torch.FloatTensor(R_variable['y_true_test'])).cpu())

        X = torch.tensor(train_inputs, requires_grad=True, dtype= torch.float32).to(device)
        X_b = torch.tensor(train_inputs_b, requires_grad=False, dtype= torch.float32).to(device)
        y_train = net_(X)
        y_train_b = net_(X_b)
        dx = torch.autograd.grad(outputs=y_train,inputs=X,grad_outputs=torch.ones_like(X)/2, retain_graph=True)[0]
        loss = float(criterion(x = X, y=y_train, dx = dx, x_=y_train_b, x_b = X_b).cpu())

        nametmp = '%smodel/'%(FolderName)
        mkdir(nametmp)
        torch.save(net_.state_dict(),"%smodel.ckpt"%(nametmp))

        R_variable['y_train'] = y_train.cpu().detach().numpy()
        R_variable['y_test'] = y_test.cpu().detach().numpy()
        R_variable['loss_train'] = [loss_train]
        R_variable['loss_test'] = [loss_test]
        R_variable['max_gap_train'] = [np.max(np.abs(R_variable['y_train']-R_variable['y_true_train']))]
        R_variable['max_gap_test'] = [np.max(np.abs(R_variable['y_test']-R_variable['y_true_test']))]
        R_variable['mean_gap_train'] = [np.mean(np.abs(R_variable['y_train']-R_variable['y_true_train']))]
        R_variable['mean_gap_test'] = [np.mean(np.abs(R_variable['y_test']-R_variable['y_true_test']))]
        R_variable['loss'] = [loss]

    def run_onestep(self):

        y_train = net_(torch.FloatTensor(train_inputs).to(device))
        loss_train = float(distance(y_train, torch.FloatTensor(R_variable['y_true_train'])).cpu())
        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        loss_test = float(distance(y_test, torch.FloatTensor(R_variable['y_true_test'])).cpu())

        X = torch.tensor(train_inputs, requires_grad=True, dtype= torch.float32).to(device)
        X_b = torch.tensor(train_inputs_b, requires_grad=False, dtype= torch.float32).to(device)
        y_train = net_(X)
        y_train_b = net_(X_b)
        dx = torch.autograd.grad(outputs=y_train,inputs=X,grad_outputs=torch.ones_like(X)/2, retain_graph=True)[0]
        loss = float(criterion(x = X, y=y_train, dx = dx, x_=y_train_b, x_b = X_b).cpu())

        R_variable['y_train'] = y_train.cpu().detach().numpy()
        R_variable['y_test'] = y_test.cpu().detach().numpy()
        R_variable['loss_train'].append(loss_train)
        R_variable['loss_test'].append(loss_test)
        R_variable['max_gap_train'].append(np.max(np.abs(R_variable['y_train']-R_variable['y_true_train'])))
        R_variable['max_gap_test'].append(np.max(np.abs(R_variable['y_test']-R_variable['y_true_test'])))
        R_variable['mean_gap_train'].append(np.mean(np.abs(R_variable['y_train']-R_variable['y_true_train'])))
        R_variable['mean_gap_test'].append(np.mean(np.abs(R_variable['y_test']-R_variable['y_true_test'])))
        R_variable['loss'].append(loss)

        optimizer = torch.optim.Adam(net_.parameters(), lr=R_variable['learning_rate'])

        for i in range(R_variable['train_size']//R_variable['batch_size']+1):

            mask = np.random.choice(R_variable['train_size'], R_variable['batch_size'], replace=False)
            mask_b = np.random.choice(R_variable['train_size_b'], R_variable['batch_size_b'], replace=False)
            X = torch.tensor(train_inputs[mask], requires_grad=True, dtype= torch.float32).to(device)
            X_b = torch.tensor(train_inputs_b[mask_b], requires_grad=True, dtype= torch.float32).to(device)
            y_train = net_(X)
            y_train_b = net_(X_b)
            dx = torch.autograd.grad(outputs=y_train,inputs=X, grad_outputs=torch.ones_like(X)/2, retain_graph=True, create_graph=True,only_inputs=True)[0]
            loss = criterion(x = X, y=y_train, dx = dx, x_=y_train_b, x_b = X_b)

            optimizer.zero_grad() 
            loss.backward()
            optimizer.step() 

        R_variable['learning_rate'] = R_variable['learning_rate'] * (1-R_variable['learning_rateDecay'])

    def run(self, step_n=1):

        # Load paremeters
        nametmp = '%smodel/model.ckpt'%(FolderName)
        net_.load_state_dict(torch.load(nametmp))
        net_.eval()

        for epoch in range(step_n):

            self.run_onestep()

            if epoch%plot_epoch==0:

                print('time elapse: %.3f' %(time.time()-t0))
                print('model, epoch: %d, loss: %f' %(epoch,R_variable['loss'][-1]))
                print('model, epoch: %d, train distance: %f' %(epoch,R_variable['loss_train'][-1]))
                print('model, epoch: %d, test distance: %f' %(epoch,R_variable['loss_test'][-1]))
                print('max gap of train inputs: %f' %(R_variable['max_gap_train'][-1]))
                print('max gap of test inputs: %f' %(R_variable['max_gap_train'][-1]))
                print('mean gap of train inputs: %f' %(R_variable['mean_gap_train'][-1]))
                print('mean gap of test inputs: %f' %(R_variable['mean_gap_train'][-1]))

                self.plot_loss()
                self.plot_y(name='%s'%(epoch))
                self.plot_gap()
                self.save_file()

                # Save parameters
                nametmp = '%smodel/'%(FolderName)
                shutil.rmtree(nametmp)
                mkdir(nametmp)
                torch.save(net_.state_dict(), "%smodel.ckpt"%(nametmp))

    def plot_loss(self):
        
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['loss_train']
        y2 = R_variable['loss_test']
        plt.plot(y1,'ro',label='Distance Train')
        plt.plot(y2,'g*',label='Distance Test')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('Distance',fontsize=15)
        fntmp = '%sdistance'%(FolderName)
        save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)

    def plot_gap(self):
        
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['max_gap_train']
        y2 = R_variable['max_gap_test']
        y3 = R_variable['mean_gap_train']
        y4 = R_variable['mean_gap_test']
        plt.plot(y1,'r*',label='Max Train')
        plt.plot(y2,'g*',label='Max Test')
        plt.plot(y3,'ro',label='Mean Train')
        plt.plot(y4,'go',label='Mean Test')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('gap',fontsize=15)
        fntmp = '%sgap'%(FolderName)
        save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)

    def plot_y(self,name=''):
        
        if R_variable['input_dim']==2:

            X = np.arange(R_variable['x_start'], R_variable['x_end'], 0.1)
            Y = np.arange(R_variable['x_start'], R_variable['x_end'], 0.1)
            X, Y = np.meshgrid(X, Y)
            xy=np.concatenate((np.reshape(X,[-1,1]),np.reshape(Y,[-1,1])),axis=1)
            Z = np.reshape(get_f(xy),[len(X),-1])

            fp = plt.figure()
            ax = fp.gca(projection='3d')
            surf = ax.plot_surface(X, Y, Z-np.min(Z), cmap=cm.coolwarm,linewidth=0, antialiased=False)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fp.colorbar(surf, shrink=0.5, aspect=5)
            ax.scatter(train_inputs[:,0], train_inputs[:,1], R_variable['y_train']-np.min(R_variable['y_train']))
            fntmp = '%s2du%s'%(FolderName,name)
            save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)

        if R_variable['input_dim']==1:

            plt.figure()
            ax = plt.gca()
            y1 = R_variable['y_test']
            y2 = R_variable['y_true_test']
            plt.plot(test_inputs,y1,'ro',label='Test')
            plt.plot(test_inputs,y2,'b*',label='True')
            plt.title('g2u',fontsize=15)        
            plt.legend(fontsize=18) 
            fntmp = '%su_m%s'%(FolderName,name)
            save_fig(plt,fntmp,ax=ax,isax=1,iseps=0)
            
    def save_file(self):
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:
            pickle.dump(R_variable, f, protocol=4)
         
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R_variable:
            if np.size(R_variable[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R_variable[para]))
        
        text_file.close()

# All parameters
R_variable={}

R_variable['input_dim'] = 1
R_variable['output_dim'] = 1
R_variable['ActFuc'] = 1  # 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
R_variable['hidden_units'] = [100,100,100]
R_variable['full_net'] = [R_variable['input_dim']] + R_variable['hidden_units'] + [R_variable['output_dim']]

R_variable['learning_rate'] = 1e-5
R_variable['learning_rateDecay'] = 2e-6

R_variable['beta'] = 1000

R_variable['astddev'] = 2/np.square(R_variable['input_dim']+R_variable['output_dim'])
R_variable['bstddev'] = 2/(R_variable['input_dim']+R_variable['output_dim'])

plot_epoch = 500
R_variable['train_size'] = 1000; 
R_variable['batch_size'] = R_variable['train_size'] 
R_variable['test_size'] = R_variable['train_size'] 

R_variable['train_size_b'] = 100; 
R_variable['batch_size_b'] = R_variable['train_size_b'] 
R_variable['test_size_b'] = R_variable['train_size_b'] 

R_variable['x_start'] = -np.pi/2
R_variable['x_end'] = np.pi/2

if R_variable['input_dim']==1:
    R_variable['test_inputs'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                      endpoint=True),[R_variable['test_size'],1])
    R_variable['train_inputs'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                      endpoint=True),[R_variable['train_size'],1])
    R_variable['test_inputs_b'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size_b'],
                                                      endpoint=True),[R_variable['test_size_b'],1])
    R_variable['train_inputs_b'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size_b'],
                                                      endpoint=True),[R_variable['train_size_b'],1])
    
else:
    R_variable['test_inputs'] = np.random.rand(R_variable['test_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['train_inputs'] = np.random.rand(R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['test_inputs_b'] = np.random.rand(R_variable['test_size_b'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['train_inputs_b'] = np.random.rand(R_variable['train_size_b'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']

mask = np.random.choice(R_variable['input_dim']*2, R_variable['test_size_b'], replace=True)
for i in range(R_variable['test_size_b']):
    if mask[i]%2==0:
        R_variable['test_inputs_b'][i][mask[i]//2] = R_variable['x_start']
    else:
        R_variable['test_inputs_b'][i][mask[i]//2] = R_variable['x_end']

mask = np.random.choice(R_variable['input_dim']*2, R_variable['train_size_b'], replace=True)
for i in range(R_variable['train_size_b']):
    if mask[i]%2==0:
        R_variable['train_inputs_b'][i][mask[i]//2] = R_variable['x_start']
    else:
        R_variable['train_inputs_b'][i][mask[i]//2] = R_variable['x_end']

def get_f(xs):
    tmp = 0
    for ii in range(R_variable['input_dim']):
        tmp += np.cos(8*xs[:,ii:ii+1]) 
    return tmp

def get_g(xs):
    tmp = 0
    for ii in range(R_variable['input_dim']):
        tmp += 64*np.cos(8*xs[:,ii:ii+1]) 
    return tmp

def get_f_t(xs):
    return torch.cos(8*xs).sum(1,keepdim=True) 

def get_g_t(xs):
    return 84*torch.cos(8*xs).sum(1,keepdim=True)

# delta u = -g
# u on edge = f

train_inputs=R_variable['train_inputs']
test_inputs=R_variable['test_inputs']
train_inputs_b = R_variable['train_inputs_b']
test_inputs_b = R_variable['test_inputs_b']
R_variable['y_true_train']=get_f(train_inputs)
R_variable['y_true_test']= get_f(test_inputs)
R_variable['y_true_train_b'] = get_f(train_inputs_b)
R_variable['y_true_test_b'] = get_f(test_inputs_b)

# Make a folder to save all output
BaseDir = 'Laplace-Ritz/'
subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
FolderName = '%s%s/'%(BaseDir,subFolderName)
mkdir(BaseDir)
mkdir(FolderName)
mkdir('%smodel/'%(FolderName))
print(subFolderName)

if not platform.system()=='Windows':
    shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, dx, x_, x_b): 
        return torch.mean(0.5*torch.pow(dx, 2).sum(1,keepdim=True)- get_g_t(x) * y) + R_variable['beta'] * torch.mean(torch.pow(x_-get_f_t(x_b), 2))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

t0 = time.time()
net_ = Network().to(device)
net_.apply(weights_init)
print(net_)

criterion = Loss().to(device)
distance = nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(net_.parameters(), lr=R_variable['learning_rate'])

model = Model()
model.run(10000)




