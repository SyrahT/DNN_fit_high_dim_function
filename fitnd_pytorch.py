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

def mkdir(fn):#Create a directory
    if not os.path.isdir(fn):
        os.mkdir(fn)
def save_fig(pltm, fntmp,fp=0,ax=0,isax=0,iseps=0,isShowPic=0):#Save the figure
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

def weights_init(m):#Initialization weight
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, R_variable['astddev'])
        m.bias.data.normal_(0, R_variable['bstddev'])
class Act_op(nn.Module):#Custom activation function
    def __init__(self):
        super(Act_op, self).__init__()
    def forward(self, x):
        return x ** 50
class Network(nn.Module):#DNN 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
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
        loss_train = float(criterion(y_train, torch.FloatTensor(R_variable['y_true_train'])).cpu())
        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        loss_test = float(criterion(y_test, torch.FloatTensor(R_variable['y_true_test'])).cpu())

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

    def run_onestep(self):

        y_test = net_(torch.FloatTensor(test_inputs).to(device))
        loss_test = float(criterion(y_test, torch.FloatTensor(R_variable['y_true_test']).to(device)).cpu())
        y_train = net_(torch.FloatTensor(train_inputs).to(device))
        loss_train = float(criterion(y_train, torch.FloatTensor(R_variable['y_true_train']).to(device)).cpu())

        R_variable['y_train'] = y_train.cpu().detach().numpy()
        R_variable['y_test'] = y_test.cpu().detach().numpy()
        R_variable['loss_test'].append(loss_test)
        R_variable['loss_train'].append(loss_train)
        R_variable['max_gap_train'].append(np.max(np.abs(R_variable['y_train']-R_variable['y_true_train'])))
        R_variable['max_gap_test'].append(np.max(np.abs(R_variable['y_test']-R_variable['y_true_test'])))
        R_variable['mean_gap_train'].append(np.mean(np.abs(R_variable['y_train']-R_variable['y_true_train'])))
        R_variable['mean_gap_test'].append(np.mean(np.abs(R_variable['y_test']-R_variable['y_true_test'])))

        optimizer = torch.optim.Adam(net_.parameters(), lr=R_variable['learning_rate'])

        for i in range(R_variable['train_size']//R_variable['batch_size']+1):

            mask = np.random.choice(R_variable['train_size'], R_variable['batch_size'], replace=False)
            y_train = net_(torch.FloatTensor(train_inputs[mask]).to(device))
            loss = criterion(y_train, torch.FloatTensor(R_variable['y_true_train'][mask]).to(device))
            
            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step() 

        R_variable['learning_rate'] = R_variable['learning_rate'] * (1-R_variable['learning_rateDecay'])

    def run(self, step_n=1):

        nametmp = '%smodel/model.ckpt'%(FolderName)
        net_.load_state_dict(torch.load(nametmp))
        net_.eval()

        for epoch in range(step_n):

            self.run_onestep()

            if epoch%plot_epoch==0:

                print('time elapse: %.3f' %(time.time()-t0))
                print('model, epoch: %d, train loss: %f' %(epoch,R_variable['loss_train'][-1]))
                print('model, epoch: %d, test loss: %f' %(epoch,R_variable['loss_test'][-1]))
                print('max gap of train inputs: %f' %(R_variable['max_gap_train'][-1]))
                print('max gap of test inputs: %f' %(R_variable['max_gap_train'][-1]))
                print('mean gap of train inputs: %f' %(R_variable['mean_gap_train'][-1]))
                print('mean gap of test inputs: %f' %(R_variable['mean_gap_train'][-1]))

                self.plot_loss()
                self.plot_y(name='%s'%(epoch))
                self.plot_gap()
                self.save_file()

                nametmp = '%smodel/'%(FolderName)
                shutil.rmtree(nametmp)
                mkdir(nametmp)
                torch.save(net_.state_dict(), "%smodel.ckpt"%(nametmp))

    def plot_loss(self):
        
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['loss_test']
        y2 = R_variable['loss_train']
        plt.plot(y1,'ro',label='Test')
        plt.plot(y2,'g*',label='Train')
        ax.set_xscale('log')
        ax.set_yscale('log')                
        plt.legend(fontsize=18)
        plt.title('loss',fontsize=15)
        fntmp = '%sloss'%(FolderName)
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
            Z = np.reshape(get_y(xy),[len(X),-1])

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

#All parameters
R_variable={}

R_variable['input_dim'] = 2
R_variable['output_dim'] = 1
R_variable['ActFuc'] = 1  # 0: ReLU; 1: Tanh; 2:Sin; 3:x**50; 4:Sigmoid
R_variable['hidden_units'] = [200,200,200]
R_variable['full_net'] = [R_variable['input_dim']] + R_variable['hidden_units'] + [R_variable['output_dim']]

R_variable['learning_rate'] = 1e-6
R_variable['learning_rateDecay'] = 2e-7

R_variable['astddev'] = np.sqrt(1/20)# For weight
R_variable['bstddev'] = np.sqrt(1/20)# For bias

plot_epoch = 500
R_variable['train_size'] = 1000; 
R_variable['batch_size'] = R_variable['train_size'] 
R_variable['test_size'] = R_variable['train_size'] 
R_variable['x_start'] = -np.pi/2
R_variable['x_end'] = np.pi/2

if R_variable['input_dim']==1:
    R_variable['test_inputs'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                      endpoint=True),[R_variable['test_size'],1])
    R_variable['train_inputs'] = np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                      endpoint=True),[R_variable['train_size'],1])
else:
    R_variable['test_inputs'] = np.random.rand(R_variable['test_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']
    R_variable['train_inputs'] = np.random.rand(R_variable['train_size'],R_variable['input_dim'])*(R_variable['x_end']-R_variable['x_start'])+R_variable['x_start']

def get_y(xs):#Function to fit
    tmp = 0
    for ii in range(R_variable['input_dim']):
        tmp += np.cos(4*xs[:,ii:ii+1]) 
    return tmp

test_inputs = R_variable['test_inputs']
train_inputs = R_variable['train_inputs']
R_variable['y_true_test'] = get_y(test_inputs)
R_variable['y_true_train'] = get_y(train_inputs)

# make a folder to save all output
BaseDir = 'tkj/'
subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
FolderName = '%s%s/'%(BaseDir,subFolderName)
mkdir(BaseDir)
mkdir(FolderName)
mkdir('%smodel/'%(FolderName))
print(subFolderName)

if  not platform.system()=='Windows':
    shutil.copy(__file__,'%s%s'%(FolderName,os.path.basename(__file__)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

t0 = time.time()
net_ = Network().to(device)
net_.apply(weights_init)
print(net_)

criterion = nn.MSELoss(reduction='mean').to(device)
optimizer = torch.optim.Adam(net_.parameters(), lr=R_variable['learning_rate'])

model = Model()
model.run(10000)




