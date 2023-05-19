"""self-written utils"""
import torch
from torch import nn
import numpy as np
import datetime
from torchsummary import summary

def get_batch_data(x_data, y_data, batch_size) -> list:
    assert len(x_data) == len(y_data), "size doesn't match"
    res_iter = []
    for i in range(len(x_data)):
        if i % batch_size == 0:
            res_iter.append([[],[]]) # [[x],[y]]
        res_iter[-1][0].append(x_data[i])
        res_iter[-1][1].append(y_data[i])
    for i in range(len(res_iter)):
        res_iter[i][0] = np.array(res_iter[i][0])
        res_iter[i][1] = np.array(res_iter[i][1])
    return res_iter
                   
filepath = ''
def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(filepath, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

                   
def evaluate(data: torch.Tensor, batch_size=10, criterion=None):
    model.eval()
    total_loss = 0
    avg_loss = []
    if criterion == None:
        criterion = nn.MSELoss()
    val_iter = get_batch_data(data[0], data[1], batch_size)
    val_iter = torch.tensor(val_iter)
    
    with torch.no_grad():
        for x, y in val_iter:
            # x_torch = torch.from_numpy(x)
            # y_torch = torch.from_numpy(y)
                   
            pred = model(x_torch)
            loss = criterion(pred, y_torch)
            
            total_loss += loss*(x_torch.shape[0])
            avg_loss.append(loss)
    return total_loss.item() / data.shape[0], avg_loss
                   

def train(model, data, criterion=None, optimizer=None, batch_size=10, epoches=10, lr=0.01, print_format='normal',
          loss_appendix=False, graph=True):
    total_loss = 0
    avg_loss = []
    if criterion == None:
        criterion = nn.MSELoss()
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_iter = get_batch_data(data[0], data[1], batch_size)
    # train_iter = np.array(train_iter, dtype = np.float32)
    
    for epoch in range(epoches):
        for batch, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x_torch = torch.from_numpy(x).to(torch.float32)
            y_torch = torch.from_numpy(y).to(torch.float32)

            pred = model(x_torch)
            loss = criterion(pred, y_torch)
            if loss_appendix:
                pass
            loss.backward()
            optimizer.step()

            total_loss += loss*x.shape[0]
            avg_loss.append(loss)
                   
            if print_format == 'time':
                logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_iter), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / x.size(0), loss, torch.exp(loss)))
            else:
                logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_iter), optimizer.param_groups[0]['lr'],
                    loss, torch.exp(loss)))
                
    name = 'I:{:3d}, H:{}, O:{:3d}, lr:{:02.2f}, epoch:{:5d}'.format(
            model.inode, model.hnode_stringformat, model.onode, optimizer.param_groups[0]['lr'], epoches)
                
    if graph:
        my_plt(avg_loss, 'loss', name)
        my_plt(avg_loss, 'loss_mean', name, mean=True)
        
    logging('\n' + str(summary(model)))
    logging('\n' + str(model))
        
    return total_loss, avg_loss
        

import matplotlib.pyplot as plt
def my_plt(ls: list, ylabel, name, line=True, mean=False):
    dtype = type(ls[0])
    if dtype is torch.Tensor:
        for i in range(len(ls)):
            ls[i] = float(ls[i].detach())
    # for i in range(len(ls)):
    #     ls[i] = float(ls[i])
    plt.subplots(figsize=(12,4))
    if mean:
        ls_sum = [0]
        for i in range(len(ls)):
            ls_sum.append(ls_sum[i] + ls[i])
        """ls_sum[1:len(ls) + 1] is the sum array"""
        ls = [ls_sum[i+1]/(i+1) for i in range(len(ls))]
    if line:
        plt.plot(np.arange(len(ls)), ls, 'r')
    else:
        plt.scatter(x=np.arange(len(ls)), y=ls, c='b')
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.xlim(-1,)
    plt.title(name.upper())
    
def line_plt(ls: list, ylabel, name):
    ls_sum = [0]
    for i in range(len(ls)):
        ls_sum.append(ls_sum[i] + ls[i])
    """ls_sum[1:len(ls) + 1] is the sum array"""
    ls = [ls_sum[i+1]/(i+1) for i in range(len(ls))]
    plt.subplots(figsize=(12,4))
    plt.plot(np.arange(len(ls)), ls, c='b')
    plt.xlabel('Iterations')
    plt.ylabel(ylabel)
    plt.xlim(-1,)
    plt.title(name.upper())
    

def log_clear():
    with open(os.path.join(filepath, 'log.txt'), 'w') as f_log:
        print('cleared.')