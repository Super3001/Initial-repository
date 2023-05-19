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

                   
def evaluate(model, data: torch.Tensor, batch_size=10, criterion=None,
             y_label = True):
    model.eval()
    total_loss = 0
    avg_loss = []
    if criterion == None:
        criterion = nn.MSELoss()
    val_iter = get_batch_data(data[0], data[1], batch_size)
    val_iter = torch.tensor(val_iter)
    
    with torch.no_grad():
        for x, y in val_iter:
            x_torch = torch.from_numpy(x)
            y_torch = torch.from_numpy(y)
            
            if y_label:
                y_torch_n = torch.zeros(pred.shape)
                for i in range(y_torch_n.shape[0]):
                    y_torch_n[i][y_torch[i]] = 1
                y_torch = y_torch_n
                   
            pred = model(x_torch)
            loss = criterion(pred, y_torch)
            
            total_loss += loss*(x_torch.shape[0])
            avg_loss.append(loss)
    return total_loss.item() / data.shape[0], avg_loss
                   
"""not using Dataloader"""
def train(model, # nn.Modules derived manual ML model
          data, # [X, y]
          criterion=None, optimizer=None, batch_size=10, epoches=10, lr=0.01, print_format='normal',
          loss_appendix=False, # if add para_cal to loss
          graph=True, 
          y_label=True, # if y is label or one-hot coding
          model_desp=False, # if model has .inode / .onode / .hnode_stringformat attrs
          val=False,
          val_data=None
          ):
    total_loss = 0
    avg_loss = []
    if criterion == None:
        criterion = nn.MSELoss()
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_iter = get_batch_data(data[0], data[1], batch_size)
    # train_iter = np.array(train_iter, dtype = np.float32)
    
    if val:
        total_loss_val = 0
        avg_loss_val = []
        val_iter = get_batch_data(val_data[0], val_data[1], 1)
        val_iter = torch.tensor(val_iter)
    
    for epoch in range(epoches):
        for batch, (x, y) in enumerate(train_iter):
            optimizer.zero_grad()
            x_torch = torch.from_numpy(x).to(torch.float32).requires_grad_()
            if y_label:
                y_torch = torch.from_numpy(y).to(torch.long)
            else:
                y_torch = torch.from_numpy(y).to(torch.float32)
            
            pred = model(x_torch)
            # pred = torch.argmax(model(x_torch), dim=1)
            if y_label:
                y_torch_n = torch.zeros(pred.shape)
                for i in range(y_torch_n.shape[0]):
                    y_torch_n[i][y_torch[i]] = 1
                y_torch = y_torch_n
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
                
        if val:
            model.eval()
            with torch.no_grad():
                avg_loss_val.append(0)
                for x, y in val_iter:
                    x_torch = torch.from_numpy(x).to(torch.float32)
                    if y_label:
                        y_torch = torch.from_numpy(y).to(torch.long)
                    else:
                        y_torch = torch.from_numpy(y).to(torch.float32)
                    
                    pred = model(x_torch)
                    # pred = torch.argmax(model(x_torch), dim=1)
                    if y_label:
                        y_torch_n = torch.zeros(pred.shape)
                        for i in range(y_torch_n.shape[0]):
                            y_torch_n[i][y_torch[i]] = 1
                        y_torch = y_torch_n
                    loss = criterion(pred, y_torch)
                    if loss_appendix:
                        pass
                    # loss.backward()
                    # optimizer.step()

                    total_loss_val += loss*x.shape[0]
                    avg_loss_val[-1] += loss
                        
                # 计算每一个epoch的平均val_loss
                loss = avg_loss_val[-1] / len(val_data)
                if print_format == 'time':
                    logging('val: | epoch {:3d} | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / x.size(0), loss, torch.exp(loss)))
                else:
                    logging('val: | epoch {:3d} | lr {:02.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, optimizer.param_groups[0]['lr'],
                        loss, torch.exp(loss)))
                
    if model_desp:
        name = 'I:{:3d}, H:{}, O:{:3d}, lr:{:02.2f}, epoch:{:5d}'.format(
            model.inode, model.hnode_stringformat, model.onode, optimizer.param_groups[0]['lr'], epoches)
    else:
        name = 'model_0'
                
    if graph:
        my_plt(avg_loss, 'loss', name)
        my_plt(avg_loss, 'loss_mean', name, mean=True)
        if val:
            my_plt(avg_loss_val, 'val_loss', name)
            my_plt(avg_loss_val, 'val_loss_mean', name, mean=True)
        
    logging('\n' + str(summary(model)))
    logging('\n' + str(model))
    
    if val:
        return (total_loss, avg_loss) , (total_loss_val, avg_loss_val)
    return total_loss, avg_loss

"""using Dataloader"""
def train_1(model, # nn.Modules derived manual ML model
          loader, # [X, y]
          criterion=None, optimizer=None, batch_size=10, epoches=10, lr=0.01, print_format='normal',
          loss_appendix=False, # if add para_cal to loss
          graph=True, 
          y_label=True, # if y is label or one-hot coding
          model_desp=False, # if has .inode / .onode / .hnode_stringformat attrs
          val=False,
          val_loader=None
          ):
    total_loss = 0
    avg_loss = []
    total_loss_val = 0
    avg_loss_val = []
    if criterion == None:
        criterion = nn.MSELoss()
    if optimizer == None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    for epoch in range(epoches):
        for batch, (x_torch, y_torch) in enumerate(loader):
            optimizer.zero_grad()
            # x_torch = torch.from_numpy(x).to(torch.float32).requires_grad_()
            
            pred = model(x_torch)
            # pred = torch.argmax(model(x_torch), dim=1)
            if y_label:
                y_torch_n = torch.zeros(pred.shape)
                for i in range(y_torch_n.shape[0]):
                    y_torch_n[i][y_torch[i]] = 1
                y_torch = y_torch_n
            loss = criterion(pred, y_torch)
            if loss_appendix:
                pass
            loss.backward()
            optimizer.step()

            total_loss += loss*x_torch.shape[0]
            avg_loss.append(loss)

            if batch % 100 == 0:   
                if print_format == 'time':
                    logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(loader), optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / x_torch.size(0), loss, torch.exp(loss)))
                else:
                    logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(loader), optimizer.param_groups[0]['lr'],
                        loss, torch.exp(loss)))
                
        if val:
            model.eval()
            with torch.no_grad():
                avg_loss_val.append(0)
                for x_torch, y_torch in val_loader:
                
                    pred = model(x_torch)
                    # pred = torch.argmax(model(x_torch), dim=1)
                    if y_label:
                        y_torch_n = torch.zeros(pred.shape)
                        for i in range(y_torch_n.shape[0]):
                            y_torch_n[i][y_torch[i]] = 1
                        y_torch = y_torch_n
                    loss = criterion(pred, y_torch)
                    if loss_appendix:
                        pass
                    # loss.backward()
                    # optimizer.step()

                    total_loss_val += loss
                    avg_loss_val[-1] += loss
                    
                loss = avg_loss_val[-1] / len(val_loader)
                # val 不按照batch计算
                if print_format == 'time':
                    logging('val: | epoch {:3d} | lr {:02.2f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / x.size(0), loss, torch.exp(loss)))
                else:
                    logging('val: | epoch {:3d} | lr {:02.2f} | '
                            'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, optimizer.param_groups[0]['lr'],
                        loss, torch.exp(loss)))
                
    if model_desp:
        name = 'I:{:3d}, H:{}, O:{:3d}, lr:{:02.2f}, epoch:{:5d}'.format(
            model.inode, model.hnode_stringformat, model.onode, optimizer.param_groups[0]['lr'], epoches)
    else:
        name = 'model_0'
                
    if graph:
        my_plt(avg_loss, 'loss', name)
        my_plt(avg_loss, 'loss_mean', name, mean=True)
        if val:
            my_plt(avg_loss_val, 'val_loss', name)
            my_plt(avg_loss_val, 'val_loss_mean', name, mean=True)
        
    logging('\n' + str(summary(model)))
    logging('\n' + str(model))
    
    if val:
        return (total_loss, avg_loss) , (total_loss_val, avg_loss_val)
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
        logging('cleared.')