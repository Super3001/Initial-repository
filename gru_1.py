import torch
import torch.nn as nn
import torch.nn.functional as F


flog = open('log.txt', 'a')

def logging(s):
    flog.write(s + '\n')
    print(s)

class GRU_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 num_classes, dropout):
        super(GRU_1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(1000, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim*2, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.num_classes = num_classes
        # self.softmax = nn.Softmax(dim=1) # 添加softmax激活函数?

    def forward(self, x):
        bat_size, seq_len = x.size()

        # Embedding
        x = self.embedding(x)
        pos = torch.arange(seq_len).repeat(bat_size, 1).to(x.device)
        x = x + self.pos_embedding(pos)

        # Transformer Encoder
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        # out = self.softmax(out)
        return out

from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        input_ids = self.ds[idx]['input_ids'].to(torch.int32)
        labels = self.ds[idx]['labels'].to(torch.float32)
        return input_ids, labels


from datasets import load_from_disk
ds = load_from_disk('./dataset')
train_dataset = MyDataset(ds['train'])

# dx = torch.cat([
#         torch.cat([
#             torch.randint(0, 20, (50, 32)).to(torch.long),
#             torch.randint(20, 40, (50, 32)).to(torch.long)
#         ], dim=1),
#         torch.cat([
#             torch.randint(10, 30, (50, 32)).to(torch.long),
#             torch.randint(30, 50, (50, 32)).to(torch.long)
#         ], dim=1)],
#         dim=0)
# from dummyDataset import DummyDataset

# dy = torch.cat([
#     torch.cat([torch.zeros(50, 27).to(torch.float32), torch.ones(50, 1).to(torch.float32)], dim=1),
#     torch.cat([torch.ones(50, 1).to(torch.float32), torch.zeros(50, 27).to(torch.float32)], dim=1)],
#     dim=0)
# train_dataset = DummyDataset(dx, dy)

# 使用DataLoader类生成可迭代的数据加载器
batch_size = 128  # [1, 16, 32 , 128]
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # using kfolds later
emotions = [
 'admiration',
 'amusement',
 'anger',
 'annoyance',
 'approval',
 'caring',
 'confusion',
 'curiosity',
 'desire',
 'disappointment',
 'disapproval',
 'disgust',
 'embarrassment',
 'excitement',
 'fear',
 'gratitude',
 'grief',
 'joy',
 'love',
 'nervousness',
 'optimism',
 'pride',
 'realization',
 'relief',
 'remorse',
 'sadness',
 'surprise',
 'neutral']

from torch.utils.data import SubsetRandomSampler, Subset
from sklearn.model_selection import KFold
import numpy

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    print(device)
    vocab_size = 30522
    # vocab_size = 1000
    hidden_dim = 80  # # mbed_dim
    num_layers = 3  # [1, 10]
    dropout = 0.1  # [0.1, 0.2]
    num_classes = 28
    model = GRU_1(input_dim=vocab_size,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               num_classes=num_classes,
                               dropout=dropout)



    # 参数初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)  # Xavier初始化
            m.bias.data.fill_(0.01)
        elif type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)  # 正交初始化
                elif 'bias' in name:
                    param.data.fill_(0.01)


    # model.apply(init_weights)
    check_ori = torch.load('model_gru1_kfold_3.ckpt')
    model.load_state_dict(check_ori)
    print(model)

    cnt = 0
    # 注册一个回调函数，用于查看梯度
    def print_grad(grad):
        global cnt
        print(cnt, 'Gradient:', torch.max(grad), torch.min(grad))
        print(torch.linalg.norm(grad))
        cnt += 1


    # 在网络的第一层全连接层上注册回调函数
    model.gru.weight_hh_l0.register_hook(print_grad)
    # model.fc.weight.register_hook(print_grad)

    # print(model)
    input_size = 64
    import torch.onnx

    #Function to Convert to ONNX
    def Convert_ONNX():

        # set the model to inference mode
        model.eval()

        # Let's create a dummy input tensor
        dummy_input = torch.randint(0,100,(1, input_size)).to(torch.long)

        # Export the model
        torch.onnx.export(model,     # model being run
                          dummy_input,    # model input (or a tuple for multiple inputs)
                          "gru-1.onnx",    # where to save the model
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names = ['modelInput'],  # the model's input names
                          output_names = ['modelOutput'], # the model's output names
                          dynamic_axes={'modelInput' : {0 : 'batch_size'},  # variable length axes
                                        'modelOutput' : {0 : 'batch_size'}})
        print(" ")
        print('Model has been converted to ONNX')

    # Convert_ONNX()
    # exit(0)

    import time
    str_now = time.strftime('%Y%m%d %H %M',time.localtime())
    
    # 训练模型
    num_epochs = 1
    # thresh = 100
    thresh = len(train_dataset)
    # arr = numpy.arange(0, thresh) # for kfold.split()
    arr = list(range(thresh))
    K = 11
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters()) # defualt: lr=1e-3
    # lr [0.2, 0.05, 0.01, 0.001, 1e-4, 1e-5, 2e-4]
    model = model.to(device)
    logging('')
    logging(str_now)
    logging('training start...')
    
    floss = open(f'.\\loss\\loss {str_now}.txt','w')
    floss.write('| kind | epoch | fold | iteration | loss |\n')
    floss.write('|:--:|:--:|:--:|:--:|:--:|\n')
    # floss.close()
    
    for epoch in range(num_epochs):
        start = time.time()
        kfold = KFold(n_splits=K, shuffle=True)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(arr)):
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
            print(type(train_idx[0]))
            # print(f'Fold {fold+1}')
            # train_sampler = SubsetRandomSampler(train_idx)
            # val_sampler = SubsetRandomSampler(val_idx)
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size)
            val_loader = DataLoader(train_subset, batch_size=batch_size)

            loss_sum = 0
            for i, (inputs, labels) in enumerate(train_loader):
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

                if (i+1) % 20 == 0:
                    print('Epoch [{}/{}], Fold [{}/{}], Step [{}/{}], Loss: {:.4f}, avgLoss: {:.4f}'
                        .format(epoch+1, num_epochs, fold, K, i+1, len(train_loader),
                                loss.item(), loss_sum / (i+1)))
                    floss.write(f'| train | {epoch} | {fold} | {i} | {loss_sum/(i+1)} | \n')

                    # 打印模型参数
                    # for name, param in model.named_parameters():
                    #     print(name)
                    #     print(param.data)
                    # ls_param = list(model.named_parameters())
                    # print(ls_param[-2][0])
                    # print(ls_param[-2][1].data)
                    
            total_val_loss = 0
            for j, (inputs, labels) in enumerate(val_loader):
                model.eval()
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    pred = model(inputs)
                    loss = criterion(pred, labels)
                    total_val_loss += loss
                    
            print('Epoch [{}/{}], Fold [{}/{}],train_Loss: {:.4f}, val_Loss: {:.4f}'
                        .format(epoch+1, num_epochs, fold, K, 
                                loss_sum / len(train_loader), total_val_loss / len(val_loader)))
            floss.write(f'| train | {epoch} | {fold} | --- | {loss_sum/len(train_loader)} | \n')
            floss.write(f'| val | {epoch} | {fold} | --- | {total_val_loss/len(val_loader)} | \n')
            
        end = time.time()
        logging(f'epoch{epoch}: using {end-start:.2f}s')
        
        
        
    # 测试模型
    examine_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    corr = 0
    for i, (inputs, labels) in enumerate(examine_dataloader):
        with torch.no_grad():
            outputs = model(inputs)
            if torch.argmax(outputs) == torch.argmax(labels):
                corr += 1
    logging(f'accuracy: {corr / len(examine_dataloader):.3f}')

    # 保存模型
    torch.save(model.state_dict(), r'model_gru1_kfold_5.ckpt')
    floss.close()
    flog.close()

