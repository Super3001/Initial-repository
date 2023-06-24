import re

fname = '.\\loss\\loss 20230623 20 22.txt'
floss = open(fname, 'r')

train_losses = []
val_losses = []
for line in floss.readlines():
    if 'train' in line:
        match = re.search(r'\d+\.\d+', line)
        if match:
            loss = eval(match.group(0))
            train_losses.append(loss)
            print(loss)
        else:
            print('no value')
    elif 'val' in line:
        match = re.search(r'\d+\.\d+', line)
        if match:
            loss = eval(match.group(0))
            val_losses.append(loss)
            print(loss)
        else:
            print('no value')
            
import matplotlib.pyplot as plt

print(len(train_losses))
print(len(val_losses))

plt.figure(0)
# plt.subplot(211)
plt.plot(list(range(len(train_losses))), train_losses)
plt.savefig(fname.replace('.txt', ' train.png'))
plt.show()
plt.figure(1)
# plt.subplot(212)
plt.plot(list(range(len(val_losses))), val_losses)
plt.savefig(fname.replace('.txt', ' val.png'))
plt.show()