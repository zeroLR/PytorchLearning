"""
Dependencies:
torch: 1.4
CUDA: 10.1
python: 3.7.6
torchvision
"""
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime

EPOCH = 2               # 進行2批運算
BATCH_SIZE = 50         # 50張為一組
LR = 0.001              # 學習速度
DOWNLOAD_MNIST = True   # 是否下載資料集         

# 使用Fashion-MNIST訓練集
train_data = torchvision.datasets.FashionMNIST(root='./FashionMnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)    # 訓練資料
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
print('\ntrain_data size: ', train_data.data.size())

test_data = torchvision.datasets.FashionMNIST(root='./FashionMnist/', train=False)  # 測試資料
print('\ntest_data size: ', test_data.data.size())

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.     # Tensor on GPU
test_y = test_data.targets[:2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),  # 輸入 1x28x28, 輸出 16x28x28, 以5x5像素的大小掃描, 1像素為間隔, 以(kernel_size-stride)/2的結果在圖片周圍新增0的像素
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)                                         # ReLU激勵函數, 池化後 16x14x14
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)                         # 輸入 16x14x14, 輸出 32x14x14, 池化後 32x7x7
        self.out = nn.Linear(32 * 7 * 7, 10)                                                                        # 輸出層, 資料集有10個分類(0~9)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (batch, 32, 7, 7) 這邊會加入batch(BATCH_SIZE)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7) 保留batch,將後面數據轉成同一維度
        output = self.out(x)
        return output

cnn = CNN()

cnn.cuda()  # 將 model 所有 parameters 和 buffers 轉成GPU可執行

optimizer = optim.Adam(cnn.parameters(), lr=LR)     # 設定優化器, 部分演算法使用 momentum動量來加速收斂
loss_func = nn.CrossEntropyLoss()                   # 損失函數

loss_print = 0      # 儲存loss值
accuracy_print = 0  # 儲存準確值

# 批量訓練
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        b_x = x.cuda()      # Tensor on GPU, size (50, 1, 28, 28)
        b_y = y.cuda()      # Tensor on GPU, size (50)
        
        output = cnn(b_x)
        loss = loss_func(output, b_y)   # 計算loss值
        optimizer.zero_grad()           # 梯度歸零
        loss.backward()                 # 反向傳遞
        optimizer.step()                # 執行優化器

        # 每50步顯示一次結果
        if step % 50 == 0:
            time = datetime.now()
            test_output = cnn(test_x)

            pred_y = torch.max(test_output, 1)[1].cuda().data   # 將計算移至GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy, ' | time: ',time)
            loss_print = loss.data.cpu().numpy()
            accuracy_print = accuracy

test_output = cnn(test_x[:20])  # 輸出20張圖

pred_y = torch.max(test_output, 1)[1].cuda().data # 將計算移至GPU

print(pred_y, 'prediction number')
print(test_y[:20], 'real number')

# 顯示預測和測試圖
f, axarr = plt.subplots(2,20)
for i in range(20):
    #  顯示預測圖
    if pred_y[i] == test_y[i]:
        axarr[0,i].imshow(train_data.data[pred_y[i]].numpy(), cmap='gray')
        axarr[0,i].text(9,-8,'%i' % pred_y[i],fontdict={'size': 10, 'color':  'blue'})
        axarr[0,i].axis('off')
        axarr[1,i].axis('off')
    else:
        axarr[0,i].imshow(train_data.data[pred_y[i]].numpy(), cmap='rainbow')
        axarr[0,i].text(9,-8,'%i' % pred_y[i],fontdict={'size': 10, 'color':  'red'})
        axarr[0,i].axis('off')
        axarr[1,i].axis('off')
    
    axarr[1,i].imshow(train_data.data[test_y[i]].numpy(), cmap='gray')  # 顯示測試圖
    axarr[1,i].text(8,-8,'%i' % test_y[i],fontdict={'size': 10, 'color':  'black'})
plt.text(-500,120,'loss: %.4f' % loss_print,fontdict={'size': 20, 'color':  'red'})
plt.text(-200,120,'accuracy: %.2f' % accuracy_print,fontdict={'size': 20, 'color':  'blue'})

plt.show()

# 保存Model
path = "torch_ModelSave"
def Model_Save():
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(cnn, path + "\\" +'FashionMnist_CNN_GPU.pkl')
    torch.save(cnn.state_dict(), path + "\\" +'FashionMnist_CNN_GPU_params.pkl')

Model_Save()