"""
Dependencies:
torch: 1.4(Stable)
numpy
matplotlib
"""
import torch
import numpy as np 
import matplotlib.pyplot as plt

# numpy.arange() 給定範圍建立陣列
print('\nnumpy.arange():',np.arange(3,7)) #[3 4 5 6]


# 建立內容為1的tensor
x_ones = torch.ones(2, 3, 4) 
"""
tensor([[[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]],

        [[1., 1., 1., 1.],
         [1., 1., 1., 1.],
         [1., 1., 1., 1.]]])
"""

print('\nx_ones:',x_ones)
print('\nx_ones.size():',x_ones.size())         # torch.size() 查看tensor大小
print('\nx_ones.size(0):',x_ones.size(0))       # 第一維度
print('\nx_ones.size(1):',x_ones.size(1))       # 第二維度
print('\nx_ones.size(2):',x_ones.size(2))       # 第三維度


x_tensor = torch.tensor([1, 2, 3, 4])           # tensor([1, 2, 3,4]) 一維tensor(張量)
x_arange2D = torch.arange(1,7).view(1,2,3)      # tensor([[[1, 2, 3], [4, 5, 6]]])二維tensor
print('\nx_tensor:\n',x_tensor)
print('\nx_arage2D:\n',x_arange2D)

# 減少tensor在dim的維度(預設為1)
print('\ntorch.squeeze(x_arage2D):\n',torch.squeeze(x_arange2D,dim=0))  # x_arange2D.squeeze(0),輸出 tensor([[1, 2, 3],[4, 5, 6]])
print(torch.squeeze(x_arange2D,dim=0).size())

# 增加tensor在dim的維度
print('\ntorch.unsqueeze(x_tensor):\n',torch.unsqueeze(x_tensor,dim=0)) # x_tensor.unsqueeze(0),輸出 tensor([[1, 2, 3,4]])
print(torch.unsqueeze(x_tensor,dim=0).size())

# 給予兩數，以兩數相差與steps-1求商為間隔執行steps步 (3, 4.75, 6.5, 8.25, 10)
x_linspace = torch.linspace(start = 3, end = 10, steps=5)
print('\ntorch.linspace(start,end,steps):',x_linspace)

# 固定隨機數
torch.manual_seed(1)
print('\ntorch.manual_seed',torch.rand(2))

