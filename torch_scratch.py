import torch

#### unfold trials

x = [[[1,2,3,4], [5,6,7,8]], [[11,12,13,14], [15,16,17,18]]]
x = torch.Tensor(x)
x = torch.reshape(x, (2,1,2,4))
x.shape
x
kernel_h, kernel_w = 2, 2
step = 1
n_channels = 1

# unfold(dimension, size, step)
windows = x.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)
print(windows.shape)
len(windows)
windows[0]
y = x.unfold(dimension = 2, size=2, step=1)
y = torch.reshape(y, (2, -1, 2, 2))