import torch 

x = torch.ones(10, 1)
print(x.squeeze(1).shape)