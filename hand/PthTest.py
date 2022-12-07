import torch


save_dict =torch.load("./result/DQN.pth")

print(save_dict['i'])
print(save_dict['q_net'])