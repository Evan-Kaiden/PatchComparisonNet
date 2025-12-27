from torchvision import datasets as dsets, transforms
from torch.utils.data import DataLoader, Subset
import torch


# cifar10_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),   
#     transforms.ToTensor(),     
#     transforms.Normalize(
#         mean=[0.4914, 0.4822, 0.4465],
#         std=[0.2470, 0.2435, 0.2616]
#     ),
# ])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    ),
])

train_mem_data = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform)

N = len(train_mem_data)

indicies = torch.randperm(N)
mem_indices = indicies[:2000]
train_indices = indicies[2000:]

mem_data = Subset(train_mem_data, mem_indices)
train_data = Subset(train_mem_data, train_indices)

trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
memloader = DataLoader(mem_data, batch_size=100, shuffle=True)
testloader = DataLoader(test_data, batch_size=16, shuffle=False)