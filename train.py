import torch
import config
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE
from tqdm import tqdm

def save_model(model,optimizer,filename = "model.pth.tar"):
    print("!!!!!! Model Saved Successfully !!!!!!")
    checkpoints = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }
    torch.save(checkpoints,filename)

train_loader = DataLoader(datasets.MNIST(root="",train=True,download=True,transform=transforms.ToTensor()),batch_size=config.BATCH_SIZE,shuffle=True)
test_loader = DataLoader(datasets.MNIST("", train=False, download=True, transform=transforms.ToTensor()),batch_size=1)

model = VAE().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

for epoch in range(config.EPOCH):
    loop = tqdm(train_loader,leave=True)
    for idx, (data,_) in enumerate(loop):
        data = data.to(config.DEVICE)
        out, pm, logVar = model(data)
        kl_divergence = 0.5 * torch.sum(1+ logVar - pm.pow(2) - logVar.exp())
        loss = F.binary_cross_entropy(out,data,size_average=False) - kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Epoch {}: Loss {}'.format(epoch, loss))

save_model(model,optimizer,filename="model.pth.tar")