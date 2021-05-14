import torch
import config
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import VAE

test_loader = DataLoader(datasets.MNIST("", train=False, download=True, transform=transforms.ToTensor()),batch_size=32)
checkpoint = "model.pth.tar"
model = VAE().to(config.DEVICE)
optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE)


def load_model(checkpoint_file,model,optimizer,lr):
    print("!!!!!! Loading Model !!!!!!")
    checkpoint =torch.load(checkpoint_file,map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


load_model(checkpoint,model,optimizer,lr=config.LEARNING_RATE)

model.eval()
with torch.no_grad():
    for idx, (data,_) in enumerate(test_loader):
        data = data.to('cuda')
        out, pm, logVar = model(data)
        save_image(data,"input.png")
        save_image(out,"output.png")
        break