# https://github.com/valeoai/FOUND

import torch

from train import train
from matchnet import Matcher
from samplier import extract_patches
from sim_classifier import sim_classify
from dataloader import trainloader, testloader, memloader


def main():

    def extractor(img):
        return extract_patches(img, kernel_size=18, stride=3)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    print("device: ", device)

    m = Matcher(10, extractor).to(device)
    opt = torch.optim.Adam(m.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=0.1, patience=5, verbose=True
    )
    
    train(epochs=100, model=m, trainloader=trainloader, 
          testloader=testloader, memloader=memloader, optimizer=opt, 
          criterion=criterion, scheduler=scheduler, device=device
          )
    
    sim_classify(m.encoder, testloader, memloader, extractor, device=device)

    torch.save(m.state_dict(), "model.pth")

if __name__ == "__main__":
    main()
