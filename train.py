import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm
from losses import SupervisedContrastiveLoss

contrastive_loss = SupervisedContrastiveLoss()

def train_one_epoch(epoch : int, model : nn.Module, trainloader : DataLoader, optimizer : Optimizer, criterion, scheduler, device=None):
    model.train()
    total = len(trainloader)
    pbar = tqdm(total=total, desc=f"Train Epoch {epoch}", leave=False)
    total_loss = 0.0
    with pbar:
        for images,targets in trainloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
                
            logits, targets = model(images, targets)   
            ce_loss = criterion(logits, targets)

            patch_embeds, B, Tq = model.encode_patches(images)
            patch_embeds = F.normalize(patch_embeds, dim=1)

            contrast_loss = contrastive_loss(patch_embeds, targets)

            loss = ce_loss  + 0.25 * contrast_loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()

            pbar.update(1)
        scheduler.step(total_loss / total)


def test(epoch: int, model : nn.Module, testloader : DataLoader, memloader : DataLoader, criterion, device=None):
    model.eval()
    
    correct_total = 0
    loss_total = 0
    total = 0

    with torch.no_grad():
        mem_iter = iter(memloader)

        total = len(testloader)
        pbar = tqdm(total=total, desc="Testing", leave=False)

        with pbar:
            for images, targets in testloader:
                images, targets = images.to(device), targets.to(device)
            
                try:
                    mem_images, cls = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(memloader)
                    mem_images, cls = next(mem_iter)

                mem_images, cls = mem_images.to(device), cls.to(device)
        

                logits = model.predict(images, mem_images, cls)

                loss = criterion(logits, targets)
                pred_labels = logits.argmax(dim=1)

                correct_total += (pred_labels == targets).sum().item()
                loss_total += loss.item() * images.size(0)
                total += images.size(0)

                pbar.update(1)

    print(f"Epoch {epoch} | Loss {loss_total / total :.3f} | Accuracy {correct_total / total :.3f} ({correct_total}/{total})")

def train(epochs : int, model : nn.Module, trainloader : DataLoader, testloader: DataLoader, memloader: DataLoader, optimizer : Optimizer, criterion, scheduler, device=None):
    for epoch in range(epochs):
        train_one_epoch(epoch, model, trainloader, optimizer, criterion, scheduler, device)
        test(epoch, model, testloader, memloader, criterion, device)