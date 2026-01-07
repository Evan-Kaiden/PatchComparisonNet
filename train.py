import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import SupervisedContrastiveLoss

import time

contrastive_loss = SupervisedContrastiveLoss()

def train_one_epoch(epoch : int, model : nn.Module, trainloader : DataLoader, optimizer : Optimizer, criterion, scheduler, device=None):
    model.train()
    total = len(trainloader)
    # pbar = tqdm(total=total, desc=f"Train Epoch {epoch}", leave=False)
    total_loss = 0.0
    # with pbar:
    for images,targets in trainloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
            
        logits, targets, selection_pen = model(images, targets)   

        patch_embeds, _, _ = model.encode_patches(images)
        patch_embeds = F.normalize(patch_embeds, dim=1)

        ce_loss = criterion(logits, targets)

        if model.base_train:
            contrast_loss = contrastive_loss(patch_embeds, targets)
            loss = ce_loss + 0.25 * contrast_loss
        elif not model.base_train:
            sel_loss = (selection_pen * torch.log(selection_pen + 1e-8)).sum(dim=1).mean()
            loss = ce_loss + 0.1 * sel_loss

        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # pbar.update(1)
        if scheduler is not None:
            scheduler.step(total_loss / total)


def test(epoch: int, model : nn.Module, testloader : DataLoader, memloader : DataLoader, criterion, device=None):
    model.eval()
    
    correct_total = 0
    loss_total = 0
    total = 0

    with torch.no_grad():
        mem_iter = iter(memloader)

        total = 0
        # pbar = tqdm(total=len(testloader), desc="Testing", leave=False)

        # with pbar:
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

            # pbar.update(1)

    acc = correct_total / total
    loss = loss_total / total
    print(f"Epoch {epoch} | Loss {loss:.3f} | Accuracy {acc:.3f} ({correct_total}/{total})")
    return loss, acc

def train(epochs : int, model : nn.Module, trainloader : DataLoader, testloader: DataLoader, memloader: DataLoader, optimizer : Optimizer, criterion, scheduler, checkpoint, config, start_epoch=0, device=None):  
    mode = "base" if model.base_train else "selector"

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_one_epoch(epoch, model, trainloader, optimizer, criterion, scheduler, device)
        loss, acc = test(epoch, model, testloader, memloader, criterion, device)
        end_time = time.time()

        epoch_time = int(end_time - start_time)
        print('Epoch took {} seconds.'.format(epoch_time))
        
        state_dict = scheduler.state_dict() if scheduler is not None else None
        state = {
                "last_mode": mode,
                
                "base_epoch": epoch + 1 if mode == "base" else checkpoint.get("base_epoch"),
                "selector_epoch": epoch + 1 if mode == "selector" else checkpoint.get("selector_epoch"),
                
                "base_test_loss": loss if mode == "base" else checkpoint.get("base_test_loss"),
                "selector_test_loss" : loss if mode == "selector" else checkpoint.get("selector_test_loss"),
                
                "base_test_acc" : acc * 100 if mode == "base" else checkpoint.get("base_test_acc"),
                "selector_test_acc" : acc * 100 if mode == "selector" else checkpoint.get("selector_test_acc"),

                "base_model_state": model.state_dict() if mode == "base" else checkpoint.get("base_model_state"),
                "selector_model_state": model.state_dict() if mode == "selector" else checkpoint.get("selector_model_state"),

                
                "base_optimizer_state": optimizer.state_dict() if mode == "base" else checkpoint.get("base_optimizer_state"),
                "selector_optimizer_state": optimizer.state_dict() if mode == "selector" else checkpoint.get("selector_optimizer_state"),

                "base_scheduler_state": state_dict if mode == "base" else checkpoint.get("base_scheduler_state"),
                "selector_scheduler_state": state_dict if mode == "selector" else checkpoint.get("selector_scheduler_state"),
                
                "config": config,
                }
        torch.save(state, os.path.join(os.path.curdir, config["run_dir"], f"state.pth"))
        checkpoint = state