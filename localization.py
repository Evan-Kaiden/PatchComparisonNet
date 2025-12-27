import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse

from mapping import map_arg
from matchnet import Matcher
from samplier import extract_patches
from dataloader import testloader, memloader

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def undoNorm(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(-1,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(-1,1,1)
    return tensor * std + mean


def heatmap_from_scores(model, image, patch_size, stride):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    img_h, img_w = image.size(-2), image.size(-1)
    H = (img_h - patch_size) // stride + 1
    W = (img_w - patch_size) // stride + 1

    patch_embeds, B, Tq = model.encode_patches(image)
    patch_embeds_bt = patch_embeds.view(B, Tq, -1)
    sel_logits = model.patch_scorer(patch_embeds_bt).squeeze(-1)
    sel_logits = sel_logits.reshape(H, W).detach().cpu().numpy()

    print(sel_logits)
    print(
    patch_embeds_bt.std(dim=1).mean().item(),
    patch_embeds_bt.std(dim=2).mean().item()
    )
    print((patch_embeds[0] - patch_embeds[1]).abs().mean())
    plt.imshow(sel_logits)
    plt.show()


"""How important is patch i for classification of class c"""
def heatmap_from_patches(model, image, memory, memory_cls, patch_size, stride, k=10, wrt=None):

    img_h, img_w = image.size(-2), image.size(-1)
    H = (img_h - patch_size) // stride + 1
    W = (img_w - patch_size) // stride + 1

    device = next(model.parameters()).device

    image = image.to(device)
    memory = memory.to(device)
    l = len(memory) // 4

    memory_batched = [memory[0:l, :, :, :], memory[l:2*l, :, :, :], memory[2*l:3*l, :, :, :], memory[3*l:, :, :, :]]
    memory_cls_batched = [memory_cls[0:l], memory_cls[l:2*l], memory_cls[2*l:3*l], memory_cls[3*l:]]

    if wrt is None:
        logits = torch.zeros((1, 10), device="mps")
        for m, m_cls in zip(memory_batched, memory_cls_batched):
            logits += model.predict(image, m, m_cls)
        
        pred = torch.argmax(logits, dim=1).squeeze(0)
        filtered_memory = memory[memory_cls == pred]
            
    else:
        logits = torch.zeros((1, 10), device="mps")
        for m, m_cls in zip(memory_batched, memory_cls_batched):
            logits += model.predict(image, m, m_cls)
        
        pred = torch.argmax(logits, dim=1).squeeze(0)
        filtered_memory = memory[memory_cls == wrt]

    img_patches = model.extractor(image)
    mem_patches = model.extractor(filtered_memory)
    B, Tq, C, ph, pw = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches_flat = img_patches.reshape(Tq, C, ph, pw).contiguous()
    mem_patches_flat = mem_patches.reshape(M * Tm, C, ph, pw).contiguous()

    img_embeds = model.encoder(img_patches_flat)
    mem_embeds = model.encoder(mem_patches_flat)

    sim_selected = (img_embeds @ mem_embeds.t()) / model.temperature
    sim_selected = torch.sum(sim_selected, dim=1)
    grid = sim_selected.reshape(H, W).detach().cpu().numpy()
    
    img = undoNorm(image).detach().cpu().squeeze(0).permute(1,2,0).numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.axis('off')
    im = ax2.imshow(grid, cmap='viridis', interpolation='bicubic')
    ax2.axis('off')

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)


    plt.tight_layout()

    if wrt == pred:
        plt.title("Predicted")
        plt.show(block=False)
    elif wrt !=9:
        plt.show(block=False)
    
    else:plt.show()
    

def undoNorm(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(-1,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(-1,1,1)
    return tensor * std + mean

def extractor(img):
    return extract_patches(img, kernel_size=kernel, stride=stride)


parser = argparse.ArgumentParser()

parser.add_argument("--run_dir", type=str)

args = parser.parse_args()

# =========== Load in Checkpoint ===========

checkpoint = torch.load(os.path.join(args.run_dir, "state.pth"), map_location="cpu")

config = checkpoint["config"]

backbone = map_arg[config["backbone"]]

kernel = config["patch_size"]
stride = config["stride"]

# =========== Load in Model Weights ===========

model = Matcher(10, extractor, backbone)
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.to("mps")

# =========== Select Image ===========

memory, cls = next(iter(memloader))

image, label = next(iter(testloader))
n = torch.randint(low=0, high=30, size=(1,)).item()
image = image[n:n+1, :, :,]

# =========== Generate Visual ===========
for i in range(10):
    heatmap_from_patches(model, image.to("mps"), memory.to("mps"), cls.to("mps"), config["patch_size"], config["stride"], wrt=i)
