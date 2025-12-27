import torch
import torch.nn.functional as F

import os
import argparse

from mapping import map_arg
from matchnet import Matcher
from samplier import extract_patches
from dataloader import testloader, memloader

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def top_k_contributors(
    image: torch.Tensor,
    model,
    memory: torch.Tensor,
    memory_cls: torch.Tensor,
    k: int = 10,
    cls_filter: str = "pred",      # "pred" or "true"
    true_label: int | None = None,
):
    """
    Returns relations only where the *query patch* is among the selector's chosen patches.
    Assumes B==1 for image.
    """
    assert image.ndim == 4 and image.size(0) == 1, "image must have shape [1, C, H, W]"

    device = next(model.parameters()).device
    image = image.to(device)
    memory = memory.to(device)

    img_patches = model.extractor(image)
    mem_patches = model.extractor(memory)
    B, Tq, C, ph, pw = img_patches.shape
    M, Tm, _, _, _ = mem_patches.shape

    img_patches_flat = img_patches.reshape(Tq, C, ph, pw).contiguous()
    mem_patches_flat = mem_patches.reshape(M * Tm, C, ph, pw).contiguous()

    img_embeds = model.encoder(img_patches_flat)
    mem_embeds = model.encoder(mem_patches_flat)

    img_embeds_bt = img_embeds.view(B, Tq, -1)
    sel_logits = model.patch_scorer(img_embeds_bt).squeeze(-1)
    K = min(model.k, Tq)
    idx = sel_logits.topk(K, dim=-1).indices.view(-1)
    selected_mask = torch.zeros(Tq, dtype=torch.bool, device=device)
    selected_mask[idx] = True

    sim_selected = (img_embeds[selected_mask] @ mem_embeds.t()) / model.temperature

    if memory_cls.ndim == 2:
        memory_cls = memory_cls.argmax(dim=1)
    memory_cls = memory_cls.view(-1).long().to(device)
    assert memory_cls.numel() == M, f"memory_cls must have length M. got {memory_cls.shape}, M={M}"

    if cls_filter == "true":
        assert true_label is not None, "true_label must be provided when cls_filter='true'"
        target_cls = int(true_label)
    else:
        with torch.no_grad():
            logits_img = model.predict(image, memory, memory_cls)
        target_cls = int(logits_img.argmax(dim=1).item())

    mem_labels = memory_cls.repeat_interleave(Tm)
    mask_mem = (mem_labels == target_cls)

    if mask_mem.any():
        sim_f = sim_selected[:, mask_mem]
        mem_idx_all = mask_mem.nonzero(as_tuple=False).view(-1)
    else:
        sim_f = sim_selected
        mem_idx_all = torch.arange(M * Tm, device=device)

    sim_flat = sim_f.reshape(-1)
    k_eff = min(k, sim_flat.numel())
    scores, top_indices = torch.topk(sim_flat, k_eff, largest=True)

    L = sim_f.shape[1]
    img_local = top_indices // L
    mem_in_mask = top_indices % L

    img_patch_idx = idx[img_local]
    mem_patch_idx = mem_idx_all[mem_in_mask]

    relations = []
    for q_idx, mp_idx, score in zip(img_patch_idx.tolist(),
                                    mem_patch_idx.tolist(),
                                    scores.tolist()):
        mem_image_idx = mp_idx // Tm
        mem_patch_in_image = mp_idx % Tm

        relations.append({
            "score": float(score),
            "target_cls": int(target_cls),
            "img_patch_index": int(q_idx),
            "mem_image_index": int(mem_image_idx),
            "mem_patch_index_in_image": int(mem_patch_in_image),
            "img_patch": img_patches_flat[q_idx].detach().cpu(),
            "mem_patch": mem_patches_flat[mp_idx].detach().cpu(),
        })

    return relations

def show_relations_on_images(
    image: torch.Tensor,
    memory: torch.Tensor,
    relations: dict,
    kernel_size: int,
    stride: int,
    denorm_fn=None,
    title: str = None,
):
    """
    Draws the query patch + memory patch rectangles for one relation dict.
    """

    num_rel = len(relations)
    fig, axes = plt.subplots(num_rel, 3, figsize=(6, 1.5 * num_rel))

    if num_rel == 1:
        axes = axes.reshape(1, 2)
    for i, relation in enumerate(relations):
        img_patch_idx = int(relation["img_patch_index"])
        mem_image_index = int(relation["mem_image_index"])
        mem_patch_idx_in_image = int(relation["mem_patch_index_in_image"])

        image = image.cpu()
        memory = memory.cpu()

        _, _, H_img, W_img = image.shape
        _, _, H_mem, W_mem = memory.shape

        n_h_img = (H_img - kernel_size) // stride + 1
        n_w_img = (W_img - kernel_size) // stride + 1

        n_h_mem = (H_mem - kernel_size) // stride + 1
        n_w_mem = (W_mem - kernel_size) // stride + 1

        Tq_expected = n_h_img * n_w_img
        Tm_expected = n_h_mem * n_w_mem

        assert img_patch_idx < Tq_expected, f"img_patch_idx {img_patch_idx} out of range {Tq_expected}"
        assert mem_patch_idx_in_image < Tm_expected, f"mem_patch_idx {mem_patch_idx_in_image} out of range {Tm_expected}"

        img_row = img_patch_idx // n_w_img
        img_col = img_patch_idx % n_w_img
        y_img = img_row * stride
        x_img = img_col * stride

        mem_row = mem_patch_idx_in_image // n_w_mem
        mem_col = mem_patch_idx_in_image % n_w_mem
        y_mem = mem_row * stride
        x_mem = mem_col * stride

        img_vis = image[0]
        mem_vis = memory[mem_image_index]

        if denorm_fn is not None:
            img_vis = denorm_fn(img_vis)
            mem_vis = denorm_fn(mem_vis)

        img_vis = img_vis.permute(1, 2, 0).detach().cpu().numpy()
        mem_vis = mem_vis.permute(1, 2, 0).detach().cpu().numpy()

        axes[i, 0].axis('off')
        axes[i, 0].text(0.5, 0.5, f"Sim Score: {relation["score"]:.4f}",
                        fontsize=8, fontweight='bold',
                        ha='center', va='center')

        axes[i, 1].imshow(img_vis)
        axes[i, 1].add_patch(Rectangle(
            (x_img, y_img), kernel_size, kernel_size,
            linewidth=1.25, edgecolor='r', facecolor='none'
        ))
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mem_vis)
        axes[i, 2].add_patch(Rectangle(
            (x_mem, y_mem), kernel_size, kernel_size,
            linewidth=1.25, edgecolor='r', facecolor='none'
        ))
        axes[i, 2].axis("off")

    plt.tight_layout(pad=0.2)
    fig.subplots_adjust(hspace=0.1)
    plt.tight_layout()
    plt.show()

def extractor(img):
    return extract_patches(img, kernel_size=kernel, stride=stride)

def undoNorm(tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(-1,1,1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(-1,1,1)
    return tensor * std + mean


parser = argparse.ArgumentParser()

parser.add_argument("--run_dir", type=str)
parser.add_argument("--top_k", type=int, default=10)

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
model = model.to("mps")

# =========== Create Visuals ===========

memory, cls = next(iter(memloader))

loader = iter(testloader)
print("finding correctly classified image")
while True:
    image, label = next(loader)
    n = torch.randint(low=0, high=30, size=(1,)).item()
    image = image[n:n+1, :, :,]
    break
    with torch.no_grad():
        logits = model.predict(image.to("mps"), memory.to("mps"), cls.to("mps"))
    mask = (torch.argmax(logits, dim=-1).cpu() == label)
    if mask.any():
        image = image[mask][0:1]
        break

print("finding top contributors")
with torch.no_grad():
    relations = top_k_contributors(image, model, memory, cls, k=args.top_k)

print("displaying relations")

show_relations_on_images(
    image,
    memory,
    relations,
    kernel_size=kernel,
    stride=stride,
    denorm_fn=undoNorm,
)

