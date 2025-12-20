# https://github.com/valeoai/FOUND

import torch

from train import train
from matchnet import Matcher
from samplier import extract_patches
from sim_classifier import sim_classify
from dataloader import trainloader, testloader, memloader

import argparse
from mapping import map_arg

import os
import json
from datetime import datetime

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir, exist_ok=True)


parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'
                                                                         'resnet101', 'resnet152', 'vgg11', 
                                                                         'vgg16', 'vgg19', 'mobilenetv2',
                                                                         'efficientnet'])
parser.add_argument('--patch_size', type=int, default=18)
parser.add_argument('--stride', type=int, default=3)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_scheduler', type=str, default='none', choices=['cosine', 'linear', 'step', 'none'])
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

def extractor(img):
    return extract_patches(img, kernel_size=args.patch_size, stride=args.stride)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

print("device: ", device)

backbone = map_arg[args.backbone]
m = Matcher(10, extractor, backbone).to(device)
opt = map_arg[args.optimizer](m.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

scheduler = args.lr_scheduler
if scheduler == "cosine":
    scheduler = map_arg[scheduler](optimizer=opt, T_max=args.epochs, eta_min=(args.lr / 100))
elif scheduler == "linear":
    scheduler = map_arg[scheduler](optimizer=opt, total_iters=args.epochs, start_factor=1, end_factor=.75)
elif scheduler == "step":
    scheduler = map_arg[scheduler](optimizer=opt, step_size=max(1, args.epochs // 10), gamma=0.5)
else:
    scheduler = None

train(epochs=args.epochs, model=m, trainloader=trainloader, 
        testloader=testloader, memloader=memloader, optimizer=opt, 
        criterion=criterion, scheduler=scheduler, device=device
        )

config = vars(args)

config.update({
    "device": device,
    "num_classes": 10,
    "optimizer_class": opt.__class__.__name__,
    "scheduler": args.lr_scheduler,
    "backbone_class": backbone.__class__.__name__,
})

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

torch.save(
    {
        "model_state": m.state_dict(),
        "config": config,
    },
    os.path.join(run_dir, "model.pth"),
)