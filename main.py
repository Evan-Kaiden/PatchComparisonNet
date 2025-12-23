# https://github.com/valeoai/FOUND

import torch

from train import train
from matchnet import Matcher
from samplier import extract_patches
from sim_classifier import sim_classify
from dataloader import trainloader, testloader, memloader

import argparse
import utils
from mapping import map_arg

import os
import json
from datetime import datetime

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
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--run_dir', type=str)
args = parser.parse_args()

def extractor(img):
    return extract_patches(img, kernel_size=args.patch_size, stride=args.stride)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

print("device: ", device)

if args.run_dir is None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

backbone = map_arg[args.backbone]
m = Matcher(10, extractor, backbone).to(device)
opt = map_arg[args.optimizer](m.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.epochs, args.lr)

config = vars(args)

start_epoch = 0
print( args.continue_train, os.path.exists(os.path.join(args.run_dir, "state.pth")))
if args.continue_train and os.path.exists(os.path.join(args.run_dir, "state.pth")):
    checkpoint = torch.load(os.path.join(args.run_dir, "state.pth"), map_location=device)
    config = checkpoint.get("config", config)

    backbone = map_arg[config["backbone"]]
    m = Matcher(10, extractor, backbone).to(device)
    opt = map_arg[config["optimizer"]](m.parameters(), lr=args.lr)
    
    m.load_state_dict(checkpoint["model_state"])
    opt.load_state_dict(checkpoint["optimizer_state"])
    scheduler = utils.get_scheduler(map_arg, opt, config['lr_scheduler'], config['epochs'], config['lr'])
    if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint.get("epoch", 0)
else:
    config.update({
        "device": device,
        "num_classes": 10,
        "epoch": start_epoch,
        "optimizer_class": opt.__class__.__name__,
        "scheduler": args.lr_scheduler,
        "backbone_class": backbone.__class__.__name__,
    })
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

train(epochs=args.epochs, model=m, trainloader=trainloader, 
      testloader=testloader, memloader=memloader, optimizer=opt, 
      criterion=criterion, scheduler=scheduler, config=config, 
      start_epoch=start_epoch, device=device)