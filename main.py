# https://github.com/valeoai/FOUND

import torch

from train import train
from matchnet import Matcher
from samplier import extract_patches
from dataloader import trainloader, testloader, memloader

import argparse
import utils
from mapping import map_arg

import os
import json
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50',
                                                                         'resnet101', 'resnet152', 'vgg11', 
                                                                         'vgg16', 'vgg19', 'mobilenetv2',
                                                                         'efficientnet'])
parser.add_argument('--patch_size', type=int, default=18)
parser.add_argument('--stride', type=int, default=3)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'rmsprop', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_scheduler', type=str, default='none', choices=['cosine', 'linear', 'step', 'none'])
parser.add_argument('--base_epochs', type=int, default=50)
parser.add_argument('--selector_epochs', type=int, default=10)
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--run_dir', type=str)
args = parser.parse_args()

def extractor(img):
    return extract_patches(img, kernel_size=args.patch_size, stride=args.stride)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"

print("device:", device)

if args.run_dir is None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir
    print(f"saving to {args.run_dir}")

backbone = map_arg[args.backbone]
m = Matcher(10, extractor, backbone).to(device)
opt = map_arg[args.optimizer](m.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()
scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.base_epochs, args.lr)

config = vars(args)

start_epoch = 0

if args.continue_train and os.path.exists(os.path.join(args.run_dir, "state.pth")):
    checkpoint = torch.load(os.path.join(args.run_dir, "state.pth"), map_location=device)
    config = checkpoint.get("config", config)
    mode = checkpoint["last_mode"]
    print(os.path.join(os.path.curdir, config["run_dir"], f"state.pth"))
    print(config)

    backbone = map_arg[config["backbone"]]
    m = Matcher(10, extractor, backbone).to(device)

    if mode == "base":
        opt = map_arg[config["optimizer"]](m.parameters(), lr=args.lr)
        m.load_state_dict(checkpoint["base_model_state"])
        opt.load_state_dict(checkpoint["base_optimizer_state"])
        scheduler = utils.get_scheduler(map_arg, opt, config['lr_scheduler'], config['base_epochs'], config['lr'])
        if "base_scheduler_state" in checkpoint and checkpoint["base_scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["base_scheduler_state"])
        start_epoch = checkpoint.get("base_epoch", 0)
        print(f"Training From Checkpoint:\n    Current Mode: {mode}\n    Epoch: {start_epoch}\n    Test Loss {checkpoint.get('base_test_loss', 0)}\n    Test Accuracy {checkpoint.get('base_test_acc', 0)}")

        train(epochs=args.base_epochs, model=m, trainloader=trainloader, 
            testloader=testloader, memloader=memloader, optimizer=opt, 
            criterion=criterion, scheduler=scheduler, checkpoint=checkpoint, config=config,
            start_epoch=start_epoch, device=device)
            
        m.scorer_train_mode()

        opt = map_arg[args.optimizer](m.patch_scorer.parameters(), lr=args.lr/10)
        scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.selector_epochs, args.lr/10)
        
        train(epochs=args.selector_epochs, model=m, trainloader=trainloader, 
            testloader=testloader, memloader=memloader, optimizer=opt, 
            criterion=criterion, scheduler=scheduler, checkpoint=checkpoint, config=config, 
            start_epoch=0, device=device)

    if mode == "selector":
        if checkpoint["selector_model_state"] is not None:
            m.load_state_dict(checkpoint["selector_model_state"])
            opt.load_state_dict(checkpoint["selector_optimizer_state"])
        else:
            m.load_state_dict(checkpoint["base_model_state"])
            opt.load_state_dict(checkpoint["base_optimizer_state"])

        
        scheduler = utils.get_scheduler(map_arg, opt, config['lr_scheduler'], config['selector_epochs'], config['lr'])
        if "selector_scheduler_state" in checkpoint and checkpoint["selector_scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["selector_scheduler_state"])
        
        start_epoch = checkpoint.get("selector_epoch", 0)
        print(f"Training From Checkpoint:\n    Current Mode: {mode}\n    Epoch: {start_epoch}\n    Test Loss {checkpoint.get('selector_test_loss', 0)}\n    Test Accuracy {checkpoint.get('selector_test_acc', 0)}")

        m.scorer_train_mode()

        opt = map_arg[args.optimizer](m.patch_scorer.parameters(), lr=args.lr/10)
        scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.selector_epochs, args.lr/10)

        train(epochs=args.selector_epochs, model=m, trainloader=trainloader, 
            testloader=testloader, memloader=memloader, optimizer=opt, 
            criterion=criterion, scheduler=scheduler, checkpoint=checkpoint, config=config,
            start_epoch=start_epoch, device=device)

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

    torch.save({
            "last_mode": "base",
            
            "base_epoch": 0,
            "selector_epoch": 0,
            
            "base_test_loss": 0,
            "selector_test_loss" : 0,
            
            "base_test_acc" : 0,
            "selector_test_acc" : 0,

            "base_model_state": None,
            "selector_model_state": None,

            
            "base_optimizer_state": None,
            "selector_optimizer_state": None,

            "base_scheduler_state": None,
            "selector_scheduler_state": None,
            
            "config": config,
            },
        os.path.join(os.path.curdir, config["run_dir"], f"state.pth")
        )

    checkpoint = torch.load(os.path.join(args.run_dir, "state.pth"), map_location=device)

    print("Training Base Network")
    train(epochs=args.base_epochs, model=m, trainloader=trainloader, 
        testloader=testloader, memloader=memloader, optimizer=opt, 
        criterion=criterion, scheduler=scheduler, checkpoint=checkpoint, config=config, 
        start_epoch=start_epoch, device=device)

    m.scorer_train_mode()

    opt = map_arg[args.optimizer](m.patch_scorer.parameters(), lr=args.lr/10)
    scheduler = utils.get_scheduler(map_arg, opt, args.lr_scheduler, args.selector_epochs, args.lr/10)

    print("Training Patch Selector")
    train(epochs=args.selector_epochs, model=m, trainloader=trainloader, 
        testloader=testloader, memloader=memloader, optimizer=opt, 
        criterion=criterion, scheduler=scheduler, checkpoint=checkpoint, config=config, 
        start_epoch=0, device=device)

