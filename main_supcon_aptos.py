"""
main_supcon_aptos.py
────────────────────────────────────────────────────────────────────────────
Stage 1 – SupCon pre-training adapted for APTOS 2019.
Drop-in replacement for the original main_supcon.py:
  • Accepts --dataset APTOS (ImageFolder layout produced by prepare_aptos.py)
  • --balance flag uses CICL's balanced sampling
  • Saves last.pth and periodic checkpoints

Usage:
  python main_supcon_aptos.py \
      --batch_size 64 \
      --data_folder /kaggle/working/aptos_imagefolder \
      --dataset APTOS \
      --epochs 50 \
      --cosine \
      --balance \
      --num_workers 4
"""

from __future__ import print_function

import os, sys, argparse, time, math, random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


# ─────────────────────────── argument parsing ────────────────────────────────

def parse_option():
    parser = argparse.ArgumentParser("CICL SupCon – APTOS 2019")

    parser.add_argument("--print_freq",  type=int, default=10)
    parser.add_argument("--save_freq",   type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs",      type=int, default=50)

    # optimisation
    parser.add_argument("--learning_rate",    type=float, default=0.05)
    parser.add_argument("--lr_decay_epochs",  type=str,   default="30,40,45")
    parser.add_argument("--lr_decay_rate",    type=float, default=0.1)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--momentum",         type=float, default=0.9)

    # model / dataset
    parser.add_argument("--model",       type=str, default="resnet50")
    parser.add_argument("--dataset",     type=str, default="APTOS")
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--size",        type=int, default=224)

    # method
    parser.add_argument("--method",      type=str, default="SupCon",
                        choices=["SupCon", "SimCLR"])
    parser.add_argument("--temp",        type=float, default=0.1)

    # flags
    parser.add_argument("--cosine",  action="store_true")
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--syncBN",  action="store_true")
    parser.add_argument("--warm",    action="store_true")
    parser.add_argument("--trial",   type=str, default="0")

    opt = parser.parse_args()

    if opt.data_folder is None:
        opt.data_folder = "./datasets/"

    opt.model_path = "./save/SupCon/{}_models".format(opt.dataset)
    opt.tb_path    = "./save/SupCon/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(x) for x in iterations]

    opt.model_name = (
        "{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}"
        .format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    )
    if opt.cosine:
        opt.model_name += "_cosine"
    if opt.batch_size > 10 or opt.warm:
        opt.warm = True
        opt.model_name += "_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder   = os.path.join(opt.tb_path,    opt.model_name)
    opt.save_folder = os.path.join(opt.model_path,  opt.model_name)
    os.makedirs(opt.tb_folder,   exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


# ─────────────────────────── data loader ─────────────────────────────────────

def set_loader(opt):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(opt.data_folder, "train"),
        transform=TwoCropTransform(train_transform)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    if opt.balance:
        ds = train_loader.dataset
        ds.dic   = {i: 0 for i in set(ds.targets)}
        ds.first = [0]
        for t in ds.targets:
            ds.dic[t] += 1
        for i, j in enumerate(ds.dic.values()):
            ds.first.append(ds.first[i] + j)
        ds.first.pop()
        opt.n_cls = len(ds.dic)

    return train_loader


# ─────────────────────────── model ───────────────────────────────────────────

def set_model(opt):
    model     = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model     = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


# ─────────────────────────── train one epoch ─────────────────────────────────

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        labels = np.array(labels).astype(np.int16)
        bsz    = labels.shape[0]
        perm   = np.random.choice(bsz, bsz, replace=False).tolist()
        images[0] = images[0][perm]
        images[1] = images[1][perm]
        labels    = labels[perm]
        images    = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = torch.Tensor(labels).cuda(non_blocking=True)

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model(images)
        f1, f2   = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == "SupCon":
            loss = criterion(features, labels)
        elif opt.method == "SimCLR":
            loss = criterion(features)
        else:
            raise ValueError(f"Unknown method: {opt.method}")

        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print(
                f"Train: [{epoch}][{idx+1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.4f} ({losses.avg:.4f})"
            )
            sys.stdout.flush()

    return losses.avg


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    opt = parse_option()
    print(f"[INFO] Dataset : {opt.dataset}")
    print(f"[INFO] Epochs  : {opt.epochs}")
    print(f"[INFO] Balanced: {opt.balance}")

    train_loader = set_loader(opt)

    if opt.balance:
        ds = train_loader.dataset
        ds.imgs    = sorted(ds.imgs,    key=lambda x: x[1])
        ds.targets = sorted(ds.targets)
        ds.samples = sorted(ds.samples, key=lambda x: x[1])
        max_item            = max(ds.dic.values())
        n_batches           = max_item * opt.n_cls // opt.batch_size
        class_items_per_batch = opt.batch_size // opt.n_cls

    model, criterion = set_model(opt)
    optimizer        = set_optimizer(opt, model)

    for epoch in range(1, opt.epochs + 1):
        if opt.balance:
            ds = train_loader.dataset
            ds.samples = sorted(ds.imgs.copy(), key=lambda x: x[1])
            indices = [
                np.random.choice(range(max_item), max_item, replace=False).tolist()
                for _ in range(opt.n_cls)
            ]
            indices = [
                [ds.first[i] + (k % ds.dic[i]) for k in j]
                for i, j in enumerate(indices)
            ]
            order = []
            for i in range(n_batches):
                for j in range(opt.n_cls):
                    order.extend(indices[j][i*class_items_per_batch:(i+1)*class_items_per_batch])
            ds.samples = (np.array(ds.samples)[order]).tolist()

        adjust_learning_rate(opt, optimizer, epoch)
        t0   = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        t1   = time.time()
        print(f"Epoch {epoch}/{opt.epochs}  loss={loss:.4f}  time={t1-t0:.1f}s")

        if epoch % opt.save_freq == 0:
            ckpt_path = os.path.join(
                opt.save_folder, f"ckpt_epoch_{epoch}.pth"
            )
            save_model(model, optimizer, opt, epoch, ckpt_path)

    # always save last checkpoint
    last_path = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, last_path)
    print(f"\n[SAVED] Encoder checkpoint → {last_path}")


if __name__ == "__main__":
    main()
