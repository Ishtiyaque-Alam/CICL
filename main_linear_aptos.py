"""
main_linear_aptos.py
────────────────────────────────────────────────────────────────────────────
Stage 2 – Linear classifier training + evaluation for APTOS 2019.
Loads the frozen SupCon encoder and trains a linear head on top.
At the end of every epoch it evaluates on the val split and prints:

  • Per-class TP / TN / FP / FN
  • Overall Accuracy
  • Macro F1-score
  • Full confusion matrix
  • sklearn classification report (precision, recall, F1 per class)

Usage:
  python main_linear_aptos.py \
      --data_folder  /kaggle/working/aptos_imagefolder \
      --ckpt         /kaggle/working/save/SupCon/APTOS_models/<name>/last.pth \
      --n_cls        5 \
      --epochs       30 \
      --batch_size   128 \
      --num_workers  4 \
      --cosine
"""

from __future__ import print_function

import os, sys, argparse, time, math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from util import AverageMeter, adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


# ─────────────────────────── argument parsing ────────────────────────────────

def parse_option():
    parser = argparse.ArgumentParser("CICL Linear – APTOS 2019")

    parser.add_argument("--print_freq",  type=int, default=10)
    parser.add_argument("--save_freq",   type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs",      type=int, default=30)

    # optimisation
    parser.add_argument("--learning_rate",   type=float, default=1.0)
    parser.add_argument("--lr_decay_epochs", type=str,   default="15,20,25")
    parser.add_argument("--lr_decay_rate",   type=float, default=0.2)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--momentum",        type=float, default=0.9)

    # model / dataset
    parser.add_argument("--model",       type=str, default="resnet50")
    parser.add_argument("--dataset",     type=str, default="APTOS")
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--n_cls",       type=int, default=5)
    parser.add_argument("--img_size",    type=int, default=224)

    # checkpoint
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to encoder checkpoint (last.pth from stage 1)")

    # flags
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--warm",   action="store_true")

    opt = parser.parse_args()

    opt.model_path = "./save/SupCon/{}_linear".format(opt.dataset)
    iterations     = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = [int(x) for x in iterations]

    opt.model_name = "{}_{}_{}_lr_{}_decay_{}_bsz_{}".format(
        opt.dataset, opt.model, opt.learning_rate,
        opt.weight_decay, opt.batch_size, "linear"
    )
    if opt.cosine:
        opt.model_name += "_cosine"
    if opt.warm:
        opt.model_name += "_warm"
        opt.warmup_from = 0.01
        opt.warm_epochs = 5
        if opt.cosine:
            eta_min         = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to   = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to   = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


# ─────────────────────────── data loaders ────────────────────────────────────

def set_loaders(opt):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(opt.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(opt.data_folder, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(opt.data_folder, "val"),
        transform=val_transform
    )

    # build class-count helpers (same as original CICL)
    train_dataset.dic   = {i: 0 for i in set(train_dataset.targets)}
    train_dataset.first = [0]
    for t in train_dataset.targets:
        train_dataset.dic[t] += 1
    for i, j in enumerate(train_dataset.dic.values()):
        train_dataset.first.append(train_dataset.first[i] + j)
    train_dataset.first.pop()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────── model ───────────────────────────────────────────

def set_model(opt):
    encoder    = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    criterion  = torch.nn.CrossEntropyLoss()

    ckpt       = torch.load(opt.ckpt, map_location="cpu")
    state_dict = ckpt["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if torch.cuda.is_available():
        encoder    = encoder.cuda()
        classifier = classifier.cuda()
        criterion  = criterion.cuda()
        cudnn.benchmark = True

    encoder.load_state_dict(state_dict)
    return encoder, classifier, criterion


# ─────────────────────────── train one epoch ─────────────────────────────────

def train_epoch(train_loader, encoder, classifier, criterion, optimizer,
                epoch, opt, max_item, n_batches, class_items_per_batch):
    encoder.eval()
    classifier.train()

    losses = AverageMeter()
    top1   = AverageMeter()

    # balanced resampling (same logic as original CICL)
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

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz    = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        with torch.no_grad():
            feats = encoder.encoder(images)
        logits = classifier(feats)
        loss   = criterion(logits, labels)

        # accuracy
        preds  = logits.argmax(dim=1)
        acc    = (preds == labels).float().mean() * 100
        losses.update(loss.item(), bsz)
        top1.update(acc.item(),   bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % opt.print_freq == 0:
            print(
                f"Train: [{epoch}][{idx+1}/{len(train_loader)}] "
                f"loss={losses.val:.4f}({losses.avg:.4f}) "
                f"acc={top1.val:.2f}({top1.avg:.2f})"
            )
            sys.stdout.flush()

    return losses.avg, top1.avg


# ─────────────────────────── validate + metrics ───────────────────────────────

@torch.no_grad()
def validate(val_loader, encoder, classifier, criterion, opt, epoch):
    encoder.eval()
    classifier.eval()

    all_preds  = []
    all_labels = []
    losses     = AverageMeter()

    for images, labels in val_loader:
        images = images.cuda(non_blocking=True)
        labels_gpu = labels.cuda(non_blocking=True)
        bsz    = labels_gpu.shape[0]

        feats  = encoder.encoder(images)
        logits = classifier(feats)
        loss   = criterion(logits, labels_gpu)
        losses.update(loss.item(), bsz)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=list(range(opt.n_cls)))

    print(f"\n{'='*60}")
    print(f"  Epoch {epoch} – Validation Results")
    print(f"{'='*60}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print()

    # Per-class TP / TN / FP / FN
    print(f"  {'Class':<14}{'TP':>6}{'TN':>8}{'FP':>8}{'FN':>8}")
    print(f"  {'-'*44}")
    for c in range(opt.n_cls):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        print(f"  DR Grade {c:<5}{tp:>6}{tn:>8}{fp:>8}{fn:>8}")
    print()

    # sklearn report
    target_names = [f"DR Grade {i}" for i in range(opt.n_cls)]
    print(classification_report(y_true, y_pred,
                                target_names=target_names,
                                zero_division=0))

    # Confusion matrix
    print("  Confusion Matrix (rows=GT, cols=Pred):")
    header = "       " + "".join(f"  P{i}" for i in range(opt.n_cls))
    print(header)
    for i, row in enumerate(cm):
        print(f"  GT {i}  " + "".join(f"{v:>4}" for v in row))
    print(f"{'='*60}\n")

    return acc, macro_f1


# ─────────────────────────── main ────────────────────────────────────────────

def main():
    opt = parse_option()
    print(f"[INFO] Dataset     : {opt.dataset}  ({opt.n_cls} classes)")
    print(f"[INFO] Encoder ckpt: {opt.ckpt}")
    print(f"[INFO] Epochs      : {opt.epochs}")

    train_loader, val_loader = set_loaders(opt)
    ds = train_loader.dataset
    ds.imgs    = sorted(ds.imgs,    key=lambda x: x[1])
    ds.targets = sorted(ds.targets)
    ds.samples = sorted(ds.samples, key=lambda x: x[1])

    max_item             = max(ds.dic.values())
    n_batches            = max_item * opt.n_cls // opt.batch_size
    class_items_per_batch = opt.batch_size // opt.n_cls

    encoder, classifier, criterion = set_model(opt)
    optimizer = set_optimizer(opt, classifier)

    best_acc  = 0.0
    best_f1   = 0.0

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        t0 = time.time()
        loss, train_acc = train_epoch(
            train_loader, encoder, classifier, criterion, optimizer,
            epoch, opt, max_item, n_batches, class_items_per_batch
        )
        t1 = time.time()
        print(f"[Epoch {epoch}] train_loss={loss:.4f}  train_acc={train_acc:.2f}%  "
              f"time={t1-t0:.1f}s")

        val_acc, val_f1 = validate(
            val_loader, encoder, classifier, criterion, opt, epoch
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(opt.save_folder, "best.pth")
            state = {
                "epoch":      epoch,
                "classifier": classifier.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "opt":        opt,
            }
            torch.save(state, ckpt_path)
            print(f"  [SAVED] Best model (acc={best_acc*100:.2f}%) → {ckpt_path}")

        if val_f1 > best_f1:
            best_f1 = val_f1

        if epoch % opt.save_freq == 0:
            ckpt_path = os.path.join(opt.save_folder, f"epoch_{epoch}.pth")
            state = {
                "epoch":      epoch,
                "classifier": classifier.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "opt":        opt,
            }
            torch.save(state, ckpt_path)

    # save last
    last_path = os.path.join(opt.save_folder, "last.pth")
    state = {
        "epoch":      opt.epochs,
        "classifier": classifier.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "opt":        opt,
    }
    torch.save(state, last_path)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Accuracy : {best_acc*100:.2f}%")
    print(f"  Best Macro F1 : {best_f1:.4f}")
    print(f"  Linear head   : {last_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
