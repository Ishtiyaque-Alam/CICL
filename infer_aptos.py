"""
infer_aptos.py
────────────────────────────────────────────────────────────────────────────
Full inference / evaluation script for CICL on APTOS 2019.

Loads the trained SupCon encoder + linear classifier checkpoint and runs
evaluation on the validation (or test) split, reporting:

  • Overall Accuracy
  • Macro F1-score  (sklearn)
  • Per-class TP, TN, FP, FN  (one-vs-rest decomposition)
  • Full confusion matrix

Usage (inside a Kaggle notebook cell):
  python infer_aptos.py \
      --data_folder /kaggle/working/aptos_imagefolder \
      --ckpt_encoder /kaggle/working/save/SupCon/APTOS_models/<name>/last.pth \
      --ckpt_linear  /kaggle/working/save/SupCon/APTOS_linear/<name>/best.pth \
      --n_cls 5 \
      --split val \
      --batch_size 64 \
      --num_workers 4
"""

import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)

# ── make sure CICL root is on sys.path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks.resnet_big import SupConResNet, LinearClassifier


def parse_args():
    p = argparse.ArgumentParser("CICL APTOS Inference")
    p.add_argument("--data_folder",   required=True)
    p.add_argument("--ckpt_encoder",  required=True,
                   help="Path to encoder checkpoint saved by main_supcon_aptos.py")
    p.add_argument("--ckpt_linear",   required=True,
                   help="Path to linear classifier checkpoint saved by main_linear_aptos.py")
    p.add_argument("--model",         default="resnet50")
    p.add_argument("--n_cls",         type=int, default=5)
    p.add_argument("--split",         default="val", choices=["train", "val"])
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--img_size",      type=int, default=224)
    return p.parse_args()


def build_loader(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = datasets.ImageFolder(
        root=os.path.join(args.data_folder, args.split),
        transform=val_transform
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return loader


def load_models(args):
    # ── encoder ──────────────────────────────────────────────────────────
    encoder = SupConResNet(name=args.model)
    ckpt_enc = torch.load(args.ckpt_encoder, map_location="cpu", weights_only=False)
    enc_state = ckpt_enc["model"]
    # strip DataParallel prefix if present
    enc_state = {k.replace("module.", ""): v for k, v in enc_state.items()}
    encoder.load_state_dict(enc_state)
    encoder = encoder.cuda()
    encoder.eval()

    # ── linear classifier ─────────────────────────────────────────────
    classifier = LinearClassifier(name=args.model, num_classes=args.n_cls)
    ckpt_lin = torch.load(args.ckpt_linear, map_location="cpu", weights_only=False)
    lin_state = ckpt_lin["classifier"]
    lin_state = {k.replace("module.", ""): v for k, v in lin_state.items()}
    classifier.load_state_dict(lin_state)
    classifier = classifier.cuda()
    classifier.eval()

    cudnn.benchmark = True
    return encoder, classifier


@torch.no_grad()
def run_inference(loader, encoder, classifier):
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        feats  = encoder.encoder(images)
        logits = classifier(feats)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred, n_cls):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

    # Per-class TP / TN / FP / FN  (one-vs-rest)
    per_class = {}
    for c in range(n_cls):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        per_class[c] = {"TP": int(tp), "TN": int(tn),
                        "FP": int(fp), "FN": int(fn)}

    return accuracy, macro_f1, cm, per_class


def print_report(accuracy, macro_f1, cm, per_class, n_cls, y_true, y_pred):
    print("\n" + "=" * 60)
    print("  CICL – APTOS 2019 Evaluation Results")
    print("=" * 60)
    print(f"  Overall Accuracy : {accuracy * 100:.2f}%")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    print()

    # sklearn full report
    target_names = [f"DR Grade {i}" for i in range(n_cls)]
    print(classification_report(y_true, y_pred,
                                target_names=target_names,
                                zero_division=0))

    print("-" * 60)
    print(f"  {'Class':<12} {'TP':>6} {'TN':>8} {'FP':>8} {'FN':>8}")
    print("-" * 60)
    for c in range(n_cls):
        m = per_class[c]
        print(f"  DR Grade {c:<3}  {m['TP']:>6} {m['TN']:>8} {m['FP']:>8} {m['FN']:>8}")
    print("-" * 60)

    print("\n  Confusion Matrix  (rows=GT, cols=Pred):")
    header = "        " + "".join(f"  P{i}" for i in range(n_cls))
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  GT {i}   " + "".join(f"{v:>4}" for v in row)
        print(row_str)
    print("=" * 60)


def main():
    args = parse_args()
    loader = build_loader(args)
    encoder, classifier = load_models(args)

    print(f"[INFO] Running inference on '{args.split}' split "
          f"({len(loader.dataset)} images)…")
    y_true, y_pred = run_inference(loader, encoder, classifier)

    accuracy, macro_f1, cm, per_class = compute_metrics(y_true, y_pred, args.n_cls)
    print_report(accuracy, macro_f1, cm, per_class, args.n_cls, y_true, y_pred)

    # Save raw arrays for further analysis
    np.save("/kaggle/working/aptos_true_labels.npy", y_true)
    np.save("/kaggle/working/aptos_pred_labels.npy", y_pred)
    print("\n[SAVED] Predictions → /kaggle/working/aptos_{true,pred}_labels.npy")


if __name__ == "__main__":
    main()
