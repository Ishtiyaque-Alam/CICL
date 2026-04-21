"""
prepare_aptos.py
────────────────────────────────────────────────────────────────────────────
Converts APTOS 2019 (CSV + flat image directory) into the ImageFolder
directory structure that CICL's DataLoader expects:

  <out_root>/
    train/
      0/  ← DR grade 0 images
      1/
      2/
      3/
      4/
    val/
      0/
      1/
      2/
      3/
      4/

Images are SYMLINKED (not copied) to save disk space.
Use --val_split 0.2 to hold out 20 % of training images as a validation set.
The Kaggle test set (no labels) is NOT touched by this script.

Usage (inside a Kaggle notebook cell):
  python prepare_aptos.py \
      --train_csv  /kaggle/input/.../train_1.csv \
      --train_dir  /kaggle/input/.../train_images/train_images \
      --out_root   /kaggle/working/aptos_imagefolder \
      --val_split  0.2 \
      --seed       42
"""

import argparse
import os
import shutil
import random
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  required=True, help="Path to train_1.csv")
    p.add_argument("--train_dir",  required=True, help="Path to flat train images folder")
    p.add_argument("--out_root",   default="/kaggle/working/aptos_imagefolder")
    p.add_argument("--val_split",  type=float, default=0.2,
                   help="Fraction of training data used as validation (default 0.2)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--img_ext",    default=".png",
                   help="Image extension in train_dir (default .png)")
    return p.parse_args()


def make_dirs(root, splits, n_classes):
    for split in splits:
        for cls in range(n_classes):
            Path(root, split, str(cls)).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path):
    """Prefer symlink; fall back to hard-link then copy (for Kaggle /tmp mounts)."""
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.train_csv)
    # Detect label column (diagnosis / label / DR_grade …)
    label_col = None
    for col in ["diagnosis", "label", "DR_grade", "level"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(
            f"Cannot find a label column in {args.train_csv}. "
            f"Columns present: {list(df.columns)}"
        )

    id_col = df.columns[0]          # first column = image id / filename
    n_classes = int(df[label_col].max()) + 1
    print(f"[INFO] {len(df)} samples  |  {n_classes} classes  |  "
          f"id='{id_col}'  label='{label_col}'")

    make_dirs(args.out_root, ["train", "val"], n_classes)

    # stratified split per class
    train_rows, val_rows = [], []
    for grade, grp in df.groupby(label_col):
        idxs = grp.index.tolist()
        random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * args.val_split))
        val_rows.extend(idxs[:n_val])
        train_rows.extend(idxs[n_val:])

    print(f"[INFO] train={len(train_rows)}  val={len(val_rows)}")

    for split_name, row_idxs in [("train", train_rows), ("val", val_rows)]:
        for idx in row_idxs:
            row   = df.loc[idx]
            img_id = str(row[id_col])
            grade  = int(row[label_col])

            # Try with and without extension
            src = Path(args.train_dir) / (img_id + args.img_ext)
            if not src.exists():
                src = Path(args.train_dir) / img_id
            if not src.exists():
                # Search by glob
                candidates = list(Path(args.train_dir).glob(img_id + ".*"))
                if not candidates:
                    print(f"[WARN] Image not found: {img_id} — skipping")
                    continue
                src = candidates[0]

            dst = Path(args.out_root) / split_name / str(grade) / src.name
            link_or_copy(src, dst)

    # Print class distribution
    print("\n[INFO] Class distribution:")
    for split in ["train", "val"]:
        counts = []
        for cls in range(n_classes):
            d = Path(args.out_root) / split / str(cls)
            counts.append(len(list(d.iterdir())))
        print(f"  {split}: " + "  ".join(f"cls{c}={n}" for c, n in enumerate(counts)))

    print(f"\n[DONE] ImageFolder structure ready at: {args.out_root}")


if __name__ == "__main__":
    main()
