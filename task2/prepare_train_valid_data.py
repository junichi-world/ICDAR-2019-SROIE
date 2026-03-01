import argparse
import glob
import os
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from PIL import Image


ALPHABET = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZ\\ '
ALLOWED_CHARS = set(ALPHABET.lower())


def _normalize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"\(\d+\)$", "", stem)


def _pick_preferred(paths: Iterable[str]) -> str:
    paths = sorted(paths)
    # Prefer files without "(n)" suffix when both canonical and duplicated names exist.
    for p in paths:
        if _normalize_stem(p) == os.path.splitext(os.path.basename(p))[0]:
            return p
    return paths[0]


def _index_files(folder: str, ext: str) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for p in glob.glob(os.path.join(folder, f"*.{ext}")):
        grouped[_normalize_stem(p)].append(p)
    return {k: _pick_preferred(v) for k, v in grouped.items()}


def _sanitize_label(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch.lower() in ALLOWED_CHARS)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _iter_annotation_lines(txt_path: str):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", maxsplit=8)
            if len(parts) < 9:
                continue
            try:
                coords = [int(float(parts[i])) for i in range(8)]
            except ValueError:
                continue
            transcript = parts[8]
            yield coords, transcript


def _crop_box(image: Image.Image, coords: List[int]) -> Image.Image:
    xs = [coords[0], coords[2], coords[4], coords[6]]
    ys = [coords[1], coords[3], coords[5], coords[7]]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(image.width, max(xs))
    bottom = min(image.height, max(ys))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("L")


def _clear_pair_files(folder: str):
    for p in glob.glob(os.path.join(folder, "*.jpg")):
        os.remove(p)
    for p in glob.glob(os.path.join(folder, "*.txt")):
        os.remove(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=os.path.join("..", "dataset", "0325updated.task1train(626p)"),
        help="source folder containing receipt .jpg and bbox+transcript .txt files",
    )
    parser.add_argument("--train-dir", default="data_train")
    parser.add_argument("--valid-dir", default="data_valid")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    task2_root = os.path.abspath(os.path.dirname(__file__))
    source = os.path.abspath(os.path.join(task2_root, args.source))
    train_dir = os.path.abspath(os.path.join(task2_root, args.train_dir))
    valid_dir = os.path.abspath(os.path.join(task2_root, args.valid_dir))

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    if args.clean:
        _clear_pair_files(train_dir)
        _clear_pair_files(valid_dir)

    jpg_map = _index_files(source, "jpg")
    txt_map = _index_files(source, "txt")
    keys = sorted(set(jpg_map.keys()) & set(txt_map.keys()))
    if not keys:
        raise FileNotFoundError(f"No matched jpg/txt receipt files found in: {source}")

    rng = random.Random(args.seed)
    rng.shuffle(keys)
    split = int(round(len(keys) * (1 - args.valid_ratio)))
    train_keys = set(keys[:split])

    train_count = 0
    valid_count = 0
    skipped_empty = 0
    skipped_bad_box = 0

    for stem in keys:
        img_path = jpg_map[stem]
        ann_path = txt_map[stem]
        image = Image.open(img_path).convert("RGB")
        out_dir = train_dir if stem in train_keys else valid_dir

        line_no = 0
        for coords, transcript in _iter_annotation_lines(ann_path):
            label = _sanitize_label(transcript)
            if not label:
                skipped_empty += 1
                continue

            crop = _crop_box(image, coords)
            if crop is None or crop.width < 2 or crop.height < 2:
                skipped_bad_box += 1
                continue

            out_stem = f"{stem}_{line_no:03d}"
            img_out = os.path.join(out_dir, out_stem + ".jpg")
            txt_out = os.path.join(out_dir, out_stem + ".txt")

            crop.save(img_out, format="JPEG", quality=95)
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(label)

            if out_dir == train_dir:
                train_count += 1
            else:
                valid_count += 1
            line_no += 1

    print(f"source receipts: {len(keys)}")
    print(f"train samples: {train_count}")
    print(f"valid samples: {valid_count}")
    print(f"skipped empty labels: {skipped_empty}")
    print(f"skipped invalid boxes: {skipped_bad_box}")
    print(f"train dir: {train_dir}")
    print(f"valid dir: {valid_dir}")


if __name__ == "__main__":
    main()
