import glob
import os
import re
import shutil
from collections import defaultdict


def normalize_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"\(\d+\)$", "", stem)


def pick_unique(paths):
    grouped = defaultdict(list)
    for p in paths:
        grouped[normalize_stem(p)].append(p)
    picked = {}
    for stem, items in grouped.items():
        items = sorted(items)
        for p in items:
            if normalize_stem(p) == os.path.splitext(os.path.basename(p))[0]:
                picked[stem] = p
                break
        else:
            picked[stem] = items[0]
    return picked


def main():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", ".."))
    source = os.path.join(repo_root, "dataset", "0325updated.task1train(626p)")
    target = os.path.join(this_dir, "original")
    os.makedirs(target, exist_ok=True)

    jpg_map = pick_unique(glob.glob(os.path.join(source, "*.jpg")))
    txt_map = pick_unique(glob.glob(os.path.join(source, "*.txt")))
    keys = sorted(set(jpg_map.keys()) & set(txt_map.keys()))

    copied = 0
    for stem in keys:
        jpg_src = jpg_map[stem]
        txt_src = txt_map[stem]
        jpg_dst = os.path.join(target, stem + ".jpg")
        txt_dst = os.path.join(target, stem + ".txt")
        shutil.copy2(jpg_src, jpg_dst)
        shutil.copy2(txt_src, txt_dst)
        copied += 1

    print(f"copied {copied} pairs")
    print(f"source: {source}")
    print(f"target: {target}")


if __name__ == "__main__":
    main()
