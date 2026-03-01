import argparse
import json
import os
import random
import re
from collections import defaultdict


LABEL_MAP = {"background": 0, "text": 1}


def normalize_stem(name):
    stem = os.path.splitext(name)[0]
    return re.sub(r"\(\d+\)$", "", stem)


def index_unique_files(folder, ext):
    grouped = defaultdict(list)
    for name in os.listdir(folder):
        if not name.lower().endswith(f".{ext.lower()}"):
            continue
        grouped[normalize_stem(name)].append(name)
    picked = {}
    for stem, names in grouped.items():
        names = sorted(names)
        # Prefer canonical filename without "(n)" suffix.
        for n in names:
            if normalize_stem(n) == os.path.splitext(n)[0]:
                picked[stem] = n
                break
        else:
            picked[stem] = names[0]
    return picked


def parse_annotation(annotation_path):
    boxes = []
    labels = []
    with open(annotation_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", maxsplit=8)
            if len(parts) < 8:
                continue
            try:
                xs = [int(float(parts[i])) for i in [0, 2, 4, 6]]
                ys = [int(float(parts[i])) for i in [1, 3, 5, 7]]
            except ValueError:
                continue

            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(LABEL_MAP["text"])
    return {"boxes": boxes, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=os.path.join("..", "..", "..", "dataset", "0325updated.task1train(626p)"),
        help="folder containing original .jpg/.txt pairs",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("..", "..", "..", "dataset", "0325updated.task1train(626p)"),
        help="folder to write TRAIN/TEST json files",
    )
    parser.add_argument("--test-every", type=int, default=25, help="use every Nth sample as test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    this_dir = os.path.abspath(os.path.dirname(__file__))
    source = os.path.abspath(os.path.join(this_dir, args.source))
    output = os.path.abspath(os.path.join(this_dir, args.output))
    os.makedirs(output, exist_ok=True)

    jpg_map = index_unique_files(source, "jpg")
    txt_map = index_unique_files(source, "txt")
    keys = sorted(set(jpg_map.keys()) & set(txt_map.keys()))
    if not keys:
        raise FileNotFoundError(f"No matching jpg/txt pairs found in {source}")

    rng = random.Random(args.seed)
    rng.shuffle(keys)

    train_ids = []
    test_ids = []
    train_images, train_objects = [], []
    test_images, test_objects = [], []

    for i, stem in enumerate(keys, start=1):
        image_path = os.path.join(source, jpg_map[stem])
        txt_path = os.path.join(source, txt_map[stem])
        objects = parse_annotation(txt_path)
        if not objects["boxes"]:
            continue

        if i % args.test_every == 0:
            test_ids.append(stem)
            test_images.append(image_path)
            test_objects.append(objects)
        else:
            train_ids.append(stem)
            train_images.append(image_path)
            train_objects.append(objects)

    with open(os.path.join(output, "TRAIN_images.json"), "w", encoding="utf-8") as f:
        json.dump(train_images, f)
    with open(os.path.join(output, "TRAIN_objects.json"), "w", encoding="utf-8") as f:
        json.dump(train_objects, f)
    with open(os.path.join(output, "TEST_images.json"), "w", encoding="utf-8") as f:
        json.dump(test_images, f)
    with open(os.path.join(output, "TEST_objects.json"), "w", encoding="utf-8") as f:
        json.dump(test_objects, f)
    with open(os.path.join(output, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f)

    # Optional ID lists for compatibility with old scripts.
    with open(os.path.join(output, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids))
    with open(os.path.join(output, "test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_ids))

    print(f"source: {source}")
    print(f"output: {output}")
    print(f"train samples: {len(train_images)}")
    print(f"test samples: {len(test_images)}")


if __name__ == "__main__":
    main()
