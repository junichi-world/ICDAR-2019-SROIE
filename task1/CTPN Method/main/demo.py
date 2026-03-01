import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from train import CtpnModel, STRIDE


def resize_for_model(image: np.ndarray, resolution: Tuple[int, int]):
    h0, w0 = image.shape[:2]
    h1, w1 = resolution
    resized = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LINEAR)
    return resized, (h0 / h1, w0 / w1)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: List[List[float]], scores: List[float], iou_thresh=0.3):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        remain = []
        for j in order[1:]:
            if iou(boxes[i], boxes[j]) <= iou_thresh:
                remain.append(j)
        order = np.array(remain, dtype=np.int64)
    return keep


def decode_predictions(y_cls, y_v, anchors, score_thresh=0.7, topk=1000):
    probs = torch.softmax(y_cls, dim=-1)[0, ..., 1].cpu().numpy()  # [H,W,K]
    y_v = y_v[0].detach().cpu().numpy()  # [H,W,K,2]

    boxes = []
    scores = []
    h, w, k = probs.shape
    for r in range(h):
        for c in range(w):
            for a in range(k):
                score = float(probs[r, c, a])
                if score < score_thresh:
                    continue

                h_anchor = float(anchors[a])
                v_c = float(y_v[r, c, a, 0])
                v_h = float(y_v[r, c, a, 1])
                cy = r * STRIDE + STRIDE / 2 + v_c * h_anchor
                hh = np.exp(v_h) * h_anchor
                x1 = c * STRIDE
                x2 = x1 + STRIDE
                y1 = cy - hh / 2
                y2 = cy + hh / 2
                boxes.append([x1, y1, x2, y2])
                scores.append(score)

    if not boxes:
        return [], []

    if len(boxes) > topk:
        idx = np.argsort(scores)[-topk:]
        boxes = [boxes[i] for i in idx]
        scores = [scores[i] for i in idx]

    keep = nms(boxes, scores, iou_thresh=0.3)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    return boxes, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--test-data-path", default="data/demo")
    parser.add_argument("--output-path", default="data/res")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresh", type=float, default=0.7)
    parser.add_argument("--n-anchor", type=int, default=10)
    parser.add_argument("--input-h", type=int, default=448)
    parser.add_argument("--input-w", type=int, default=224)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device(args.device)
    model = CtpnModel(n_anchor=args.n_anchor, pretrained_backbone=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    anchors = torch.tensor([5 * (2 ** (i / 2)) for i in range(args.n_anchor)])
    tfm = transforms.ToTensor()

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        image_paths.extend(glob.glob(os.path.join(args.test_data_path, ext)))
    image_paths = sorted(set(image_paths))
    print(f"Find {len(image_paths)} images")

    with torch.no_grad():
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"skip unreadable image: {p}")
                continue
            resized, (h_scale, w_scale) = resize_for_model(img, (args.input_h, args.input_w))
            x = tfm(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
            y_cls, y_v, _ = model(x)
            boxes, scores = decode_predictions(
                y_cls, y_v, anchors, score_thresh=args.score_thresh
            )

            stem = os.path.splitext(os.path.basename(p))[0]
            out_img = os.path.join(args.output_path, os.path.basename(p))
            out_txt = os.path.join(args.output_path, stem + ".txt")

            with open(out_txt, "w", encoding="utf-8") as f:
                for b, s in zip(boxes, scores):
                    x1 = int(round(b[0] * w_scale))
                    y1 = int(round(b[1] * h_scale))
                    x2 = int(round(b[2] * w_scale))
                    y2 = int(round(b[3] * h_scale))
                    line = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{s:.6f}\n"
                    f.write(line)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            cv2.imwrite(out_img, img)
            print(f"wrote: {out_txt}")


if __name__ == "__main__":
    main()
