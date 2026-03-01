import argparse
import csv
import glob
import os

import cv2
import torch
from PIL import Image
from torch.autograd import Variable

import dataset
import models.crnn as crnn
import utils


ALPHABET = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZ\\ '


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def predict_this_box(image, model, alphabet):
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((200, 32))
    image = transformer(image)
    if next(model.parameters()).is_cuda:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print(f"{raw_pred:<30} => {sim_pred:<30}")
    return sim_pred


def load_images_to_predict(model_path, force_cpu=False):
    img_h = 32
    nclass = len(ALPHABET) + 1
    nhiddenstate = 256
    use_cuda = torch.cuda.is_available() and not force_cpu
    map_location = "cuda" if use_cuda else "cpu"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = crnn.CRNN(img_h, 1, nclass, nhiddenstate)
    if use_cuda:
        model = model.cuda()

    print(f"loading pretrained model from {model_path}")
    state = torch.load(model_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    _ensure_dir("test_result")
    image_paths = sorted(glob.glob("data_test/*.jpg"))
    if not image_paths:
        print("No images found in data_test/*.jpg")
        return

    for jpg in image_paths:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        bbox_path = os.path.join("boundingbox", stem + ".txt")
        if not os.path.exists(bbox_path):
            print(f"skip {stem}: missing {bbox_path}")
            continue

        image = Image.open(jpg).convert("L")
        words_list = []
        with open(bbox_path, "r", encoding="utf-8") as boxes:
            for line in csv.reader(boxes):
                if len(line) < 8:
                    continue
                box = [int(string, 10) for string in line[0:8]]
                box_img = image.crop((box[0], box[1], box[4], box[5]))
                words = predict_this_box(box_img, model, ALPHABET)
                words_list.append(words)

        out_path = os.path.join("test_result", stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as resultfile:
            for line in words_list:
                resultfile.write(line + "\n")
        print(f"wrote: {out_path}")


def process_txt():
    _ensure_dir("task2_result")
    old_files = sorted(glob.glob("test_result/*.txt"))

    for old_file in old_files:
        new = []
        with open(old_file, "r", encoding="utf-8") as old:
            for line in csv.reader(old):
                if not line or not line[0]:
                    continue
                value = line[0].strip()
                if " " in value:
                    value = value.split(" ")[0]
                new.append([value.upper()])

        out_name = os.path.basename(old_file)
        out_path = os.path.join("task2_result", out_name)
        with open(out_path, "w", encoding="utf-8") as newfile:
            for row in new:
                if row:
                    newfile.write(row[0] + "\n")
        print(f"wrote: {out_path}")


def for_task3():
    _ensure_dir("for_task3")
    pred_files = sorted(glob.glob("test_result/*.txt"))
    for pred_file in pred_files:
        boxfile = os.path.join("boundingbox", os.path.basename(pred_file))
        if not os.path.exists(boxfile):
            print(f"skip {pred_file}: missing bbox file {boxfile}")
            continue
        box = []
        with open(boxfile, "r", encoding="utf-8") as boxes:
            for line in csv.reader(boxes):
                if len(line) < 8:
                    continue
                box.append([int(string, 10) for string in line[0:8]])

        words = []
        with open(pred_file, "r", encoding="utf-8") as prediction:
            for line in csv.reader(prediction):
                words.append(line)
        words = [s if len(s) != 0 else [" "] for s in words]

        out_path = os.path.join("for_task3", os.path.basename(boxfile))
        with open(out_path, "w", encoding="utf-8", newline="") as newfile:
            csv_out = csv.writer(newfile)
            for a, b in zip(box, words):
                csv_out.writerow(a + b)
        print(f"wrote: {out_path}")


def draw():
    _ensure_dir("task2_result_draw")
    txt_files = sorted(glob.glob("for_task3/*.txt"))
    for txt in txt_files:
        stem = os.path.splitext(os.path.basename(txt))[0]
        img_path = os.path.join("test_original", stem + ".jpg")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip {txt}: missing image {img_path}")
            continue
        with open(txt, "r", encoding="utf-8") as txt_file:
            for line in csv.reader(txt_file):
                if len(line) < 8:
                    continue
                box = [int(string, 10) for string in line[0:8]]
                label = line[8].upper() if len(line) >= 9 else ""
                cv2.rectangle(image, (box[0], box[1]), (box[4], box[5]), (0, 255, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    image, label, (box[0], box[1]), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                )
        out_path = os.path.join("task2_result_draw", stem + ".jpg")
        cv2.imwrite(out_path, image)
        print(f"wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["predict", "process", "for_task3", "draw", "all"],
        default="predict",
    )
    parser.add_argument("--model-path", default="./expr/netCRNN_190_423.pth")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    try:
        if args.mode == "predict":
            load_images_to_predict(args.model_path, force_cpu=args.cpu)
        elif args.mode == "process":
            process_txt()
        elif args.mode == "for_task3":
            for_task3()
        elif args.mode == "draw":
            draw()
        elif args.mode == "all":
            load_images_to_predict(args.model_path, force_cpu=args.cpu)
            process_txt()
            for_task3()
    except FileNotFoundError as e:
        print(e)
        print("Place pretrained CRNN weights at ./expr/netCRNN_190_423.pth or use --model-path.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
