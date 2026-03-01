import csv
import glob
import os
import shutil

def get_data():
    jpg_files = sorted(glob.glob("original/*.jpg"))
    txt_files = [os.path.splitext(s)[0] + ".txt" for s in jpg_files]

    for file in txt_files:
        boxes = []
        with open(file, "r", encoding="utf-8", newline="") as lines:
            for line in csv.reader(lines):
                if len(line) < 8:
                    continue
                xs = [int(float(line[i])) for i in [0, 2, 4, 6]]
                ys = [int(float(line[i])) for i in [1, 3, 5, 7]]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
        out_name = os.path.basename(file)
        with open(os.path.join("mlt", "label", out_name), "w+", newline="") as labelFile:
            wr = csv.writer(labelFile)
            wr.writerows(boxes)

    for jpg in jpg_files:
        shutil.copy(jpg, os.path.join("mlt", "image"))


if __name__ == "__main__":
    get_data()
