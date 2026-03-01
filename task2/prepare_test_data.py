import glob
import os
import shutil


def _find_single_dir(root, pattern):
    matches = [p for p in glob.glob(os.path.join(root, pattern)) if os.path.isdir(p)]
    if not matches:
        raise FileNotFoundError(f"No directory matched: {os.path.join(root, pattern)}")
    if len(matches) > 1:
        matches = sorted(matches)
    return matches[0]


def main():
    task2_root = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(task2_root, ".."))
    dataset_root = os.path.join(repo_root, "dataset")

    src_images = _find_single_dir(dataset_root, "task1&2_test*")
    src_boxes = _find_single_dir(dataset_root, "text.task1&2-test*")

    dst_data_test = os.path.join(task2_root, "data_test")
    dst_bbox = os.path.join(task2_root, "boundingbox")
    dst_test_original = os.path.join(task2_root, "test_original")

    os.makedirs(dst_data_test, exist_ok=True)
    os.makedirs(dst_bbox, exist_ok=True)
    os.makedirs(dst_test_original, exist_ok=True)

    copied = 0
    for img in sorted(glob.glob(os.path.join(src_images, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img))[0]
        bbox = os.path.join(src_boxes, stem + ".txt")
        if not os.path.exists(bbox):
            continue

        shutil.copy2(img, os.path.join(dst_data_test, os.path.basename(img)))
        shutil.copy2(bbox, os.path.join(dst_bbox, os.path.basename(bbox)))
        shutil.copy2(img, os.path.join(dst_test_original, os.path.basename(img)))
        copied += 1

    print(f"copied {copied} test image/bbox pairs")
    print(f"data_test: {dst_data_test}")
    print(f"boundingbox: {dst_bbox}")
    print(f"test_original: {dst_test_original}")


if __name__ == "__main__":
    main()
