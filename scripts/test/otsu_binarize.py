import os
import argparse
import cv2

CUSTOM_IMG_DIR = "data/custom/author_0/physical"


def parse_args():
    parser = argparse.ArgumentParser(description="Otsu binarization")
    parser.add_argument("--img-dir", default=CUSTOM_IMG_DIR)
    return parser.parse_args()


def otsu_binarize(path, output_path=None):
    """
    Binarizes all non-.bmp images within a directory, then saves them as a new
    .bmp image each.
    """

    if path.lower().endswith(".bmp"):
        return path

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarized = cv2.bitwise_not(binarized)

    output_path = output_path or os.path.splitext(path)[0] + ".bmp"
    cv2.imwrite(output_path, binarized)
    return output_path


if __name__ == "__main__":
    args = parse_args()
    non_bmp = [
        os.path.join(args.img_dir, f)
        for f in os.listdir(args.img_dir)
        if not f.lower().endswith(".bmp")
    ]

    for path in non_bmp:
        out = otsu_binarize(path)
        print(f"Saved to {out}")
