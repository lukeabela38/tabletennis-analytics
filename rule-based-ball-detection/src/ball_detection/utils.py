import cv2
import numpy as np

def import_image(path: str) -> np.array:
    return cv2.imread(path)

def export_image(path: str, img: np.array) -> None:
    cv2.imwrite(path, img) 