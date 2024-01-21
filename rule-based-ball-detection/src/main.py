import numpy as np
from ball_detection import utils, processing

def main() -> int:

    path: str = "images/inputs/behind.jpeg"
    img: np.array = utils.import_image(path)
    
    ### processing

    img = processing.hough_circle_transform(img, singular_output=True)

    ###

    utils.export_image("images/outputs/behind.jpeg", img)
    
    return 0

if __name__ == "__main__":
    main()