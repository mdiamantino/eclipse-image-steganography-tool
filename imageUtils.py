import cv2


def getImage(img_path):
    """
    Loads image into memory as a OpenCV image.
    :param img_path: Path to image [STR]
    :return: OpenCV grayscale image [NUMPY NDARRAY]
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image was not found!")
    else:
        return img
