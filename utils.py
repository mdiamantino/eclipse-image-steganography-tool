import cv2


def getImage(img_path):
    """
    Loads image into memory as a OpenCV image.
    :param img_path: Path to image [STR]
    :return: OpenCV grayscale image [NUMPY NDARRAY]
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Image was not found!")
    else:
        return img


def stringToBinary(string_text):
    """
    Converts utf-8 text to its binary form (8 bit per character).
    :param string_text: UTF-8 text [STRING]
    :return: Array containing each bit of the binary form of each character
             [LIST OF STRINGS]
    """
    map_of_binary_str = map(lambda x: str(format(ord(x), '08b')),
                            string_text)
    return [item for sublist in map_of_binary_str for item in sublist]


def getBlockFromBlockIndex(img, width, block_index, block_size=8):
    i = 8 * (block_index % (width // block_size))
    j = 8 * (block_index // (width // block_size))
    return img[j:j + 8, i:i + 8]


def getYCrCbFromOriginalImg(img):
    YCrCbImage = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(YCrCbImage)
    return (y, cr, cb)


def getOriginalImgFromYCrCb(y, cr, cb):
    ycrcb_format_img = cv2.merge((y, cr, cb))
    standard_format_img = cv2.cvtColor(ycrcb_format_img, cv2.COLOR_YCR_CB2BGR)
    return standard_format_img
