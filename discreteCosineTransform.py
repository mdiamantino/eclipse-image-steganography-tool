import itertools
import warnings

import cv2
import numpy as np
from bitstring import BitArray

import imageUtils
import settings
from EncryptionUtils import encryptMessage


class DCT:
    def __init__(self, cover_image_path, cipher_msg):
        self.__cover_image_path_ = cover_image_path
        self.__cover_image_ = imageUtils.getImage(cover_image_path)
        self.__height_, self.__width_ = self.__cover_image_.shape[:2]
        self.__padded_cover_image_ = self.verifyAndApplyPadding()

        self.__cipher_msg_ = cipher_msg
        self.__bin_message_ = BitArray(bytes=cipher_msg).bin
        self.verifyCiphertextSize()

        self.__block_list_ = None
        self.__length_ = None

    # VERIFICATION METHODS =======================================================

    def verifyCiphertextSize(self):
        """
        Verifies that the lenght of the message to hide
        is less than the available place in the image.
        Eventually warns if the message lenght is > 10% this storage.
        """
        tot_blocks = (self.__height_ * self.__width_) // 64
        message_length = len(self.__bin_message_)
        if tot_blocks < message_length:
            raise OverflowError("Cannot embed. Message is too long!")
        elif tot_blocks / 10 < message_length:
            warning = "Message occupies ≈ {}% of the pic. " \
                      "A smaller text is preferred (< 10%)".format(
                round(message_length / tot_blocks * 100))
            warnings.warn(warning)

    def verifyAndApplyPadding(self):
        """
        Checks and eventually resizes image
        if any side length is not a multiple of 8.
        :return: OpenCV grayscale image with sides multiple of 8 [NUMPY NDARRAY]
        """
        cover_image = self.__cover_image_
        if self.__height_ % 8 != 0 or self.__width_ % 8 != 0:
            cover_image = cv2.resize(cover_image, (
                self.__width_ + (8 - self.__width_ % 8),
                self.__height_ + (8 - self.__height_ % 8)))
            cv2.imwrite(self.__cover_image_path_, cover_image)
        return cover_image

    # BREAK METHODS =======================================================

    @staticmethod
    def breakImageIntoBlocks(img, height, width):
        """
        Breaks the cover-image into a sequence of 8x8 blocks,
        from top left to bottom right.
        :param img: Cover-image to break into n 8x8 blocks.
        :return: List of blocks of pixels [LIST OF NUMPY NDARRAY]
        """
        return [img[j: j + 8, i: i + 8] for (j, i) in
                itertools.product(range(0, height, 8),
                                  range(0, width, 8))]

    def recomposeImage(self, block_size=8):
        """
        Inverse of 'breakImageIntoBlocks'.
        Builds image from sorted list of blocks of pixels.
        :param block_size: Side length of a block of pixels [INT]
        :return: Original image [NUMPY NDARRAY]
        """
        full_image = np.zeros(shape=(self.__height_, self.__width_), dtype=np.uint8)
        for i in range(len(self.__block_list_)):
            curr_col_index = block_size * (i % (self.__width_ // block_size))
            curr_line_index = block_size * (i // (self.__width_ // block_size))
            full_image[curr_line_index:curr_line_index + block_size,
            curr_col_index:curr_col_index + block_size] = self.__block_list_[i]
        return full_image

    # QUANTIZATION METHODS ===================================================

    @staticmethod
    def getQuantizedBlock(block):
        """
        Centers values of a block, runs it through DCT func and quantizes it.
        :param block: 8x8 block of pixels [NUMPY NDARRAY]
        :return:Quantized 8x8 block of pixels [NUMPY NDARRAY]
        """
        img_block = (np.subtract(block, 128))
        dct_block = cv2.dct(img_block.astype(np.float64))
        dct_block /= settings.QUANTIZATION_TABLE
        return dct_block

    @staticmethod
    def getOriginalBlockFromQuantized(quantized_block):
        """
        Inverse of "getQuantizedBlock".
        :param quantized_block: Quantized 8x8 block of pixels [NUMPY NDARRAY]
        :return: Original 8x8 block of pixels [NUMPY NDARRAY]
        """
        dct_block = quantized_block
        dct_block *= settings.QUANTIZATION_TABLE
        unquantized_block = cv2.idct(dct_block)
        return np.add(unquantized_block, 128)

    # ENCODE/DECODE METHODS  ===================================================

    def encode(self):
        img = self.__padded_cover_image_
        msg_len = len(self.__bin_message_)
        self.__block_list_ = self.breakImageIntoBlocks(img, self.__height_, self.__width_)
        for msg_bit_index in range(msg_len):
            dct_block = self.getQuantizedBlock(self.__block_list_[msg_bit_index])
            coeff = dct_block[0][0]
            message_bit = self.__bin_message_[msg_bit_index]
            print(coeff)


if __name__ == "__main__":
    message = encryptMessage("hello world", "passwordd")
    d = DCT("data/testimage.jpg", message)
    d.verifyAndApplyPadding()
    d.encode()
