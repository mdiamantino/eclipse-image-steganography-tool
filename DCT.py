import itertools
import random
import warnings

import cv2
import numpy as np
from bitstring import BitArray

import settings
import utils


class DCT:
    def __init__(self, cover_image_path, cipher_msg):
        self.__cover_image_path_ = cover_image_path
        self.__cover_image_ = utils.getImage(cover_image_path)
        self.verifyAndApplyPadding()
        self.__height_, self.__width_ = self.__cover_image_.shape[:2]
        self.__cipher_text_ = cipher_msg
        self.__bin_message_ = BitArray(bytes=cipher_msg).bin
        self.__message_length_ = len(self.__bin_message_)
        self.verifyCiphertextSize()

        self.__block_list_ = None

    # VERIFICATION METHODS ============================================================

    def verifyCiphertextSize(self):
        """
        Verifies that the length of the message to hide
        is shorter than the maximum available space in the image.
        Warning is raised if the message length is > (10% of the available capacity).
        """
        area = self.__height_ * self.__width_
        if area < 64:  # 64 -> since each quantized block is a matrix of 8x8
            raise ValueError(
                "The chosen cover image is too small (area < 64 px)")
        tot_blocks = self.__height_ * self.__width_ // 64
        if self.__message_length_ > tot_blocks:
            raise OverflowError("Cannot embed. Message is too long!")
        if self.__message_length_ > tot_blocks / 10:
            purcentage_of_occupied_storage = round(
                self.__message_length_ / tot_blocks * 100)
            warning = f"Message occupies ≈ " \
                      f"{purcentage_of_occupied_storage}% of the pic. " \
                      "A smaller text is preferred (< 10%)"
            warnings.warn(warning)

    def verifyAndApplyPadding(self):
        """
        Checks and eventually resizes image applying a padding
        if any side length is not a multiple of 8.
        The original image is eventually replaced by the padded
        (with sides multiple of 8) image.
        """
        original_height, original_width = self.__cover_image_.shape[:2]
        if original_height % 8 != 0 or original_width % 8 != 0:
            self.__cover_image_ = cv2.resize(self.__cover_image_, (
                original_width + (8 - original_width % 8),
                original_height + (8 - original_height % 8)))
            cv2.imwrite(self.__cover_image_path_, self.__cover_image_)

    # BREAK/RECOMPOSE METHODS =========================================================

    @staticmethod
    def breakImageIntoBlocks(img, height, width):
        """
        Breaks the coverimage into a sequence of 8x8 blocks,
        from top left to bottom right.
        :param img: Coverimage to break into n 8x8 blocks.
        :return: List of blocks of pixels [LIST OF NUMPY NDARRAY]
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Cannot break a non np.array image")
        return [img[j: j + 8, i: i + 8] for (j, i) in
                itertools.product(range(0, height, 8),
                                  range(0, width, 8))]

    def recomposeImage(self):
        """
        Inverse of method 'breakImageIntoBlocks':
        Recompose the image from the sorted list of blocks of pixels '__block_list_'.
        :return: Re-composed image [NUMPY NDARRAY]
        """
        if len(self.__block_list_) == 0:
            raise ValueError
        full_image = np.zeros(shape=(self.__height_, self.__width_),
                              dtype=np.uint8)  # Builds empty image
        for i in range(len(self.__block_list_)):  # Filling image
            curr_col_index = 8 * (i % (self.__width_ // 8))
            curr_line_index = 8 * (i // (self.__width_ // 8))
            full_image[
            curr_line_index:curr_line_index + 8,
            curr_col_index:curr_col_index + 8
            ] = self.__block_list_[i]
        return full_image

    # QUANTIZATION METHODS ============================================================

    @staticmethod
    def getQuantizedBlock(block):
        """
        Centers values of a block, runs it through DCT func. and quantizes it.
        :param block: 8x8 block of pixels [NUMPY NDARRAY]
        :return:Quantized 8x8 block of pixels [NUMPY NDARRAY]
        """
        img_block = np.subtract(block, 128)
        dct_block = cv2.dct(img_block.astype(np.float64))
        dct_block[0][0] /= settings.QUANTIZATION_TABLE[0][0]
        dct_block[0][0] = np.round(dct_block[0][0])
        return dct_block

    @staticmethod
    def getOriginalBlockFromQuantized(quantized_block):
        """
        Inverse of "getQuantizedBlock".
        :param quantized_block: Quantized 8x8 block of pixels [NUMPY NDARRAY]
        :return: Original 8x8 block of pixels [NUMPY NDARRAY]
        """
        dct_block = quantized_block
        dct_block[0][0] *= settings.QUANTIZATION_TABLE[0][0]
        unquantized_block = cv2.idct(dct_block)
        return np.add(unquantized_block, 128)

    # LENGTH EMBED/EXTRACT METHODS ====================================================

    def lengthToBinary(self):
        """
        Gives binary form of the length and adds a separator to that representation.
        :return: Binary representation of the length + separator to embed [LIST OF STR]
        """
        if self.__message_length_ % 8 != 0:
            raise ValueError("Message length is not multiple of 8")
        msg_length = int(
            self.__message_length_ / 8)  # Decimal representation of the length
        n_required_bits = msg_length.bit_length()
        tmp = f"0{n_required_bits}b"
        binary_length = format(msg_length, tmp)
        return list(binary_length) + utils.stringToBinary(
            settings.LENGTH_MSG_SEPARATOR)

    def embedMsglength(self):
        """
        Inserts the length of the message and end symbol (in binary form)
        at the beginning of the picture.
        At the end, '__block_list_' will be the image with embedded length,
        as list of blocks of pixels (from top left to bottom right) [LIST OF
                                                                    NUMPY NDARRAY]
        """
        mess_len_to_binary = self.lengthToBinary()
        for block_index, length_bit_to_embed in enumerate(mess_len_to_binary):
            quantized_block = self.getQuantizedBlock(
                self.__block_list_[block_index])
            if quantized_block[0][0] % 2 == 1 and int(length_bit_to_embed) == 0:
                quantized_block[0][0] -= 1
            elif quantized_block[0][0] % 2 == 0 and int(
                    length_bit_to_embed) == 1:
                quantized_block[0][0] += 1
            self.__block_list_[
                block_index] = self.getOriginalBlockFromQuantized(
                quantized_block)

    @staticmethod
    def extractMsglength(img, width):
        """
        Extracts the length (in bits) of the message embedded in the stegoimage.
        :param img: Stegoimage from which to extract the message [NUMPY NDARRAY]
        :param width: Width of the stegoimage [INT]
        :return: length of the message to extract from the stegoimage [INT]
        """
        block_index = 0
        separator_found = False
        decoded_length = ""
        while (not separator_found) and (
                block_index < settings.MAX_BITS_TO_ENCODE_LENGTH):
            block = utils.getBlockFromBlockIndex(img, width, block_index)
            unquantized_block = DCT.getQuantizedBlock(block)
            decoded_length += str(int(unquantized_block[0][0] % 2))
            if len(decoded_length) > 8:
                current_letter = str(chr(int(decoded_length[-8:], 2)))
                if current_letter == settings.LENGTH_MSG_SEPARATOR:
                    separator_found = True
            block_index += 1
        return int(''.join(decoded_length[:-8]), 2) * 8

    # ENCODE/DECODE MESSAGE METHODS ===================================================

    @staticmethod
    def getRandomBlocksFromMsgLength(seed, binary_msg_length, height, width):
        """
        Generates a random sequence of indices, (interpreted as the position
        of bits of the message to embed/extract in/from the cover/stegoimage.
        :param seed: Chosen seed [INT]
        :param binary_msg_length: length of the message to extract
                                  from the stegoimage [INT]
        :param height: Height of the cover/stegoimage [INT]
        :param width: Width of the cover/stegoimage [INT]
        :return: List of indeces [LIST OF INT]
        """
        tot_blocks = height * width // 64
        random.seed(seed)
        chosen_blocks_indices = random.sample(
            range(settings.MAX_BITS_TO_ENCODE_LENGTH, tot_blocks),
            binary_msg_length)
        return chosen_blocks_indices

    def encode_r(self, output_path, seed):
        """
        Embed message into cover image:
        1 - Embed the length.
        2 - Embed the message equally distributing bits according
            to a random sequence of indices
            and inserting them in the LSB of DCT quantized blocks.
        :param output_path: Path of stegoimage that will be generated
                            after message insertion [STR]
        :param seed: Chosen seed [INT]
        :return: Stegoimage [NUMPY NDARRAY]
        """
        y, cr, cb = utils.getYCrCbFromOriginalImg(self.__cover_image_)
        mess_len = len(self.__cipher_text_)
        positions_lst = self.getRandomBlocksFromMsgLength(seed, mess_len * 8,
                                                          self.__height_, self.__width_)
        self.__block_list_ = self.breakImageIntoBlocks(cb, self.__height_,
                                                       self.__width_)
        self.embedMsglength()
        for message_index, block_index in enumerate(positions_lst):
            block = self.__block_list_[block_index]
            dct_block = self.getQuantizedBlock(block)
            coeff = int(dct_block[0][0])

            message_bit = self.__bin_message_[message_index]
            if (coeff % 2) == 1 and int(message_bit) == 0:
                dct_block[0][0] -= 1
            elif (coeff % 2) == 0 and int(message_bit) == 1:
                dct_block[0][0] += 1

            self.__block_list_[
                block_index] = self.getOriginalBlockFromQuantized(
                dct_block)
        modified_cb = self.recomposeImage()
        final_img_standard_format = utils.getOriginalImgFromYCrCb(y,
                                                                  cr,
                                                                  modified_cb)
        cv2.imwrite(output_path, final_img_standard_format)
        return final_img_standard_format

    @staticmethod
    def decode_r(setego_img_path, seed):
        """
        Extract a message from a stegoimage:
        1 - Extract the length of the message.
        2 - Generate the random sequence of indices.
        3 - Extract the message.
        :param stego_img_path: Path of stegoimage
                               from which to extract the message [STR]
        :param seed: Chosen seed [INT]
        :return: Message hidden in the stegoimage [BYTE STR]
        """
        original_stego_img = utils.getImage(setego_img_path)
        _, _, cb = utils.getYCrCbFromOriginalImg(original_stego_img)
        height, width = original_stego_img.shape[:2]
        msg_length = DCT.extractMsglength(cb, width)
        positions_lst = DCT.getRandomBlocksFromMsgLength(
            seed, msg_length, height, width)
        decoded_msg = "0b"
        block_list = DCT.breakImageIntoBlocks(cb, height, width)
        for message_index, block_index in enumerate(positions_lst):
            block_index = positions_lst[message_index]
            block = block_list[block_index]
            dct_block = DCT.getQuantizedBlock(block)
            coeff = int(dct_block[0][0])
            decoded_msg += str(
                coeff % 2)  # Adding to the message the currently read bit
        return BitArray(decoded_msg).bytes


if __name__ == "__main__":
    from EncryptionUtils import encryptMessage, decryptMessage

    message = "HELLO THIS IS A LONG MESSAGE"
    encrypted = encryptMessage(message, "password")
    d = DCT("data/test_image.jpg", encrypted)
    encoded = d.encode_r("data/ycrcb_output.png", 20)
    decoded = DCT.decode_r("data/ycrcb_output.png", 20)
    decoded_message = decryptMessage(decoded, "password")
    print(decoded_message)
