import itertools
import random
import warnings

import cv2
import numpy as np
from bitstring import BitArray

import settings
import utils


# TODO: Simply methods
# TODO: Add comments
# TODO: Modify docstrings

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

    # VERIFICATION METHODS ===============================================================

    def verifyCiphertextSize(self):
        """
        Verifies that the length of the message to hide
        is shorter than the maximum available space in the image.
        Warning is raised if the message length is > (10% of the available capacity).
        """
        tot_blocks = self.__height_ * self.__width_ // 64  # 64 -> since each quantized block is a matrix of 8x8
        if self.__message_length_ > tot_blocks:
            raise OverflowError("Cannot embed. Message is too long!")
        elif self.__message_length_ > tot_blocks / 10:
            purcentage_of_occupied_storage = round(self.__message_length_ / tot_blocks * 100)
            warning = f"Message occupies ≈ {purcentage_of_occupied_storage}% of the pic. " \
                      "A smaller text is preferred (< 10%)"
            warnings.warn(warning)

    def verifyAndApplyPadding(self):
        """
        Checks and eventually resizes image applying a padding
        if any side length is not a multiple of 8.
        The original image is eventually replaced by the padded (with sides multiple of 8) image.
        """
        original_height, original_width = self.__cover_image_.shape[:2]
        if original_height % 8 != 0 or original_width % 8 != 0:
            self.__cover_image_ = cv2.resize(self.__cover_image_, (
                original_width + (8 - original_width % 8),
                original_height + (8 - original_height % 8)))
            cv2.imwrite(self.__cover_image_path_, self.__cover_image_)

    # BREAK/RECOMPOSE METHODS ============================================================

    def breakImageIntoBlocks(self, img, height, width):
        """
        Breaks the coverimage into a sequence of 8x8 blocks,
        from top left to bottom right.
        :param img: Coverimage to break into n 8x8 blocks.
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

    # QUANTIZATION METHODS ===============================================================

    @staticmethod
    def getQuantizedBlock(block):
        """
        Centers values of a block, runs it through DCT func and quantizes it.
        :param block: 8x8 block of pixels [NUMPY NDARRAY]
        :return:Quantized 8x8 block of pixels [NUMPY NDARRAY]
        """
        img_block = (np.subtract(block, 128))
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

    # LENGTH EMBED/EXTRACT METHODS =======================================================

    def lengthToBinary(self):
        """
        Gives binary form of the length and adds a separator to it.
        :return: Bits of the length + separator to embed [LIST OF STR]
        """
        assert self.__message_length_ % 8 == 0
        msg_length = int(self.__message_length_ / 8)
        n_required_bits = msg_length.bit_length()
        if n_required_bits % 8 != 0:
            n_required_bits = (1 + (n_required_bits // 8)) * 8
        tmp = "0{}b".format(n_required_bits)
        binary_length = format(msg_length, tmp)
        return list(binary_length) + utils.stringToBinary(settings.LENGTH_MSG_SEPARATOR)

    def embedMsglength(self):
        """
        Inserts the length of the message and end symbol (in binary form)
        at the beginning of the picture.
        :return: Image with embedded length, as list of blocks of pixels
        (from top left to bottom right) [LIST OF NUMPY NDARRAY]
        """
        mess_len_to_binary = self.lengthToBinary()
        for block_index in range(len(mess_len_to_binary)):
            quantized_block = self.getQuantizedBlock(
                self.__block_list_[block_index])
            length_bit_to_embed = mess_len_to_binary[block_index]
            if quantized_block[0][0] % 2 == 1 and int(length_bit_to_embed) == 0:
                quantized_block[0][0] -= 1
            elif quantized_block[0][0] % 2 == 0 and int(length_bit_to_embed) == 1:
                quantized_block[0][0] += 1
            self.__block_list_[block_index] = self.getOriginalBlockFromQuantized(
                quantized_block)
        return self.__block_list_

    @staticmethod
    def getBlockFromBlockIndex(img, width, block_index, block_size=8):
        i = 8 * (block_index % (width // block_size))
        j = 8 * (block_index // (width // block_size))
        return img[j:j + 8, i:i + 8]

    @staticmethod
    def extractMsglength(img, width):
        """
        Extracts the length (in bits) of the message embedded in the stegoimage.
        :param img: Stegoimage from which to extract the message [NUMPY NDARRAY]
        :param width: Width of the stegoimage [INT]
        :return: length of the message to extract from the stegoimage [INT]
        """
        block_index, curr_bit = 0, 0
        separator_found = False
        letter_buffer = ""
        message_len = list()
        while (not separator_found) and (block_index < settings.MAX_BITS_TO_ENCODE_LENGTH):
            block = DCT.getBlockFromBlockIndex(img, width, block_index)
            unquantized_block = DCT.getQuantizedBlock(block)
            letter_buffer += str(int(unquantized_block[0][0] % 2))

            curr_bit += 1
            if curr_bit == 8:
                current_letter = str(chr(int(letter_buffer, 2)))
                curr_bit = 0
                if current_letter == settings.LENGTH_MSG_SEPARATOR:
                    separator_found = True
                else:
                    message_len.append(letter_buffer)
                letter_buffer = ""
            block_index += 1
        binary_len = [item for sublist in message_len for item in sublist]
        return int(''.join(binary_len), 2) * 8

    # ENCODE/DECODE MESSAGE METHODS =======================================================

    @staticmethod
    def getRandomBlocksFromMsglength(seed, binary_msg_length, height, width):
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
        chosen_blocks_indices = random.sample(range(settings.MAX_BITS_TO_ENCODE_LENGTH, tot_blocks),
                                              binary_msg_length)
        return chosen_blocks_indices

    def encode_r(self, output_path, seed):
        """
        Embed a message into a cover image:
        1 - Embed the length.
        2 - Embed the message equally distributing bits according
            to a random sequence of indeces
            and inserting them in the LSB of DCT quantized blocks.
        :param output_path: Path of stegoimage that will be generated
                            after message insertion [STR]
        :param seed: Chosen seed [INT]
        :return: Stegoimage [NUMPY NDARRAY]
        """
        img = self.__cover_image_
        YCrCbImage = cv2.cvtColor(self.__cover_image_, cv2.COLOR_BGR2YCR_CB)
        y, cr, img = cv2.split(YCrCbImage)
        mess_len = len(self.__cipher_text_)
        dic = self.getRandomBlocksFromMsglength(seed, mess_len * 8,
                                                self.__height_, self.__width_)
        self.__block_list_ = self.breakImageIntoBlocks(img, self.__height_,
                                                       self.__width_)
        self.__block_list_ = self.embedMsglength()
        for message_index in range(len(dic)):
            block_index = dic[message_index]
            block = self.__block_list_[block_index]
            dct_block = self.getQuantizedBlock(block)
            coeff = int(dct_block[0][0])

            message_bit = self.__bin_message_[message_index]
            if (coeff % 2) == 1 and int(message_bit) == 0:
                dct_block[0][0] -= 1
            elif (coeff % 2) == 0 and int(message_bit) == 1:
                dct_block[0][0] += 1

            self.__block_list_[block_index] = self.getOriginalBlockFromQuantized(
                dct_block)
        rgb_final_img = self.recomposeImage()
        final_img = cv2.merge((y, cr, rgb_final_img))

        cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_YCR_CB2BGR))
        return cv2.cvtColor(final_img, cv2.COLOR_YCR_CB2BGR)

    def decode_r(self, original_stego_img, seed):
        """
        Extract a message from a stegoimage:
        1 - Extract the length of the message.
        2 - Generate the random sequence of indeces.
        3 - Extract the message.
        :param stego_img_path: Path of stegoimage
                               from which to extract the message [STR]
        :param seed: Chosen seed [INT]
        :return: Message hiddent in the stegoimage [STR]
        """
        # original_stego_img = cv2.imread(stego_img_path, cv2.IMREAD_GRAYSCALE)

        height, width = original_stego_img.shape[:2]
        YCrCbImage = cv2.cvtColor(original_stego_img, cv2.COLOR_BGR2YCR_CB)
        y, cr, img = cv2.split(YCrCbImage)
        msg_length = DCT.extractMsglength(img, width)
        dic = DCT.getRandomBlocksFromMsglength(seed, msg_length, height, width)
        decoded_msg = "0b"
        block_list = self.breakImageIntoBlocks(img, height, width)
        for message_index in range(len(dic)):
            block_index = dic[message_index]
            block = block_list[block_index]
            dct_block = self.getQuantizedBlock(block)
            coeff = int(dct_block[0][0])
            decoded_msg += str(coeff % 2)  # Adding to the message the currently read bit
        return BitArray(decoded_msg).bytes


if __name__ == "__main__":
    from EncryptionUtils import encryptMessage, decryptMessage

    message = "HELLO THIS IS A LONG MESSAGE"
    encrypted = encryptMessage(message, "password")
    d = DCT("data/test_image.jpg", encrypted)
    encoded = d.encode_r("data/ycrcb_output.jpg", 20)
    decoded = d.decode_r(encoded, 20)
    decoded_message = decryptMessage(decoded, "password")
    print(decoded_message)
