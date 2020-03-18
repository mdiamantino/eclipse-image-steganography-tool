from __future__ import print_function, unicode_literals

from eclipse.src.cover_image_builder import CoverImageBuilder
from eclipse.src.discrete_cosine_transform_tool import DCT
from eclipse.src.encryption_utils import encrypt_message, decrypt_message


def embed(original_image_path: str,
          stego_image_output_path: str,
          message: str,
          password: str,
          chosen_seed: int) -> str:
    """
    :param original_image_path: Original image path [STR]
    :param stego_image_output_path: Path of the output stegoimage [STR]
    :param message: Message to hide into the stegoimage [STR]
    :param password: Password to encrypt the message [STR]
    :param chosen_seed: A seed for the uniform bits distribution in the image [INT]
    :return: path to the cover image
    """
    cib = CoverImageBuilder(original_image_path)
    cib.build_cover_image()
    cover_image_path = cib.get_output_path()
    encrypted_message = encrypt_message(message, password)
    embedder = DCT(cover_image_path, encrypted_message)
    embedder.embed_msg(stego_image_output_path, chosen_seed)
    return cover_image_path


def extract(stego_img_path: str, password: str, chosen_seed: int) -> str:
    """
    Extract the hidden message from the stegoimage.
    :param stego_img_path: Path to the stego image [STR]
    :param password: Password the message was encrypted with [STR]
    :param chosen_seed: Chosen seed [INT]
    :return: Extracted and decrypted message [STR]
    """
    extracted_encrypted_message = DCT.extract_msg(stego_img_path, chosen_seed)
    decrypted_message = decrypt_message(extracted_encrypted_message, password)
    return decrypted_message
