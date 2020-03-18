from __future__ import print_function, unicode_literals

from PyInquirer import prompt

import eclipse.common.cli_questions as cli_q
from eclipse.common.utils import shredTraces
from eclipse.src.cover_image_builder import CoverImageBuilder
from eclipse.src.discrete_cosine_transform_tool import DCT
from eclipse.src.encryption_utils import encrypt_message, decrypt_message


def extract(stego_img_path: str, password: str, seed: int) -> str:
    """
    Extract the hidden message from the stegoimage.
    :param stego_img_path: Path to the stego image [STR]
    :param password: Password the message was encrypted with [STR]
    :param seed: Chosen seed [INT]
    :return: Extracted and decrypted message [STR]
    """
    extracted_encrypted_message = DCT.extract_msg(stego_img_path, seed)
    decrypted_message = decrypt_message(extracted_encrypted_message, password)
    return decrypted_message


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


def command_line_interface_main():
    # Main operations
    operation_answer = prompt(cli_q.operation_questions)
    if operation_answer["operation"] == "Embed message":

        # Embedding message
        extract_answers = prompt(cli_q.embed_questions)
        cover_image_path = embed(
            original_image_path=extract_answers['original_image_path'],
            stego_image_output_path=extract_answers['stego_image_output_path'],
            message=extract_answers['message_to_hide'],
            password=extract_answers['password'],
            chosen_seed=extract_answers['seed'])
        print("[*] Message successfully hidden in the image")

        # Deleting
        shred_answers = prompt(cli_q.shred_questions)
        if shred_answers['shred_original_image']:
            shredTraces(path_of_file_to_delete=extract_answers['original_image_path'])
            print('[*] Original image successfully deleted')
        if shred_answers['shred_cover_image']:
            shredTraces(path_of_file_to_delete=cover_image_path)
            print('[*] Cover image successfully deleted')
    else:
        # Extracting
        extract_answers = prompt(cli_q.extract_questions)
        message = extract(stego_img_path=extract_answers['stego_img_path'],
                          password=extract_answers['password'],
                          seed=extract_answers['seed'])
        print("[*] Extracted hidden message: '%s'" % message)

        if prompt(cli_q.save_message_question)['save_message']:
            msg_path = prompt(cli_q.save_path_question)['message_path']
            with open(msg_path, 'w') as message_file:
                message_file.write(message)
        if prompt(cli_q.delete_stego_image_question)['shred_stego_image']:
            # Deleting
            shredTraces(extract_answers['stego_img_path'])
            print('[*] Stego image successfully deleted')
