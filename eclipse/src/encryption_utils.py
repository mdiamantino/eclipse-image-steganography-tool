__author__ = "Mark Diamantino Caribé"

from base64 import urlsafe_b64encode
from os import urandom
from zlib import compress, decompress

from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from eclipse.common.settings import SALT_LEN, DK_LEN, COUNT


def gen_salted_key_from_password(salt: bytes, password: str) -> bytes:
    """
    Generates the salted key given a password
    :param salt: Random generated salt [BYTE STR]
    :param password: Password from which key will be derived [STR]
    :return: Password derived key [STR]
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=DK_LEN,
        salt=salt,
        iterations=COUNT,
        backend=default_backend()
    )
    key = urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_message(message_to_encrypt: str, password: str) -> bytes:
    """
    Encrypts a message given a password.
    :param message_to_encrypt: Message to encrypt [STR]
    :param password: Password from which key will be derived [STR]
    :return: Salt concatenated to the cipher message [STR]
    """
    encoded_to_encrypt = message_to_encrypt.encode('utf-8')
    compressed_to_encrypt = compress(encoded_to_encrypt)
    salt = urandom(SALT_LEN)
    key = gen_salted_key_from_password(salt, password)
    cipher = Fernet(key).encrypt(compressed_to_encrypt)
    return salt + cipher


def decrypt_message(cipher_message: bytes, password: str) -> str:
    """
    Decrypts a cipher given the password.
    :param cipher_message: Salt concatenated to the cipher message [STR]
    :param password: Password from which key will be derived [STR]
    :return: Decrypted message [STR]
    """
    key = gen_salted_key_from_password(salt=cipher_message[:SALT_LEN],
                                       password=password)
    pt = Fernet(key).decrypt(cipher_message[SALT_LEN:])
    original_message = decompress(pt)
    return original_message.decode('utf-8')
