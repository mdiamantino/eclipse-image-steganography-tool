__author__ = "Mark Diamantino Caribé"

import zlib

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

from settings import SALT_LEN, DK_LEN, COUNT

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def getSaltedKeyFromPassword(salt, password):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=DK_LEN,
        salt=salt,
        iterations=COUNT,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

def encryptMessage(message_to_encrypt, password):
    encoded_to_encrypt = message_to_encrypt.encode('utf-8')
    compressed_to_encrypt = zlib.compress(encoded_to_encrypt)
    salt = os.urandom(SALT_LEN)
    key = getSaltedKeyFromPassword(salt, password.encode())
    cipher = Fernet(key).encrypt(compressed_to_encrypt)
    return salt + cipher


def decryptMessage(cipher_message, password):
    key = getSaltedKeyFromPassword(salt=cipher_message[:SALT_LEN],
                                   password=password.encode())
    pt = Fernet(key).decrypt(cipher_message[SALT_LEN:])
    original_message = zlib.decompress(pt)
    return original_message.decode('utf-8')
