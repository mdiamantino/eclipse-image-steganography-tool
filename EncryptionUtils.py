__author__ = "Mark Diamantino Caribé"
import zlib

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

from settings import SALT_LEN, DK_LEN, COUNT


def getSaltedKeyFromPassword(password):
    salt = get_random_bytes(SALT_LEN)
    key = PBKDF2(password, salt, dkLen=DK_LEN, count=COUNT)
    return salt, key


def getKeyFromSaltAndPassword(salt, password):
    key = PBKDF2(password, salt, dkLen=DK_LEN, count=COUNT)
    return key


def encryptMessage(message_to_encrypt, password):
    encoded_to_encrypt = message_to_encrypt.encode('utf-8')
    compressed_to_encrypt = zlib.compress(encoded_to_encrypt)
    salt, key = getSaltedKeyFromPassword(password)
    encoder = AES.new(key, AES.MODE_CBC)
    ct_bytes = encoder.encrypt(pad(compressed_to_encrypt, AES.block_size))
    return salt + encoder.iv + ct_bytes


def decryptMessage(cipher_message, password):
    key = getKeyFromSaltAndPassword(salt=cipher_message[:SALT_LEN], password=password)
    iv = cipher_message[SALT_LEN:SALT_LEN + 16]
    message_to_decrypt = cipher_message[SALT_LEN + 16:]
    decoder = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(decoder.decrypt(message_to_decrypt), AES.block_size)
    original_message = zlib.decompress(pt)
    return original_message.decode('utf-8')
