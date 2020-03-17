import random
import string
from unittest import TestCase

from EncryptionUtils import encryptMessage, decryptMessage


def randomString(stringLength):
    letters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(letters) for _ in range(stringLength))

class Test(TestCase):
    @staticmethod
    def testApplication(num_tests):
        for _ in range(num_tests):
            message_to_encrypt = randomString(random.randint(1, 1000))
            password = randomString(random.randint(1, 40))
            res = encryptMessage(message_to_encrypt, password)
            assert message_to_encrypt == decryptMessage(res, password)

    def test_encrypt_message(self):
        self.testApplication(1)
