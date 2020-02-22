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
