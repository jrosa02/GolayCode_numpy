import unittest
import GolayCoder
import GolayTransmission


class Test_TestGolayTransmission(unittest.TestCase):

    def test_clean_transmission(self):
        glt = GolayTransmission.GolayTransmission()
        gl = GolayCoder.GolayCoder_np()

        with open("text.txt", "rb") as input_file:
            bytestring = input_file.read()

        bitarray = glt.bytes_2_boolarray(bytestring)

        bitarray = glt._addpadding(bitarray)

        wordarray = glt._boolarray_to_12bit_words(bitarray)

        codewords = gl.encode_wordarray(wordarray)

        decoded = gl.
        
            


if __name__ == "__main__":
    unittest.main()