# GolayTransmission.py

import numpy as np

import numpy.testing as npt
import unittest

class GolayTransmission:


    def __init__(self) -> None:
        pass

    def bytes_2_boolarray(self, bytes2ndarray: bytes) -> np.ndarray[bool]:

        output = np.ndarray((len(bytes2ndarray)*8), dtype=bool)

        index: int = 0

        for byte in bytes2ndarray:
            for i in range(8):
                output[index] = (byte<<i) & 0x80
                index += 1

        return output

    def _addpadding(self, input: np.ndarray[bool])->np.ndarray[bool]:
        modulo = (input.shape[0] % 12)
        if modulo == 0:
            return input

        x = np.concatenate((input, np.zeros((12-modulo, 1), dtype=bool)))
        return x
    
    def _charlist2boolarray(self, string: list[np.uint8])->np.ndarray[bool]:
        buffer = np.zeros((8*len(string), 1), dtype=bool)

        for i, char in enumerate(string):
            buffer[i*8:(i+1)*8, :] = self._char2boolarray(char)

        return buffer
    
    def _char2boolarray(self, char: np.uint8)->np.ndarray[bool]:
        buffer = np.zeros((8, 1), dtype=bool)

        for j in range(8):
            buffer[7 - j, :] = (char >> j) & 1

        return buffer

    def _boolarray_to_12bit_words(self, boolarray: np.ndarray[bool])->np.ndarray[np.ndarray[bool]]:
        padded_array : np.ndarray[bool]= self._addpadding(boolarray)
        n_rows = padded_array.shape[0] // 12
        output = padded_array.reshape((n_rows, 12))
        return output


class Test_TestGolayCoder(unittest.TestCase):

    def test_bytes_2_boolarray_OK(self):
        golay = GolayTransmission()
        test_input = bytes(b'stop')

        exp_output = np.array([0, 1, 1, 1, 0, 0, 1, 1,
                               0, 1, 1, 1, 0, 1, 0, 0, 
                               0, 1, 1, 0, 1, 1, 1, 1, 
                               0, 1, 1, 1, 0, 0, 0, 0],
                               dtype=bool)

        output = golay.bytes_2_boolarray(test_input)

        npt.assert_array_equal(
            output, exp_output
        )

    def test_char2boolarray_simple(self):
        golay = GolayTransmission()

        input_char = np.uint8(67)

        exp_output = np.array(
            [0, 1, 0, 0, 0, 0, 1, 1],
            dtype=bool,
        )
        exp_output = exp_output.reshape((8, 1))

        npt.assert_array_equal(
            golay._char2boolarray(input_char), exp_output
        )

    def test_charlist2boolarray_simple(self):
        golay = GolayTransmission()

        input_char = [np.uint8(67), np.uint8(3)]

        exp_output = np.array(
            [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            dtype=bool,
        )
        exp_output = exp_output.reshape((-1, 1))

        npt.assert_array_equal(
            golay._charlist2boolarray(input_char), exp_output
        )

    def test_addpadding_simple(self):
        golay = GolayTransmission()

        input_array = np.zeros((10, 1), dtype=bool)

        exp_output = np.zeros((12, 1))
        output = golay._addpadding(input_array)

        npt.assert_array_equal(
            output, exp_output
        )


    def test_boolarray2_12bit_words_simple(self):
        golay = GolayTransmission()

        input_array = np.zeros((26, 1), dtype=bool)
        input_array[12] = True

        exp_output = np.zeros((3, 12))
        exp_output[1, 0] = True

        npt.assert_array_equal(
            golay._boolarray_to_12bit_words(input_array), exp_output
        )


if __name__ == "__main__":
    unittest.main()
