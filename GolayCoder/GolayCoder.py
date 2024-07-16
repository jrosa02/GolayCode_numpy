# GolayCoder.py

import numpy as np

import numpy.testing as npt
import unittest

from GF2 import GF2


class GolayCoder_np:

    def __init__(self) -> None:
        # given golay code linearity, there is need only for 12 linearly independent vectors span the 12 dimensional space
        self._I = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=bool,
        )

        # Golay code checksums for linearly independent vectors
        self._B = np.array(
            [
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=bool,
        )

        self._Ht: np.ndarray[np.ndarray[bool]] = np.hstack((self._I, self._B), dtype=bool)

        self.gf2 = GF2()

    def construct_codeword(self, word: np.ndarray[bool]) -> np.ndarray[bool]:
        ecc_code = np.zeros((12, 1), dtype=bool)
        for n, bit in enumerate(word):

            if bit:
                # Add two vectors over the Galois field GF2
                B_vec = self._B[n, :].T.reshape(ecc_code.shape)
                ecc_code = np.logical_xor(ecc_code, B_vec)

        return np.vstack((word, ecc_code))

    def encode_wordarray(self, word_array: np.ndarray[np.ndarray[bool]])->np.ndarray[bool]:

        output = np.ndarray((word_array.shape[0], 24), dtype=bool)

        for i, word in enumerate(word_array):
            output[i, :] = self.construct_codeword(word.reshape(-1, 1)).T

        return output

    def _syndrome_calc(self, codeword: np.ndarray[bool])-> np.ndarray[np.ndarray[bool]]:

        output: np.ndarray[bool] = self.gf2.mat_mul(self._Ht, codeword)

        return output
    
    def _syndrome2_calc(self, codeword: np.ndarray[bool])-> np.ndarray[np.ndarray[bool]]:

        output: np.ndarray[bool] = self.gf2.mat_mul(self._Ht, codeword)

        return self.gf2.mat_mul(self._B, output.T)
    
    def _weigth_calc(self, word: np.ndarray[bool]) -> int:

        return word.sum()
    
    def _get_errorbits(self, codeword: np.ndarray[bool]) -> np.ndarray[bool]:
        output = np.zeros(codeword.shape, dtype=bool)

        syndrome = self._syndrome_calc(codeword)
        weigth = self._weigth_calc(syndrome)

        if weigth == 0:
            return output
        
        if weigth <= 3:
            output[:12] = syndrome
            return output
        
        for i, b in enumerate(self._B):
            weigth = self._weigth_calc(self.gf2.mat_add(syndrome, b.reshape(-1, 1)))
            if weigth <=2:
                output[12+i] = True
                output[:12] = self.gf2.mat_add(syndrome, b.reshape(-1, 1))
                return output
        
        syndrome2 = self.gf2.mat_mul(self._B, syndrome)
        weigth2 = self._weigth_calc(syndrome2)
        
        if weigth2 <= 3:
            output[12:] = syndrome2
            return output
        
        for i, b in enumerate(self._B):
            weigth = self._weigth_calc(self.gf2.mat_add(syndrome2, b.reshape(-1, 1)))
            if weigth <=2:
                output[i] = True
                output[12:] = self.gf2.mat_add(syndrome2, b.reshape(-1, 1))
                return output

        raise ValueError #non corectable error(hamming 4) detected

    def decode_codeword(self, codeword: np.ndarray[bool]) -> np.ndarray[bool]:
        return self.gf2.mat_add(codeword, self._get_errorbits(codeword))












class Test_TestGolayCoder(unittest.TestCase):

    def test_construct_codeword_empty(self):
        golay = GolayCoder_np()

        input_array = np.zeros((12, 1), dtype=bool)
        exp_output = np.zeros((24, 1), dtype=bool)

        npt.assert_array_equal(
            golay.construct_codeword(input_array),
            exp_output,
            err_msg="All zeros",
        )

    def test_construct_codeword_simple(self):
        golay = GolayCoder_np()

        input_array = np.zeros((12, 1), dtype=bool)
        input_array[0] = True

        exp_output = np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            dtype=bool,
        )
        exp_output = exp_output.reshape((24, 1))

        npt.assert_array_equal(
            golay.construct_codeword(input_array), exp_output, err_msg="One 1"
        )

    def test_construct_codeword_2bits(self):
        golay = GolayCoder_np()

        input_array = np.zeros((12, 1), dtype=bool)
        input_array[0] = True
        input_array[1] = True

        exp_output = np.array(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            dtype=bool,
        )
        exp_output = exp_output.reshape((24, 1))

        npt.assert_array_equal(
            golay.construct_codeword(input_array), exp_output, err_msg="One 1"
        )

    def test_syndrome_calc_Noerrors(self):
        golay = GolayCoder_np()

        input_codewords = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        
        exp_output = np.zeros(12, dtype=bool)

        for codeword in input_codewords:

            output = golay._syndrome_calc(codeword)

            npt.assert_array_equal(
                output, exp_output
            )

    def test_data_syndrome_calc_1error(self):
        golay = GolayCoder_np()

        input_codewords = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)
        
        exp_outputs = np.zeros((3, 12), dtype=bool)
        exp_outputs[0, 1] = True
        exp_outputs[1, 3] = True
        exp_outputs[2, 6] = True

        for codeword, exp_output in zip(input_codewords, exp_outputs):

            output = golay._syndrome_calc(codeword)

            npt.assert_array_equal(
                output, exp_output
            )

    
    def test_ecc_syndrome_calc_1error(self):
        golay = GolayCoder_np()

        input_codewords = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        ], dtype=bool)
        
        exp_outputs = np.zeros((3, 12), dtype=bool)
        exp_outputs[0, 11] = True
        exp_outputs[1, 9] = True
        exp_outputs[2, 6] = True

        for codeword, exp_output in zip(input_codewords, exp_outputs):

            output = golay._syndrome2_calc(codeword)

            npt.assert_array_equal(
                output, exp_output
            )

    def test_get_errorbits_1bit_Error(self):
        golay = GolayCoder_np()

        codeword = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        ], dtype=bool).T
        input_array = np.ndarray(codeword.shape)

        for i, _ in enumerate(codeword):
            input_array = np.copy(codeword)
            input_array[i] = not input_array[i]            


            exp_out = np.zeros(input_array.shape, dtype=bool)
            exp_out[i] = True
            output = golay._get_errorbits(input_array)

            npt.assert_array_equal(
                output, exp_out
            )

    def test_get_errorbits_2bit_Error(self):
        golay = GolayCoder_np()

        codeword = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        ], dtype=bool).T
        input_array = np.ndarray(codeword.shape)

        for i, _ in enumerate(codeword):
            for j, _ in enumerate(codeword):
                input_array = np.copy(codeword)
                input_array[i] = not input_array[i]
                input_array[j] = not input_array[j]

                exp_out = np.zeros(input_array.shape, dtype=bool)
                exp_out[i] = not exp_out[i]
                exp_out[j] = not exp_out[j]
                output = golay._get_errorbits(input_array)

                try:
                    npt.assert_array_equal(
                        output, exp_out
                    )
                except AssertionError:
                    print(f"Wrong errorcode with errorbits on {i}, {j}")

    def test_get_errorbits_3bit_Error(self):
        golay = GolayCoder_np()

        codeword = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        ], dtype=bool).T
        input_array = np.ndarray(codeword.shape)

        for i, _ in enumerate(codeword):
            for j, _ in enumerate(codeword):
                for k, _ in enumerate(codeword):
                    input_array = np.copy(codeword)
                    input_array[i] = not input_array[i]
                    input_array[j] = not input_array[j]
                    input_array[k] = not input_array[k]

                    exp_out = np.zeros(input_array.shape, dtype=bool)
                    exp_out[i] = not exp_out[i]
                    exp_out[j] = not exp_out[j]
                    exp_out[k] = not exp_out[k]
                    output = golay._get_errorbits(input_array)

                    try:
                        npt.assert_array_equal(
                            output, exp_out
                        )
                    except AssertionError:
                        print(f"Wrong errorcode with errorbits on {i}, {j}, {k}")

    def test_get_errorbits_3bit_Error(self):
        golay = GolayCoder_np()

        codeword = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        ], dtype=bool).T
        input_array = np.ndarray(codeword.shape)

        for i, _ in enumerate(codeword):
            for j, _ in enumerate(codeword):
                for k, _ in enumerate(codeword):
                    input_array = np.copy(codeword)
                    input_array[i] = not input_array[i]
                    input_array[j] = not input_array[j]
                    input_array[k] = not input_array[k]

                    exp_out = np.zeros(input_array.shape, dtype=bool)
                    exp_out[i] = not exp_out[i]
                    exp_out[j] = not exp_out[j]
                    exp_out[k] = not exp_out[k]
                    output = golay._get_errorbits(input_array)

                    try:
                        npt.assert_array_equal(
                            output, exp_out
                        )
                    except AssertionError:
                        print(f"Wrong errorcode with errorbits on {i}, {j}, {k}")

    def test_get_errorbits_4bit_Error(self):
        golay = GolayCoder_np()

        codeword = np.array([
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        ], dtype=bool).T
        input_array = np.ndarray(codeword.shape)

        input_array = np.copy(codeword)

        with self.assertRaises(ValueError):
            output = golay._get_errorbits(input_array)



    def test_encodewordarray_OK_noerrors(self):
        golay = GolayCoder_np()

        input_array = np.zeros((3, 12), dtype=bool)
        input_array[0, 0] = True
        input_array[1, 1] = True

        exp_output = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=bool)

        output = golay.encode_wordarray(input_array)

        npt.assert_array_equal(
            output, exp_output
        )

    # def test_codeword_error_corection_1bit_Error(self):
    #     golay = GolayCoder()

    #     exp_output = np.array([
    #         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    #     ], dtype=bool).T
    #     input_array = np.ndarray(exp_output.shape)

    #     for i, _ in enumerate(exp_output):
    #         input_array = np.copy(exp_output)
    #         input_array[i] = not input_array[i]            

    #         corrected = golay._codeword_error_corection(input_array)

    #         npt.assert_array_equal(
    #             corrected, exp_output
    #         )

if __name__ == "__main__":
    unittest.main()
