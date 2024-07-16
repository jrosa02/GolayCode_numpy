# GF2.py

import numpy as np

import numpy.testing as npt
import unittest

class GF2:
    def __init__(self) -> None:
        pass

    def bit_vec_add(self, b1: int, b2: int):
        return b1^b2
    
    def bit_vec_mul(self, b1: int, b2: int):
        acc: bool = 0
        bitsum = self.bit_vec_add(b1, b2)
        while bitsum:
            acc ^= (bitsum & 1)
            bitsum  = bitsum >> 1
        return acc
    
    def bit_mat_mul(self, bm1: np.ndarray, bm2: np.ndarray):

        output: np.ndarray = np.ndarray(bm1.shape)
        if bm1.shape != bm2.shape:
            raise ValueError(f"Matrix shapes forbid bit_mat_mul, M1.shape:{bm1.shape} != M2.shape:{bm2.shape}")
        
        for i, _ in enumerate(bm1):
            output[i] = (output[i] << 1) | self.bit_vec_mul(bm1, bm2)

        return output

    def mat_add(
        self, 
        M1: np.ndarray[np.ndarray[bool]], 
        M2: np.ndarray[np.ndarray[bool]]
    ) -> np.ndarray[np.ndarray[bool]]:
        
        if M1.shape != M2.shape:
            raise ValueError(f"Matrix shapes differ M1.shape:{M1.shape} != M2.shape:{M2.shape}")
        
        return np.logical_xor(M1, M2, dtype=bool)

    def mat_mul(        
        self,
        M1: np.ndarray[np.ndarray[bool]], 
        M2: np.ndarray[np.ndarray[bool]]
    ) -> np.ndarray[np.ndarray[bool]]:
        
        if M1.shape[1] != M2.shape[0]:
            raise ValueError(f"Matrix shapes forbid matmul, M1.shape:{M1.shape} != M2.shape:{M2.shape}")
        
        M1 = M1.astype(int)
        M2 = M2.astype(int)

        return np.mod(M1 @ M2, 2).astype(bool)


class Test_TestGolayCoder(unittest.TestCase):

    def test_add_matrix_OK(self):
        gf2 = GF2()
        M1 = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0]], dtype=bool)

        M2 = np.array([[1, 1, 0, 0],
                       [0, 0, 0, 1],
                       [1, 0, 1, 0]], dtype=bool)

        expect_output = np.array([[1, 0, 1, 0], 
                                  [0, 1, 1, 1], 
                                  [0, 0, 0, 0]], dtype=bool)

        npt.assert_array_equal(
            gf2.mat_add(M1, M2),
            expect_output,
        )

    def test_add_matrix_incorrect_shapes(self):
        gf2 = GF2()
        M1 = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0]], dtype=bool)

        M2 = np.array([[1, 1, 0, 0],
                       [1, 0, 1, 0]], dtype=bool)

        with self.assertRaises(ValueError):
            gf2.mat_add(M1, M2)

    def test_mul_matrix_incorrect_shapes(self):
        gf2 = GF2()
        M1 = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0]], dtype=bool)

        M2 = np.array([[1, 1, 0, 0],
                       [1, 0, 1, 0]], dtype=bool)

        with self.assertRaises(ValueError):
            gf2.mat_mul(M1, M2)

    def test_mul_matrix_OK1(self):
        gf2 = GF2()
        M1 = np.array([[0, 1],
                       [0, 1]], dtype=bool)

        M2 = np.array([[1, 1],
                       [1, 0]], dtype=bool)
        
        exp_output = np.array([[1, 0],
                               [1, 0]], dtype=bool)
        npt.assert_array_equal(
            gf2.mat_mul(M1, M2),
            exp_output,
        )

    def test_mul_matrix_OK2(self):
        gf2 = GF2()
        M1 = np.array([[0, 1],
                       [1, 1]], dtype=bool)

        M2 = np.array([[1, 1],
                       [1, 1]], dtype=bool)
        
        exp_output = np.array([[1, 1],
                               [0, 0]], dtype=bool)
        
        npt.assert_array_equal(
            gf2.mat_mul(M1, M2),
            exp_output,
        )
    
    def test_bit_vec_add_OK(self):
        gf2 = GF2()
        b1 = 0b1010101
        b2 = 0b0110111

        exp_out = 0b1100010

        out = gf2.bit_vec_add(b1, b2)

        self.assertEqual(exp_out, out)

    def test_bit_vec_mul_OK(self):
        gf2 = GF2()
        b1 = 0b1010101
        b2 = 0b0110111

        exp_out = 0b1

        out = gf2.bit_vec_mul(b1, b2)

        self.assertEqual(exp_out, out)

    def test_bit_mat_mul_OK(self):
        gf2 = GF2()

        b1 = 0b1010101
        b2 = 0b0110111

        exp_out = 0b1

        out = gf2.bit_mat_mul(b1, b2)

        self.assertEqual(exp_out, out)

if __name__ == "__main__":
    unittest.main()
