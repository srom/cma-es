import unittest

import numpy as np
import tensorflow as tf

from .model import CMA


class TestCMA(unittest.TestCase):

    def test_six_hump_camel_fn(self):
        tf.random.set_seed(444)

        def fitness_fn(x):
            """
            Six-Hump Camel Function
            https://www.sfu.ca/~ssurjano/camel6.html
            """
            return (
                (4 - 2.1 * x[:,0]**2 + x[:,0]**4 / 3) * x[:,0]**2 +
                x[:,0] * x[:,1] +
                (-4 + 4 * x[:,1]**2) * x[:,1]**2
            )

        cma = CMA(
            initial_solution=[1.5, 2.4],
            initial_step_size=0.5,
            fitness_function=fitness_fn
        )
        cma.search(100)

        x1, x2 = cma.best_solution()

        cond = (
            (
                np.isclose(x1, 0.0898, rtol=1e-3) and
                np.isclose(x2, -0.7126, rtol=1e-3)
            ) or
            (
                np.isclose(x1, -0.0898, rtol=1e-3) and
                np.isclose(x2, 0.7126, rtol=1e-3)
            )
        )
        self.assertTrue(cond)


if __name__ == '__main__':
    unittest.main()
