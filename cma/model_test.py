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

    def test_branin_fn(self):
        tf.random.set_seed(444)

        def fitness_fn(x):
            """
            Branin Function
            https://www.sfu.ca/~ssurjano/branin.html
            """
            a = 1.
            b = 5.1 / (4 * np.pi**2)
            c = 5. / np.pi
            r = 6.
            s = 10.
            t = 1 / (8 * np.pi)
            return (
                (a * (x[:,1] - b * x[:,0]**2 + c * x[:,0] - r))**2 +
                s * (1 - t) * tf.cos(x[:,0]) + s
            )

        cma = CMA(
            initial_solution=[3., 7.],
            initial_step_size=1.,
            fitness_function=fitness_fn
        )
        cma.search(100)

        x1, x2 = cma.best_solution()

        cond = (
            (
                np.isclose(x1, -np.pi, rtol=1e-3) and
                np.isclose(x2, 12.275, rtol=1e-3)
            ) or
            (
                np.isclose(x1, np.pi, rtol=1e-3) and
                np.isclose(x2, 2.275, rtol=1e-3)
            ) or
            (
                np.isclose(x1, 9.42478, rtol=1e-3) and
                np.isclose(x2, 2.475, rtol=1e-3)
            )
        )
        self.assertTrue(cond)


if __name__ == '__main__':
    unittest.main()
