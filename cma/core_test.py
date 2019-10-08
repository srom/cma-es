import unittest

import numpy as np
import tensorflow as tf

from .core import CMA


class TestCMA(unittest.TestCase):

    def setUp(self, *args, **kwargs):
        super().setUp(*args, **kwargs)

        tf.keras.backend.clear_session()
        tf.random.set_seed(444)

    def test_six_hump_camel_fn(self):
        num_max_epochs = 100

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
        cma.search(num_max_epochs)

        x1, x2 = cma.best_solution()

        # Assert global minimum has been reached
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

        # Early stopping occured
        self.assertTrue(cma.generation < num_max_epochs)

    def test_branin_fn(self):
        num_max_epochs = 100

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
            initial_solution=[-2., 7.],
            initial_step_size=1.,
            fitness_function=fitness_fn
        )
        cma.search(num_max_epochs)

        x1, x2 = cma.best_solution()

        # Assert global minimum has been reached
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

        # Early stopping occured
        self.assertTrue(cma.generation < num_max_epochs)

    def test_schwefel_fn(self):
        num_max_epochs = 100

        def fitness_fn(x):
            """
            Schwefel Function
            https://www.sfu.ca/~ssurjano/branin.html
            """
            dimension = tf.cast(tf.shape(x)[1], tf.float64)
            return 418.9829 * dimension - tf.reduce_sum(x * tf.sin(tf.sqrt(tf.abs(x))), axis=1)

        # NOTE: Fails if the initial solution is too far from the optimal solution
        # e.g. [400., 400., -400., 400.] fails to find the global minimum and
        # settles for (420.9687, 420.9687, -302.5249, 420.9687) instead
        cma = CMA(
            initial_solution=[400., 400., 400., 400.],
            initial_step_size=50.,
            fitness_function=fitness_fn,
            enforce_bounds=[[-500, 500]] * 4,
        )
        cma.search(num_max_epochs)

        x1, x2, x3, x4 = cma.best_solution()

        # Assert global minimum has been reached
        cond = (
            np.isclose(x1, 420.9687, rtol=1e-3) and
            np.isclose(x2, 420.9687, rtol=1e-3) and
            np.isclose(x3, 420.9687, rtol=1e-3) and
            np.isclose(x4, 420.9687, rtol=1e-3)
        )
        self.assertTrue(cond)


if __name__ == '__main__':
    unittest.main()
