import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class CMA(object):
    """
    TensorFlow 2.0 implementation of the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    This implementation is essentially following "The CMA Evolution Strategy: A Tutorial",
    https://arxiv.org/abs/1604.00772
    """

    def __init__(
        self,
        initial_solution,
        initial_step_size,
        fitness_function,
    ):
        if not isinstance(initial_solution, (np.ndarray, list)):
            raise ValueError('Initial solution must be a list or numpy array')
        elif np.ndim(initial_solution) != 1:
            ndim = np.ndim(initial_solution)
            raise ValueError(f'Initial solution must be a 1D array but got an array of dim {ndim}')
        elif not np.isscalar(initial_step_size) or initial_step_size <= 0:
            raise ValueError(f'Initial step size must be a number greater than zero')
        elif not callable(fitness_function):
            raise ValueError(f'Fitness function must be callable')

        self.initial_solution = initial_solution
        self.dimension = len(initial_solution)
        self.initial_step_size = initial_step_size
        self.fitness_fn = fitness_function

        self.initialized = False
        self.generation = 0

    def init(self):
        # ------------------------
        # Non-trainable parameters
        # ------------------------
        # Solution dimension
        self.N = tf.constant(self.dimension, dtype=tf.float64)
        # Population size
        self.λ = tf.floor(tf.math.log(self.N) * 3 + 8)
        # Number of surviving individuals from one generation to the next
        self.μ = tf.floor(self.λ / 2)
        # Recombination weights
        self.weights = tf.concat([
            tf.math.log(self.μ + 0.5) - tf.math.log(tf.range(1, self.μ + 1)),
            tf.zeros(shape=(self.λ - self.μ,), dtype=tf.float64),
        ], axis=0)
        self.weights = (self.weights / tf.reduce_sum(self.weights))[:,tf.newaxis]
        # Variance-effective size of mu
        self.μeff = tf.reduce_sum(self.weights) ** 2 / tf.reduce_sum(self.weights ** 2)
        # Time constant for cumulation for C
        self.cc = (4 + self.μeff / self.N) / (self.N + 4 + 2 * self.μeff / self.N)
        # Time constant for cumulation for sigma control
        self.cσ = (self.μeff + 2) / (self.N + self.μeff + 5)
        # Learning rate for rank-one update of C
        self.c1 = 2 / ((self.N + 1.3)**2 + self.μeff)
        # Learning rate for rank-μ update of C
        self.cμ = 2 * (self.μeff - 2 + 1 / self.μeff) / ((self.N + 2)**2 + 2 * self.μeff / 2)
        # Damping for sigma
        self.damps = 1 + 2 * tf.maximum(0, tf.sqrt((self.μeff - 1) / (self.N + 1)) - 1) + self.cσ
        # Expectation of ||N(0,I)||
        self.chiN = tf.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))

        # --------------------
        # Trainable parameters
        # --------------------
        # Mean
        self.m = tf.Variable(tf.constant(self.initial_solution, dtype=tf.float64))
        # Step-size
        self.σ = tf.Variable(tf.constant(self.initial_step_size, dtype=tf.float64))
        # Covariance matrix
        self.C = tf.Variable(tf.eye(num_rows=self.N, dtype=tf.float64))
        # Evolution path for σ
        self.p_σ = tf.Variable(tf.zeros((self.N,), dtype=tf.float64))
        # Evolution path for C
        self.p_C = tf.Variable(tf.zeros((self.N,), dtype=tf.float64))

        self.initialized = True
        return self

    def search(self, max_num_epochs):
        if not self.initialized:
            self.init()

        for epoch in range(max_num_epochs):
            self.generation += 1

            # ----------------------------------------
            # (1) Sample a new population of solutions
            # ----------------------------------------
            population_dist = tfp.distributions.MultivariateNormalTriL(
                loc=self.m,
                scale_tril=tf.linalg.cholesky(tf.square(self.σ) * self.C)
            )
            x = population_dist.sample(tf.cast(self.λ, tf.int32))

            # ------------------------------------------------
            # (2) Selection and Recombination: Moving the Mean
            # ------------------------------------------------
            # Evaluate and sort solutions
            f_x = self.fitness_fn(x)
            x_sorted = tf.gather(x, tf.argsort(f_x))

            # The new mean is a weighted average of the top-μ solutions
            x_diff = (x_sorted - self.m)
            x_avg = tf.reduce_sum(tf.multiply(x_diff, self.weights), axis=0)
            self.m.assign_add(x_avg)

            # ----------------------------------
            # (3) Adapting the Covariance Matrix
            # ----------------------------------
            y = x_avg / self.σ

            # Udpdate evolution path for Rank-one-Update
            p_C = (
                (1 - self.cc) * self.p_C +
                tf.sqrt(self.cc * (2 - self.cc) * self.μeff) * y
            )
            self.p_C.assign(p_C)

            # Compute Rank-μ-Update
            C_m = tf.map_fn(
                fn=lambda e: e * tf.transpose(e),
                elems=(x_diff / self.σ)[:,tf.newaxis],
            )
            y_s = tf.reduce_sum(
                tf.multiply(C_m, self.weights[:,tf.newaxis]),
                axis=0,
            )

            # Combine Rank-one-Update and Rank-μ-Update
            p_C_matrix = self.p_C[:,tf.newaxis]
            C = (
                (1 - self.c1 - self.cμ) * self.C +
                self.c1 * p_C_matrix * tf.transpose(p_C_matrix) +
                self.cμ * y_s
            )
            self.C.assign(C)

            # ---------------------
            # (4) Step-size control
            # ---------------------
            self.p_σ.assign((
                (1 - self.cσ) * self.p_σ +
                tf.sqrt(self.cσ * (2 - self.cσ) * self.μeff) * y
            ))
            sigma = self.σ * tf.exp((self.cσ / self.damps) * ((tf.norm(self.p_σ) / self.chiN) - 1))
            self.σ.assign(sigma)

            # Check termination criteria and terminate early if necessary
            if self.termination_criterion_met():
                break

        return self

    def best_solution(self):
        return self.m.read_value().numpy()

    def termination_criterion_met(self):
        return False
