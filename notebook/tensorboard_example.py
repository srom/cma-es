import logging
import os

import tensorflow as tf

if os.getcwd().split(os.sep)[-1] == 'notebook':
    os.chdir('..')

from cma import CMA


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s) %(message)s")

    tf.random.set_seed(123)

    max_epochs = 500

    log_dir = 'logs/griewank_function'
    summary_writer = tf.summary.create_file_writer(log_dir)

    def logging_function(cma, logger):
        fitness = cma.best_fitness()

        # Write best fitness to the tensorboard summary log
        with summary_writer.as_default():
            tf.summary.scalar('fitness', fitness, step=cma.generation)

        # Periodically log progress
        if cma.generation % 10 == 0:
            logger.info(f'Generation {cma.generation} - fitness {fitness}')

        if cma.termination_criterion_met or cma.generation == max_epochs:
            sol = cma.best_solution()
            logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')

    cma = CMA(
        initial_solution=[100.] * 10,
        initial_step_size=600.,
        fitness_function=fitness_fn,
        enforce_bounds=[[-600, 600]] * 10,
        callback_function=logging_function,
    )
    cma.search(max_epochs)


def fitness_fn(x):
    """
    Griewank Function
    https://www.sfu.ca/~ssurjano/griewank.html
    """
    dimension = tf.shape(x)[1].numpy()

    s, p = [], []
    for i in range(dimension):
        s.append(x[:,i]**2)
        p.append(tf.cos(x[:,i] / tf.sqrt(tf.cast(i, dtype=tf.float64) + 1)))

    return 1. + (1. / 4000) * tf.reduce_sum(s, axis=0) - tf.reduce_prod(p, axis=0)


if __name__ == '__main__':
    main()
