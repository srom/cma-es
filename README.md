Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
--------------------------------------------------------

A TensorFlow 2 implementation.

## What is CMA-ES?

> The CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is an evolutionary algorithm for difficult non-linear non-convex black-box optimisation problems in continuous domain. It is considered as state-of-the-art in evolutionary computation and has been adopted as one of the standard tools for continuous optimisation in many (probably hundreds of) research labs and industrial environments around the world. 

[The CMA Evolution Strategy][1]

## Installation

The package is [available on PyPI](https://pypi.org/project/cma-es/) and can be installed with pip:

```sh
pip install cma-es
```

Alternatively, `cma-es` can also be installed from [conda-forge](https://anaconda.org/conda-forge/cma-es):

```sh
conda install -c conda-forge cma-es
```

## Example Usage

### 1. Define the fitness function

CMA attempts to __minimize__ a user-defined fitness function. 

Function signature:

```    
Args:
  x: tf.Tensor of shape (M, N)

Returns:
  Fitness evaluations: tf.Tensor of shape (M,)
```

Where `M` is the number of solutions to evaluate and `N` is the dimension of a single solution.

```python
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
```

![Figure1: Six-Hump Camel Function](six_hump_camel_fn.png?raw=true)

### 2. Configure CMA-ES

```python
from cma import CMA

cma = CMA(
    initial_solution=[1.5, -0.4],
    initial_step_size=1.0,
    fitness_function=fitness_fn,
)
```

The initial solution and initial step size (i.e. initial standard deviation of the search distribution) are problem specific.

The population size is automatically set by default, but it can be overidden by specifying the parameter `population_size`.

For bounded constraint optimization problems, the parameter `enforce_bounds` can be set, e.g. `enforce_bounds=[[-2, 2], [-1, 1]]` for a 2D function.

### 3. Run the optimizer

The search method runs until the maximum number of generation is reached or until one of the early termination criteria is met. By default, the maximum number of generations is 500.

```python
best_solution, best_fitness = cma.search()
```

The notebook [`Example 1 - Six Hump Camel Function`][4] goes into more details, including ways to plot the optimization path such as in the figure below.

![Figure 2: Optimization path](cma_trace.png?raw=true)

## Logging

A user-defined callback function can be specified to inspect variables during the search.

It is mainly intended for logging purpose, e.g:

```python
max_epochs = 500

def logging_function(cma, logger):
    if cma.generation % 10 == 0:
        fitness = cma.best_fitness()
        logger.info(f'Generation {cma.generation} - fitness {fitness}')

    if cma.termination_criterion_met or cma.generation == max_epochs:
        sol = cma.best_solution()
        fitness = cma.best_fitness()
        logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')

cma = CMA(
    initial_solution=[1.5, -0.4],
    initial_step_size=1.0,
    fitness_function=fitness_fn,
    callback_function=logging_function,
)
cma.search(max_epochs)
```

Check out an example logging progress to TensorBoard: [tensorboard_example.py][6]

## Running on GPU

By virtue of using TensorFlow, CMA should make use of available GPUs without any code change. 

If you run into issues, check the official [TF documentation](https://www.tensorflow.org/guide/gpu).

## More examples

- Jupyter notebooks with examples are available:
  - [Example 1 - Six-Hump Camel Function][4]
  - [Example 2 - Schwefel Function][5]
  - [Example 3 - Logging to TensorBoard][6]
- Unit tests provide a few more examples: `cma/core_test.py`

## Resources

- [CMA-ES at Wikipedia][3]
- [The CMA Evolution Strategy][1]
- [The CMA Evolution Strategy: A Tutorial][2]

[1]: http://cma.gforge.inria.fr/
[2]: https://arxiv.org/abs/1604.00772
[3]: https://en.wikipedia.org/wiki/CMA-ES
[4]: https://nbviewer.jupyter.org/github/srom/cma-es/blob/master/notebook/Example%201%20-%20Six%20Hump%20Camel%20Function.ipynb
[5]: https://nbviewer.jupyter.org/github/srom/cma-es/blob/master/notebook/Example%202%20-%20Schwefel%20Function.ipynb
[6]: https://github.com/srom/cma-es/blob/master/notebook/tensorboard_example.py
