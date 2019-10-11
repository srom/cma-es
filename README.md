Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
--------------------------------------------------------

A Tensorflow v2 implementation.

## Example

### 1. Define the fitness function

The CMA class expects fitness functions with the following signature:

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

![Six-Hump Camel Function](six_hump_camel_fn.png?raw=true)

### 2. Configure CMA-ES

The initial solution and initial step size (i.e. initial standard deviation of the search normal distribution) are problem specific.

```python
cma = CMA(
    initial_solution=[1.5, -0.4],
    initial_step_size=1.0,
    fitness_function=fitness_fn,
)
```

### 3. Run the search

The search method runs until the maximum number of generation is reached or until one of the early termination criteria is met. By default, the maximum number of generations is 500.

```python
cma.search();
```

Retrieve the best solution and its fitness value:

```python
num_generations = cma.generation
best_solution = cma.best_solution()
best_fitness = cma.best_fitness()
```

The notebook `notebook/Example 1 - Six Hump Camel Function.ipynb` contains more details about this example, including ways to plot the evolution path of the algorithm such as in the figure below.

![Six-Hump Camel Function](cma_trace.png?raw=true)

## More examples

- Jupyter notebooks with examples are available in the `notebook/` folder.
- Unit tests also provide a few more examples: `cma/core_test.py`

## Resources

- The CMA Evolution Strategy - http://cma.gforge.inria.fr/
- The CMA Evolution Strategy: A Tutorial - https://arxiv.org/abs/1604.00772
