import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import chi2
import tensorflow as tf


def plot_3d_surface(
    fitness_fn,
    xlim,
    ylim,
    zlim=None,
    view_init=None,
    mean=None,
    solutions=None,
    show_axes=True,
    fig=None,
    ax=None,
    figsize=(15, 8),
):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    a = np.linspace(*xlim, 100)
    b = np.linspace(*ylim, 100)

    A, B = np.meshgrid(a, b)
    grid_values = tf.convert_to_tensor([[u, v] for u, v in zip(np.ravel(A), np.ravel(B))])
    zs = fitness_fn(grid_values).numpy()
    Z = zs.reshape(A.shape)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    if mean is not None:
        ax.scatter3D(
            [mean[0]],
            [mean[1]],
            [fitness_fn(tf.convert_to_tensor([mean])).numpy()[0]],
            depthshade=False,
            marker='+',
            color='red',
            s=50,
        )

    if solutions is None:
        solutions = []

    for i, solution in enumerate(solutions):
        ax.scatter3D(
            [solution[0]],
            [solution[1]],
            [fitness_fn(tf.convert_to_tensor([solution])).numpy()[0]],
            depthshade=False,
            marker='o',
            color='red' if (i+1) <= len(solutions) / 2 else 'grey',
            s=30,
        )

    ax.plot_surface(A, B, Z, cmap='cool', alpha=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)

    if zlim is not None:
        ax.set_zlim(zlim)

    if view_init is not None:
        ax.view_init(*view_init)

    if not show_axes:
        plt.axis('off')

    return fig, ax


def plot_2d_contour(
    fitness_fn,
    xlim,
    ylim,
    mean=None,
    solutions=None,
    levels=25,
    show_axes=True,
    show_color_scale=True,
    fig=None,
    ax=None,
    figsize=(15, 8),
):
    if fig is None:
        fig = plt.figure(figsize=figsize)

    if ax is None:
        ax = fig.add_subplot(111)

    a = np.linspace(*xlim, 100)
    b = np.linspace(*ylim, 100)

    A, B = np.meshgrid(a, b)
    grid_values = tf.convert_to_tensor([[u, v] for u, v in zip(np.ravel(A), np.ravel(B))])
    zs = fitness_fn(grid_values).numpy()
    Z = zs.reshape(A.shape)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    if mean is not None:
        ax.plot(
            mean[0],
            mean[1],
            marker='+',
            color='black',
            markersize=12,
            linestyle='None',
            label='mean',
        )

    if solutions is None:
        solutions = []

    mu = int(np.floor(len(solutions) / 2))
    for i, solution in enumerate(solutions):
        if i == 0:
            label = 'population (selected)'
        elif i == mu:
            label = 'population (discarded)'
        else:
            label = None

        ax.plot(
            solution[0],
            solution[1],
            marker='o',
            color='white',
            markersize=8 if (i+1) <= len(solutions) / 2 else 5,
            linestyle='None',
            markeredgecolor='grey' if (i+1) <= len(solutions) / 2 else None,
            label=label
        )

    cs = ax.contourf(A, B, Z, levels=levels, cmap='cool')

    if fig is not None and show_color_scale:
        fig.colorbar(cs, ax=ax)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(False)

    if mean is not None or len(solutions) > 0:
        ax.legend()

    if not show_axes:
        plt.axis('off')

    return fig, ax


def draw_confidence_ellipse(
    ax,
    mean,
    eigenvectors,
    eigenvalues,
    confidence=0.95,
    facecolor='None',
    edgecolor='black',
    **kwargs,
):
    """
    Draw a covariance error ellipse, i.e. an iso-contour of the multivariate normal distribution.

    A 95% confidence ellipse (default) shows where 95% of sampled points will fall.

    Ref: https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    """
    if not np.isscalar(confidence) or confidence <= 0 or confidence >= 1:
        raise ValueError('Confidence must be a number between 0 and 1')

    chi2_val = chi2.isf(q=1. - confidence, df=2)

    width = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height = 2 * np.sqrt(chi2_val * eigenvalues[1])

    # Counter clockwise angle in degrees between the y-axis and the
    # second principal axis of the covariance matrix.
    # Note: the angle between the x-axis and the first principal axis is the same,
    # thus angle_deg([1, 0], eigenvectors[0]) is equivalent.
    angle = angle_deg([0, 1], eigenvectors[1])

    ellipse = Ellipse(
        xy=(mean[0], mean[1]),
        width=width,
        height=height,
        angle=angle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )
    ax.add_patch(ellipse);


def plot_generations(generations, cma_trace, fitness_fn, xlim, ylim, num_columns=3):
    num_rows = int(np.ceil(len(generations) / num_columns))
    f, axes = plt.subplots(
        num_rows,
        num_columns,
        sharex=True,
        sharey=True,
        figsize=(16, 5 * num_rows),
    )
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(generations):
            ax.remove()
            continue

        generation = generations[i]
        trace = cma_trace[generation]
        m = trace['m']
        B = trace['B']
        l = trace['σ']**2 * np.diagonal(trace['D'])**2
        population = trace['population']

        plot_2d_contour(
            fitness_fn,
            xlim=xlim,
            ylim=ylim,
            mean=m,
            solutions=population,
            show_color_scale=False,
            fig=f,
            ax=ax,
        );

        draw_confidence_ellipse(
            ax,
            mean=m,
            eigenvectors=B,
            eigenvalues=l,
            confidence=0.95,
        )

        if i > 0:
            ax.get_legend().remove()

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.get_xaxis().set_major_formatter(FormatStrFormatter('%.2f'))
        ax.get_yaxis().set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(f'Generation {generation}')

    return f, axes


def plot_mean_coordinates(trace, num_columns=2, figsize=(15, 6)):
    means = np.vstack([t['m'] for t in trace])
    generations = range(len(means))

    num_rows = int(np.ceil(means.shape[1] / num_columns))
    _fig_size = (figsize[0], figsize[1] * num_rows)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=_fig_size)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= means.shape[1]:
            ax.remove()
            continue

        ax.plot(generations, means[:,i])
        ax.set_xlabel('Generation')
        ax.set_title(f'$X_{i+1}$')
        ax.grid(True)

    fig.suptitle('Evolution of the mean\n', fontsize='x-large');

    return fig, axes


def angle_rad(u, v):
    """
    Counter-clockwise angle in radian between vectors u and v.
    """
    a = u / np.linalg.norm(u, 2)
    b = v / np.linalg.norm(v, 2)
    return np.arctan2(
        a[0] * b[1] - a[1] * b[0],
        a[0] * b[0] + a[1] * b[1],
    )


def angle_deg(u, v):
    """
    Counter-clockwise angle in degrees between vectors u and v.
    """
    return angle_rad(u, v) * (180 / np.pi)
