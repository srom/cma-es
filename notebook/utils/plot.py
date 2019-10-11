import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import chi2


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
    y = lambda x1, x2: fitness_fn(np.array([[x1, x2]]))[0]

    A, B = np.meshgrid(a, b)
    zs = np.array([y(u, v) for u, v in zip(np.ravel(A), np.ravel(B))])
    Z = zs.reshape(A.shape)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    if mean is not None:
        ax.scatter3D(
            [mean[0]],
            [mean[1]],
            [y(mean[0], mean[1])],
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
            [y(solution[0], solution[1])],
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
    y = lambda x1, x2: fitness_fn(np.array([[x1, x2]]))[0]

    A, B = np.meshgrid(a, b)
    zs = np.array([y(u, v) for u, v in zip(np.ravel(A), np.ravel(B))])
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
            label = f'population (top {mu})'
        elif i == mu:
            label = f'population (bottom {mu})'
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
    Draw confidence ellipse representing the covariance matrix. See [1] for a walkthrough.

    [1] https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    """
    if not np.isscalar(confidence) or 0 > confidence > 1:
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


def plot_generations(generations, cma_trace, fitness_fn, xlim, ylim):
    num_rows = int(np.ceil(len(generations) / 2))
    f, axes = plt.subplots(num_rows, 2, figsize=(18, 8 * num_rows));
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(generations):
            ax.remove();
            break

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

        ax.set_title(f'Generation {generation}');

    return f, axes


def plot_mean_coordinates(means):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    generations = range(len(means))

    for i, ax in enumerate(axes):
        ax.plot(generations, means[:,i])
        ax.set_xlabel('Generation')
        ax.set_title(f'$x_{i+1}$')
        ax.grid(True)

    fig.suptitle('Evolution of the mean\n', fontsize='xx-large');

    return fig, axes


def angle_rad(u, v):
    """
    Counter-clockwise angle in radian between vectors u and v.
    """
    a, b = u / np.linalg.norm(u, 2), v / np.linalg.norm(v, 2)
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def angle_deg(u, v):
    """
    Counter-clockwise angle in degrees between vectors u and v.
    """
    angle = angle_rad(u, v) * (180 / np.pi)
    return angle
