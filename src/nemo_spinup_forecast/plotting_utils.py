"""Plotting helpers for the Jumper notebook."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_snapshots(simus: Sequence, names: Sequence[str]):
    """
    Plot the first time step for each simulation.

    Parameters
    ----------
    simus : sequence
        Simulation objects.
    names : sequence of str
        Labels for each simulation.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, len(simus), figsize=(20, 4), squeeze=False)
    axes = axes.flatten()

    for ax, simu, name in zip(axes, simus, names, strict=True):
        if simu.z_size is not None:
            im = ax.pcolor(simu.simulation[0, 0])
            ax.set_title(f"Surface {name}")
        else:
            im = ax.pcolor(simu.simulation[0])
            ax.set_title(f"{name}")
        plt.colorbar(im, ax=ax)

    plt.show()
    return fig, axes


def plot_pca_diagnostics(simus: Sequence, names: Sequence[str], colors: Sequence[str]):
    """
    Plot explained variance ratios and leading components for each simulation.

    Parameters
    ----------
    simus : sequence
        Simulation objects.
    names : sequence of str
        Labels for each simulation.
    colors : sequence of str
        Colors to use for components.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(3, len(simus), figsize=(20, 10), squeeze=False)

    for i, simu in enumerate(simus):
        axes[0, i].plot(simu.pca.explained_variance_ratio_ * 100, "ko", markersize=4)
        axes[0, i].set_title(f"Explained Variance Ratio - {names[i]}")

        axes[1, i].plot(
            simu.components[:, 0], color=colors[i], alpha=0.9, label="1st comp"
        )
        axes[1, i].plot(
            simu.components[:, 1], color=colors[i], alpha=0.4, label="2nd comp"
        )
        axes[1, i].set_title(f"Components - {names[i]}")
        axes[1, i].legend()

        if simu.z_size is not None:
            im = axes[2, i].pcolor(simu.get_component(0)[0])
            plt.colorbar(im, ax=axes[2, i])
            axes[2, i].set_title(f"1st PC of the surface - {names[i]}")
        else:
            im = axes[2, i].pcolor(simu.get_component(0))
            plt.colorbar(im, ax=axes[2, i])
            axes[2, i].set_title(f"1st PC - {names[i]}")

    fig.suptitle("PCA INFO")
    plt.show()
    return fig, axes


def plot_rmse_depth_profile(
    values: Sequence[np.ndarray],
    depth,
    names: Sequence[str],
    colors: Sequence[str],
    title: str,
):
    """
    Plot RMSE error bars versus depth.

    Parameters
    ----------
    values : sequence of ndarray
        RMSE values for each variable.
    depth : array-like
        Depth coordinates (e.g., ``array.deptht``).
    names : sequence of str
        Labels for each variable.
    colors : sequence of str
        Colors for each variable.
    title : str
        Title for the plot.

    Returns
    -------
    tuple
        Matplotlib ``(fig, ax)``.
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    for i, (val, name, color) in enumerate(zip(values, names, colors, strict=True)):
        if i == 0:
            plt.errorbar(
                np.mean(val),
                depth[0],
                xerr=np.std(val),
                fmt=".",
                label=name,
                color=color,
                ecolor="grey",
            )
        else:
            plt.errorbar(
                np.mean(val, axis=1),
                depth,
                xerr=np.std(val, axis=1),
                fmt=".",
                label=name,
                color=color,
                ecolor="grey",
            )

    plt.title(title)
    plt.ylabel("Depth")
    plt.xlabel("RMSE")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
    return fig, ax


def plot_rmse_maps(maps: Sequence[np.ndarray], names: Sequence[str]):
    """
    Plot RMSE maps for each variable.

    Parameters
    ----------
    maps : sequence of ndarray
        RMSE maps for each variable.
    names : sequence of str
        Labels for each variable.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, len(maps), figsize=(20, 5), squeeze=False)
    axes = axes.flatten()

    for ax, rmse_map, name in zip(axes, maps, names, strict=True):
        if len(np.shape(rmse_map)) == 2:
            im = ax.pcolor(rmse_map)
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Mean rmse map - {name}")
        else:
            im = ax.pcolor(np.nanmean(rmse_map, axis=0))
            plt.colorbar(im, ax=ax)
            ax.set_title(f"Mean rmse map - {name}")

    plt.show()
    return fig, axes


def plot_reconstructions(maps: Sequence[np.ndarray], names: Sequence[str]):
    """
    Plot reconstructed fields for each variable.

    Parameters
    ----------
    maps : sequence of ndarray
        Reconstructed arrays (time, [z,] y, x).
    names : sequence of str
        Labels for each variable.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, len(maps), figsize=(20, 4), squeeze=False)
    axes = axes.flatten()

    for ax, simu, name in zip(axes, maps, names, strict=True):
        if len(np.shape(simu)) > 3:
            im = ax.pcolor(simu[0, 0])
            ax.set_title(f"Surface {name}")
        else:
            im = ax.pcolor(simu[0])
            ax.set_title(f"{name}")
        plt.colorbar(im, ax=ax)

    plt.show()
    return fig, axes


def plot_bar_with_errors(
    categories: Sequence[str],
    means: Sequence[float],
    errors: Sequence[float],
    title: str,
    ylabel: str,
):
    """
    Plot a bar chart with error bars.

    Parameters
    ----------
    categories : sequence
        Category labels.
    means : sequence
        Mean values for each category.
    errors : sequence
        Error values for each category.
    title : str
        Title for the plot.
    ylabel : str
        Y-axis label.

    Returns
    -------
    tuple
        Matplotlib ``(fig, ax)``.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(categories, means, yerr=errors, capsize=5, color=["tab:blue", "grey"])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Categories")
    plt.show()
    return fig, ax


def plot_depth_error_profiles(
    depth,
    mean_ref: Sequence[np.ndarray],
    std_ref: Sequence[np.ndarray],
    mean_pred: Sequence[np.ndarray],
    std_pred: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[str],
    title: str,
):
    """
    Plot absolute error profiles for multiple variables.

    Parameters
    ----------
    depth : array-like
        Depth coordinates.
    mean_ref : sequence of ndarray
        Mean reference errors per variable.
    std_ref : sequence of ndarray
        Standard deviation of reference errors per variable.
    mean_pred : sequence of ndarray
        Mean predicted errors per variable.
    std_pred : sequence of ndarray
        Standard deviation of predicted errors per variable.
    labels : sequence of str
        Labels for each variable.
    colors : sequence of str
        Colors for each variable.
    title : str
        Figure title.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 6), squeeze=False)
    axes = axes.flatten()

    for i, label in enumerate(labels):
        axes[i].plot(
            depth,
            mean_ref[i],
            color="black",
            label=label,
            linestyle="dashed",
            alpha=0.6,
        )
        axes[i].fill_between(
            depth,
            mean_ref[i] + std_ref[i],
            mean_ref[i] - std_ref[i],
            color="black",
            alpha=0.1,
        )

        axes[i].plot(depth, mean_pred[i], color=colors[i], label=label)
        axes[i].fill_between(
            depth,
            mean_pred[i] + std_pred[i],
            mean_pred[i] - std_pred[i],
            color=colors[i],
            alpha=0.2,
        )
        axes[i].set_xlabel("Depth")
        axes[i].set_ylabel("Mean Error")
        axes[i].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_depth_prediction_reference(
    depth,
    mean_pred: Sequence[np.ndarray],
    mean_ref: Sequence[np.ndarray],
    titles: Sequence[str],
):
    """
    Plot prediction vs reference profiles for multiple variables.

    Parameters
    ----------
    depth : array-like
        Depth coordinates.
    mean_pred : sequence of ndarray
        Predicted mean profiles per variable.
    mean_ref : sequence of ndarray
        Reference mean profiles per variable.
    titles : sequence of str
        Titles for each subplot.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(1, len(titles), figsize=(15, 4), squeeze=False)
    axes = axes.flatten()

    for ax, title, pred, ref in zip(axes, titles, mean_pred, mean_ref, strict=True):
        ax.plot(depth, pred, label="predictions")
        ax.plot(depth, ref, label="reference")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Average over depth")
    plt.show()
    return fig, axes


def plot_mean_profiles(
    mean_pred: Sequence[np.ndarray],
    mean_ref: Sequence[np.ndarray],
    names: Sequence[str],
    colors: Sequence[str],
):
    """
    Plot mean profiles for predictions and references.

    Parameters
    ----------
    mean_pred : sequence of ndarray
        Predicted mean profiles.
    mean_ref : sequence of ndarray
        Reference mean profiles.
    names : sequence of str
        Labels for each variable.
    colors : sequence of str
        Colors for each variable.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(len(mean_pred), 1, figsize=(10, 8), squeeze=False)
    axes = axes.flatten()
    for ax, pred, ref, name, color in zip(
        axes, mean_pred, mean_ref, names, colors, strict=True
    ):
        ax.plot(pred, color=color, label=name)
        ax.plot(ref, color="grey", label="ref", linestyle="dashed")
        ax.set_title(f"Mean profiles - {name}")
        ax.legend()

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_component_timeseries(
    ref,
    pred,
    names: Sequence[str],
    colors: Sequence[str],
    comp: int,
    total_len: int,
):
    """
    Plot component time series for reference and predicted data.

    Parameters
    ----------
    ref : sequence
        Reference Simulation.
    pred : sequence
        Predicted component DataFrames.
    names : sequence of str
        Labels for each variable.
    colors : sequence of str
        Colors for each variable.
    comp : int
        Component index to plot.
    total_len : int
        Length for the x-axis.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)``.
    """
    fig, axes = plt.subplots(len(ref), 1, figsize=(10, 8), squeeze=False)
    axes = axes.flatten()

    for ax, simu, pred_item, name, color in zip(
        axes, ref, pred, names, colors, strict=True
    ):
        ax.plot(simu.components[:, comp], color="grey", linestyle="dashed", label="ref")
        ax.plot(
            np.arange(0, total_len),
            pred_item.iloc[:, comp],
            color=color,
            alpha=0.9,
            label=name,
        )
        ax.set_title(f"Components - {name}")
        ax.legend()
    fig.suptitle("PCA INFO")
    plt.show()
    return fig, axes
