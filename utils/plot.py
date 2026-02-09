import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict

# --- Private Utility Functions ---

def _as_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a detached numpy array."""
    return x.detach().float().cpu().numpy()

def _symmetric_clim(arr: np.ndarray, percentile: float = 99.0) -> Tuple[float, float]:
    """Calculate robust symmetric color limits around zero."""
    a = np.percentile(np.abs(arr), percentile)
    a = a if a > 0 else np.max(np.abs(arr)) + 1e-8
    return -a, a

def _setup_save(out_path: Optional[str]):
    """Ensure directory exists if out_path is provided."""
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

# --- Generic Plotting Primitives (The "Higher Layer") ---

def _plot_heatmap(
    arr: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    cbar_label: str,
    figsize: Tuple[float, float],
    cmap: str,
    vmin: float,
    vmax: float,
    out_path: Optional[str] = None,
    aspect: str = "auto",
    origin: str = "lower",
    yticklabels: Optional[List[str]] = None,
    xticklabels: Optional[List[str]] = None,
):
    """Generic heatmap plotting function."""
    _setup_save(out_path)
    plt.figure(figsize=figsize)
    
    plt.imshow(arr, aspect=aspect, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=cbar_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if yticklabels:
        plt.yticks(np.arange(len(yticklabels)), yticklabels)
    if xticklabels:
        plt.xticks(np.arange(len(xticklabels)), xticklabels)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()

def _plot_barchart(
    values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    figsize: Tuple[float, float],
    out_path: Optional[str] = None,
    xticklabels: Optional[List[str]] = None,
    xtick_rotation: int = 45,
    xtick_ha: str = "right",
    xtick_fontsize: int = 8
):
    """Generic bar chart plotting function."""
    _setup_save(out_path)
    plt.figure(figsize=figsize)
    
    xs = np.arange(len(values))
    plt.bar(xs, values)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if xticklabels:
        plt.xticks(xs, xticklabels, rotation=xtick_rotation, ha=xtick_ha, fontsize=xtick_fontsize)
    else:
        plt.xticks(xs, fontsize=xtick_fontsize)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()

def _plot_linechart(
    values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    figsize: Tuple[float, float],
    out_path: Optional[str] = None,
):
    """Generic line chart plotting function."""
    _setup_save(out_path)
    plt.figure(figsize=figsize)
    
    plt.plot(values)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()


# --- For Integrated Gradients / Saliency Maps ---

def plot_attr_heatmap(
    attr: torch.Tensor,
    title: str,
    signed: bool = True,
    feature_names: list[str] | None = None,
    time_axis_label: str = "time",
    out_path: str = None,
):
    """
    Plots attribution heatmap (T, F) by transposing and calling _plot_heatmap.
    attr: (T, F) tensor
    """
    arr = _as_numpy(attr)
    
    # Data and style preparation
    if signed:
        vmin, vmax = _symmetric_clim(arr, percentile=99.0)
        cmap = "bwr"
        plot_arr = arr.T
        cbar_label = "attribution"
    else:
        arr = np.abs(arr)
        vmin, vmax = 0, np.percentile(arr, 99.0)
        cmap = "viridis"
        plot_arr = arr.T
        cbar_label = "attribution (|.|)"
    
    yticklabels = None
    if feature_names is not None and len(feature_names) == arr.shape[1]:
        yticklabels = feature_names

    _plot_heatmap(
        arr=plot_arr,
        title=title,
        x_label=time_axis_label,
        y_label="feature",
        cbar_label=cbar_label,
        figsize=(8, 3.2),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        out_path=out_path,
        yticklabels=yticklabels
    )

def plot_time_profile(
    attr: torch.Tensor, 
    title: str, 
    signed: bool = False, 
    out_path: str = None,
    scroll_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
):
    """
    Plots mean attribution over features (T,).
    If scroll_xy (x_vec, y_vec) is provided, plots them on a secondary y-axis.
    """
    attr_profile = attr.abs().mean(dim=1) if not signed else attr.mean(dim=1)
    attr_profile_np = _as_numpy(attr_profile)
    
    _setup_save(out_path)
    fig, ax1 = plt.subplots(figsize=(10, 3))
    
    color = 'tab:blue'
    ax1.set_xlabel('time', fontdict={'fontsize': 'small'})
    y_label = "mean |attr|" if not signed else "mean attr"
    ax1.set_ylabel(y_label, color=color, fontdict={'fontsize': 'small'})
    ax1.plot(attr_profile_np, color=color, label=y_label)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0) # Attribution is >= 0

    ax1.grid(True, axis='y', linestyle=':', color='gray', alpha=0.7)
    ax1.grid(True, axis='x', linestyle=':', color='gray', alpha=0.7)

    # Plot scroll_xy on a secondary axis if provided
    if scroll_xy is not None:
        scroll_x_np = _as_numpy(scroll_xy[0])
        scroll_y_np = _as_numpy(scroll_xy[1])

        # Create secondary (right) axis
        ax2 = ax1.twinx()
        ax2.grid(False)

        color = 'tab:red'
        ax2.set_ylabel('Scroll Position', color=color, fontdict={'fontsize': 'small'})
        ax2.plot(scroll_x_np, color=color, linestyle='--', label='scroll x')
        ax2.plot(scroll_y_np, color='tab:green', linestyle=':', label='scroll y')
        ax2.tick_params(axis='y', labelcolor=color)

        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(
            lines1 + lines2, 
            labels1 + labels2, 
            loc='upper left', 
            bbox_to_anchor=(1.1, 1), 
            fontsize='x-small'
        )
    plt.title(title)
    fig.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_feature_profile(attr: torch.Tensor, title: str, y_label: str = "mean |attr|", out_path: str = None, feature_names: list[str] | None = None, signed: bool = False):
    """
    Plots mean attribution over time (F,) by calling _plot_barchart.
    """
    # Data preparation
    x = attr.abs().mean(dim=0) if not signed else attr.mean(dim=0)
    if not y_label:
        y_label = "mean |attr|" if not signed else "mean attr"
    
    xticklabels = None
    if feature_names is not None and len(feature_names) == len(x):
        xticklabels = feature_names

    _plot_barchart(
        values=_as_numpy(x),
        title=title,
        x_label="feature",
        y_label=y_label,
        figsize=(8, 2.2),
        out_path=out_path,
        xticklabels=xticklabels
    )


# --- For Attention Rollout / Attention Flow Maps ---

def plot_square_heatmap(
    mat: torch.Tensor,
    title: str,
    x_label: str,
    y_label: str,
    out_path: str | None = None,
    cmap: str = "viridis",
    vmax_percentile: float = 99.0,
):
    """
    Plots a square (N, N) heatmap by calling _plot_heatmap.
    """
    # Data and style preparation
    arr = _as_numpy(mat)
    vmin, vmax = 0.0, np.percentile(arr, vmax_percentile)
    vmax = max(vmax, 1e-12)  # avoid zero-range

    _plot_heatmap(
        arr=arr,
        title=title,
        x_label=x_label,
        y_label=y_label,
        cbar_label="rollout weight",
        figsize=(6, 5.2),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        out_path=out_path
    )


def plot_token_importance(
    mat: torch.Tensor,
    reduce: str,
    title: str,
    x_label: str,
    out_path: str | None = None,
    xticklabels: list[str] | None = None,
):
    """
    Builds a 1-D importance vector from a rollout matrix (N, N) and calls _plot_barchart.
    """
    # Data preparation
    arr = _as_numpy(mat)
    if reduce == "col":
        vec = arr.mean(axis=0)  # influence of each source token overall
    elif reduce == "row":
        vec = arr.mean(axis=1)  # susceptibility of each target token overall
    else:
        raise ValueError("reduce must be 'col' or 'row'")

    _plot_barchart(
        values=vec,
        title=title,
        x_label=x_label,
        y_label="rollout importance",
        figsize=(6.4, 2.4),
        out_path=out_path,
        xticklabels=xticklabels
    )


# --- For Occlusion Sensitivity ---

def plot_occlusion_sensitivity(
    attr: torch.Tensor,
    title: str,
    xticklabels: List[str],
    y_label: str = "similarity drop",
    out_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 2.5)
):
    """
    Plots occlusion sensitivity results (dict) by calling _plot_barchart.
    """
    # Data preparation
    values = _as_numpy(attr)
    
    _plot_barchart(
        values=values,
        title=title,
        x_label="feature",
        y_label=y_label,
        figsize=figsize,
        out_path=out_path,
        xticklabels=xticklabels,
        xtick_rotation=60 # A bit more rotation can be good for long feature names
    )

def plot_scroll_x_for_users(
    scroll_x_dict: Dict[int, torch.Tensor],
    out_dir: Optional[str] = None
):
    """
    Plots scroll x time series for multiple users on the same chart.
    scroll_x_dict: dict of {user_id: scroll_x_tensor}
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 4))
    _setup_save(out_dir)

    num_users = len(scroll_x_dict)

    # Get a list of N unique colors from a colormap.
    # 'jet' or 'hsv' are good choices for many distinct colors.
    # 'viridis' is better for colorblind-friendliness.
    # colors = plt.cm.jet(np.linspace(0, 1, num_users))

    for i, (user_id, x_position) in enumerate(scroll_x_dict.items()):
        ax.plot(
            _as_numpy(x_position),
            label=f"User {user_id}",
            linewidth=2,
            # color=colors[i]
        )

    ax.set_xlabel('Time (timesteps)', fontdict={'fontsize': 'small'})
    ax.set_ylabel('x Position (Normalized)', fontdict={'fontsize': 'small'})
    ax.set_title('Horizontal Finger Position Over Time for Different Users')
    ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.7)
    ax.grid(True, axis='x', linestyle=':', color='gray', alpha=0.7)
    ax.set_ylim(0, 1) # Set y-axis limits from 0 to 1
    ax.legend(
        loc='upper left', 
        bbox_to_anchor=(1.05, 1), 
        title='Users', 
        fontsize='small'
    )

    plt.tight_layout()
    if out_dir:
        plt.savefig(out_dir, dpi=180, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
