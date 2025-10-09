import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def _as_numpy(x: torch.Tensor):
    return x.detach().float().cpu().numpy()

def _symmetric_clim(arr: np.ndarray, percentile=99.0):
    # robust symmetric color limits around zero
    a = np.percentile(np.abs(arr), percentile)
    a = a if a > 0 else np.max(np.abs(arr)) + 1e-8
    return -a, a

def plot_attr_heatmap(
    attr: torch.Tensor,
    title: str,
    signed: bool = True,
    feature_names: list[str] | None = None,
    time_axis_label: str = "time",
    out_path: str = None,
):
    """
    attr: (T, F) tensor
    """
    arr = _as_numpy(attr)
    plt.figure(figsize=(8, 3.2))
    if signed:
        vmin, vmax = _symmetric_clim(arr, percentile=99.0)
        cmap = "bwr"
    else:
        arr = np.abs(arr)
        vmin, vmax = 0, np.percentile(arr, 99.0)
        cmap = "viridis"

    plt.imshow(arr.T, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="attribution" + ("" if signed else " (|.|)"))
    plt.xlabel(time_axis_label)
    plt.ylabel("feature")
    plt.title(title)
    if feature_names is not None and len(feature_names) == arr.shape[1]:
        plt.yticks(np.arange(len(feature_names)), feature_names)
    else:
        plt.yticks(np.arange(arr.shape[1]))
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()

def plot_time_profile(attr: torch.Tensor, title: str, signed: bool = False, out_path: str = None):
    """
    Mean attribution over features → (T,)
    """
    x = attr.abs().mean(dim=1) if not signed else attr.mean(dim=1)
    x = _as_numpy(x)
    plt.figure(figsize=(8, 2.2))
    plt.plot(x)
    plt.xlabel("time")
    plt.ylabel("mean |attr|" if not signed else "mean attr")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()

def plot_feature_profile(attr: torch.Tensor, title: str, out_path: str = None, feature_names: list[str] | None = None, signed: bool = False):
    """
    Mean attribution over time → (F,)
    """
    x = attr.abs().mean(dim=0) if not signed else attr.mean(dim=0)
    x = _as_numpy(x)
    plt.figure(figsize=(8, 2.2))
    plt.bar(np.arange(len(x)), x)
    plt.xlabel("feature")
    plt.ylabel("mean |attr|" if not signed else "mean attr")
    if feature_names is not None and len(feature_names) == len(x):
        plt.xticks(np.arange(len(x)), feature_names, rotation=45, ha="right")
    else:
        plt.xticks(np.arange(len(x)))
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        plt.show()