import math
import os
import random
import argparse
import pickle
import pandas as pd
from typing import Literal
from xml.parsers.expat import model
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.animation as animation
from adjustText import adjust_text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R
from itertools import combinations

from model.dataset import (
    HUMITrainDataset,
    FETATrainDataset,
    HUMITestDataset,
    FETATestDataset,
)
from model.behaveformer import BehaveFormer
from utils.config import Config
from utils.utils import read_pickle
from imu_visualisation import get_phone_box
from xai import project_prototypes


# --- Configuration ---
CATALOG_PATH = "work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/p16_0.1_0.1_0.01/checkpoints/prototype_catalog.pkl"
OUTPUT_DIR = "prototype_vis/p16_0.1_0.1_0.01"
TARGET_EPOCH = 100


def _load_catalog(catalog_path, target_epoch=None):
    """Helper to load the catalog and drill down to the correct epoch."""
    print(f"Loading prototype catalog from {catalog_path}...")
    with open(catalog_path, "rb") as f:
        full_catalog = pickle.load(f)

    if isinstance(list(full_catalog.keys())[0], int) and isinstance(
        full_catalog[list(full_catalog.keys())[0]], dict
    ):
        epoch = target_epoch if target_epoch else max(full_catalog.keys())
        print(f"Loading prototypes from Epoch {epoch}...")
        if epoch not in full_catalog:
            print(f"Error: Epoch {epoch} not found in catalog!")
            return None
        return full_catalog[epoch]
    return full_catalog


def _create_comet_animation(
    x_arrays, y_arrays, titles, super_title, save_path, tail_length=5
):
    """Core rendering function handling all Matplotlib and Animation logic."""
    num_plots = len(x_arrays)
    if num_plots == 0:
        print("No valid data to animate.")
        return

    # Setup the plot axes
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    lines, dots = [], []
    max_frames = max([len(x) for x in x_arrays] + [0])

    for idx in range(num_plots):
        ax = axes[idx]

        # Initialize empty line (comet tail) and dot (finger)
        (line,) = ax.plot([], [], color="royalblue", linewidth=2.5, alpha=0.8)
        (dot,) = ax.plot(
            [],
            [],
            "ro",
            markersize=10,
            zorder=5,
            label="Finger Position" if idx == 0 else "",
        )

        lines.append(line)
        dots.append(dot)

        # Fix geometric distortion and lock to absolute screen coordinates
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.invert_yaxis()
        ax.set_aspect(19.5 / 9.0)

        ax.set_title(titles[idx], fontsize=10, fontweight="bold")
        ax.set_xlabel("Screen X")
        ax.set_ylabel("Screen Y")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right")

    plt.suptitle(super_title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    # --- Animation Update Function ---
    def update(frame):
        updated_artists = []
        for i in range(num_plots):
            f = min(frame, len(x_arrays[i]) - 1)

            if f == len(x_arrays[i]) - 1:
                start_idx = 0
                lines[i].set_alpha(0.4)
            else:
                start_idx = max(0, f - tail_length)
                lines[i].set_alpha(0.8)

            lines[i].set_data(
                x_arrays[i][start_idx : f + 1], y_arrays[i][start_idx : f + 1]
            )
            dots[i].set_data([x_arrays[i][f]], [y_arrays[i][f]])

            updated_artists.extend([lines[i], dots[i]])

        return updated_artists

    print(f"Generating animation with {max_frames} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames + 20, interval=200, blit=True
    )
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow", fps=5)
        print(f"Saved to {save_path}")


def _create_sync_comet_3d_animation(
    x_arrays, y_arrays, imu_arrays, titles, super_title, save_path, tail_length=5, layout_mode="standard"
):
    num_plots = len(x_arrays)
    if num_plots == 0:
        print("No valid data to animate.")
        return

    # ==========================================
    # 1. THE LAYOUT ROUTER (Block-Triangle Mapping)
    # ==========================================
    layout = [
        {"name": "scroll", "height": 2},
        {"name": "gyro", "height": 1},
        {"name": "mag", "height": 1},  
        {"name": "phone3d", "height": 1.5}, 
    ]
    row_map = {row["name"]: idx for idx, row in enumerate(layout)}

    if layout_mode == "standard":
        nrows, ncols = len(layout), num_plots
        height_ratios = [row["height"] for row in layout]
        fig = plt.figure(figsize=(4 * num_plots, sum(height_ratios) * 2.5))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=height_ratios)

        def get_slice(row_idx, item_idx):
            return np.s_[row_idx, item_idx]

    elif layout_mode == "turing":
        if num_plots != 6:
            print("Error: Turing mode requires exactly 6 plots.")
            return
            
        # We double the rows (Top half for Targets, Bottom half for Prototypes)
        nrows = len(layout) * 2
        # 8 columns ensures we can center the top block over the bottom two perfectly
        ncols = 8
        height_ratios = [row["height"] for row in layout] * 2
        
        # Make the figure wider to accommodate the 8 columns beautifully
        fig = plt.figure(figsize=(12, sum(height_ratios) * 2))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=height_ratios)

        # Draw the visual divider down the middle (between P and Q sides)
        line = plt.Line2D([0.5, 0.5], [0.02, 0.94], transform=fig.transFigure, color="black", linewidth=1, alpha=0.7)
        fig.add_artist(line)

        def get_slice(row_idx, item_idx):
            # Is this sequence a Target (Top Triangle) or a Prototype (Bottom Triangle)?
            is_top_level = (item_idx == 0 or item_idx == 3)
            
            # Shift the row down if it belongs to the bottom Prototypes
            r = row_idx if is_top_level else len(layout) + row_idx

            # Map the columns (Every block gets EXACTLY 2 columns of width!)
            # --- LEFT PANEL (Target P side) ---
            if item_idx == 0: return np.s_[r, 1:3]  # Target P (Centered Top)
            if item_idx == 1: return np.s_[r, 0:2]  # P Proto 1 (Bottom Left)
            if item_idx == 2: return np.s_[r, 2:4]  # P Proto 2 (Bottom Right)
            
            # --- RIGHT PANEL (Target Q side) ---
            if item_idx == 3: return np.s_[r, 5:7]  # Target Q (Centered Top)
            if item_idx == 4: return np.s_[r, 4:6]  # Q Proto 1 (Bottom Left)
            if item_idx == 5: return np.s_[r, 6:8]  # Q Proto 2 (Bottom Right)

    # ==========================================
    # 2. SHARED DATA PREPARATION
    # ==========================================
    axes_scroll, axes_3d, axes_gyr, axes_mag = [], [], [], []
    lines, dots, phones, normal_lines = [], [], [], []
    gyr_cursors, mag_cursors = [], []
    r0_inverses, all_quats = [], []
    max_frames = max([len(x) for x in x_arrays] + [len(i) for i in imu_arrays] + [0])

    if "phone3d" in row_map:
        print("Pre-computing 3D sensor fusion...")
        for imu_seq in imu_arrays:
            acc = imu_seq[:, 0:3] * 10.0
            gyr = imu_seq[:, 12:15]
            mag = imu_seq[:, 24:27] * 1000.0
            madgwick = Madgwick(acc=acc, gyr=gyr, mag=mag)
            Q = madgwick.Q
            norms = np.linalg.norm(Q, axis=1)
            Q[(norms < 1e-6) | np.isnan(norms)] = [1.0, 0.0, 0.0, 0.0]
            all_quats.append(Q)

    # ==========================================
    # 3. SHARED SUBPLOT GENERATION
    # ==========================================
    for idx in range(num_plots):
        if "scroll" in row_map:
            ax_s = fig.add_subplot(gs[get_slice(row_map["scroll"], idx)])
            (line,) = ax_s.plot([], [], color="royalblue", linewidth=2.5, alpha=0.8)
            (dot,) = ax_s.plot([], [], "ro", markersize=10, zorder=5, label="Finger" if idx == 0 and layout_mode == "standard" else "")
            lines.append(line)
            dots.append(dot)
            ax_s.set_xlim([0.0, 1.0]); ax_s.set_ylim([0.0, 1.0]); ax_s.invert_yaxis(); ax_s.set_aspect(19.5 / 9.0)
            ax_s.set_title(titles[idx], fontsize=10 if layout_mode == "standard" else 11, fontweight="bold")
            ax_s.grid(True, linestyle="--", alpha=0.4)
            if idx == 0 and layout_mode == "standard": ax_s.legend(loc="upper right")

        if "phone3d" in row_map:
            ax_3d = fig.add_subplot(gs[get_slice(row_map["phone3d"], idx)], projection="3d")
            ax_3d.set_xlim([-0.6, 0.6]); ax_3d.set_ylim([-0.6, 0.6]); ax_3d.set_zlim([-0.6, 0.6])
            ax_3d.view_init(elev=45, azim=-90); ax_3d.set_axis_off()
            # Title for all blocks so users know what they are looking at in Turing mode
            ax_3d.set_title("Relative Phone Orientation", fontsize=10, color="black")

            Q = all_quats[idx]
            rot = R.from_quat([Q[0, 1], Q[0, 2], Q[0, 3], Q[0, 0]])
            r0_inverses.append(rot.inv())

            rotated_faces = [[rot.apply(v) for v in face] for face in get_phone_box()[1]]
            phone = Poly3DCollection(rotated_faces, alpha=0.2, facecolors="cyan", edgecolors="black")
            ax_3d.add_collection3d(phone)
            phones.append(phone)

            pointer_tip = rot.apply([0, 0, 0.8])
            (normal_line,) = ax_3d.plot([0, pointer_tip[0]], [0, pointer_tip[1]], [0, pointer_tip[2]], color="red", linewidth=3, zorder=10)
            normal_lines.append(normal_line)

        if "gyro" in row_map:
            ax_g = fig.add_subplot(gs[get_slice(row_map["gyro"], idx)])
            time_steps = np.arange(len(imu_arrays[idx]))
            gyr_data = imu_arrays[idx][:, 12:15]
            ax_g.plot(time_steps, gyr_data[:, 0], label="X", color="tab:red", alpha=0.7)
            cursor_g = ax_g.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
            gyr_cursors.append(cursor_g)
            
            # Small context titles for Turing blocks
            ax_g.set_title("Gyroscope (rad/s)", fontsize=10, color="black")
            ax_g.set_ylim([1.2, -1.2])
            ax_g.set_xlim([0, len(time_steps) - 1]); ax_g.grid(True, linestyle=":", alpha=0.6)

        if "mag" in row_map:
            ax_m = fig.add_subplot(gs[get_slice(row_map["mag"], idx)])
            time_steps = np.arange(len(imu_arrays[idx]))
            mag_fd_data = imu_arrays[idx][:, 30:33]
            ax_m.plot(time_steps, mag_fd_data[:, 1], color="tab:green", alpha=0.7, label="Y")
            cursor_m = ax_m.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
            mag_cursors.append(cursor_m)
            
            # Small context titles for Turing blocks
            ax_m.set_title("Magnetometer 1st Derivative", fontsize=10, color="black")
            ax_g.set_ylim([1.75, -1.75])
            ax_m.set_xlim([0, len(time_steps) - 1]); ax_m.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(super_title, fontsize=16 if layout_mode == "standard" else 20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95] if layout_mode == "turing" else [0, 0, 1, 1])

    # ==========================================
    # 4. SHARED ANIMATION UPDATE
    # ==========================================
    def update(frame):
        updated_artists = []
        for i in range(num_plots):
            len_s = len(x_arrays[i])
            len_i = len(imu_arrays[i])

            fi = min(frame, len_i - 1)
            progress = (fi + 1) / len_i
            fs = math.ceil(progress * len_s) - 1

            if "scroll" in row_map:
                start_idx = 0 if fs == len_s - 1 else max(0, fs - tail_length)
                lines[i].set_alpha(0.4 if fs == len_s - 1 else 0.8)
                lines[i].set_data(x_arrays[i][start_idx : fs + 1], y_arrays[i][start_idx : fs + 1])
                dots[i].set_data([x_arrays[i][fs]], [y_arrays[i][fs]])
                updated_artists.extend([lines[i], dots[i]])

            if "phone3d" in row_map:
                Q = all_quats[i]
                rot = R.from_quat([Q[fi, 1], Q[fi, 2], Q[fi, 3], Q[fi, 0]])
                rot = rot * r0_inverses[i]

                VISUAL_SCALAR = 4.0
                amplified_rot = R.from_rotvec(rot.as_rotvec() * VISUAL_SCALAR)

                rotated_faces = [[amplified_rot.apply(v) for v in face] for face in get_phone_box()[1]]
                phones[i].set_verts(rotated_faces)

                pointer_tip = amplified_rot.apply([0, 0, 0.8])
                normal_lines[i].set_data_3d([0, pointer_tip[0]], [0, pointer_tip[1]], [0, pointer_tip[2]])
                updated_artists.extend([phones[i], normal_lines[i]])

            if "gyro" in row_map:
                gyr_cursors[i].set_xdata([fi])
                updated_artists.append(gyr_cursors[i])

            if "mag" in row_map:
                mag_cursors[i].set_xdata([fi])
                updated_artists.append(mag_cursors[i])

        return updated_artists

    print(f"Generating {'hierarchical block ' if layout_mode == 'turing' else ''}animation with {max_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=max_frames + 20, interval=150, blit=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow", fps=5)
        print(f"Saved to {save_path}")

def _save_sync_comet_3d_last_frame(
    x_arrays, y_arrays, imu_arrays, titles, super_title, save_path, tail_length=5, layout_mode="standard"
):
    """Clean static renderer: plots the full trajectory and final 3D state instantly."""
    num_plots = len(x_arrays)
    if num_plots == 0:
        print("No valid data to plot.")
        return

    # ==========================================
    # 1. THE LAYOUT ROUTER
    # ==========================================
    layout = [
        {"name": "scroll", "height": 2},
        {"name": "gyro", "height": 1.2},
        {"name": "mag", "height": 1.2},  
        # {"name": "phone3d", "height": 1.5}, 
    ]
    row_map = {row["name"]: idx for idx, row in enumerate(layout)}

    if layout_mode == "standard":
        nrows, ncols = len(layout), num_plots
        height_ratios = [row["height"] for row in layout]
        fig = plt.figure(figsize=(4 * num_plots, sum(height_ratios) * 2.5))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=height_ratios)

        def get_slice(row_idx, item_idx):
            return np.s_[row_idx, item_idx]

    elif layout_mode == "turing":
        nrows = len(layout) * 2
        ncols = 8
        height_ratios = [row["height"] for row in layout] * 2
        fig = plt.figure(figsize=(12, sum(height_ratios) * 2))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, height_ratios=height_ratios)

        line = plt.Line2D([0.5, 0.5], [0.02, 0.94], transform=fig.transFigure, color="black", linewidth=1, alpha=0.7)
        fig.add_artist(line)

        def get_slice(row_idx, item_idx):
            is_top_level = (item_idx == 0 or item_idx == 3)
            r = row_idx if is_top_level else len(layout) + row_idx
            if item_idx == 0: return np.s_[r, 1:3]
            if item_idx == 1: return np.s_[r, 0:2]
            if item_idx == 2: return np.s_[r, 2:4]
            if item_idx == 3: return np.s_[r, 5:7]
            if item_idx == 4: return np.s_[r, 4:6]
            if item_idx == 5: return np.s_[r, 6:8]

    # ==========================================
    # 2. PLOT GENERATION (Pure Static)
    # ==========================================
    for idx in range(num_plots):
        x_data = x_arrays[idx]
        y_data = y_arrays[idx]
        imu_data = imu_arrays[idx] if len(imu_arrays) > idx else None
        
        # --- Scroll Plot ---
        if "scroll" in row_map:
            ax_s = fig.add_subplot(gs[get_slice(row_map["scroll"], idx)])
            ax_s.set_xlim([0.0, 1.0]); ax_s.set_ylim([0.0, 1.0]); ax_s.invert_yaxis(); ax_s.set_aspect(19.5 / 9.0)
            ax_s.set_title(titles[idx], fontsize=14 if layout_mode == "standard" else 11, pad=18, fontweight="bold")
            ax_s.grid(True, linestyle="--", alpha=0.4)
            
            # Draw the trajectory as gradient segments
            num_segments = len(x_data) - 1
            for j in range(num_segments):
                # Scale alpha linearly from 0.1 (oldest) to 1.0 (newest)
                alpha_val = 0.3 + 0.5 * (j / max(1, num_segments - 1))
                
                # 1. Draw the line segment
                ax_s.plot(
                    x_data[j : j + 2], 
                    y_data[j : j + 2], 
                    color="royalblue", 
                    linewidth=1.5, 
                    alpha=alpha_val
                )
            
            # Drop the large red dot at the very last point (the current finger position)
            ax_s.plot(x_data[-1], y_data[-1], "ro", label="Finger", markersize=10, zorder=5)
            ax_s.legend(loc="upper right") if idx == 0 else None

        if imu_data is not None:
            time_steps = np.arange(len(imu_data))
            
            # --- Gyro Plot ---
            if "gyro" in row_map:
                ax_g = fig.add_subplot(gs[get_slice(row_map["gyro"], idx)])
                ax_g.plot(time_steps, imu_data[:, 12], color="tab:red", alpha=0.7, label="g_x") # Gyro X
                ax_g.set_title("Gyroscope Reading vs Time", fontsize=12)
                ax_g.grid(True, linestyle=":", alpha=0.6)
                ax_g.set_ylim([-0.4, 0.4])

                ax_g.legend(loc="upper right") if idx == 0 else None

            # --- Mag Plot ---
            if "mag" in row_map:
                ax_m = fig.add_subplot(gs[get_slice(row_map["mag"], idx)])
                ax_m.plot(time_steps, imu_data[:, 31], color="tab:green", alpha=0.7, label="m_fd_y") # Mag Y (FD)
                ax_m.set_title("Magnetometer 1st Derivative vs Time", fontsize=12)
                ax_m.grid(True, linestyle=":", alpha=0.6)
                ax_m.set_ylim([-1.75, 1.75])

                ax_m.legend(loc="upper right") if idx == 0 else None
                

            # --- 3D Phone Plot (Calculated strictly for the last frame) ---
            if "phone3d" in row_map:
                ax_3d = fig.add_subplot(gs[get_slice(row_map["phone3d"], idx)], projection="3d")
                ax_3d.set_xlim([-0.6, 0.6]); ax_3d.set_ylim([-0.6, 0.6]); ax_3d.set_zlim([-0.6, 0.6])
                ax_3d.view_init(elev=45, azim=-90); ax_3d.set_axis_off()
                ax_3d.set_title("Final Phone Orientation", fontsize=10, color="black")

                # Madgwick for the whole sequence to get stable orientation, but we only use the last frame
                acc = imu_data[:, 0:3] * 10.0
                gyr = imu_data[:, 12:15]
                mag = imu_data[:, 24:27] * 1000.0
                madgwick = Madgwick(acc=acc, gyr=gyr, mag=mag)
                Q = madgwick.Q

                norms = np.linalg.norm(Q, axis=1)
                Q[(norms < 1e-6) | np.isnan(norms)] = [1.0, 0.0, 0.0, 0.0]
                
                # Base rotation (first frame) vs Final rotation (last frame)
                r0 = R.from_quat([Q[0, 1], Q[0, 2], Q[0, 3], Q[0, 0]])
                r_final = R.from_quat([Q[-1, 1], Q[-1, 2], Q[-1, 3], Q[-1, 0]])
                
                relative_rot = r_final * r0.inv()
                amplified_rot = R.from_rotvec(relative_rot.as_rotvec() * 4.0)

                rotated_faces = [[amplified_rot.apply(v) for v in face] for face in get_phone_box()[1]]
                phone = Poly3DCollection(rotated_faces, alpha=0.2, facecolors="cyan", edgecolors="black")
                ax_3d.add_collection3d(phone)

                pointer_tip = amplified_rot.apply([0, 0, 0.8])
                ax_3d.plot([0, pointer_tip[0]], [0, pointer_tip[1]], [0, pointer_tip[2]], color="red", linewidth=3, zorder=10)

    plt.suptitle(super_title, fontsize=16 if layout_mode == "standard" else 20, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95] if layout_mode == "turing" else [0, 0, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved static frame to {save_path}")
    
    plt.close(fig)

def plot_different_prototypes_3d_static(
    catalog_path,
    prototypes_to_plot,
    target_epoch,
    type: Literal["Similar", "Different"],
    tail_length=15, # Extended tail length so you can see more of the path in a static image
):
    """Wrapper function to load data and call the static PNG renderer."""
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog:
        return

    x_arrays, y_arrays, imu_arrays, titles = [], [], [], []

    for p_idx in prototypes_to_plot:
        if p_idx not in catalog:
            print(f"Warning: Prototype {p_idx} not found in catalog.")
            continue

        entry = catalog[p_idx]
        u = entry["source_user"]
        s = entry["source_sess"]
        q = entry["source_seq"]

        try:
            scroll_seq = entry["data"][0]
            imu_seq = entry["data"][1]

            x_arrays.append(scroll_seq[:, 0])
            y_arrays.append(scroll_seq[:, 1])
            imu_arrays.append(imu_seq)

            titles.append(f"Prototype {p_idx}\n(User {u}, Sess {s}, Seq {q})")
        except Exception as e:
            print(f"Failed to load data for P{p_idx}. Error: {e}")

    if not x_arrays:
        print("No valid prototypes found to plot.")
        return

    save_path = os.path.join(
        OUTPUT_DIR, f"static_{type.lower()}_prototypes_3D_prototypes-{','.join(map(str, prototypes_to_plot))}.png"
    )
    super_title = (
        f"Final Frame 3D Trajectories of {type} Prototypes (Epoch {target_epoch})"
    )

    _save_sync_comet_3d_last_frame(
        x_arrays,
        y_arrays,
        imu_arrays,
        titles,
        super_title,
        save_path,
        tail_length,
    )

def visualize_clean_catalog(catalog_path, mode="grid"):
    # 1. Load the Catalog
    if not os.path.exists(catalog_path):
        print(f"Error: File not found at {catalog_path}")
        return

    with open(catalog_path, "rb") as f:
        full_catalog = pickle.load(f)

    # Handle Master Catalog (Epochs)
    if isinstance(list(full_catalog.keys())[0], int) and isinstance(
        full_catalog[list(full_catalog.keys())[0]], dict
    ):
        # last_epoch = max(full_catalog.keys())
        target_epoch = 200  # Or use max(full_catalog.keys())
        print(f"Visualizing Epoch {target_epoch}...")
        catalog = full_catalog[target_epoch]
    else:
        catalog = full_catalog

    # --- STEP 2: Filter Duplicates (The Key Fix) ---
    unique_prototypes = []
    seen_sources = set()

    # Iterate through sorted keys to find unique sources
    for proto_id in sorted(catalog.keys()):
        entry = catalog[proto_id]
        u, s, q = (
            entry["source_user"],
            entry["source_sess"] if "source_sess" in entry else entry["source_session"],
            entry["source_seq"],
        )

        source_sig = (u, s, q)

        if source_sig not in seen_sources:
            seen_sources.add(source_sig)
            unique_prototypes.append(
                {"id": proto_id, "entry": entry, "sig": source_sig}
            )

    num_unique = len(unique_prototypes)
    print(
        f"Found {len(catalog)} total prototypes, simplified to {num_unique} UNIQUE prototypes."
    )

    # --- STEP 3: Visualization ---

    if mode == "grid":
        # GRID MODE: One subplot per unique prototype
        cols = 4  # Fewer columns since we have fewer plots now
        rows = (num_unique // cols) + (1 if num_unique % cols != 0 else 0)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]  # Handle case of 1 subplot

        for i, item in enumerate(unique_prototypes):
            ax = axes[i]
            entry = item["entry"]
            pid = item["id"]
            u, s, q = item["sig"]

            scroll_seq = entry["data"][0]

            # Plot X-Axis (Horizontal)
            # Assuming Column 0 is X. Change index if X is elsewhere.
            ax.plot(scroll_seq[:, 0], color="tab:blue", linewidth=1.5, label="X-Axis")

            # Styling
            ax.set_title(f"Proto {pid}\nU{u}-S{s}-Seq{q}", fontsize=8)
            ax.grid(True, alpha=0.3)
            # Remove axis ticks for cleaner look
            ax.set_xticks([])

        # Cleanup empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        # plt.savefig(os.path.join(OUTPUT_DIR, f'unique_grid_epoch_{target_epoch}.png'), dpi=150)

    elif mode == "overlay":
        # OVERLAY MODE: All lines on ONE plot
        plt.figure(figsize=(10, 6))

        for item in unique_prototypes:
            entry = item["entry"]
            pid = item["id"]

            scroll_seq = entry["data"][0]
            imu_seq = entry["data"][1]

            # if pid == 27:
            #     continue
            # Plot with low alpha (transparency) to see density
            plt.plot(scroll_seq[:, 0], alpha=0.6, linewidth=2, label=f"P{pid}")
            # plt.plot(imu_seq[:, 31], alpha=0.6, linewidth=2, label=f"P{pid}")

        plt.title(
            # f"Overlay of {num_unique} Unique Prototype Behaviors (Magnetometer fd_y)",
            f"Overlay of {num_unique} Unique Prototype Behaviors (Finger Scroll X-Axis)",
            fontsize=14,
        )
        plt.xlabel("Time Step")
        # plt.ylabel("Magnetometer fd_y")
        plt.ylabel("Scroll Position (X-Axis)")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        # plt.savefig(
        #     os.path.join(OUTPUT_DIR, f"unique_overlay_epoch_{target_epoch}.png"),
        #     dpi=150,
        # )

    print("Done.")
    plt.show()


def animate_different_prototypes(
    catalog_path,
    prototypes_to_plot,
    target_epoch,
    type: Literal["Similar", "Different"],
    tail_length=5,
):
    """Visualizes different prototypes side-by-side."""
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog:
        return

    x_arrays, y_arrays, titles = [], [], []

    for p_idx in prototypes_to_plot:
        if p_idx not in catalog:
            print(f"Warning: Prototype {p_idx} not found in catalog.")
            continue

        entry = catalog[p_idx]
        u = entry["source_user"]
        s = entry["source_sess"]
        q = entry["source_seq"]

        try:
            scroll_seq = entry["data"][0]
            x_arrays.append(scroll_seq[:, 0])
            y_arrays.append(scroll_seq[:, 1])
            titles.append(f"Prototype {p_idx}\n(User {u}, Sess {s}, Seq {q})")
        except Exception as e:
            print(f"Failed to load data for P{p_idx}. Error: {e}")

    save_path = os.path.join(
        OUTPUT_DIR, f"comet_trajectories_{type.lower()}_prototypes.gif"
    )
    _create_comet_animation(
        x_arrays,
        y_arrays,
        titles,
        f"Comet Tail Scroll Trajectories of {type} Prototypes (Epoch {target_epoch})",
        save_path,
        tail_length,
    )


def animate_different_prototypes_3d(
    catalog_path,
    prototypes_to_plot,
    target_epoch,
    type: Literal["Similar", "Different"],
    tail_length=5,
):
    """Visualizes different prototypes side-by-side using the 4-row 3D and sensor dashboard."""
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog:
        return

    x_arrays, y_arrays, imu_arrays, titles = [], [], [], []

    for p_idx in prototypes_to_plot:
        if p_idx not in catalog:
            print(f"Warning: Prototype {p_idx} not found in catalog.")
            continue

        entry = catalog[p_idx]
        u = entry["source_user"]
        s = entry["source_sess"]
        q = entry["source_seq"]

        try:
            scroll_seq = entry["data"][0]
            imu_seq = entry["data"][1]  # Extract the IMU data!

            x_arrays.append(scroll_seq[:, 0])
            y_arrays.append(scroll_seq[:, 1])
            imu_arrays.append(imu_seq)

            titles.append(f"Prototype {p_idx}\n(User {u}, Sess {s}, Seq {q})")
        except Exception as e:
            print(f"Failed to load data for P{p_idx}. Error: {e}")

    if not x_arrays:
        print("No valid prototypes found to animate.")
        return

    save_path = os.path.join(
        OUTPUT_DIR, f"comet_trajectories_{type.lower()}_prototypes_3D_prototypes{','.join(map(str, prototypes_to_plot))}.gif"
    )
    super_title = (
        f"Synchronized 3D Trajectories of {type} Prototypes (Epoch {target_epoch})"
    )

    _create_sync_comet_3d_animation(
        x_arrays,
        y_arrays,
        imu_arrays,
        titles,
        super_title,
        save_path,
        tail_length,
    )


def animate_prototype_neighbors(
    catalog_path, prototype_id, target_epoch, tail_length=5
):
    """Visualizes the Top-K nearest neighbors for a SINGLE prototype side-by-side."""
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog:
        return

    if prototype_id not in catalog:
        print(f"Error: Prototype {prototype_id} not found in epoch {target_epoch}!")
        return

    neighbors = catalog[prototype_id].get("neighbors", [])
    if not neighbors:
        print(f"Error: No 'neighbors' key found for Prototype {prototype_id}.")
        return

    x_arrays, y_arrays, titles = [], [], []

    for idx, n_data in enumerate(neighbors):
        u, s, q = n_data["source_user"], n_data["source_sess"], n_data["source_seq"]
        sim = n_data.get("cosine_similarity", 0.0)

        try:
            scroll_seq = n_data["data"][0]
            x_arrays.append(scroll_seq[:, 0])
            y_arrays.append(scroll_seq[:, 1])
            titles.append(
                f"Neighbor {idx+1}\n(User {u}, Sess {s}, Seq {q})\nSim: {sim:.3f}"
            )
        except Exception as e:
            print(f"Failed to load data for Neighbor {idx+1}. Error: {e}")

    save_path = os.path.join(OUTPUT_DIR, f"prototype_{prototype_id}_neighbors.gif")
    super_title = f"Top {len(neighbors)} Neighbors for Prototype {prototype_id} (Epoch {target_epoch})"
    _create_comet_animation(
        x_arrays, y_arrays, titles, super_title, save_path, tail_length
    )


def animate_prototype_neighbors_3d(
    catalog_path, prototype_id, target_epoch, tail_length=5
):
    """Extracts neighbors and passes BOTH Scroll and IMU data to the 3D animator."""
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog or prototype_id not in catalog:
        print(f"Error: Prototype {prototype_id} not found in epoch {target_epoch}!")
        return

    neighbors = catalog[prototype_id].get("neighbors", [])
    if not neighbors:
        return

    x_arrays, y_arrays, imu_arrays, titles = [], [], [], []

    for idx, n_data in enumerate(neighbors):
        u, s, q = n_data["source_user"], n_data["source_sess"], n_data["source_seq"]
        sim = n_data.get("cosine_similarity", 0.0)

        try:
            scroll_seq = n_data["data"][0]
            imu_seq = n_data["data"][1]  # Extract the IMU data!

            x_arrays.append(scroll_seq[:, 0])
            y_arrays.append(scroll_seq[:, 1])
            imu_arrays.append(imu_seq)
            titles.append(f"Neighbor {idx+1}\n(U{u}, S{s}, Q{q})\nSim: {sim:.3f}")
        except Exception as e:
            print(f"Failed to load data for Neighbor {idx+1}. Error: {e}")

    save_path = os.path.join(OUTPUT_DIR, f"prototype_{prototype_id}_neighbors_3D.gif")
    super_title = f"Top {len(neighbors)} Neighbors for Prototype {prototype_id} (Epoch {target_epoch})"

    _create_sync_comet_3d_animation(
        x_arrays, y_arrays, imu_arrays, titles, super_title, save_path, tail_length
    )


def explain_test_user_behavior(
    model,
    test_dataset,
    device,
    catalog_path,
    target_epoch,
    imu_type="all",
    top_k=3,
    target_index=0,
):
    """
    Takes an unseen test user, gets the 16D similarity vector output,
    plots the vector as a bar chart, and animates the user side-by-side with the top matching prototypes.
    """
    model.eval()
    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog:
        return

    # 1. Grab a single unseen test sequence
    print(f"Extracting Test User from batch index {target_index}...")

    test_batch = test_dataset.get_sample_from_user(target_index)

    if test_batch is None:
        print(
            f"Error: Target index {target_index} is out of bounds for the test dataset."
        )
        return

    run_id = f"batch_{target_index}"

    # Test dataloader yields (scroll_batch, imu_batch, labels)
    # Isolate just the first user's sequence from the batch
    test_scroll = test_batch[0][0:1]  # Shape: (1, seq_len, 8)

    if imu_type != "none":
        test_imu = test_batch[1][0:1]  # Shape: (1, seq_len, 36)

    # 2. Forward Pass: Get the 16D Output Vector
    with torch.no_grad():
        if imu_type != "none":
            output_vector, _ = model(
                [test_scroll.to(device).float(), test_imu.to(device).float()]
            )
        else:
            output_vector, _ = model(test_scroll.to(device).float())

    # Convert to 1D numpy array
    sim_vector = output_vector[0].cpu().numpy()

    # 3. Find the Top-K Prototypes
    # NOTE: If your model outputs DISTANCE, use argsort() (lowest is best).
    # If it outputs SIMILARITY, use argsort()[::-1] (highest is best).
    # Assuming Similarity here:
    top_indices = sim_vector.argsort()[::-1][:top_k]

    print(f"Test User's 16D Output Vector: {sim_vector}")
    print(f"Top {top_k} Matching Prototypes: {top_indices}")

    # ==========================================
    # 4. PLOT 1: The "Behavioral Barcode" (Static Bar Chart)
    # ==========================================
    plt.figure(figsize=(10, 4))
    colors = [
        "red" if i in top_indices else "royalblue" for i in range(len(sim_vector))
    ]
    bars = plt.bar(range(len(sim_vector)), sim_vector, color=colors, alpha=0.8)

    plt.title(
        f"Test User {target_index} - Similarity Scores With Prototypes",
        fontweight="bold",
    )
    plt.xlabel("Prototype ID")
    plt.ylabel("Model Output (Similarity)")
    plt.xticks(range(len(sim_vector)), [f"P{i}" for i in range(len(sim_vector))])
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    barcode_path = os.path.join(
        OUTPUT_DIR, f"test_user_barcode_epoch_{target_epoch}_{run_id}.png"
    )
    plt.savefig(barcode_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Behavioral Barcode to {barcode_path}")

    # ==========================================
    # 5. PLOT 2: The Physical Comparison (Animation)
    # ==========================================
    x_arrays, y_arrays, imu_arrays, titles = [], [], [], []

    # A. Add the Test User First (The Target)
    x_arrays.append(test_scroll[0].cpu().numpy()[:, 0])
    y_arrays.append(test_scroll[0].cpu().numpy()[:, 1])
    if imu_type != "none":
        imu_arrays.append(test_imu[0].cpu().numpy())
    titles.append(f"TARGET: Test User {target_index}\nScroll Sequence")

    # B. Add the Top-K Prototypes from the Catalog
    for p_idx in top_indices:
        if p_idx not in catalog:
            print(f"Warning: P{p_idx} missing from catalog!")
            continue

        entry = catalog[p_idx]
        scroll_seq = entry["data"][0]

        x_arrays.append(scroll_seq[:, 0])
        y_arrays.append(scroll_seq[:, 1])

        if imu_type != "none":
            imu_seq = entry["data"][1]
            imu_arrays.append(imu_seq)

        score = sim_vector[p_idx]
        titles.append(f"TOP MATCH: Prototype {p_idx}\nSimilarity Score: {score:.3f}")

    # Route to your existing layout manager animator!
    anim_path = os.path.join(
        OUTPUT_DIR, f"test_user_explanation_epoch_{target_epoch}_{run_id}.png"
    )
    super_title = f""

    _save_sync_comet_3d_last_frame(
        x_arrays, y_arrays, imu_arrays, titles, super_title, anim_path, tail_length=5
    )

def explain_multiple_users_behavioral_barcodes_2x2(
    model,
    test_dataset,
    device,
    target_epoch,
    target_indices,         
    highlight_protos=[3, 15], 
    imu_type="all",
    run_id="run_1",
    output_dir="prototype_vis"
):
    model.eval()
    sim_vectors_list = []

    with torch.no_grad():
        for target_index in target_indices:
            # 1. Fetch the sample for the user
            sample = test_dataset.get_sample_from_user(target_index)
            
            # 2. Extract the first sequence (using [0] instead of [0:1] to avoid double-batching)
            scroll_seq = sample[0][0]
            scroll_tensor = torch.tensor(scroll_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            if len(sample) > 1 and imu_type != "none":
                imu_seq = sample[1][0]
                imu_tensor = torch.tensor(imu_seq, dtype=torch.float32).unsqueeze(0).to(device)
                
                output = model([scroll_tensor, imu_tensor])
            else:
                output = model(scroll_tensor)
                
            # 4. Extract similarities, and STRICTLY enforce the 16-prototype limit!
            similarities = output[0].squeeze(0).cpu().numpy()
            sim_vectors_list.append(similarities.tolist())

    # --- THE 2x2 GRID SETUP ---
    # figsize adjusted to (12, 6) so the barcodes aren't squished horizontally
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Flatten the 2x2 axes matrix into a 1D list so we can loop through it easily
    axes = axes.flatten() 

    for i, ax in enumerate(axes):
        # Safety catch: if you pass fewer than 4 users, hide the empty subplots
        if i >= len(sim_vectors_list):
            ax.set_visible(False)
            continue
            
        sim_vector = sim_vectors_list[i]
        user_idx = target_indices[i]

        colors = ["darkorange" if j in highlight_protos else "lightgray" for j in range(len(sim_vector))]

        ax.bar(
            range(len(sim_vector)), 
            sim_vector, 
            color=colors, 
            edgecolor='dimgray', 
            linewidth=1.0, 
            alpha=0.9
        )

        ax.set_title(f"Test User {user_idx}", fontweight="bold", fontsize=11)
        
        # Only draw the Y-axis label on the left-most plots (i = 0 and 2)
        if i % 2 == 0:
            ax.set_ylabel("Similarity", fontsize=10)
        
        ax.set_ylim([-1.1, 1.1]) 
        ax.axhline(0, color='black', linewidth=1.2) 
        
        ax.set_xticks(range(len(sim_vector)))
        ax.set_xticklabels([f"P{j}" for j in range(len(sim_vector))], fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # A single, centered X-axis label for the entire figure
    fig.text(0.5, -0.02, "Prototype ID", ha='center', fontweight="bold", fontsize=11)
    
    # Adjust layout so the shared text doesn't overlap
    plt.tight_layout(rect=[0, 0, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    barcode_path = os.path.join(
        output_dir, f"grid_2x2_barcodes_epoch_{target_epoch}_{run_id}.png"
    )
    plt.savefig(barcode_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved 2x2 Behavioral Barcodes to {barcode_path}")

def visualize_user_behavior_consistency_humidb(
    model, test_dataset, device, target_user_idx, target_epoch, imu_type="all"
):
    """
    Passes ALL sequences for a specific HuMIdb test user through the model and plots a heatmap
    to show how stable their behavioral signature is over time, using load_data.
    """
    if test_dataset.dataset_name != "HuMIdb":
        print("Error: This function is currently optimized specifically for HuMIdb.")
        return

    model.eval()
    sim_vectors = []
    labels = []

    print(f"Extracting all data for User {target_user_idx} using load_data()...")

    # HuMIdb has fixed numbers of sessions and sequences
    num_sessions = test_dataset.num_sessions
    num_seqs = test_dataset.num_seqs

    for s_idx in range(num_sessions):
        for q_idx in range(num_seqs):

            # ---> Using load_data as requested! <---
            sample = test_dataset.load_data(target_user_idx, s_idx, q_idx)

            labels.append(f"S{s_idx}-Q{q_idx}")

            # Prepare tensors and add the missing batch dimension
            test_scroll = torch.tensor(sample[0]).unsqueeze(0).to(device).float()

            with torch.no_grad():
                if imu_type != "none":
                    test_imu = torch.tensor(sample[1]).unsqueeze(0).to(device).float()
                    out, _ = model([test_scroll, test_imu])
                else:
                    out, _ = model(test_scroll)

            sim_vectors.append(out[0].cpu().numpy())

    if not sim_vectors:
        print(f"No data found for User {target_user_idx}")
        return

    sim_matrix = np.array(sim_vectors)  # Shape: (Num_Sequences, 16)
    num_protos = sim_matrix.shape[1]

    # ==========================================
    # Plot the Heatmap
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Use coolwarm colormap: Red = 1.0 (Match), Blue = -1.0 (Opposite), White = 0.0 (Unrelated)
    plt.imshow(sim_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(label="Model Output (Similarity Score)")
    plt.xticks(range(num_protos), [f"P{i}" for i in range(num_protos)])
    plt.yticks(range(len(labels)), labels, fontsize=8)

    plt.title(
        f"Similarity Vector Consistency over Different Scroll Sequences\n(Test User {target_user_idx})",
        fontweight="bold",
    )
    plt.xlabel("Prototype ID")
    plt.ylabel("Session & Sequence Index")

    # Add a grid to make distinct blocks clearer
    plt.gca().set_xticks([x - 0.5 for x in range(1, num_protos)], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    plt.grid(which="minor", color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"test_user_{target_user_idx}_consistency_heatmap_epoch_{target_epoch}.png",
    )
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved Heatmap to {save_path}")


def visualize_pure_latent_spread(
    model,
    dataloader,
    device,
    imu_type,
    num_samples=2000,
    catalog_path=None,
    target_epoch=None,
):
    model.eval()
    all_latents = []
    samples_collected = 0

    print("1. Extracting background data latents...")
    with torch.no_grad():
        for i, item in enumerate(dataloader):
            # Unpack the triplet exactly as train.py does
            anchor, positive, negative, _ = item

            # Forward pass to get latent vector (index 1 of model output)
            if imu_type != "none":
                _, latent = model(
                    [anchor[0].to(device).float(), anchor[1].to(device).float()]
                )
            else:
                _, latent = model(anchor[0].to(device).float())

            all_latents.append(latent.cpu().numpy())
            samples_collected += anchor[0].shape[0]

            if samples_collected >= num_samples:
                break

    # Combine background data
    data_latents = np.concatenate(all_latents, axis=0)[:num_samples]

    print("2. Extracting prototype representations...")
    num_protos = model.prototype_layer.prototypes.shape[0]

    # --- Load physical anchors if catalog and epoch are provided ---
    if catalog_path and target_epoch:
        print(
            f"   -> Loading physical data anchors from Catalog (Epoch {target_epoch})"
        )
        catalog = _load_catalog(catalog_path, target_epoch)
        if not catalog:
            print("   -> Failed to load catalog. Falling back to raw model weights.")
            proto_weights = model.prototype_layer.prototypes.data.cpu()
        else:
            proto_latents = []

            with torch.no_grad():
                for i in range(num_protos):
                    if i not in catalog:
                        print(f"Warning: P{i} not in catalog. Using zeros.")
                        proto_latents.append(
                            np.zeros(model.prototype_layer.prototypes.shape[1])
                        )
                        continue

                    raw_data = catalog[i]["data"]

                    # Pass the exact physical sequence through the CURRENT model
                    if imu_type != "none":
                        scroll_tensor = (
                            torch.as_tensor(raw_data[0]).unsqueeze(0).to(device).float()
                        )
                        imu_tensor = (
                            torch.as_tensor(raw_data[1]).unsqueeze(0).to(device).float()
                        )
                        _, latent = model([scroll_tensor, imu_tensor])
                    else:
                        scroll_arr = (
                            raw_data[0]
                            if isinstance(raw_data, (tuple, list))
                            else raw_data
                        )
                        scroll_tensor = (
                            torch.as_tensor(scroll_arr).unsqueeze(0).to(device).float()
                        )
                        _, latent = model(scroll_tensor)

                    proto_latents.append(latent.cpu().numpy()[0])

            # Convert our physical latents to tensor so we can normalize exactly like before
            proto_weights = torch.tensor(np.array(proto_latents))
    else:
        print("   -> Extracting raw floating weights directly from model.")
        proto_weights = model.prototype_layer.prototypes.data.cpu()

    # Normalize prototypes
    proto_weights = F.normalize(proto_weights, p=2, dim=1).numpy()

    # Normalize data latents
    data_latents = data_latents / np.linalg.norm(data_latents, axis=1, keepdims=True)
    combined_latents = np.vstack([data_latents, proto_weights])

    perplexities = [5, 10, 30]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print("3. Running t-SNE for different perplexities...")
    for idx, p in enumerate(perplexities):
        print(f"   -> Processing perplexity {p}...")
        tsne = TSNE(
            n_components=2,
            perplexity=p,
            metric="cosine",
            random_state=42,
            init="random",
        )
        combined_2d = tsne.fit_transform(combined_latents)

        data_2d = combined_2d[:-num_protos]
        proto_2d = combined_2d[-num_protos:]

        ax = axes[idx]
        ax.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c="lightgray",
            alpha=0.4,
            s=15,
            label="Training Data",
        )
        ax.scatter(
            proto_2d[:, 0],
            proto_2d[:, 1],
            c="red",
            marker="*",
            s=250,
            edgecolor="black",
            label="Prototypes",
        )

        texts = []
        for i in range(num_protos):
            text = ax.text(
                proto_2d[i, 0],
                proto_2d[i, 1],
                f"P{i}",
                fontsize=11,
                fontweight="bold",
                color="darkred",
            )
            texts.append(text)

        # Let adjust_text repel the labels and draw connecting lines
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="-", color="gray", lw=0.8, alpha=0.7, shrinkA=5, shrinkB=2
            ),
            expand_points=(1.5, 1.5),
        )

        ax.set_title(f"Perplexity = {p}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc="upper right")

    plt.suptitle(
        (
            f"Pure Latent Space Spread of {num_protos} Prototypes (Epoch {target_epoch})"
            if target_epoch
            else ""
        ),
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"pure_latent_spread_epoch_{target_epoch}.png"),
        dpi=200,
    )
    print(f"\nDone! Saved as pure_latent_spread_epoch_{target_epoch}.png")


def generate_turing_test_pairs(test_dataset, num_same=10, num_diff=10):
    pairs = []
    num_users = test_dataset.num_users
    num_sessions = test_dataset.num_sessions
    num_seqs = test_dataset.num_seqs

    # --- 1. Define the Quality Filter ---
    def is_high_quality(u, s, q):
        try:
            sample = test_dataset.load_data(u, s, q)
            scroll = sample[0]
            
            # Rule 1: Check Scroll Activity (Must be > 60% active frames)
            # Padded rows are exactly zeros. We count rows that have at least one non-zero value.
            active_scroll_frames = np.sum(np.any(scroll != 0, axis=1))
            if active_scroll_frames < (len(scroll) * 0.60): 
                return False
            
            # Rule 2: Check IMU Variance (Ensure the phone was actually moving)
            if len(sample) > 1:
                imu = sample[1]
                gyro_data = imu[:, 12:15]
                mag_derivative_data = imu[:, 30:33]
                
                # If Gyro is dead/missing
                if np.var(gyro_data) < 1e-5: 
                    return False
                    
                # If Mag First Derivative is dead/missing
                if np.var(mag_derivative_data) < 1e-5:
                    return False
                    
            return True
        except Exception:
            return False

    # --- 2. Define a safe miner to fetch good sequences ---
    def get_valid_seq(user_idx=None):
        max_attempts = 20 # Prevent infinite loops if a user has terrible data
        for _ in range(max_attempts):
            u = user_idx if user_idx is not None else random.randint(0, num_users - 1)
            s = random.randint(0, num_sessions - 1)
            q = random.randint(0, num_seqs - 1)
            
            if is_high_quality(u, s, q):
                return (u, s, q)
                
        # Fallback if we fail 20 times
        print(f"Warning: Struggled to find high-quality data for User {user_idx}. Using fallback.")
        u = user_idx if user_idx is not None else random.randint(0, num_users - 1)
        return (u, 0, 0)

    print("Mining high-quality sequences for the Turing Test...")

    # --- 3. Build the 'Same' Pairs ---
    for _ in range(num_same):
        pair_found = False
        
        while not pair_found:
            # 1. Pick a random user and get their first good sequence
            u = random.randint(0, num_users - 1)
            p_meta = get_valid_seq(user_idx=u)
            
            # 2. Try up to 20 times to find a DIFFERENT good sequence for this same user
            for _ in range(20):
                q_meta = get_valid_seq(user_idx=u)
                if p_meta != q_meta:
                    pair_found = True
                    break # Success! Break the inner loop.
                    
            # 3. If we tried 20 times and failed, the inner loop ends, 
            # pair_found remains False, and the outer loop automatically 
            # restarts to pick a brand NEW user!
            
        pairs.append({"P_meta": p_meta, "Q_meta": q_meta, "Ground_Truth": "Same"})

    # --- 4. Build the 'Different' Pairs ---
    for _ in range(num_diff):
        # Pull two completely different users
        u_p, u_q = random.sample(range(num_users), 2)
        p_meta = get_valid_seq(user_idx=u_p)
        q_meta = get_valid_seq(user_idx=u_q)
        
        pairs.append({"P_meta": p_meta, "Q_meta": q_meta, "Ground_Truth": "Different"})

    random.shuffle(pairs)
    return pairs

def generate_turing_test_study(model, test_dataset, device, catalog_path, target_epoch, imu_type="all", output_folder=None, num_trials=20):
    if output_folder is None:
        print("Error: Output folder must be specified to save Turing Test trials.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    model.eval()

    catalog = _load_catalog(catalog_path, target_epoch)
    if not catalog: return

    pairs = generate_turing_test_pairs(test_dataset, num_same=num_trials // 2, num_diff=num_trials - num_trials // 2)
    answer_key = []

    print(f"Generating {len(pairs)} Turing Test Trials...")

    for trial_idx, trial in enumerate(pairs, start=1):
        print(f"Processing Trial {trial_idx}/{len(pairs)}...")

        def extract_and_infer(meta):
            u, s, q = meta
            sample = test_dataset.load_data(u, s, q)
            scroll_tensor = torch.tensor(sample[0]).unsqueeze(0).to(device).float()
            if imu_type != "none":
                imu_tensor = torch.tensor(sample[1]).unsqueeze(0).to(device).float()
                with torch.no_grad(): out, _ = model([scroll_tensor, imu_tensor])
            else:
                with torch.no_grad(): out, _ = model(scroll_tensor)
            sim_vector = out[0].cpu().numpy()
            top_2_idx = sim_vector.argsort()[::-1][:2]
            return sample, top_2_idx, sim_vector

        sample_P, top2_P, sim_P = extract_and_infer(trial["P_meta"])
        sample_Q, top2_Q, sim_Q = extract_and_infer(trial["Q_meta"])

        x_arrays, y_arrays, imu_arrays, titles = [], [], [], []

        # --- Target P ---
        x_arrays.append(sample_P[0][:, 0]); y_arrays.append(sample_P[0][:, 1])
        if imu_type != "none": imu_arrays.append(sample_P[1])
        titles.append("TARGET SEQUENCE P")

        for p_idx in top2_P:
            entry = catalog[p_idx]
            x_arrays.append(entry["data"][0][:, 0]); y_arrays.append(entry["data"][0][:, 1])
            if imu_type != "none": imu_arrays.append(entry["data"][1])
            titles.append(f"Seq P {'Top' if p_idx == top2_P[0] else 'Second'} Match: Proto {p_idx}\n(Sim: {sim_P[p_idx]:.2f})")

        # --- Target Q ---
        x_arrays.append(sample_Q[0][:, 0]); y_arrays.append(sample_Q[0][:, 1])
        if imu_type != "none": imu_arrays.append(sample_Q[1])
        titles.append("TARGET SEQUENCE Q")

        for p_idx in top2_Q:
            entry = catalog[p_idx]
            x_arrays.append(entry["data"][0][:, 0]); y_arrays.append(entry["data"][0][:, 1])
            if imu_type != "none": imu_arrays.append(entry["data"][1])
            titles.append(f"Seq Q {'Top' if p_idx == top2_Q[0] else 'Second'} Match: Proto {p_idx}\n(Sim: {sim_Q[p_idx]:.2f})")

        trial_name = f"Trial_{trial_idx:02d}"
        super_title = f"Visual Comparison Test - {trial_name}\nAre Sequence P and Sequence Q performed by the SAME person?"
        save_path = os.path.join(output_folder, f"{trial_name}.gif")

        # --> Call the layout engine using layout_mode="turing"
        _create_sync_comet_3d_animation(x_arrays, y_arrays, imu_arrays, titles, super_title, save_path, tail_length=5, layout_mode="turing")

        answer_key.append({"Trial": trial_name, "P_User_ID": trial["P_meta"][0], "Q_User_ID": trial["Q_meta"][0], "Ground_Truth": trial["Ground_Truth"]})

    df = pd.DataFrame(answer_key)
    df.to_csv(os.path.join(output_folder, "answer_key.csv"), index=False)
    print(f"All {num_trials} trials generated! Answer key saved.")


def quantify_signature_stability(model, test_dataset, device, imu_type="all"):
    """
    Calculates both Intra/Inter Distance and Feature-Wise Variance for IJCB.
    """
    model.eval()
    
    all_user_vectors = {} # Dictionary to hold vectors for each user
    num_users = test_dataset.num_users
    num_sessions = test_dataset.num_sessions
    num_seqs = test_dataset.num_seqs
    
    print("Extracting all test vectors for stability analysis...")
    
    # 1. Collect all 16-D Similarity Vectors per user
    with torch.no_grad():
        for u in range(num_users):
            user_vectors = []
            for s in range(num_sessions):
                for q in range(num_seqs):
                    try:
                        sample = test_dataset.load_data(u, s, q)
                        test_scroll = torch.tensor(sample[0]).unsqueeze(0).to(device).float()
                        
                        if imu_type != "none":
                            test_imu = torch.tensor(sample[1]).unsqueeze(0).to(device).float()
                            out, _ = model([test_scroll, test_imu])
                        else:
                            out, _ = model(test_scroll)
                            
                        user_vectors.append(out[0].cpu().numpy())
                    except Exception:
                        continue # Skip missing data
            if user_vectors:
                all_user_vectors[u] = np.array(user_vectors)

    # 2. Calculate Intra-User Distances (User compared to themselves)
    intra_distances = []
    feature_variances = []
    
    for u, vectors in all_user_vectors.items():
        if len(vectors) > 1:
            # Pairwise Euclidean distances between all of a user's own sequences
            for v1, v2 in combinations(vectors, 2):
                dist = np.linalg.norm(v1 - v2)
                intra_distances.append(dist)
            
            # XAI Metric: Standard deviation of each prototype column
            std_per_feature = np.std(vectors, axis=0)
            feature_variances.append(np.mean(std_per_feature))

    # 3. Calculate Inter-User Distances (User compared to everyone else)
    inter_distances = []
    user_ids = list(all_user_vectors.keys())
    
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            u1_vectors = all_user_vectors[user_ids[i]]
            u2_vectors = all_user_vectors[user_ids[j]]
            
            # Compare every sequence of U1 against every sequence of U2
            for v1 in u1_vectors:
                for v2 in u2_vectors:
                    dist = np.linalg.norm(v1 - v2)
                    inter_distances.append(dist)

    # 4. Final Aggregation
    mean_intra = np.mean(intra_distances)
    std_intra = np.std(intra_distances)
    
    mean_inter = np.mean(inter_distances)
    std_inter = np.std(inter_distances)
    
    mean_feature_std = np.mean(feature_variances)

    print("\n" + "="*50)
    print("IJCB BEHAVIORAL STABILITY METRICS")
    print("="*50)
    print(f"Intra-User Distance (Mean ± SD): {mean_intra:.4f} ± {std_intra:.4f}")
    print(f"Inter-User Distance (Mean ± SD): {mean_inter:.4f} ± {std_inter:.4f}")
    print("-" * 50)
    print(f"Distance Gap (Inter - Intra):    {mean_inter - mean_intra:.4f}")
    print(f"Fisher Ratio (Separability):     {((mean_inter - mean_intra)**2) / (std_intra**2 + std_inter**2):.4f}")
    print("-" * 50)
    print(f"XAI Feature-Wise Stability (Mean prototype StdDev): {mean_feature_std:.4f}")
    print("="*50)

    return intra_distances, inter_distances, mean_intra, mean_inter, mean_feature_std


def plot_distance_distributions(intra_distances, inter_distances, output_path):
    plt.figure(figsize=(10, 6))
    
    # Plot the smoothed bell curves (KDE) with filled areas
    sns.kdeplot(intra_distances, fill=True, color="royalblue", label="Genuine (Intra-User) Distribution", alpha=0.5, linewidth=2)
    sns.kdeplot(inter_distances, fill=True, color="crimson", label="Impostor (Inter-User) Distribution", alpha=0.5, linewidth=2)
    
    # Styling for publication
    plt.title("Distribution of Genuine vs. Impostor Behavioural Distances", fontsize=14, fontweight='bold')
    plt.xlabel("Euclidean Distance", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300) # High DPI for journal submission
    plt.show()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", choices=["humi", "feta"], required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--weights", help="Path to the saved .pt or .tar model weights", required=True
    )
    args = parser.parse_args()

    # 1. Load Config
    config_data = Config(args.config).get_config_dict()
    hyperparams = config_data["hyperparams"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Extract Hyperparameters
    batch_size = hyperparams["batch_size"]
    epoch_batch_count = hyperparams["epoch_batch_count"]
    action_type = config_data["data"]["action_type"]
    imu_type = hyperparams["imu_type"]

    # 3. Initialize Dataset and Dataloader
    print(f"Initializing Dataset: {args.dataname}...")
    if args.dataname == "humi":
        splits = read_pickle(
            os.path.join(config_data["folders"]["data_dir"], "splits.pickle")
        )
        train_dataset = HUMITrainDataset(
            batch_size=batch_size,
            epoch_batch_count=epoch_batch_count,
            action=action_type,
            training_file=os.path.join(
                config_data["folders"]["root_dir"],
                config_data["folders"]["data_dir"],
                "training_scroll_imu_data_all.pickle",
            ),
            user_list=splits["training"],
            imu_type=imu_type,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataset = HUMITestDataset(
            action=action_type,
            validation_file=os.path.join(
                config_data["folders"]["data_dir"], "testing_scroll_imu_data_all.pickle"
            ),
            imu_type=imu_type,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    elif args.dataname == "feta":
        splits = read_pickle(
            os.path.join(config_data["folders"]["data_dir"], "splits.pickle")
        )
        train_dataset = FETATrainDataset(
            data_root=os.path.join(config_data["folders"]["data_dir"], "processed"),
            batch_size=batch_size,
            epoch_batch_count=epoch_batch_count,
            action=action_type,
            user_list=splits["training"],
            imu_type=imu_type,
            data_type="float64",
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=4
        )
        test_dataset = FETATestDataset(
            data_root=os.path.join(config_data["folders"]["data_dir"], "processed"),
            action=action_type,
            user_list=splits["testing"],
            imu_type=imu_type,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # 4. Initialize Model
    print("Initializing BehaveFormer...")
    scroll_feature_dim = hyperparams["scroll_feature_dim"]
    imu_feature_dim = hyperparams["num_imu"] * 12 if hyperparams["num_imu"] > 0 else 0
    model = BehaveFormer(
        scroll_feature_dim,
        imu_feature_dim,
        config_data["data"]["scroll_sequence_len"],
        config_data["data"]["imu_sequence_len"],
        hyperparams["target_len"],
        hyperparams["gre_k"],
        hyperparams["scroll_temporal_heads"],
        hyperparams["scroll_channel_heads"],
        hyperparams["imu_temporal_heads"],
        hyperparams["imu_channel_heads"],
        hyperparams["num_prototypes"],
        imu_type=imu_type,
    ).to(device)

    # 5. Load Weights gracefully (handles both .pt state_dict and .tar checkpoint dicts)
    print(f"Loading weights from {args.weights}...")
    checkpoint = torch.load(args.weights, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 6. Run Visualizations
    # visualize_pure_latent_spread(
    #     model,
    #     train_dataloader,
    #     device,
    #     imu_type,
    #     catalog_path=CATALOG_PATH,
    #     target_epoch=TARGET_EPOCH,
    # )
    # for i in [35]:  # Run multiple times to see different test users
    #     explain_test_user_behavior(
    #         model,
    #         test_dataset,
    #         device,
    #         catalog_path=CATALOG_PATH,
    #         target_epoch=TARGET_EPOCH,
    #         imu_type=imu_type,
    #         top_k=1,
    #         target_index=i,
    #     )
    # explain_multiple_users_behavioral_barcodes_2x2(
    #     model,
    #     test_dataset,
    #     device,
    #     target_epoch=TARGET_EPOCH,
    #     target_indices=[0, 13, 21, 56],
    #     highlight_protos=[3, 15],
    #     imu_type=imu_type,
    #     output_dir=OUTPUT_DIR,
    # )
    # visualize_user_behavior_consistency_humidb(
    #     model,
    #     test_dataset,
    #     device,
    #     target_user_idx=56,  # Change this index to analyze different test users
    #     target_epoch=TARGET_EPOCH,
    #     imu_type=imu_type,
    # )
    # generate_turing_test_study(
    #     model,
    #     test_dataset,
    #     device,
    #     catalog_path=CATALOG_PATH,
    #     target_epoch=TARGET_EPOCH,
    #     imu_type=imu_type,
    #     output_folder=os.path.join(OUTPUT_DIR, "turing_test_trials"),
    #     num_trials=20,
    # )

    # plot_different_prototypes_3d_static(
    #     CATALOG_PATH, prototypes_to_plot=[3, 15], target_epoch=TARGET_EPOCH, type="Different"
    # )
    # animate_different_prototypes_3d(
    #     CATALOG_PATH, prototypes_to_plot=[5, 7, 11, 13], target_epoch=TARGET_EPOCH, type="Different"
    # )
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=0, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=3, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=5, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=6, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=8, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=10, target_epoch=TARGET_EPOCH)
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=15, target_epoch=TARGET_EPOCH)

    intra_distances, inter_distances, mean_intra, mean_inter, mean_feature_std = quantify_signature_stability(model, test_dataset, device, imu_type=imu_type)
    plot_distance_distributions(intra_distances, inter_distances, output_path=os.path.join(OUTPUT_DIR, "consistency_test_distance_distribution.png"))

if __name__ == "__main__":
    main()
