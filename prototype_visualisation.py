import math
import os
import argparse
import pickle
from typing import Literal
from xml.parsers.expat import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.animation as animation
from adjustText import adjust_text
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation as R

from model.dataset import HUMITrainDataset, FETATrainDataset
from model.behaveformer import BehaveFormer
from utils.config import Config
from utils.utils import read_pickle
from imu_visualisation import get_phone_box
from xai import project_prototypes


# --- Configuration ---
CATALOG_PATH = "work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20260305_022958/checkpoints/prototype_catalog.pkl"
OUTPUT_DIR = "prototype_vis"


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
    x_arrays, y_arrays, imu_arrays, titles, super_title, save_path, tail_length=5
):
    """Core rendering function handling synchronized 2D Scroll and 3D Phone animations."""
    num_plots = len(x_arrays)
    if num_plots == 0:
        print("No valid data to animate.")
        return

    # Setup the plot axes: 1 row for scroll, 1 row for 3D phone, 1 row for gyro, 1 row for mag fd
    fig = plt.figure(figsize=(4 * num_plots, 16))

    axes_scroll, axes_3d, axes_gyr, axes_mag = [], [], [], []
    lines, dots, phones, normal_lines = [], [], [], []
    r0_inverses = (
        []
    )  # <-- Track the inverse of Frame 0 for each plot so that every phone starts in the same relative orientation
    gyr_cursors, mag_cursors = [], []  # Track the moving vertical lines
    max_frames = max([len(x) for x in x_arrays] + [len(i) for i in imu_arrays] + [0])
    all_quats = []

    # 1. Pre-compute Quaternions (Doing this inside the animation loop is too slow)
    print("Pre-computing 3D sensor fusion...")
    for imu_seq in imu_arrays:
        # Reverse scaling based on dataset.py logic
        acc = imu_seq[:, 0:3] * 10.0
        gyr = imu_seq[:, 12:15]
        mag = imu_seq[:, 24:27] * 1000.0

        madgwick = Madgwick(acc=acc, gyr=gyr, mag=mag)
        Q = madgwick.Q

        # Sanitize broken Quaternions
        # Calculate the norm (length) of every quaternion in the sequence
        norms = np.linalg.norm(Q, axis=1)
        # Find any that are 0 length, or turned into NaNs from zero-padded sensor data
        bad_quats = (norms < 1e-6) | np.isnan(norms)
        # Override broken ones with a default "Identity" quaternion (ahrs expects [w, x, y, z])
        Q[bad_quats] = [1.0, 0.0, 0.0, 0.0]

        all_quats.append(Q)

    # 2. Setup the Subplots
    for idx in range(num_plots):
        # --- Row 1: 2D Scroll ---
        ax_s = fig.add_subplot(4, num_plots, idx + 1)
        (line,) = ax_s.plot([], [], color="royalblue", linewidth=2.5, alpha=0.8)
        (dot,) = ax_s.plot(
            [], [], "ro", markersize=10, zorder=5, label="Finger" if idx == 0 else ""
        )
        lines.append(line)
        dots.append(dot)

        ax_s.set_xlim([0.0, 1.0])
        ax_s.set_ylim([0.0, 1.0])
        ax_s.invert_yaxis()
        ax_s.set_aspect(19.5 / 9.0)
        ax_s.set_title(titles[idx], fontsize=10, fontweight="bold")
        ax_s.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax_s.legend(loc="upper right")
        axes_scroll.append(ax_s)

        # --- Row 2: 3D Phone ---
        ax_3d = fig.add_subplot(4, num_plots, idx + 1 + num_plots, projection="3d")
        ax_3d.set_xlim([-0.6, 0.6])
        ax_3d.set_ylim([-0.6, 0.6])
        ax_3d.set_zlim([-0.6, 0.6])
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("Relative Phone Orientation", fontsize=9)

        # Set the camera angle and remove axes for a cleaner look
        ax_3d.view_init(elev=0, azim=-90)
        ax_3d.set_axis_off()

        # Grab the Frame 0 Quaternion for initialisation of the phone model and arrow to avoid jumping cameras
        Q = all_quats[idx]
        rot = R.from_quat([Q[0, 1], Q[0, 2], Q[0, 3], Q[0, 0]])

        # Calculate and save the inverse to "zero out" the starting position
        r0_inverses.append(rot.inv())

        vertices, faces = get_phone_box()
        phone = Poly3DCollection(
            faces, alpha=0.2, facecolors="cyan", edgecolors="black"
        )
        ax_3d.add_collection3d(phone)
        phones.append(phone)

        (normal_line,) = ax_3d.plot(
            [0, 0], [0, 0], [0, 0.8], color="red", linewidth=3, zorder=10
        )
        normal_lines.append(normal_line)
        axes_3d.append(ax_3d)

        # --- Row 3: Gyroscope Setup ---
        ax_g = fig.add_subplot(4, num_plots, idx + 1 + 2 * num_plots)
        time_steps = np.arange(len(imu_arrays[idx]))
        gyr_data = imu_arrays[idx][:, 12:15]  # Extract raw Gyro

        ax_g.plot(time_steps, gyr_data[:, 0], label="X", color="tab:red", alpha=0.7)
        # ax_g.plot(time_steps, gyr_data[:, 1], label="Y", color="tab:green", alpha=0.7)
        # ax_g.plot(time_steps, gyr_data[:, 2], label="Z", color="tab:blue", alpha=0.7)

        # Create the moving time cursor at frame 0
        cursor_g = ax_g.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        gyr_cursors.append(cursor_g)

        ax_g.set_title("Gyroscope (rad/s)", fontsize=9)
        ax_g.set_xlim([0, len(time_steps) - 1])
        ax_g.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_g.legend(loc="upper right", fontsize=7)
        axes_gyr.append(ax_g)

        # --- Row 4: Magnetometer FD (1st Derivative) Setup ---
        ax_m = fig.add_subplot(4, num_plots, idx + 1 + 3 * num_plots)

        # Use 30:33 (mag fd)
        mag_fd_data = imu_arrays[idx][:, 30:33]

        # Plot the features exactly as the model sees them
        # ax_m.plot(time_steps, mag_fd_data[:, 0], color="tab:red", alpha=0.7, label="X")
        ax_m.plot(
            time_steps, mag_fd_data[:, 1], color="tab:green", alpha=0.7, label="Y"
        )
        # ax_m.plot(time_steps, mag_fd_data[:, 2], color="tab:blue", alpha=0.7, label="Z")

        # Create the moving time cursor at frame 0
        cursor_m = ax_m.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
        mag_cursors.append(cursor_m)

        ax_m.set_title(
            "Magnetometer 1st Derivative (m_fd)", fontsize=9, fontweight="bold"
        )
        ax_m.set_xlim([0, len(time_steps) - 1])
        ax_m.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax_m.legend(loc="upper right", fontsize=7)
        axes_mag.append(ax_m)

    plt.suptitle(super_title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 3. Synchronized Animation Update
    def update(frame):
        updated_artists = []
        for i in range(num_plots):
            len_s = len(x_arrays[i])
            len_i = len(imu_arrays[i])

            # 1. IMU drives the clock directly since IMU has longer sequences
            fi = min(frame, len_i - 1)

            # 2. Scroll maps proportionally to the IMU progress
            progress = (fi + 1) / len_i
            fs = math.ceil(progress * len_s) - 1

            # --- Update 2D Scroll ---
            if fs == len_s - 1:
                start_idx = 0
                lines[i].set_alpha(0.4)
            else:
                start_idx = max(0, fs - tail_length)
                lines[i].set_alpha(0.8)

            lines[i].set_data(
                x_arrays[i][start_idx : fs + 1], y_arrays[i][start_idx : fs + 1]
            )
            dots[i].set_data([x_arrays[i][fs]], [y_arrays[i][fs]])

            # --- Update 3D Phone ---
            Q = all_quats[i]
            rot = R.from_quat([Q[fi, 1], Q[fi, 2], Q[fi, 3], Q[fi, 0]])
            # Apply the inverse of Frame 0 to keep the phone's starting orientation consistent across different sequences
            rot = rot * r0_inverses[i]
            vertices, faces = get_phone_box()
            rotated_faces = [[rot.apply(v) for v in face] for face in faces]

            phones[i].set_verts(rotated_faces)

            # Calculate the new tip position of the normal vector (pointing out of the screen) after rotation
            pointer_tip = rot.apply([0, 0, 0.8])
            normal_lines[i].set_data_3d(
                [0, pointer_tip[0]], [0, pointer_tip[1]], [0, pointer_tip[2]]
            )

            # --- Update Time Cursors for Gyr and Mag ---
            # set_xdata moves the vertical line to the current IMU frame
            gyr_cursors[i].set_xdata([fi])
            mag_cursors[i].set_xdata([fi])

            updated_artists.extend(
                [
                    lines[i],
                    dots[i],
                    phones[i],
                    normal_lines[i],
                    gyr_cursors[i],
                    mag_cursors[i],
                ]
            )

        return updated_artists

    print(f"Generating synchronized 3D animation with {max_frames} frames...")
    ani = animation.FuncAnimation(
        fig, update, frames=max_frames + 20, interval=150, blit=False
    )
    plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="pillow", fps=5)
        print(f"Saved to {save_path}")


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
        OUTPUT_DIR, f"comet_trajectories_{type.lower()}_prototypes_3D.gif"
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
            "Pure Latent Space Spread of 16 Prototypes" + f" (Epoch {target_epoch})"
            if target_epoch
            else ""
        ),
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pure_latent_spread.png"), dpi=200)
    print("\nDone! Saved as pure_latent_spread.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataname", choices=["humi", "feta"], required=True)
    # parser.add_argument("-c", "--config", required=True)
    # parser.add_argument(
    #     "--weights", help="Path to the saved .pt or .tar model weights", required=True
    # )
    # args = parser.parse_args()

    # # 1. Load Config
    # config_data = Config(args.config).get_config_dict()
    # hyperparams = config_data["hyperparams"]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # 2. Extract Hyperparameters
    # batch_size = hyperparams["batch_size"]
    # epoch_batch_count = hyperparams["epoch_batch_count"]
    # action_type = config_data["data"]["action_type"]
    # imu_type = hyperparams["imu_type"]

    # # 3. Initialize Dataset and Dataloader
    # print(f"Initializing Dataset: {args.dataname}...")
    # if args.dataname == "humi":
    #     splits = read_pickle(
    #         os.path.join(config_data["folders"]["data_dir"], "splits.pickle")
    #     )
    #     train_dataset = HUMITrainDataset(
    #         batch_size=batch_size,
    #         epoch_batch_count=epoch_batch_count,
    #         action=action_type,
    #         training_file=os.path.join(
    #             config_data["folders"]["root_dir"],
    #             config_data["folders"]["data_dir"],
    #             "training_scroll_imu_data_all.pickle",
    #         ),
    #         user_list=splits["training"],
    #         imu_type=imu_type,
    #     )
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    # elif args.dataname == "feta":
    #     splits = read_pickle(
    #         os.path.join(config_data["folders"]["data_dir"], "splits.pickle")
    #     )
    #     train_dataset = FETATrainDataset(
    #         data_root=os.path.join(config_data["folders"]["data_dir"], "processed"),
    #         batch_size=batch_size,
    #         epoch_batch_count=epoch_batch_count,
    #         action=action_type,
    #         user_list=splits["training"],
    #         imu_type=imu_type,
    #         data_type="float64",
    #     )
    #     train_dataloader = DataLoader(
    #         train_dataset, batch_size=batch_size, num_workers=4
    #     )

    # # 4. Initialize Model
    # print("Initializing BehaveFormer...")
    # scroll_feature_dim = hyperparams["scroll_feature_dim"]
    # imu_feature_dim = hyperparams["num_imu"] * 12 if hyperparams["num_imu"] > 0 else 0
    # model = BehaveFormer(
    #     scroll_feature_dim,
    #     imu_feature_dim,
    #     config_data["data"]["scroll_sequence_len"],
    #     config_data["data"]["imu_sequence_len"],
    #     hyperparams["target_len"],
    #     hyperparams["gre_k"],
    #     hyperparams["scroll_temporal_heads"],
    #     hyperparams["scroll_channel_heads"],
    #     hyperparams["imu_temporal_heads"],
    #     hyperparams["imu_channel_heads"],
    #     hyperparams["num_prototypes"],
    #     imu_type=imu_type,
    # ).to(device)

    # # 5. Load Weights gracefully (handles both .pt state_dict and .tar checkpoint dicts)
    # print(f"Loading weights from {args.weights}...")
    # checkpoint = torch.load(args.weights, map_location=device)
    # if "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"])
    # else:
    #     model.load_state_dict(checkpoint)

    # project_prototypes(model, train_dataloader, device, epoch=220, save_dir=CATALOG_PATH.removesuffix("prototype_catalog.pkl"), imu_type=imu_type)

    # # 6. Run Visualizations
    # visualize_pure_latent_spread(
    #     model,
    #     train_dataloader,
    #     device,
    #     imu_type,
    #     catalog_path=CATALOG_PATH,
    #     target_epoch=220,
    # )
    animate_different_prototypes_3d(
        CATALOG_PATH, prototypes_to_plot=[0, 6, 8, 15], target_epoch=220, type="Similar"
    )
    animate_different_prototypes_3d(
        CATALOG_PATH, prototypes_to_plot=[0, 3, 5, 10], target_epoch=220, type="Different"
    )
    # animate_prototype_neighbors_3d(CATALOG_PATH, prototype_id=8, target_epoch=220)
    # animate_prototype_neighbors(CATALOG_PATH, prototype_id=3, target_epoch=220)
    # animate_prototype_neighbors(CATALOG_PATH, prototype_id=5, target_epoch=220)
    # animate_prototype_neighbors(CATALOG_PATH, prototype_id=6, target_epoch=220)
    # animate_prototype_neighbors(CATALOG_PATH, prototype_id=8, target_epoch=220)
    # animate_prototype_neighbors(CATALOG_PATH, prototype_id=15, target_epoch=220)


if __name__ == "__main__":
    main()
