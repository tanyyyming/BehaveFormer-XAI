import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.animation as animation
from adjustText import adjust_text

# Import your existing modules
from model.dataset import HUMITrainDataset, FETATrainDataset
from model.behaveformer import BehaveFormer
from utils.config import Config
from utils.utils import read_pickle


# --- Configuration ---
CATALOG_PATH = "work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20260305_022958/checkpoints/prototype_catalog.pkl"
OUTPUT_DIR = "prototype_vis"


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


def animate_comet_trajectories(catalog_path, prototypes_to_plot, tail_length=5):
    print(f"Loading prototype catalog from {catalog_path}...")
    with open(catalog_path, "rb") as f:
        full_catalog = pickle.load(f)

    # Handle nested epoch dictionary if present
    if isinstance(list(full_catalog.keys())[0], int) and isinstance(full_catalog[list(full_catalog.keys())[0]], dict):
        target_epoch = max(full_catalog.keys())
        catalog = full_catalog[target_epoch]
    else:
        catalog = full_catalog

    # Setup the plot axes
    fig, axes = plt.subplots(1, len(prototypes_to_plot), figsize=(5 * len(prototypes_to_plot), 6))
    if len(prototypes_to_plot) == 1:
        axes = [axes]

    # Store data for the animation loop
    all_x = []
    all_y = []
    lines = []
    dots = []
    max_frames = 0

    for idx, p_idx in enumerate(prototypes_to_plot):
        ax = axes[idx]
        
        if p_idx not in catalog:
            print(f"Warning: Prototype {p_idx} not found in catalog.")
            all_x.append(np.array([]))
            all_y.append(np.array([]))
            continue
            
        entry = catalog[p_idx]
        u = entry["source_user"]
        s = entry.get("source_sess", entry.get("source_session"))

        try:
            # Extract directly from the catalog's stored data
            scroll_seq = entry["data"][0]  
            raw_x = scroll_seq[:, 0]
            raw_y = scroll_seq[:, 1]
            
            all_x.append(raw_x)
            all_y.append(raw_y)
            max_frames = max(max_frames, len(raw_x))
            
            # Initialize empty line (the comet tail) and dot (the finger)
            line, = ax.plot([], [], color='royalblue', linewidth=2.5, alpha=0.8)
            dot, = ax.plot([], [], 'ro', markersize=10, zorder=5, label='Finger Position')
            
            lines.append(line)
            dots.append(dot)

            # Lock to absolute screen coordinates [0, 1] for true physical scale
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.invert_yaxis() # Phone screens have Y=0 at the top

            # Fix the geometric distortion! 
            # Aspect parameter = (Visual Height of 1 Unit) / (Visual Width of 1 Unit)
            # 19.5 / 9 is a standard modern phone screen ratio (~2.16)
            PHONE_ASPECT_RATIO = 19.5 / 9.0  
            ax.set_aspect(PHONE_ASPECT_RATIO)

            ax.set_title(f"Prototype {p_idx}\n(User {u}, Sess {s})", fontweight='bold')
            ax.set_xlabel("Screen X")
            ax.set_ylabel("Screen Y")
            ax.grid(True, linestyle='--', alpha=0.4)
            if idx == 0:
                ax.legend(loc='upper right')

        except Exception as e:
            print(f"Failed to load data for P{p_idx}. Error: {e}")
            all_x.append(np.array([]))
            all_y.append(np.array([]))

    plt.suptitle("Comet Tail Scroll Trajectories", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # --- Animation Update Function ---
    def update(frame):
        updated_artists = []
        for i in range(len(prototypes_to_plot)):
            if len(all_x[i]) == 0:
                continue
            
            # If one sequence finishes earlier than the others, hold it on its last frame
            f = min(frame, len(all_x[i]) - 1)
            
            # The Magic Logic:
            # If we reached the absolute end of the sequence, show the WHOLE trajectory.
            # Otherwise, only show the recent `tail_length` window to hide the mess.
            if f == len(all_x[i]) - 1:
                start_idx = 0
                lines[i].set_alpha(0.4) # Dim the final whole-trajectory slightly
            else:
                start_idx = max(0, f - tail_length)
                lines[i].set_alpha(0.8)
            
            # Update the trailing line
            lines[i].set_data(all_x[i][start_idx:f+1], all_y[i][start_idx:f+1])
            
            # Update the finger dot
            dots[i].set_data([all_x[i][f]], [all_y[i][f]])
            
            updated_artists.extend([lines[i], dots[i]])
            
        return updated_artists

    print(f"Generating animation with {max_frames} frames... This may take a minute.")
    # Extra frames added to the end so it pauses on the fully revealed image before looping
    ani = animation.FuncAnimation(fig, update, frames=max_frames + 20, interval=200, blit=True)
    plt.show()
    
    # Save as GIF using Pillow
    save_path = os.path.join(OUTPUT_DIR, "comet_trajectories.gif")
    ani.save(save_path, writer='pillow', fps=5)
    print(f"\nSaved animation to {save_path}!")


def visualize_pure_latent_spread(model, dataloader, device, imu_type, num_samples=2000):
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

    print("2. Extracting prototype weights...")
    proto_weights = model.prototype_layer.prototypes.data.cpu()
    proto_weights = F.normalize(proto_weights, p=2, dim=1).numpy()
    num_protos = proto_weights.shape[0]

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
                color="darkred"
            )
            texts.append(text)
            
        # Let adjust_text repel the labels and draw connecting lines
        adjust_text(
            texts, 
            ax=ax, 
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.7, shrinkA=5, shrinkB=2),
            expand_points=(1.5, 1.5) # Adds a little extra breathing room around the stars
        )

        ax.set_title(f"Perplexity = {p}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(loc="upper right")

    plt.suptitle(
        "Pure Latent Space Spread of 16 Prototypes (No PCA)",
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

    # # 6. Run Visualizations
    # visualize_pure_latent_spread(model, train_dataloader, device, imu_type)
    # animate_comet_trajectories(CATALOG_PATH, prototypes_to_plot=[0, 6, 8, 15])
    animate_comet_trajectories(CATALOG_PATH, prototypes_to_plot=[0, 3, 5, 10])


if __name__ == "__main__":
    main()
