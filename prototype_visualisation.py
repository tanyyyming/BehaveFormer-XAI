import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
CATALOG_PATH = "work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20260211_174037/checkpoints/prototype_catalog.pkl"
OUTPUT_DIR = "prototype_vis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# Run either mode
# visualize_clean_catalog(CATALOG_PATH, mode='grid')
visualize_clean_catalog(CATALOG_PATH, mode="overlay")
