import pickle
import torch
import numpy as np
import torch.nn.functional as F
from model.dataset import HUMITestDataset
from model.behaveformer import BehaveFormer
from utils.plot import *

class Xai:
    # --- Constants ---
    SCROLL_FEATURE_NAMES = ["x","y","fft_x","fft_y","fd_x","fd_y","sd_x","sd_y"]
    IMU_FEATURE_NAMES = [
        # accel (a_)
        "a_x","a_y","a_z","a_fft_x","a_fft_y","a_fft_z","a_fd_x","a_fd_y","a_fd_z","a_sd_x","a_sd_y","a_sd_z",
        # gyro (g_)
        "g_x","g_y","g_z","g_fft_x","g_fft_y","g_fft_z","g_fd_x","g_fd_y","g_fd_z","g_sd_x","g_sd_y","g_sd_z",
        # mag (m_)
        "m_x","m_y","m_z","m_fft_x","m_fft_y","m_fft_z","m_fd_x","m_fd_y","m_fd_z","m_sd_x","m_sd_y","m_sd_z",
    ]
    BASELINE_SCROLL = torch.tensor([0.5] * 2 + [0.0] * 6)  # for scroll features
    BASELINE_IMU = torch.tensor([0.0] * 36)                # for imu features

    # --- Private Plotting Helpers ---
    @staticmethod
    def _plot_attribution_analysis(
        attr_tensor: Optional[torch.Tensor],
        title_prefix: str,
        feature_names: list[str],
        scroll: Optional[torch.Tensor] = None,
        signed: bool = True,
        out_dir: Optional[str] = None
    ):
        """
        Helper to bundle the 3 standard attribution plots:
        1. plot_attr_heatmap
        2. plot_time_profile
        3. plot_feature_profile
        """
        if attr_tensor is None:
            return

        # Create save paths if out_dir is provided
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            heatmap_path = os.path.join(out_dir, "heatmap.png")
            time_path = os.path.join(out_dir, "time_profile.png")
            feat_path = os.path.join(out_dir, "feature_profile.png")
        else:
            heatmap_path, time_path, feat_path = None, None, None

        plot_attr_heatmap(
            attr_tensor,
            title=f"{title_prefix} IG",
            signed=signed,
            feature_names=feature_names,
            out_path=heatmap_path
        )
        
        scroll_xy_data = None
        if scroll is not None:
            scroll_xy_data = (scroll[:, 0], scroll[:, 1])
        plot_time_profile(
            attr_tensor,
            title=f"{title_prefix} time profile",
            signed=False,
            out_path=time_path,
            scroll_xy=scroll_xy_data
        )

        plot_feature_profile(
            attr_tensor,
            title=f"{title_prefix} feature profile",
            feature_names=feature_names,
            signed=False,
            out_path=feat_path
        )
    
    @staticmethod
    def _plot_aggregated_attribution_analysis(
        mean_scroll_feat_importance: torch.Tensor,
        mean_scroll_time_importance: torch.Tensor,
        scroll_features: list[str],
        mean_imu_feat_importance: Optional[torch.Tensor],
        mean_imu_time_importance: Optional[torch.Tensor],
        imu_features: Optional[list[str]],
        out_dir: Optional[str]
    ):
        """
        Plots the aggregated feature and time profiles from the
        results of aggregate_integrated_gradients.
        """
        # --- Define base output path ---
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # --- SCROLL ---
        # -- Plot Ranked Feature Profile --
        sorted_indices_scroll = mean_scroll_feat_importance.argsort(descending=True)
        sorted_scroll_features = [scroll_features[idx] for idx in sorted_indices_scroll]
        sorted_scroll_importance = mean_scroll_feat_importance[sorted_indices_scroll]
        
        plot_feature_profile(
            sorted_scroll_importance.unsqueeze(0),
            title="Global Scroll Feature Importance (Ranked)",
            feature_names=sorted_scroll_features, # Use sorted labels
            signed=False,
            out_path=os.path.join(out_dir, "genuine_scroll_ranked_features.png") if out_dir else None
        )

        # -- Plot Time Profile --
        plot_time_profile(
            mean_scroll_time_importance.unsqueeze(1),
            title="Global Scroll Time Importance",
            signed=False,
            out_path=os.path.join(out_dir, "genuine_scroll_time_profile.png") if out_dir else None
        )

        # --- IMU ---
        if mean_imu_feat_importance is not None and mean_imu_time_importance is not None:
            # -- Plot Ranked Feature Profile --
            sorted_indices_imu = mean_imu_feat_importance.argsort(descending=True)
            sorted_imu_features = [imu_features[idx] for idx in sorted_indices_imu]
            sorted_imu_importance = mean_imu_feat_importance[sorted_indices_imu]

            plot_feature_profile(
                sorted_imu_importance.unsqueeze(0),
                title="Global IMU Feature Importance (Ranked)",
                feature_names=sorted_imu_features,
                signed=False,
                out_path=os.path.join(out_dir, "genuine_imu_ranked_features.png") if out_dir else None
            )

            # -- Plot Time Profile --
            plot_time_profile(
                mean_imu_time_importance.unsqueeze(1),
                title="Global IMU Time Importance",
                signed=False,
                out_path=os.path.join(out_dir, "genuine_imu_time_profile.png") if out_dir else None
            )

    @staticmethod
    def _plot_attention_analysis(
        attn_t: Optional[torch.Tensor],
        attn_c: Optional[torch.Tensor],
        title_prefix: str,
        channel_feature_names: Optional[list[str]],
        out_dir: Optional[str] = None
    ):
        """
        Helper to bundle the 2 standard attention rollout/flow plots for both
        temporal (T) and channel (C) attention.
        1. plot_square_heatmap
        2. plot_token_importance
        """
        # Create save paths if out_dir is provided
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            t_heatmap_path = os.path.join(out_dir, "temporal_heatmap.png")
            t_importance_path = os.path.join(out_dir, "temporal_importance.png")
            c_heatmap_path = os.path.join(out_dir, "channel_heatmap.png")
            c_importance_path = os.path.join(out_dir, "channel_importance.png")
        else:
            t_heatmap_path, t_importance_path = None, None
            c_heatmap_path, c_importance_path = None, None

        # --- Temporal Plots ---
        if attn_t is not None:
            plot_square_heatmap(
                attn_t,
                title=f"{title_prefix} - temporal",
                x_label="source time",
                y_label="target time",
                out_path=t_heatmap_path
            )
            plot_token_importance(
                attn_t, reduce="col",
                title=f"{title_prefix} temporal importance (source→all)",
                x_label="time index",
                out_path=t_importance_path
            )
        
        # --- Channel Plots ---
        if attn_c is not None:
            plot_square_heatmap(
                attn_c,
                title=f"{title_prefix} - channel",
                x_label="source channel",
                y_label="target channel",
                out_path=c_heatmap_path
            )
            plot_token_importance(
                attn_c, reduce="col",
                title=f"{title_prefix} channel importance (source→all)",
                x_label="channel",
                xticklabels=channel_feature_names,
                out_path=c_importance_path
            )

    @staticmethod
    def compute_integrated_gradients_negmean(
        model,
        scroll,                 # (1, T_s, F_s)
        imu=None,               # (1, T_i, F_i) or None
        enroll_vectors=None,    # (E, D)
        baseline_scroll=None,
        baseline_imu=None,
        steps=50,
        distance_type="euclidean"   # "euclidean" or "cosine"
    ):
        """
        Integrated gradients using s = -(1/E) * sum_i distance(embedding, enroll_vectors[i]).
        Returns attributions for scroll and imu.
        """
        assert enroll_vectors is not None, "Need the full set of enrolment embeddings"
        model.eval()

        if baseline_scroll is None:
            baseline_scroll = Xai.BASELINE_SCROLL
        if imu is not None and baseline_imu is None:
            baseline_imu = Xai.BASELINE_IMU
        # move enrolment vectors to device and add batch dim for broadcasting
        enroll_vectors = enroll_vectors.to(scroll.device)  # shape (E, D)

        accum_grad_scroll = torch.zeros_like(scroll)
        accum_grad_imu = torch.zeros_like(imu) if imu is not None else None

        for alpha in np.linspace(0.0, 1.0, steps):
            interp_scroll = baseline_scroll + alpha * (scroll - baseline_scroll)
            interp_scroll.requires_grad_(True)
            if imu is not None:
                interp_imu = baseline_imu + alpha * (imu - baseline_imu)
                interp_imu.requires_grad_(True)
                z = model([interp_scroll.float(), interp_imu.float()])   # (1, D)
            else:
                interp_imu = None
                z = model(interp_scroll.float())                # (1, D)

            score = Xai.compute_similarity_score(z, enroll_vectors, distance_type=distance_type)

            score.backward()
            accum_grad_scroll += interp_scroll.grad.detach()
            if imu is not None:
                accum_grad_imu += interp_imu.grad.detach()
            model.zero_grad()

        # scale by path length
        attr_scroll = (scroll - baseline_scroll) * accum_grad_scroll / steps
        attr_scroll = attr_scroll.squeeze(0)
        if imu is not None:
            attr_imu = (imu - baseline_imu) * accum_grad_imu / steps
            attr_imu = attr_imu.squeeze(0)
        else:
            attr_imu = None

        return attr_scroll, attr_imu

    @staticmethod
    def compute_attention_rollout(attn_maps: list[torch.Tensor], residual_beta=0.2):
        # attn_maps: list of attention map of the form (B, H, L, L)
        B, _, L, _ = attn_maps[0].shape
        eye = torch.eye(L, device=attn_maps[0].device).unsqueeze(0).expand(B, L, L) # (B, L, L)
        R = eye.clone()
        for a in attn_maps:
            A = a.mean(dim=1)                  # avg heads -> (B, L, L)
            # Add residual and (re)normalize per paper: A' = (1-β)A + βI
            A = (1.0 - residual_beta) * A + residual_beta * eye
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)  # row norm
            R = A @ R
        
        # try only return the averaged attention map of the last layer
        # return attn_maps[-1].mean(dim=1) / attn_maps[-1].mean(dim=1).sum(dim=-1, keepdim=True)  # (B, L, L)
        return R  # (B, L, L)

    @staticmethod
    def compute_attention_flow(attn_maps: list[torch.Tensor]):
        """
        Implements 'Attention Flow' per Abnar & Zuidema (2020)
        attn_maps: list of attention weights (B, H, N, N) or (B, N, N)
        returns: (B, N, N) flow matrix
        """
        # Average over heads
        A_list = [a.mean(dim=1) if a.dim() == 4 else a for a in attn_maps]

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        logA = [torch.log(a + eps) for a in A_list]

        # Sum log-attention across layers (i.e. product of attention matrices)
        log_sum = torch.stack(logA, dim=0).sum(dim=0)  # (B, N, N)

        # Apply softmax row-wise to normalize
        flow = torch.softmax(log_sum, dim=-1)
        return flow
    
    @staticmethod
    def compute_similarity_score(test_vector, enroll_vectors, distance_type="euclidean"):
        """
        Compute similarity score between test_vector and enroll_vectors. 
        The score is such that the higher the score, the more similar.

        test_vector: (1, F)
        enroll_vectors: (E, F)
        """
        # Compute distance to each enrolment embedding
        if distance_type == "euclidean":
            # (E,) distances
            dist = torch.linalg.norm(test_vector - enroll_vectors, dim=-1)  # broadcasting to (E, F)
            score = -(dist.mean())                             # scalar
        elif distance_type == "cosine":
            t_norm = torch.nn.functional.normalize(test_vector, dim=1)      # (1, F)
            E_norm = torch.nn.functional.normalize(enroll_vectors, dim=1)  # (E, F)
            # Cosine distance = 1 - cos
            cos_sim = torch.matmul(E_norm, t_norm.t()).squeeze()   # (E,)
            score = cos_sim.mean()                                # negative cosine distance is just cos similarity
        return score

    @staticmethod
    def mask_channels(x: torch.Tensor, feat_idx, baseline):
        """
        Replace the selected features feat_idx with baseline.
        x: (1, T, F)
        baseline: scalar, (F,), or (1,1,F)
        """
        x = x.clone()
        if isinstance(baseline, torch.Tensor):
            if baseline.dim() == 1:
                baseline = baseline.view(1, 1, -1)
            baseline = baseline.to(x.device).to(x.dtype)
            x[:, :, feat_idx] = baseline[:, :, feat_idx]
        else:
            x[:, :, feat_idx] = baseline
        return x

    @staticmethod
    def mask_time_range(x: torch.Tensor, t0, t1, baseline):
        """
        Zero/baseline a time slice [t0, t1) along the time dimension.
        x: (1, T, F)
        baseline: scalar, (F,), or (1,1,F)
        """
        x = x.clone()
        if isinstance(baseline, torch.Tensor):
            if baseline.dim() == 1:
                baseline = baseline.view(1, 1, -1)
            baseline = baseline.to(x.device).to(x.dtype)
            x[:, t0:t1, :] = baseline
        else:
            x[:, t0:t1, :] = baseline
        return x

    @staticmethod
    def use_integrated_gradients(
        feature_embeddings, 
        test_dataset: HUMITestDataset,
        model: BehaveFormer, 
        num_enroll_sessions, 
        user_id=0,
        out_dir: Optional[str] = None
    ):
        """
        feature_embeddings: (num_users, num_sessions, num_seqs, feature_dim)
        """
        num_users, num_sessions, num_seqs, _ = feature_embeddings.size()

        enroll_vectors = feature_embeddings[user_id, :num_enroll_sessions]
        sess_idx = 0  # use first session for testing
        genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(user_id, sess_idx, 0)
        imposter_scroll, imposter_imu = test_dataset.get_sample_from_user((user_id + 1) % num_users, sess_idx, 0)

        genuine_scroll_attr, genuine_imu_attr = Xai.compute_integrated_gradients_negmean(
            model,
            scroll=genuine_scroll,
            imu=genuine_imu,
            enroll_vectors=enroll_vectors,
        )

        imposter_scroll_attr, imposter_imu_attr = Xai.compute_integrated_gradients_negmean(
            model,
            scroll=imposter_scroll,
            imu=imposter_imu,
            enroll_vectors=enroll_vectors,
        )
        
        # Get feature names from constants
        scroll_features = Xai.SCROLL_FEATURE_NAMES if genuine_scroll_attr.shape[1] == 8 else None
        imu_features = Xai.IMU_FEATURE_NAMES if genuine_imu_attr is not None and genuine_imu_attr.shape[1] == 36 else None

        # -------- Genuine sample --------
        Xai._plot_attribution_analysis(
            genuine_scroll_attr,
            title_prefix=f"Genuine scroll (user {user_id})",
            feature_names=scroll_features,
            scroll=genuine_scroll.squeeze(0),
            out_dir=os.path.join(out_dir, "integrated_gradients", f"user_{user_id}", "genuine_scroll") if out_dir else None
        )
        if genuine_imu_attr is not None:
            Xai._plot_attribution_analysis(
                genuine_imu_attr,
                title_prefix=f"Genuine IMU (user {user_id})",
                feature_names=imu_features,
                scroll=genuine_scroll.squeeze(0).repeat_interleave(2, dim=0),
                out_dir=os.path.join(out_dir, "integrated_gradients", f"user_{user_id}", "genuine_imu") if out_dir else None
            )

        # -------- Impostor sample --------
        Xai._plot_attribution_analysis(
            imposter_scroll_attr,
            title_prefix=f"Impostor scroll (to user {user_id})",
            feature_names=scroll_features,
            scroll=imposter_scroll.squeeze(0),
            out_dir=os.path.join(out_dir, "integrated_gradients", f"user_{user_id}", "impostor_scroll") if out_dir else None
        )
        if imposter_imu_attr is not None:
            Xai._plot_attribution_analysis(
                imposter_imu_attr,
                title_prefix=f"Impostor IMU (to user {user_id})",
                feature_names=imu_features,
                scroll=imposter_scroll.squeeze(0).repeat_interleave(2, dim=0),
                out_dir=os.path.join(out_dir, "integrated_gradients", f"user_{user_id}", "impostor_imu") if out_dir else None
            )

    def aggregate_integrated_gradients(
        feature_embeddings, 
        test_dataset: HUMITestDataset,
        model: BehaveFormer, 
        num_enroll_sessions, 
        distance_type="euclidean",
        out_dir: Optional[str] = None
    ):
        """
        Calculates the mean normalized feature importance across all users
        in the test set.
        
        feature_embeddings: Must be the reshaped tensor 
                            (num_users, num_sessions, num_seqs, feature_dim)
        """
        num_users = test_dataset.num_users
        all_scroll_feat_importances = []
        all_imu_feat_importances = []
        all_scroll_time_importances = []
        all_imu_time_importances = []

        # Get feature names from constants
        scroll_features = Xai.SCROLL_FEATURE_NAMES
        imu_features = Xai.IMU_FEATURE_NAMES
        
        print(f"Starting global ig attribution calculation for {num_users} users...")
        
        for user_id in range(num_users):
            enroll_vectors = feature_embeddings[user_id, :num_enroll_sessions]

            genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(
                user_id, 0, 0
            )

            genuine_scroll_attr, genuine_imu_attr = Xai.compute_integrated_gradients_negmean(
                model,
                scroll=genuine_scroll,
                imu=genuine_imu,
                enroll_vectors=enroll_vectors,
                baseline_scroll=Xai.BASELINE_SCROLL,
                baseline_imu=Xai.BASELINE_IMU,
                distance_type=distance_type
            )

            # Collapse either dimension (feature / time)
            # (F,)
            scroll_feat_importance = genuine_scroll_attr.abs().mean(dim=0)
            # (T,)
            scroll_time_importance = genuine_scroll_attr.abs().mean(dim=1)
            
            # Normalize (so it sums to 1) and store
            if scroll_feat_importance.sum() > 0:
                all_scroll_feat_importances.append(scroll_feat_importance / scroll_feat_importance.sum())
            if scroll_time_importance.sum() > 0:
                all_scroll_time_importances.append(scroll_time_importance / scroll_time_importance.sum())

            if genuine_imu_attr is not None:
                imu_feat_importance = genuine_imu_attr.abs().mean(dim=0)
                imu_time_importance = genuine_imu_attr.abs().mean(dim=1)
                if imu_feat_importance.sum() > 0:
                    all_imu_feat_importances.append(imu_feat_importance / imu_feat_importance.sum())
                if imu_time_importance.sum() > 0:
                    all_imu_time_importances.append(imu_time_importance / imu_time_importance.sum())

        print("Starting aggregation...")

        # --- Aggregate Scroll ---
        mean_scroll_feat_importance = torch.stack(all_scroll_feat_importances, dim=0).mean(dim=0)
        mean_scroll_time_importance = torch.stack(all_scroll_time_importances, dim=0).mean(dim=0)

        # --- Aggregate IMU ---
        mean_imu_feat_importance, mean_imu_time_importance = None, None
        if len(all_imu_feat_importances) > 0:
            mean_imu_feat_importance = torch.stack(all_imu_feat_importances, dim=0).mean(dim=0)
            mean_imu_time_importance = torch.stack(all_imu_time_importances, dim=0).mean(dim=0)
        
        print("Aggregation done. Plotting...")
        # --- Call the internal plotting function ---
        Xai._plot_aggregated_attribution_analysis(
            mean_scroll_feat_importance, mean_scroll_time_importance, scroll_features,
            mean_imu_feat_importance, mean_imu_time_importance, imu_features,
            out_dir=os.path.join(out_dir, "integrated_gradients", "aggregated_results") if out_dir else None
        )

    @staticmethod
    def use_attention_rollout(
        test_dataset: HUMITestDataset, 
        model: BehaveFormer, 
        user_id=0,
        out_dir: Optional[str] = None
    ):
        """
        Use attention rollout to get attention maps for scroll and imu.
        """
        scroll, imu = test_dataset.get_sample_from_user(user_id, 0, 0)

        with torch.no_grad():
            _, scroll_t_attn_maps, scroll_c_attn_maps = model.behave_transformer(scroll.float(), return_attn_weights=True)
            if imu is not None:
                _, imu_t_attn_maps, imu_c_attn_maps = model.imu_transformer(imu.float(), return_attn_weights=True)
        
        scroll_t_rollout = Xai.compute_attention_rollout(scroll_t_attn_maps).squeeze(0)
        scroll_c_rollout = Xai.compute_attention_rollout(scroll_c_attn_maps).squeeze(0)
        
        imu_t_rollout, imu_c_rollout = None, None
        if imu is not None:
            imu_t_rollout = Xai.compute_attention_rollout(imu_t_attn_maps).squeeze(0)
            imu_c_rollout = Xai.compute_attention_rollout(imu_c_attn_maps).squeeze(0)

        # Get feature names from constants
        scroll_features = Xai.SCROLL_FEATURE_NAMES if scroll_c_rollout.shape[0] == 8 else None
        imu_features = Xai.IMU_FEATURE_NAMES if imu_c_rollout is not None and imu_c_rollout.shape[0] == 36 else None

        # Visualize SCROLL
        Xai._plot_attention_analysis(
            scroll_t_rollout, scroll_c_rollout,
            title_prefix=f"Scroll Attention Rollout (user {user_id})",
            channel_feature_names=scroll_features,
            out_dir=os.path.join(out_dir, "attention_rollout", f"user_{user_id}", "scroll_rollout") if out_dir else None
        )

        # Visualize IMU
        Xai._plot_attention_analysis(
            imu_t_rollout, imu_c_rollout,
            title_prefix=f"IMU Attention Rollout (user {user_id})",
            channel_feature_names=imu_features,
            out_dir=os.path.join(out_dir, "attention_rollout", f"user_{user_id}", "imu_rollout") if out_dir else None
        )

    @staticmethod
    def use_attention_flow(
        test_dataset: HUMITestDataset, 
        model: BehaveFormer, 
        user_id=0,
        out_dir: Optional[str] = None
    ):
        """
        Use attention flow to get attention maps for scroll and imu.
        """
        # Get a sample from the dataset
        scroll, imu = test_dataset.get_sample_from_user(user_id, 0, 0)

        # Get the raw attention maps from the model
        with torch.no_grad():
            _, scroll_t_attn_maps, scroll_c_attn_maps = model.behave_transformer(scroll.float(), return_attn_weights=True)
            if imu is not None:
                _, imu_t_attn_maps, imu_c_attn_maps = model.imu_transformer(imu.float(), return_attn_weights=True)
        
        # Compute the attention flow matrices
        scroll_t_flow = Xai.compute_attention_flow(scroll_t_attn_maps).squeeze(0)
        scroll_c_flow = Xai.compute_attention_flow(scroll_c_attn_maps).squeeze(0)
        
        imu_t_flow, imu_c_flow = None, None
        if imu is not None:
            imu_t_flow = Xai.compute_attention_flow(imu_t_attn_maps).squeeze(0)
            imu_c_flow = Xai.compute_attention_flow(imu_c_attn_maps).squeeze(0)

        # Get feature names from constants
        scroll_features = Xai.SCROLL_FEATURE_NAMES if scroll_c_flow.shape[0] == 8 else None
        imu_features = Xai.IMU_FEATURE_NAMES if imu_c_flow is not None and imu_c_flow.shape[0] == 36 else None

        # Visualize SCROLL (reusing the same plot helper)
        Xai._plot_attention_analysis(
            scroll_t_flow, scroll_c_flow,
            title_prefix=f"Scroll (user {user_id}) - Attention Flow",
            channel_feature_names=scroll_features,
            out_dir=os.path.join(out_dir, "attention_flow", f"user_{user_id}", "scroll_flow") if out_dir else None
        )

        # Visualize IMU (reusing the same plot helper)
        Xai._plot_attention_analysis(
            imu_t_flow, imu_c_flow,
            title_prefix=f"IMU (user {user_id}) - Attention Flow",
            channel_feature_names=imu_features,
            out_dir=os.path.join(out_dir, "attention_flow", f"user_{user_id}", "imu_flow") if out_dir else None
        )

    @staticmethod
    def use_occlusion_sensitivity(
        feature_embeddings, 
        test_dataset: HUMITestDataset, 
        model: BehaveFormer, 
        num_enroll_sessions,
        which: str,
        user_id=0,
        out_dir: Optional[str] = None
    ):
        enroll_vectors = feature_embeddings[user_id, :num_enroll_sessions]
        genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(user_id, 0, 0)
        
        model.eval()
        if genuine_imu is not None:
            test_vector = model([genuine_scroll.float(), genuine_imu.float()])
        else:
            test_vector = model(genuine_scroll.float())

        s0 = Xai.compute_similarity_score(
            test_vector, enroll_vectors, distance_type="euclidean"
        )
        out = {}
        feat_dict = {name: idx for idx, name in enumerate(Xai.IMU_FEATURE_NAMES)} if which == "imu" else {name: idx for idx, name in enumerate(Xai.SCROLL_FEATURE_NAMES)}
        for name, idxs in feat_dict.items():
            if which == "imu" and genuine_imu is not None:
                masked_input = [genuine_scroll.float(), Xai.mask_channels(genuine_imu, idxs, baseline=Xai.BASELINE_IMU).float()]
            elif which == "scroll":
                masked_scroll = Xai.mask_channels(genuine_scroll, idxs, baseline=Xai.BASELINE_SCROLL).float()
                masked_input = [masked_scroll, genuine_imu.float()] if genuine_imu is not None else masked_scroll
            else:
                continue
            
            model_output = model(masked_input)
            s_mask = Xai.compute_similarity_score(
                model_output, enroll_vectors, distance_type="euclidean"
            ).item()
            out[name] = s0 - s_mask
        
        # --- Plotting ---
        feature_names = list(out.keys())
        attr = torch.tensor(list(out.values()))

        out_path = os.path.join(out_dir, "occlusion_sensitivity", f"user_{user_id}", f"{which}.png") if out_dir else None
        plot_occlusion_sensitivity(
            attr=attr,
            title=f"Occlusion Sensitivity (User {user_id}, {which})",
            xticklabels=feature_names,
            y_label="similarity drop",
            out_path=out_path
        )
    
    @staticmethod
    def aggregate_occlusion_sensitivity(
        feature_embeddings, 
        test_dataset: HUMITestDataset,
        model: BehaveFormer, 
        num_enroll_sessions,
        which: str,
        out_dir: Optional[str] = None
    ):
        """
        Calculates the mean normalized occlusion sensitivity across all users
        in the test set and plots the aggregated results.
        
        feature_embeddings: Must be the reshaped tensor 
                            (num_users, num_sessions, num_seqs, feature_dim)
        """
        num_users = test_dataset.num_users
        all_scores = []

        # Get baseline and feature names
        if which == "imu":
            baseline = Xai.BASELINE_IMU
            feature_names = Xai.IMU_FEATURE_NAMES
            feat_dict = {name: idx for idx, name in enumerate(feature_names)}
        else: # "scroll"
            baseline = Xai.BASELINE_SCROLL
            feature_names = Xai.SCROLL_FEATURE_NAMES
            feat_dict = {name: idx for idx, name in enumerate(feature_names)}
        
        print(f"Starting global occlusion sensitivity calculation for {num_users} users (type: {which})...")
        
        for user_id in range(num_users):
            enroll_vectors = feature_embeddings[user_id, :num_enroll_sessions]
            genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(user_id, 0, 0)
            
            model.eval()
            if genuine_imu is not None:
                test_vector = model([genuine_scroll.float(), genuine_imu.float()])
            else:
                test_vector = model(genuine_scroll.float())

            s0 = Xai.compute_similarity_score(
                test_vector, enroll_vectors, distance_type="euclidean"
            )
            
            out = {}
            for name, idxs in feat_dict.items():
                if which == "imu" and genuine_imu is not None:
                    masked_input = [genuine_scroll.float(), Xai.mask_channels(genuine_imu, idxs, baseline=baseline).float()]
                elif which == "scroll":
                    masked_scroll = Xai.mask_channels(genuine_scroll, idxs, baseline=baseline).float()
                    masked_input = [masked_scroll, genuine_imu.float()] if genuine_imu is not None else masked_scroll
                else:
                    continue
                
                model_output = model(masked_input)
                s_mask = Xai.compute_similarity_score(
                    model_output, enroll_vectors, distance_type="euclidean"
                ).item()
                out[name] = s0 - s_mask

            # Normalize and store scores
            scores = torch.tensor(list(out.values()))
            # We take abs() because a negative drop is also important (it means
            # occluding the feature *helped* similarity, so it was important)
            norm_scores = scores.abs() / (scores.abs().sum() + 1e-8)
            all_scores.append(norm_scores)

        print("Aggregation done. Plotting...")
        
        # --- Aggregate and Plot ---
        mean_scores = torch.stack(all_scores, dim=0).mean(dim=0)
        
        sorted_indices = mean_scores.argsort(descending=True)
        sorted_scores = mean_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        plot_path = None
        if out_dir:
            agg_dir = os.path.join(out_dir, "occlusion_sensitivity", "aggregated_results")
            os.makedirs(agg_dir, exist_ok=True)
            plot_path = os.path.join(agg_dir, f"{which}_ranked.png")

        plot_feature_profile(
            sorted_scores.unsqueeze(0),
            title=f"Global Occlusion Sensitivity (Ranked, {'IMU' if which == 'imu' else 'Scroll'})",
            y_label="mean |Δoutput|",
            feature_names=sorted_names,
            signed=False,
            out_path=plot_path
        )

    @staticmethod
    def plot_scroll_x(
        test_dataset: HUMITestDataset,
        user_id_list: list[int],
        out_dir: Optional[str] = None
    ):
        """
        Plots the horizontal finger position over time for different users.
        """
        sess_idx = 0  # use first session for testing
        user_to_scroll_x = {}
        for user_id in user_id_list:
            genuine_scroll, _ = test_dataset.get_sample_from_user(user_id, sess_idx, 0)
            user_to_scroll_x[user_id] = genuine_scroll.squeeze(0)[:, 0]
        plot_scroll_x_for_users(user_to_scroll_x, os.path.join(out_dir, "scroll_x_profiles.png") if out_dir else None)


def project_prototypes(model, train_dataloader, device, epoch, save_dir, imu_type):
    """
    For hard-projecting prototypes onto real data after each epoch during prototype-based training.
    1. Collects latent vectors from the training set.
    2. Updates prototype weights to match the nearest real data.
    3. Saves a 'Catalog' mapping prototype IDs to source metadata.
    """
    model.eval()

    all_latents = []
    all_meta = []

    # 1. Collect Candidates (Latent Vectors) from Training Data
    # We iterate through the dataloader to get a representative sample of the data distribution
    with torch.no_grad():
        for _, item in enumerate(train_dataloader):
            # Unpack the new return tuple (Note the extra metadata arg)
            anchor, _, _, anchor_meta = item

            if imu_type != 'none':
                _, latent_vector = model([anchor[0].to(device).float(), anchor[1].to(device).float()])
            else:
                _, latent_vector = model(anchor[0].to(device).float())

            all_latents.append(latent_vector.cpu())

            # Store metadata for each sample in this batch
            # anchor_meta is a dict of lists (batch_size), we need to unroll it
            batch_size = latent_vector.shape[0]
            for b in range(batch_size):
                meta_item = {
                    'user_idx': anchor_meta['user_idx'][b].item(),
                    'sess_idx': anchor_meta['sess_idx'][b].item(),
                    'seq_idx': anchor_meta['seq_idx'][b].item()
                }
                all_meta.append(meta_item)

    # Concatenate all collected latents: (N_samples, target_len)
    all_latents = torch.cat(all_latents, dim=0)

    # 2. Normalize for Cosine Similarity
    # Shape: (N_samples, target_len)
    candidates_norm = F.normalize(all_latents, p=2, dim=1, eps=1e-6)
    # Shape: (num_prototypes, target_len)
    prototypes_norm = F.normalize(model.prototype_layer.prototypes.data.cpu(), p=2, dim=1, eps=1e-6)

    # 3. Calculate Similarity Matrix
    # Shape: (num_prototypes, N_samples)
    similarity_matrix = torch.mm(prototypes_norm, candidates_norm.t())

    # 4. Find Nearest Neighbors
    # For each prototype, find the index of the single best matching training sample
    best_match_indices = torch.argmax(similarity_matrix, dim=1)

    # 5. HARD UPDATE & CATALOGING
    catalog = {}

    for proto_idx, best_match_idx in enumerate(best_match_indices):
        best_match_idx = best_match_idx.item()

        # (1) Update the Prototype Weight to be exactly the real latent vector
        # We use the ORIGINAL latent vector (not normalized) to preserve magnitude info if needed,
        # though for cosine sim, direction is what matters.
        new_weight = all_latents[best_match_idx]
        model.prototype_layer.prototypes.data[proto_idx] = new_weight.to(device)

        best_match_user_idx, best_match_sess_idx, best_match_seq_idx = (
            all_meta[best_match_idx]["user_idx"],
            all_meta[best_match_idx]["sess_idx"],
            all_meta[best_match_idx]["seq_idx"],
        )

        raw_data = train_dataloader.dataset.load_data(
            best_match_user_idx,
            best_match_sess_idx,
            best_match_seq_idx
        )

        # (2) Log the Projection Information
        catalog[proto_idx] = {
            "epoch": epoch,
            "source_user": best_match_user_idx,
            "source_session": best_match_sess_idx,
            "source_seq": best_match_seq_idx,
            "cosine_similarity": similarity_matrix[proto_idx, best_match_idx].item(),
            "data": raw_data  # Store the actual raw data used for this prototype
        }

    # 6. Save the Catalog
    master_catalog_path = os.path.join(save_dir, "prototype_catalog.pkl")
    
    # 1. Load existing master catalog if it exists
    if os.path.exists(master_catalog_path):
        try:
            with open(master_catalog_path, 'rb') as f:
                full_catalog = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            # Handle corrupt/empty files safely
            full_catalog = {}
    else:
        full_catalog = {}

    # 2. Update with current epoch data
    full_catalog[epoch] = catalog
    
    # 3. Write back to disk (Overwrite)
    with open(master_catalog_path, 'wb') as f:
        pickle.dump(full_catalog, f)
