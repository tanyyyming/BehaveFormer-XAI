import torch
import numpy as np
from model.dataset import HUMITestDataset
from model.behaveformer import BehaveFormer
from utils.plot import *

class Xai:
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
            baseline_scroll = torch.zeros_like(scroll)
        if imu is not None and baseline_imu is None:
            baseline_imu = torch.zeros_like(imu)
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

            # Compute distance to each enrolment embedding
            if distance_type == "euclidean":
                # (E,) distances
                dist = torch.norm(z - enroll_vectors, p=2, dim=1)  # broadcasting z to (E, D)
                score = -(dist.mean())                             # scalar
            elif distance_type == "cosine":
                z_norm = torch.nn.functional.normalize(z, dim=1)      # (1, D)
                E_norm = torch.nn.functional.normalize(enroll_vectors, dim=1)  # (E, D)
                # Cosine distance = 1 - cos
                cos_sim = torch.matmul(E_norm, z_norm.t()).squeeze()   # (E,)
                score = cos_sim.mean()                                # negative cosine distance is just cos similarity
            else:
                raise ValueError("distance_type must be 'euclidean' or 'cosine'")

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
    def use_integrated_gradients(feature_embeddings, test_dataset: HUMITestDataset, 
                                 model: BehaveFormer, num_enroll_sessions, user_id=0):
        """
        feature_embeddings: (num_users, num_sessions, num_seqs, feature_dim)
        """
        # For humidb, num_seqs = 1
        num_users, num_sessions, num_seqs, _ = feature_embeddings.size()

        enroll_vectors = feature_embeddings[user_id, :num_enroll_sessions] # (num_enroll_sessions, num_seqs, feature_dim)
        genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(user_id, 3, 0)
        imposter_scroll, imposter_imu = test_dataset.get_sample_from_user((user_id + 1) % num_users, 1, 0)

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

        # Optional: feature labels (adjust if yours differ)
        scroll_feature_names = ["x","y","fft_x","fft_y","fd_x","fd_y","sd_x","sd_y"]  # if F_s==8
        # For IMU, example layout (36 features): 3 sensors × 12 features each
        imu_feature_names = [
            # accel (a_)
            "a_x","a_y","a_z","a_fft_x","a_fft_y","a_fft_z","a_fd_x","a_fd_y","a_fd_z","a_sd_x","a_sd_y","a_sd_z",
            # gyro (g_)
            "g_x","g_y","g_z","g_fft_x","g_fft_y","g_fft_z","g_fd_x","g_fd_y","g_fd_z","g_sd_x","g_sd_y","g_sd_z",
            # mag (m_)
            "m_x","m_y","m_z","m_fft_x","m_fft_y","m_fft_z","m_fd_x","m_fd_y","m_fd_z","m_sd_x","m_sd_y","m_sd_z",
        ] if genuine_imu_attr is not None and genuine_imu_attr.shape[1] == 36 else None

        # -------- Genuine sample --------
        plot_attr_heatmap(genuine_scroll_attr,
                        title=f"Genuine scroll IG (user {user_id})",
                        signed=True,
                        feature_names=scroll_feature_names)

        plot_time_profile(genuine_scroll_attr,
                        title=f"Genuine scroll time profile (user {user_id})",
                        signed=False)

        plot_feature_profile(genuine_scroll_attr,
                            title=f"Genuine scroll feature profile (user {user_id})",
                            feature_names=scroll_feature_names,
                            signed=False)

        if genuine_imu_attr is not None:
            plot_attr_heatmap(genuine_imu_attr,
                            title=f"Genuine IMU IG (user {user_id})",
                            signed=True,
                            feature_names=imu_feature_names)
            plot_time_profile(genuine_imu_attr,
                            title=f"Genuine IMU time profile (user {user_id})",
                            signed=False)
            plot_feature_profile(genuine_imu_attr,
                                title=f"Genuine IMU feature profile (user {user_id})",
                                feature_names=imu_feature_names,
                                signed=False)

        # -------- Impostor sample --------
        plot_attr_heatmap(imposter_scroll_attr,
                        title=f"Impostor scroll IG (to user {user_id} template)",
                        signed=True,
                        feature_names=scroll_feature_names)

        plot_time_profile(imposter_scroll_attr,
                        title=f"Impostor scroll time profile (to user {user_id})",
                        signed=False)

        plot_feature_profile(imposter_scroll_attr,
                            title=f"Impostor scroll feature profile (to user {user_id})",
                            feature_names=scroll_feature_names,
                            signed=False)

        if imposter_imu_attr is not None:
            plot_attr_heatmap(imposter_imu_attr,
                            title=f"Impostor IMU IG (to user {user_id} template)",
                            signed=True,
                            feature_names=imu_feature_names)
            plot_time_profile(imposter_imu_attr,
                            title=f"Impostor IMU time profile (to user {user_id})",
                            signed=False)
            plot_feature_profile(imposter_imu_attr,
                                title=f"Impostor IMU feature profile (to user {user_id})",
                                feature_names=imu_feature_names,
                                signed=False)

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
    def use_attention_rollout(test_dataset: HUMITestDataset, model: BehaveFormer, user_id=0):
        """
        Use attention rollout to get attention maps for scroll and imu.
        """
        # get example sample
        scroll, imu = test_dataset.get_sample_from_user(user_id, 3, 0)

        with torch.no_grad():
            _, scroll_t_attn_maps, scroll_c_attn_maps = model.behave_transformer(scroll.float(), return_attn_weights=True)
            if imu is not None:
                _, imu_t_attn_maps, imu_c_attn_maps = model.imu_transformer(imu.float(), return_attn_weights=True)
        
        scroll_t_rollout = Xai.compute_attention_rollout(scroll_t_attn_maps).squeeze(0) # (L, L)
        scroll_c_rollout = Xai.compute_attention_rollout(scroll_c_attn_maps).squeeze(0) # (F, F)
        imu_t_rollout, imu_c_rollout = None, None
        if imu is not None:
            imu_t_rollout = Xai.compute_attention_rollout(imu_t_attn_maps).squeeze(0) # (L, L)
            imu_c_rollout = Xai.compute_attention_rollout(imu_c_attn_maps).squeeze(0) # (F, F)

        # 2) (optional) feature names for nicer x-axis on channel plots
        scroll_feature_names = ["x","y","fft_x","fft_y","fd_x","fd_y","sd_x","sd_y"] if scroll_c_rollout.shape[0] == 8 else None
        imu_feature_names = None
        if imu_c_rollout is not None and imu_c_rollout.shape[0] == 36:
            imu_feature_names = [
                # accel 12
                "a_x","a_y","a_z","a_fft_x","a_fft_y","a_fft_z","a_fd_x","a_fd_y","a_fd_z","a_sd_x","a_sd_y","a_sd_z",
                # gyro 12
                "g_x","g_y","g_z","g_fft_x","g_fft_y","g_fft_z","g_fd_x","g_fd_y","g_fd_z","g_sd_x","g_sd_y","g_sd_z",
                # mag 12
                "m_x","m_y","m_z","m_fft_x","m_fft_y","m_fft_z","m_fd_x","m_fd_y","m_fd_z","m_sd_x","m_sd_y","m_sd_z",
            ]

        # 3) Visualize SCROLL (temporal + channel)
        plot_square_heatmap(
            scroll_t_rollout,
            title=f"Scroll temporal rollout (user {user_id})",
            x_label="source time",
            y_label="target time",
        )
        plot_token_importance(
            scroll_t_rollout, reduce="col",
            title=f"Scroll temporal importance (source→all)",
            x_label="time index",
        )

        plot_square_heatmap(
            scroll_c_rollout,
            title=f"Scroll channel rollout (user {user_id})",
            x_label="source channel",
            y_label="target channel",
        )
        plot_token_importance(
            scroll_c_rollout, reduce="col",
            title=f"Scroll channel importance (source→all)",
            x_label="channel",
            xticklabels=scroll_feature_names,
        )

        # 4) Visualize IMU (if present)
        if imu_t_rollout is not None:
            plot_square_heatmap(
                imu_t_rollout,
                title=f"IMU temporal rollout (user {user_id})",
                x_label="source time",
                y_label="target time",
            )
            plot_token_importance(
                imu_t_rollout, reduce="col",
                title=f"IMU temporal importance (source→all)",
                x_label="time index",
            )

        if imu_c_rollout is not None:
            plot_square_heatmap(
                imu_c_rollout,
                title=f"IMU channel rollout (user {user_id})",
                x_label="source channel",
                y_label="target channel",
            )
            plot_token_importance(
                imu_c_rollout, reduce="col",
                title=f"IMU channel importance (source→all)",
                x_label="channel",
                xticklabels=imu_feature_names,
            )