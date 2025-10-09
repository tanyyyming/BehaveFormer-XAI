import torch
import numpy as np
from model.dataset import HUMITestDataset
from model.behaveformer import BehaveFormer
from utils.plot import plot_attr_heatmap, plot_time_profile, plot_feature_profile

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
        genuine_scroll, genuine_imu = test_dataset.get_sample_from_user(user_id, 1, 0)
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

