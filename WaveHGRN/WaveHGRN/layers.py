
import numpy as np
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
import math
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import NMF
np.set_printoptions(threshold=np.inf)
from training.tools import *
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




class AdaptiveWavelet2DLayer(nn.Module):
    """
    Adaptive Cross-Channel 2D Wavelet Decomposition Layer
    Implements Eq. (3) and Eq. (5) from the paper.
    """

    def __init__(self, in_features, init_l_filter, init_h_filter, constraint_threshold=0.05):
        super(AdaptiveWavelet2DLayer, self).__init__()

        self.in_features = in_features  # C
        self.theta = len(init_l_filter)  # Filter length
        self.constraint_threshold = constraint_threshold  # tau

        # Padding for the temporal dimension to achieve 'same' padding
        self.padding_t = (self.theta - 1) // 2

        # 2D Kernels: [Out_Channels, In_Channels, Theta, Feature_Dim]
        # Shape: [C, 1, theta, C]
        self.conv_l = nn.Conv2d(in_channels=1, out_channels=in_features,
                                kernel_size=(self.theta, in_features),
                                stride=(1, 1),
                                padding=(self.padding_t, 0),
                                bias=True)

        self.conv_h = nn.Conv2d(in_channels=1, out_channels=in_features,
                                kernel_size=(self.theta, in_features),
                                stride=(1, 1),
                                padding=(self.padding_t, 0),
                                bias=True)

        # Off-diagonal Mask to constrain inter-channel interference
        # mask[c, 0, :, c] = 0 means the primary feature (diagonal) is not clamped
        mask = torch.ones_like(self.conv_l.weight.data)
        for c in range(in_features):
            mask[c, 0, :, c] = 0
        self.register_buffer('off_diag_mask', mask)

        # Initialize with Daubechies coefficients
        wavelet_l = torch.tensor(init_l_filter, dtype=torch.float).flip(0)
        wavelet_h = torch.tensor(init_h_filter, dtype=torch.float).flip(0)
        self._init_weights(self.conv_l.weight, wavelet_l)
        self._init_weights(self.conv_h.weight, wavelet_h)

        self.activation = nn.Tanh()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def _init_weights(self, weight_tensor, wavelet_coeffs):
        with torch.no_grad():
            # Initialize off-diagonals with minute noise (Eq. 4)
            nn.init.uniform_(weight_tensor, -0.01, 0.01)
            # Initialize diagonals with standard wavelet basis
            for c in range(self.in_features):
                weight_tensor[c, 0, :, c] = wavelet_coeffs

    def get_constrained_weights(self, conv_layer):
        """ Applies the soft-constraint (tau) to off-diagonal interactions """
        w = conv_layer.weight
        w_clamped = torch.clamp(w, -self.constraint_threshold, self.constraint_threshold)
        # Use original weight for diagonal, clamped weight for off-diagonal
        return w * (1 - self.off_diag_mask) + w_clamped * self.off_diag_mask

    def forward(self, x):
        """
        Args:
            x: [N, 1, T, C] (Batch, Channel_In, Time, Features)
        Returns:
            x_l, x_h: [N, C, T/2]
        """
        # 1. Generate effective weights using the mask (Avoids in-place errors)
        w_l = self.get_constrained_weights(self.conv_l)
        w_h = self.get_constrained_weights(self.conv_h)

        # 2. Convolution: [N, 1, T, C] * [C, 1, theta, C] -> [N, C, T, 1]
        out_l = F.conv2d(x, w_l, bias=self.conv_l.bias, stride=(1, 1), padding=(self.padding_t, 0))
        out_h = F.conv2d(x, w_h, bias=self.conv_h.bias, stride=(1, 1), padding=(self.padding_t, 0))

        # 3. Activation and dimension reduction: [N, C, T, 1] -> [N, C, T]
        out_l = self.activation(out_l).squeeze(-1)
        out_h = self.activation(out_h).squeeze(-1)

        # 4. Dyadic Downsampling (Eq. 5): [N, C, T] -> [N, C, T/2]
        x_l = self.pool(out_l)
        x_h = self.pool(out_h)

        return x_l, x_h


class WaveletModule(nn.Module):
    """
    Multilevel Recursive Decomposition Framework
    Implements the structural logic of Eq. (6).
    """

    def __init__(self, num_features, num_levels=3):
        super(WaveletModule, self).__init__()
        self.num_levels = num_levels  # K

        # Daubechies 4 (db4) coefficients
        self.base_l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        self.base_h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]

        # Shared Adaptive 2D layer across all levels K
        self.shared_wavelet_layer = AdaptiveWavelet2DLayer(
            in_features=num_features,
            init_l_filter=self.base_l_filter,
            init_h_filter=self.base_h_filter
        )

        self.instance_norm = nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x):
        """
        Args:
            x: [T, N, C] (Time, Batch, Features)
        Returns:
            List of tensors: [X^(1,H), ..., X^(K,H), X^(K,L)]
            Each element has shape [T_k, N, C]
        """
        T, N, C = x.shape

        # 1. Instance Normalization (Standardizing financial time series)
        # [T, N, C] -> [N, C, T]
        x_norm = self.instance_norm(x.permute(1, 2, 0))

        # 2. Reshape for 2D Conv: [N, 1, T, C]
        x_curr = x_norm.permute(0, 2, 1).unsqueeze(1)

        high_freq_list = []

        # 3. Recursive Decomposition
        for k in range(self.num_levels):
            # x_l, x_h are [N, C, T_next]
            x_l, x_h = self.shared_wavelet_layer(x_curr)

            # Store High Frequency (Detail): [N, C, T_k] -> [T_k, N, C]
            high_freq_list.append(x_h.permute(2, 0, 1))

            # Prepare Low Frequency (Approximation) for next level
            if k < self.num_levels - 1:
                # [N, C, T_next] -> [N, T_next, C] -> [N, 1, T_next, C]
                x_curr = x_l.permute(0, 2, 1).unsqueeze(1)
            else:
                # Final Low Frequency (Trend)
                x_final_low = x_l.permute(2, 0, 1)

        # Final Set S: {High_1, ..., High_K, Low_K}
        return high_freq_list + [x_final_low]


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Input Shape: (Time, Batch, Feature)
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [T, N, F]
        return x + self.pe[:x.size(0), :]


class SingleScaleHyperInference(nn.Module):
    """
    Inference module for a single frequency band (k-th level).
    Implements: Transformer Encoding, Affinity Calculation, and Denoising.
    """

    def __init__(self, input_c, n_hid, n_heads=4, dropout=0.1):
        super(SingleScaleHyperInference, self).__init__()
        self.n_hid = n_hid  # F

        # Temporal dependency extraction (Eq. 7)
        self.input_proj = nn.Linear(input_c, n_hid)
        self.pos_encoder = PositionalEncoding(n_hid)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_hid, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, C_dict, W_Q, W_K, beta):
        """
        Args:
            x: [T_k, N, C]
            C_dict: Shared hyperedge dictionary [M, F]
            W_Q, W_K: Shared projection layers
            beta: Sparsification hyperparameter
        Returns:
            I_k: Sparse incidence matrix [N, M]
            H_k: Node feature matrix [N, F]
        """
        T_k, N, C = x.shape

        # 1. Node Feature Extraction (Eq. 7)
        # [T_k, N, C] -> [T_k, N, F]
        x_proj = self.input_proj(x)
        x_pe = self.pos_encoder(x_proj)
        # Transformer: [T_k, N, F]
        H_seq = self.transformer_encoder(x_pe)
        # Global Average Pooling over Time: [N, F]
        H_k = torch.mean(H_seq, dim=0)

        # 2. Hyperedge Affinity Calculation (Eq. 8)
        # H_Q: [N, F], C_K: [M, F]
        H_Q = W_Q(H_k)
        C_K = W_K(C_dict)

        # Scaled dot-product with Sigmoid
        # score: [N, M]
        score = torch.matmul(H_Q, C_K.transpose(0, 1)) / math.sqrt(self.n_hid)
        S_k = torch.sigmoid(score)

        # 3. Hypergraph Sparsification & Denoising (Eq. 9, 10)
        # Global average intensity mu: scalar
        mu_k = torch.mean(S_k)

        # Hard thresholding (Eq. 10)
        mask = (S_k >= beta * mu_k).float()
        I_k = S_k * mask

        return I_k, H_k


class Hyper_rel_model(nn.Module):
    """
    Multilevel Hyperedge Inference Module with shared Dictionary.
    """

    def __init__(self, num_levels, input_c, n_hid, M=20, lambda_param=1.2):
        super(Hyper_rel_model, self).__init__()
        self.M = M
        self.n_hid = n_hid
        self.beta = lambda_param
        self.num_scales = num_levels + 1

        # Shared Latent Hyperedge Prototype Dictionary (Sec 4.3.1)
        # Matrix C: [M, F]
        self.C = nn.Parameter(torch.Tensor(M, n_hid))
        nn.init.xavier_uniform_(self.C)

        # Shared Projections for alignment (Eq. 8)
        self.W_Q = nn.Linear(n_hid, n_hid, bias=False)
        self.W_K = nn.Linear(n_hid, n_hid, bias=False)

        # Scale-specific encoders (each scale has unique temporal patterns)
        self.scale_encoders = nn.ModuleList([
            SingleScaleHyperInference(input_c, n_hid)
            for _ in range(self.num_scales)
        ])

    def compute_orth_loss(self):
        """
        Orthogonality Regularization (Eq. 11)
        Ensures diverse risk contagion pathways.
        """
        # gram: [M, M]
        gram = torch.matmul(self.C, self.C.t())
        # Frobenius norm: scalar
        gram_fro = torch.norm(gram, p='fro')
        # Identity matrix: [M, M]
        I_id = torch.eye(self.M, device=self.C.device)

        # Loss: || (CC^T)/||CC^T||_F - I ||_F^2
        normalized_gram = gram / (gram_fro + 1e-8)
        loss_orth = torch.norm(normalized_gram - I_id, p='fro') ** 2
        return loss_orth

    def forward(self, x_list):
        """
        Args:
            x_list: List of tensors from Wavelet module [x_1H, ..., x_KL], each [T_k, N, C]
        Returns:
            incidence_list: List of [N, M]
            node_feat_list: List of [N, F]
            loss_orth: Scalar
        """
        incidence_list = []
        node_feat_list = []

        for i, x_k in enumerate(x_list):
            I_k, H_k = self.scale_encoders[i](x_k, self.C, self.W_Q, self.W_K, self.beta)
            incidence_list.append(I_k)
            node_feat_list.append(H_k)

        loss_orth = self.compute_orth_loss()

        return incidence_list, node_feat_list, loss_orth





class HGRN_Module(nn.Module):
    """
    Hypergraph Routing Network (HGRN).
    Implements dynamic message passing and bottom-up residual fusion.
    """

    def __init__(self, n_hid, n_hid2, M, num_experts, d_k, d_mid=32, num_scales=4):
        """
        Args:
            n_hid: Input node feature dimension F.
            n_hid2: Output projected dimension F'.
            M: Number of hyperedges.
            num_experts: Number of risk pattern operators G.
            d_k: Dimension of query/key space d_model.
            num_scales: Total frequency bands (K+1).
        """
        super(HGRN_Module, self).__init__()
        self.M = M
        self.d_k = d_k
        self.num_scales = num_scales

        # --- 1. Frequency Subspace State Encoding (Eq. 13) ---
        self.state_mlp = nn.Sequential(
            nn.Linear(n_hid, d_mid),
            nn.Tanh(),  # delta in paper
            nn.Linear(d_mid, d_k)
        )

        # --- 2. Adaptive Hyperedge Weight Routing (Eq. 14, 16) ---
        # Matrix Omega: [G, M]
        self.experts = nn.Parameter(torch.Tensor(num_experts, M))
        nn.init.xavier_uniform_(self.experts)
        # W_proj: [M, d_model]
        self.w_proj = nn.Linear(M, d_k, bias=False)

        # --- 3. Hypergraph Convolution Projection (Eq. 17) ---
        # Theta: [F, F']
        self.theta = nn.Linear(n_hid, n_hid2, bias=False)
        self.conv_activation = nn.LeakyReLU(0.2)

        # --- 4. Structured Fusion (Eq. 19) ---
        # Learnable scalar gates lambda^(k)
        self.lambdas = nn.Parameter(torch.zeros(num_scales))

    def forward(self, incidence_matrices, node_features_list):
        """
        Args:
            incidence_matrices: List of [N, M] tensors (I_k).
            node_features_list: List of [N, F] tensors (H_k).
        Returns:
            H_final: Fused representation [N, F'].
        """
        convoluted_list = []

        # Process each frequency subspace independently
        for k in range(len(node_features_list)):
            I_k = incidence_matrices[k]  # [N, M]
            H_k = node_features_list[k]  # [N, F]
            N = H_k.size(0)

            # === Step 1: State Encoding (Eq. 12, 13) ===
            # Holistic market feature g_k: [1, F]
            g_k = torch.mean(H_k, dim=0, keepdim=True)
            # State query q_k: [1, d_k]
            q_k = self.state_mlp(g_k)

            # === Step 2: Weight Routing (Eq. 14, 15, 16) ===
            # Key vectors k_g: [G, d_k]
            k_experts = self.w_proj(self.experts)
            # Matching scores: [1, G]
            scores = torch.matmul(q_k, k_experts.transpose(0, 1)) / math.sqrt(self.d_k)
            alpha = F.softmax(scores, dim=1)
            # Combined weights w_k: [M]
            w_k = torch.matmul(alpha, self.experts).squeeze(0)

            # === Step 3: Hypergraph Convolution (Eq. 17) ===
            # 3.1 Feature Projection: [N, F] -> [N, F']
            H_proj = self.theta(H_k)

            # 3.2 Degree Calculation (Dynamic)
            # Node degree D_v, Edge degree D_e
            d_v = torch.sum(I_k, dim=1) + 1e-6
            d_e = torch.sum(I_k, dim=0) + 1e-6

            # 3.3 Message Passing: D_v^-0.5 * I * W * D_e^-1 * I^T * D_v^-0.5 * H_proj
            # Left & Right Norm: [N]
            norm_v = torch.pow(d_v, -0.5).unsqueeze(1)  # [N, 1]

            # Aggregation: Node -> Edge [M, F']
            H_norm = H_proj * norm_v
            H_edge = torch.matmul(I_k.t(), H_norm)

            # Scaling at Edge level: W * D_e^-1
            # w_k: [M], d_e: [M]
            edge_scaling = w_k * torch.pow(d_e, -1.0)
            H_edge_scaled = H_edge * edge_scaling.unsqueeze(1)

            # Dispersion: Edge -> Node [N, F']
            H_node = torch.matmul(I_k, H_edge_scaled)
            H_tilde = self.conv_activation(H_node * norm_v)

            convoluted_list.append(H_tilde)

        # === Step 4: Structured Fusion (Bottom-up Eq. 18, 19) ===
        # convoluted_list is [H_high1, H_high2, ..., H_highK, H_lowK]
        low_idx = len(convoluted_list) - 1

        # 4.1 Initialization with the lowest frequency (Trend)
        H_accum = convoluted_list[low_idx]  # [N, F']

        # 4.2 Progressive Residual Correction (from HK down to H1)
        # We iterate backwards from the last high-freq scale to the first
        for i in range(low_idx - 1, -1, -1):
            H_k = convoluted_list[i]
            # Learnable gate lambda
            gate = torch.tanh(self.lambdas[i])
            # Eq. 19: Accumulate residual
            H_accum = H_accum + gate * H_k

        return H_accum


class HGRN_Module(nn.Module):
    """
    Hypergraph Routing Network (HGRN) with Routing Equilibrium Loss.
    Implements dynamic message passing and Eq. (22) for routing diversity.
    """

    def __init__(self, n_hid, n_hid2, M, num_experts, d_k, d_mid=32, num_scales=4):
        """
        Args:
            n_hid: Input node feature dimension F.
            n_hid2: Output projected dimension F'.
            M: Number of hyperedges.
            num_experts: Number of risk pattern operators G.
            d_k: Dimension of query/key space d_model.
            num_scales: Total frequency bands (K+1).
        """
        super(HGRN_Module, self).__init__()
        self.M = M
        self.d_k = d_k
        self.num_scales = num_scales
        self.G = num_experts  # Number of experts

        # --- 1. State Encoding ---
        self.state_mlp = nn.Sequential(
            nn.Linear(n_hid, d_mid),
            nn.Tanh(),
            nn.Linear(d_mid, d_k)
        )

        # --- 2. Weight Routing ---
        self.experts = nn.Parameter(torch.Tensor(num_experts, M))
        nn.init.xavier_uniform_(self.experts)
        self.w_proj = nn.Linear(M, d_k, bias=False)

        # --- 3. Hypergraph Convolution ---
        self.theta = nn.Linear(n_hid, n_hid2, bias=False)
        self.conv_activation = nn.LeakyReLU(0.2)

        # --- 4. Structured Fusion ---
        self.lambdas = nn.Parameter(torch.zeros(num_scales))

    def forward(self, incidence_matrices, node_features_list):
        """
        Args:
            incidence_matrices: List of [N, M] tensors (I_k).
            node_features_list: List of [N, F] tensors (H_k).
        Returns:
            H_final: Fused representation [N, F'].
        """
        convoluted_list = []
        routing_loss = 0.0

        for k in range(len(node_features_list)):
            I_k = incidence_matrices[k]
            H_k = node_features_list[k]

            # === Step 1: State Encoding ===
            g_k = torch.mean(H_k, dim=0, keepdim=True)
            q_k = self.state_mlp(g_k)  # [1, d_k]

            # === Step 2: Weight Routing & Routing Loss (Eq. 22) ===
            k_experts = self.w_proj(self.experts)  # [G, d_k]
            scores = torch.matmul(q_k, k_experts.transpose(0, 1)) / math.sqrt(self.d_k)
            alpha = F.softmax(scores, dim=1)  # [1, G]

            # --- Calculate Routing Loss for this band (Eq. 22) ---
            # alpha is [1, G], calculate variance and mean across G
            alpha_mean = torch.mean(alpha, dim=1)
            alpha_var = torch.var(alpha, dim=1)
            # L_route_k = Var(alpha) / (E[alpha])^2
            routing_loss_k = alpha_var / (alpha_mean ** 2 + 1e-8)
            routing_loss += routing_loss_k.item()

            # Reconstruction of W_k
            w_k = torch.matmul(alpha, self.experts).squeeze(0)

            # === Step 3: Hypergraph Convolution ===
            H_proj = self.theta(H_k)
            d_v = torch.sum(I_k, dim=1) + 1e-6
            d_e = torch.sum(I_k, dim=0) + 1e-6
            norm_v = torch.pow(d_v, -0.5).unsqueeze(1)

            H_norm = H_proj * norm_v
            H_edge = torch.matmul(I_k.t(), H_norm)
            edge_scaling = w_k * torch.pow(d_e, -1.0)
            H_edge_scaled = H_edge * edge_scaling.unsqueeze(1)

            H_node = torch.matmul(I_k, H_edge_scaled)
            H_tilde = self.conv_activation(H_node * norm_v)
            convoluted_list.append(H_tilde)

        # === Step 4: Structured Fusion ===
        low_idx = len(convoluted_list) - 1
        H_accum = convoluted_list[low_idx]
        for i in range(low_idx - 1, -1, -1):
            gate = torch.tanh(self.lambdas[i])
            H_accum = H_accum + gate * convoluted_list[i]

        return H_accum, routing_loss / len(node_features_list)


