''' Define the HGTAN model '''
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from training.tools import *
import torch.nn.functional as F
from WaveHGRN.layers import *
import torch.fft


class WaveHGRN(nn.Module):
    """
    Wavelet-based Adaptive Hypergraph Routing Network (WaveHGRN).
    Final prediction model combining multi-scale features and backbone trends.
    """

    def __init__(
            self,
            num_stock,  # N
            tem_dim,  # F (Hidden dimension from Hyper-inference)
            n_hid,  # F' (Dimension after Hyper-conv)
            n_class,  # Number of classes (usually 2 for trend prediction)
            feature,  # C (Input feature dimension)
            dropout,  # Dropout rate
            scale_num,  # K (Wavelet levels)
            hyper_edge,  # M (Number of hyperedges)
            path_num,  # G (Number of routing experts)
            window_size,  # T
            mem_dim  # d_model (Routing query/key dimension)
    ):
        super(WaveHGRN, self).__init__()
        self.dropout = dropout
        self.scale_num = scale_num

        # 1. Wavelet Decomposition Module (Sec 4.2)
        self.WaveletModule = WaveletModule(num_features=feature, num_levels=scale_num)

        # 2. Hyperedge Correlation Inference Module (Sec 4.3)
        self.Hyper_rel_model = Hyper_rel_model(
            num_levels=scale_num,
            input_c=feature,
            n_hid=tem_dim,
            M=hyper_edge,
            lambda_param=1.2  # beta
        )

        # 3. Hypergraph Routing Network (Sec 4.4)
        self.hgrn = HGRN_Module(
            n_hid=tem_dim,
            n_hid2=n_hid,
            M=hyper_edge,
            num_experts=path_num,
            d_k=mem_dim,
            num_scales=scale_num + 1
        )

        # 4. Output Layer (Sec 4.5, Eq. 20)
        # Input to this layer is [H_accum || H_backbone]
        # H_accum dimension: n_hid (F')
        # H_backbone dimension: tem_dim (F)
        self.tgt_word_prj = nn.Linear(n_hid + tem_dim, n_class)

        # Normalization layers
        self.ln_final = nn.LayerNorm(n_hid + tem_dim)

    def forward(self, src_seq):
        """
        Args:
            src_seq: Input market signals [T, N, C]
        Returns:
            output: Classification logits or probabilities [N, n_class]
            orth_loss: Regularization loss from hyperedge dictionary
        """
        # --- Stage 1: Adaptive Wavelet Transform ---
        # Fre_features: List of [T_k, N, C]
        Fre_features = self.WaveletModule(src_seq)

        # --- Stage 2: Hyperedge Inference ---
        # incidence_matrices: List of [N, M]
        # node_features_list: List of [N, F]
        # total_orth_loss: Scalar
        incidence_matrices, node_features_list, total_orth_loss = self.Hyper_rel_model(Fre_features)

        # --- Stage 3: Hypergraph Routing & Fusion ---
        # H_accum: Final fused representation [N, F'] (h_accum_n_1 in paper)
        H_accum, routing_loss = self.hgrn(incidence_matrices, node_features_list)

        # --- Stage 4: Output and Concatenation (Eq. 20) ---
        # h_backbone: Node features from the lowest frequency band (h_n_K)
        # Note: In WaveletModule, the last element is always the Lowest frequency (Trend)
        h_backbone = node_features_list[-1]  # [N, F]

        # Concatenate fused features with backbone trend: [N, F' + F]
        combined_output = torch.cat((H_accum, h_backbone), dim=-1)

        # Apply normalization and dropout
        combined_output = self.ln_final(combined_output)
        combined_output = F.dropout(combined_output, self.dropout, training=self.training)

        # Final projection (Eq. 20)
        # seq_logit shape: [N, n_class]
        seq_logit = F.elu(self.tgt_word_prj(combined_output))

        output = F.log_softmax(seq_logit, dim=1)


        return output, total_orth_loss, routing_loss
