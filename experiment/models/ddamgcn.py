import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


class Dynamic_Delay_Aware_Module(nn.Module):
    """
    Dynamic Delay-Aware Module (DDAM) for identifying key traffic nodes
    and dynamically modeling propagation time delays across the network.
    """
    def __init__(self, k_num, channels, length, max_delay=3, dropout=0.1):
        super(Dynamic_Delay_Aware_Module, self).__init__()
        self.k_num = k_num
        self.channels = channels
        self.length = length
        self.max_delay = min(max_delay, max(1, length - 1))

        # Node importance scoring layer
        self.score_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2 if channels > 1 else 1),
            nn.ReLU(),
            nn.Linear(channels // 2 if channels > 1 else 1, 1),
        )

        # Multi-delay projections for delay-aware attention
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.delay_weights = nn.Parameter(torch.ones(self.max_delay + 1) / (self.max_delay + 1))
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, C, N, T)

        Returns
        -------
        torch.Tensor
            Shape (B, C, N, T)
        """
        B, C, N, T = x.shape

        # Permute to (B, N, T, C) for spatial-temporal processing
        x_perm = x.permute(0, 2, 3, 1)

        # 1. Identify key nodes dynamically
        node_summary = x_perm.mean(dim=2)  # (B, N, C)
        scores = self.score_mlp(node_summary).squeeze(-1)  # (B, N)
        k = min(self.k_num, N)
        topk_indices = torch.topk(scores, k=k, dim=1).indices  # (B, k)

        # Gather key node representations across the node dimension
        idx = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B, C, k, T)
        key_x = torch.gather(x, dim=2, index=idx)  # (B, C, k, T)
        key_x_perm = key_x.permute(0, 2, 3, 1)  # (B, k, T, C)

        # 2. Dynamic Propagation Delay Modeling across lags tau
        q = self.q_proj(x_perm)  # (B, N, T, C)
        scale = math.sqrt(C)

        delay_outputs = []
        norm_delay_weights = torch.softmax(self.delay_weights, dim=0)

        for tau in range(self.max_delay + 1):
            if tau == 0:
                delayed_key = key_x_perm
            else:
                # Causal padding along time dimension T (dim=2)
                delayed_key = F.pad(key_x_perm[:, :, :-tau, :], (0, 0, tau, 0))

            k_mat = self.k_proj(delayed_key)  # (B, k, T, C)
            v_mat = self.v_proj(delayed_key)  # (B, k, T, C)

            # Cross-attention between all nodes and key nodes at time t with delay tau
            # Transpose to (B, T, N, C) and (B, T, k, C)
            q_t = q.permute(0, 2, 1, 3)  # (B, T, N, C)
            k_t = k_mat.permute(0, 2, 1, 3)  # (B, T, k, C)
            v_t = v_mat.permute(0, 2, 1, 3)  # (B, T, k, C)

            attn = torch.softmax(torch.matmul(q_t, k_t.transpose(-2, -1)) / scale, dim=-1)  # (B, T, N, k)
            msg = torch.matmul(attn, v_t)  # (B, T, N, C)
            delay_outputs.append(norm_delay_weights[tau] * msg)

        # Aggregate delay-aware messages
        aggregated_msg = sum(delay_outputs).permute(0, 2, 1, 3)  # (B, N, T, C)
        projected = self.out_proj(aggregated_msg)

        # Residual connection and LayerNorm
        out_perm = self.norm(x_perm + self.dropout(projected))
        return out_perm.permute(0, 3, 1, 2).contiguous()  # (B, C, N, T)


class MSTGCN(nn.Module):
    """
    Multiscale Spatiotemporal Graph Convolution Module.
    Combines multiscale temporal convolutions with spatial graph convolutions
    guided by static/adaptive topology and dynamic trend correlation.
    """
    def __init__(self, in_channels, out_channels, num_nodes, length, K=3, Kt=3, dropout=0.3):
        super(MSTGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.length = length
        self.K = K
        self.Kt = Kt

        # Multiscale temporal convolutions (scale 1, scale Kt, scale 2*Kt - 1)
        k1 = 1
        k2 = Kt
        k3 = 2 * Kt - 1

        self.tconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k1), padding=(0, k1 // 2))
        self.tconv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k2), padding=(0, k2 // 2))
        self.tconv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, k3), padding=(0, k3 // 2))

        self.temporal_fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

        # Spatial Chebyshev / diffusion polynomial weights of order K
        self.spatial_weights = nn.Parameter(torch.empty(K, out_channels, out_channels))
        nn.init.xavier_normal_(self.spatial_weights)

        # Trend vs Adjacency weighting parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, Trend_Matrix, A):
        """
        Parameters
        ----------
        v : torch.Tensor
            (B, in_channels, N, T)
        Trend_Matrix : torch.Tensor
            (B, T, N, N)
        A : torch.Tensor
            (N, N) or (B, N, N)

        Returns
        -------
        torch.Tensor
            (B, out_channels, N, T)
        """
        res = self.residual(v)

        # 1. Multiscale Temporal Convolution
        tc1 = F.relu(self.tconv1(v))
        tc2 = F.relu(self.tconv2(v))
        tc3 = F.relu(self.tconv3(v))
        v_temp = F.relu(self.temporal_fuse(torch.cat([tc1, tc2, tc3], dim=1)))  # (B, out_channels, N, T)

        # 2. Spatial Graph Convolution with Multihop / Order K
        v_perm = v_temp.permute(0, 3, 2, 1)  # (B, T, N, out_channels)
        B, T, N, C = v_perm.shape

        # Effective graph combining static/adaptive topology and dynamic trend correlation
        alpha = torch.sigmoid(self.alpha)
        if A.dim() == 2:
            g_static = torch.einsum('nm,btmc->btnc', A, v_perm)
        else:
            g_static = torch.einsum('bnm,btmc->btnc', A, v_perm)

        g_trend = torch.einsum('btnm,btmc->btnc', Trend_Matrix, v_perm)
        h1 = alpha * g_static + (1.0 - alpha) * g_trend

        # Chebyshev recurrence
        hops = [v_perm, h1]
        for k in range(2, self.K):
            if A.dim() == 2:
                hop_static = torch.einsum('nm,btmc->btnc', A, hops[-1])
            else:
                hop_static = torch.einsum('bnm,btmc->btnc', A, hops[-1])
            hop_trend = torch.einsum('btnm,btmc->btnc', Trend_Matrix, hops[-1])
            h_next = 2.0 * (alpha * hop_static + (1.0 - alpha) * hop_trend) - hops[-2]
            hops.append(h_next)

        # Project hops through spatial weights
        out_spatial = torch.zeros_like(v_perm)
        for k in range(self.K):
            out_spatial = out_spatial + torch.einsum('btnc,co->btno', hops[k], self.spatial_weights[k])

        out_spatial = out_spatial.permute(0, 3, 2, 1)  # (B, out_channels, N, T)

        # 3. Residual & Normalization
        out = res + self.dropout(F.relu(out_spatial))
        out_norm = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out_norm.contiguous()


class Encoder(nn.Module):
    def __init__(self, num_nodes, k_num, l, length=12,
                 in_dim=1, residual_channels=32, dilation_channels=32, K=3, Kt=3, dropout=0.3):
        super(Encoder, self).__init__()
        self.l = max(2, l)
        self.block1 = nn.ModuleList()
        self.block1.append(MSTGCN(in_dim, dilation_channels, num_nodes, length, K=K, Kt=Kt, dropout=dropout))
        for _ in range(self.l - 2):
            self.block1.append(MSTGCN(dilation_channels, dilation_channels, num_nodes, length, K=K, Kt=Kt, dropout=dropout))
        self.DDM_encoder = Dynamic_Delay_Aware_Module(k_num, in_dim, length, dropout=dropout)

    def forward(self, v, Trend_Matrix, A):
        v = self.DDM_encoder(v)
        for block in self.block1:
            v = block(v, Trend_Matrix, A)
        return v


class Decoder(nn.Module):
    def __init__(self, num_nodes, k_num, l, length=12,
                 residual_channels=32, dilation_channels=32, K=3, Kt=3, dropout=0.3):
        super(Decoder, self).__init__()
        self.l = max(1, l - 1)
        self.block = nn.ModuleList()
        for _ in range(self.l):
            self.block.append(MSTGCN(dilation_channels, dilation_channels, num_nodes, length, K=K, Kt=Kt, dropout=dropout))
        self.DDM_decoder = Dynamic_Delay_Aware_Module(k_num, dilation_channels, length, dropout=dropout)

    def forward(self, v, Trend_Matrix, A):
        v = self.DDM_decoder(v)
        for block in self.block:
            v = block(v, Trend_Matrix, A)
        return v


class DDAMGCN(BaseModel):
    """
    Multiscale Spatiotemporal Graph Convolutional Networks with Dynamic Delay Awareness (DDAMGCN)
    Reference: Zhu et al., IEEE Transactions on Neural Networks and Learning Systems.

    Input:
        x: (Batch, History, Nodes, Features)
    Output:
        prediction: (Batch, Horizon, Nodes)
    """
    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
        hidden_dim=32,
        dilation_channels=32,
        k_num=10,
        l=2,
        dropout=0.3,
        supports=None,
        kernel_size=2,
        K=3,
        Kt=3,
        **kwargs,
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.k_num = min(k_num, num_nodes)
        self.l = l
        self.dropout = dropout
        self.supports = supports
        self.dilation_channels = dilation_channels
        self.K = K
        self.Kt = Kt

        # Adaptive graph parameter matrix
        self.h = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.uniform_(self.h, a=0.0, b=1e-4)

        # Trend correlation projections
        self.conv1d_1 = nn.Conv2d(in_channels=input_dim, out_channels=dilation_channels, kernel_size=1)
        self.conv1d_2 = nn.Conv2d(in_channels=input_dim, out_channels=dilation_channels, kernel_size=1)

        # Encoder and Decoder stacks
        self.encoder = Encoder(
            num_nodes=num_nodes,
            k_num=self.k_num,
            l=l,
            length=history,
            in_dim=input_dim,
            dilation_channels=dilation_channels,
            K=K,
            Kt=Kt,
            dropout=dropout,
        )

        self.decoder = Decoder(
            num_nodes=num_nodes,
            k_num=self.k_num,
            l=l,
            length=history,
            dilation_channels=dilation_channels,
            K=K,
            Kt=Kt,
            dropout=dropout,
        )

        # Final projection layer to map to (B, Horizon, Nodes)
        self.conv1 = nn.Conv2d(
            dilation_channels,
            horizon,
            kernel_size=(1, history),
            padding=(0, 0),
            bias=True,
        )

    def forward(self, x, adj=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (Batch, History, Nodes, Features)
        adj : torch.Tensor, optional
            Graph adjacency matrix (Nodes, Nodes), optional.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape (Batch, Horizon, Nodes)
        """
        self.validate_input(x)
        B, T, N, F_dim = x.shape
        device = x.device

        # Permute to (B, F, N, T) for convolutional channels
        v = x.permute(0, 3, 2, 1)

        # 1. Compute dynamic Trend Correlation Matrix
        trend1 = F.relu(self.conv1d_1(v)).permute(0, 3, 2, 1)  # (B, T, N, dilation_channels)
        trend2 = F.relu(self.conv1d_2(v)).permute(0, 3, 2, 1)  # (B, T, N, dilation_channels)
        scale = math.sqrt(self.dilation_channels)
        Trend_Matrix = torch.softmax(torch.matmul(trend2, trend1.transpose(-2, -1)) / scale, dim=-1)  # (B, T, N, N)

        # 2. Build spatial adjacency matrix A combining adaptive h and given adj
        if adj is not None:
            base_adj = adj.to(device)
        elif self.supports is not None and len(self.supports) > 0 and self.supports[0] is not None:
            base_adj = self.supports[0].to(device)
        else:
            base_adj = torch.eye(self.num_nodes, device=device)

        A = torch.relu(self.h) + base_adj
        degree = torch.sum(A, dim=-1, keepdim=True)
        degree = torch.clamp(degree, min=1e-5)
        A = A / degree
        A = F.dropout(A, p=self.dropout, training=self.training)

        # 3. Encoder-Decoder transformation
        v = self.encoder(v, Trend_Matrix, A)
        v = self.decoder(v, Trend_Matrix, A)

        # 4. Output forecasting head
        out = self.conv1(v)  # (B, Horizon, N, 1)
        out = out.squeeze(-1)  # (B, Horizon, N)
        return out
