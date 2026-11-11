import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel


def asym_adj(adj):
    """
    Asymmetric row-normalization of adjacency matrix.
    Supports 2D (N, M) and 3D (B, N, M) tensors.
    """
    row_sum = adj.sum(-1, keepdim=True)
    row_sum = torch.clamp(row_sum, min=1e-8)
    return adj / row_sum


class InstanceNorm(nn.Module):
    """
    Instance Normalization across spatial and feature dimensions.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[-2, -1], keepdim=True)
        var = x.var(dim=[-2, -1], unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GCN(nn.Module):
    """
    Graph Convolution operation supporting 2D or 3D adjacency matrices.
    """
    def __init__(self):
        super(GCN, self).__init__()

    def forward(self, x, adj):
        if adj.dim() == 2:
            x = torch.einsum('nm,bmd->bnd', adj, x)
        elif adj.dim() == 3:
            x = torch.einsum('bnm,bmd->bnd', adj, x)
        else:
            raise ValueError(f'Wrong Adj Matrix Dimension! Expected 2 or 3, got {adj.dim()}.')
        return x.contiguous()


class VariationalDropout(nn.Module):
    """
    Variational Dropout reusing the same mask across time steps in a sequence.
    Device-agnostic and shape-resilient.
    """
    def __init__(self, p=0.1):
        super(VariationalDropout, self).__init__()
        self.dropout_rate = p
        self.mask = None

    def forward(self, x, time_step=0):
        if self.training and self.dropout_rate > 0:
            if time_step == 0 or self.mask is None:
                self.mask = torch.bernoulli(
                    torch.full_like(x, 1.0 - self.dropout_rate)
                ) / (1.0 - self.dropout_rate)
            return self.mask * x
        return x


class MetaDGCN(nn.Module):
    """
    Meta-learning Dynamic Graph Convolution Network with node-adaptive parameter pooling.
    Memory-efficient contraction order prevents huge intermediate tensor allocations.
    """
    def __init__(self, hop_num_k, in_features, out_dim, graph_num, node_emb_dim):
        super(MetaDGCN, self).__init__()

        self.hop_num_k = hop_num_k
        self.in_features = in_features
        self.graph_num = graph_num
        self.node_emb_dim = node_emb_dim

        self.gconv = GCN()

        self.weights_pool = nn.Parameter(
            torch.empty([node_emb_dim, (graph_num * hop_num_k + 1) * in_features, out_dim])
        )
        nn.init.xavier_normal_(self.weights_pool)

        self.bias_pool = nn.Parameter(torch.empty([node_emb_dim, out_dim]))
        nn.init.xavier_normal_(self.bias_pool)

    def forward(self, x, graph_list, node_emb):
        if node_emb.dim() == 2:
            node_emb = node_emb.unsqueeze(0).expand(x.shape[0], -1, -1)

        out = [x]
        for i in range(self.graph_num):
            x1 = x
            for k in range(self.hop_num_k):
                x2 = self.gconv(x1, graph_list[i])
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)

        # Memory-efficient contraction: contracting h with weights_pool first yields
        # (B, N, D, O) which avoids materializing the massive (B, N, I, O) intermediate tensor.
        h_w = torch.einsum('bni,dio->bndo', h, self.weights_pool)
        bias = torch.einsum('bnd,do->bno', node_emb, self.bias_pool)
        h = torch.einsum('bnd,bndo->bno', node_emb, h_w) + bias

        return h


class DynamicGraphQualification(nn.Module):
    """
    Dynamic Graph Qualification (DGQ) module for qualifying dynamic graph topology.
    """
    def __init__(self, node_emb_dim, delta=2.):
        super(DynamicGraphQualification, self).__init__()

        self.delta = delta
        self.threshold_pool = nn.Parameter(torch.empty(node_emb_dim, 1))
        nn.init.xavier_normal_(self.threshold_pool)
        self.norm_phi = InstanceNorm()

    def qualify(self, current_node_emb, prev_node_emb, static_node_emb):
        b, n, _ = current_node_emb.size()
        graph = torch.einsum('bnd,bmd->bnm', current_node_emb, prev_node_emb)
        static_graph_mask = (torch.relu(torch.einsum('nd,md->nm', static_node_emb, static_node_emb)) > 0).float()
        probs = asym_adj(torch.relu(torch.einsum('bnm,nm->bnm', graph, static_graph_mask)) + 1e-10)

        threshold = torch.sigmoid(torch.einsum('bnd,dg->bng', current_node_emb, self.threshold_pool))
        threshold_baseline = torch.diagonal(probs, dim1=-2, dim2=-1).unsqueeze(-1)
        threshold = threshold_baseline * threshold
        x = probs - threshold
        pos_mask = (x >= 0).float()
        neg_mask = 1.0 - pos_mask

        pos_mask = torch.sigmoid(x) * pos_mask
        tau_beta = torch.exp(self.norm_phi(pos_mask) * self.delta)
        phi = tau_beta * pos_mask + tau_beta * neg_mask

        return phi

    def forward(self, static_node_emb, current_node_emb, prev_node_emb):
        return self.qualify(
            current_node_emb=current_node_emb,
            prev_node_emb=prev_node_emb,
            static_node_emb=static_node_emb
        )


class SpatialTemporalCorrelationEnhancement(nn.Module):
    """
    Spatial-Temporal Correlation Enhancement (STCE) module.
    """
    def __init__(self, node_emb_dim, hidden_emb_dim, cross_attn_hidden_dim=64, corr_enhance_mode='ST', dropout=0.1):
        super(SpatialTemporalCorrelationEnhancement, self).__init__()

        self.corr_enhance_mode = corr_enhance_mode

        if 'S' in self.corr_enhance_mode:
            self.linear_q = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_k = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_v = nn.Linear(node_emb_dim, cross_attn_hidden_dim)
            self.linear_out = nn.Linear(cross_attn_hidden_dim, node_emb_dim)
            self.linear_out2 = nn.Linear(node_emb_dim, node_emb_dim)
            self.dropout1 = VariationalDropout(p=dropout)
            self.dropout2 = VariationalDropout(p=dropout)

        if 'T' in self.corr_enhance_mode:
            self.reset_proj = nn.Linear(hidden_emb_dim, node_emb_dim)

        self.scale = cross_attn_hidden_dim ** -0.5

    def spatial_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, time_step):
        q = self.linear_q(current_dynamic_node_emb)
        k = self.linear_k(prev_dynamic_node_emb)
        v = self.linear_v(prev_dynamic_node_emb)

        cross_attn = torch.einsum('bnh,bmh->bnm', q, k) * self.scale
        norm_cross_attn = torch.softmax(cross_attn, dim=-1)

        current_sce_node_emb = torch.einsum('bnm,bmh->bnh', norm_cross_attn, v)
        current_sce_node_emb = self.dropout1(current_dynamic_node_emb + self.linear_out(current_sce_node_emb), time_step)
        current_sce_node_emb = self.dropout2(current_dynamic_node_emb + self.linear_out2(current_sce_node_emb), time_step)

        return current_sce_node_emb

    def temporal_correlation_enhancement(self, current_dynamic_node_emb, prev_dynamic_node_emb, z):
        z_hat = torch.sigmoid(self.reset_proj(z))
        current_tce_node_emb = z_hat * prev_dynamic_node_emb + (1.0 - z_hat) * current_dynamic_node_emb
        return current_tce_node_emb

    def forward(self, current_dynamic_node_emb, prev_dynamic_node_emb, prev_enhanced_node_emb, z=None, time_step=0):
        if self.corr_enhance_mode == 'ST':
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, z)
        elif self.corr_enhance_mode == 'TS':
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)
            else:
                current_enhanced_node_emb = current_dynamic_node_emb
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_enhanced_node_emb, prev_enhanced_node_emb, time_step)
        elif self.corr_enhance_mode == 'S':
            current_enhanced_node_emb = self.spatial_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, time_step)
        elif self.corr_enhance_mode == 'T':
            if z is not None:
                current_enhanced_node_emb = self.temporal_correlation_enhancement(current_dynamic_node_emb, prev_dynamic_node_emb, z)
            else:
                current_enhanced_node_emb = current_dynamic_node_emb
        else:
            current_enhanced_node_emb = current_dynamic_node_emb

        return current_enhanced_node_emb


class DynamicNodeGeneration(nn.Module):
    """
    Generates dynamic node representations combining static node embeddings, temporal states, and hidden representations.
    """
    def __init__(self, node_emb_dim, time_emb_dim, hidden_emb_dim, graph_num=1):
        super(DynamicNodeGeneration, self).__init__()

        self.graph_num = graph_num
        gamma_src_dim = time_emb_dim * 2
        self.state_proj = nn.Linear(hidden_emb_dim, node_emb_dim)
        self.gamma_pool = nn.Parameter(torch.empty(gamma_src_dim, graph_num, node_emb_dim))
        nn.init.xavier_normal_(self.gamma_pool)

    def get_gamma(self, gamma_pool, emb):
        if emb.dim() == 2:
            gamma = torch.einsum('bd,dh->bh', emb, gamma_pool).unsqueeze(1)
        elif emb.dim() == 3:
            gamma = torch.einsum('bnd,dh->bnh', emb, gamma_pool)
        gamma = torch.sigmoid(gamma)
        return gamma

    def forward(self, static_node_emb, time_emb, state):
        state_emb = self.state_proj(state)
        current_dynamic_node_emb = []
        for i in range(self.graph_num):
            src_emb = torch.cat([time_emb[0], time_emb[1]], dim=-1)
            gamma = self.get_gamma(self.gamma_pool[:, i, :], src_emb)
            current_reserve_emb = gamma * static_node_emb + (1.0 - gamma) * state_emb
            current_dynamic_node_emb.append(current_reserve_emb)
        current_dynamic_node_emb = torch.stack(current_dynamic_node_emb, dim=1).squeeze(dim=1)
        return current_dynamic_node_emb


class MetaDGCRU(nn.Module):
    """
    MetaDG Convolutional Recurrent Unit (GRU Cell).
    """
    def __init__(self, hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num=1, use_mask=True, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True,
                 corr_enhance_mode='ST', dropout=0.3):
        super(MetaDGCRU, self).__init__()

        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.refine_dynamic_node = refine_dynamic_node
        self.refine_dynamic_graph = refine_dynamic_graph

        self.dynamic_node_generation = DynamicNodeGeneration(
            node_emb_dim=node_emb_dim,
            time_emb_dim=time_emb_dim,
            hidden_emb_dim=hidden_dim,
            graph_num=graph_num,
        )

        if self.refine_dynamic_node and self.refine_dynamic_graph:
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim, hidden_dim, cross_attn_hidden_dim=64, corr_enhance_mode=corr_enhance_mode)
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2.)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim, hidden_dim, cross_attn_hidden_dim=64, corr_enhance_mode=corr_enhance_mode)
            self.stce_m = SpatialTemporalCorrelationEnhancement(node_emb_dim, hidden_dim, cross_attn_hidden_dim=64, corr_enhance_mode=corr_enhance_mode)
        elif self.refine_dynamic_node:
            self.stce_p = SpatialTemporalCorrelationEnhancement(node_emb_dim, hidden_dim, cross_attn_hidden_dim=64, corr_enhance_mode=corr_enhance_mode)
            self.stce_g = SpatialTemporalCorrelationEnhancement(node_emb_dim, hidden_dim, cross_attn_hidden_dim=64, corr_enhance_mode=corr_enhance_mode)
        elif self.refine_dynamic_graph:
            self.phi = DynamicGraphQualification(node_emb_dim=node_emb_dim, delta=2.)

        self.use_mask = use_mask
        self.mask_pool = mask_pool
        if self.use_mask:
            self.dropout = nn.Dropout(p=dropout)

        gcn_in_dim = in_dim + hidden_dim
        self.gate = MetaDGCN(hop_num_k, gcn_in_dim, out_dim * 2, graph_num, node_emb_dim)
        self.update = MetaDGCN(hop_num_k, gcn_in_dim, out_dim, graph_num, node_emb_dim)

    def get_mask(self, time_emb, node_emb, mask_pool):
        mask_pool_prod = torch.einsum('bns,dsh->bdnh', node_emb, mask_pool)
        mask_emb = torch.einsum('bd,bdnh->bnh', time_emb, mask_pool_prod)
        graph_mask = torch.einsum('bnh,bmh->bnm', mask_emb, mask_emb)
        graph_mask = torch.relu(graph_mask)
        return graph_mask

    def forward(self, x, state, graph_list, static_node_emb, time_emb, continuous_time_emb,
                prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z, time_step):
        current_dynamic_node_emb = self.dynamic_node_generation(static_node_emb, time_emb, state)
        B, _, _ = current_dynamic_node_emb.size()

        if prev_dynamic_node_emb.dim() == 2:
            prev_dynamic_node_emb = prev_dynamic_node_emb.unsqueeze(0).expand(B, -1, -1)
        if prev_stce_p is not None and prev_stce_p.dim() == 2:
            prev_stce_p = prev_stce_p.unsqueeze(0).expand(B, -1, -1)
        if prev_stce_g is not None and prev_stce_g.dim() == 2:
            prev_stce_g = prev_stce_g.unsqueeze(0).expand(B, -1, -1)
        if prev_stce_m is not None and prev_stce_m.dim() == 2:
            prev_stce_m = prev_stce_m.unsqueeze(0).expand(B, -1, -1)

        if self.refine_dynamic_node and self.refine_dynamic_graph:
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)
                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)

                if self.use_mask and continuous_time_emb is not None and self.mask_pool is not None:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph

                current_stce_m = self.stce_m(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_m, z=z, time_step=time_step)
                current_phi = self.phi(static_node_emb, current_stce_m, prev_stce_m)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)

                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        elif self.refine_dynamic_node:
            current_stce_m = None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                current_stce_p = self.stce_p(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_p, z=z, time_step=time_step)
                current_stce_g = self.stce_g(current_dynamic_node_emb, prev_dynamic_node_emb, prev_stce_g, z=z, time_step=time_step)
                graph = torch.einsum('bnd,bmd->bnm', current_stce_g, current_stce_g)
                if self.use_mask and continuous_time_emb is not None and self.mask_pool is not None:
                    graph_mask = self.get_mask(continuous_time_emb, current_stce_g, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        elif self.refine_dynamic_graph:
            current_stce_p, current_stce_g, current_stce_m = None, None, None
            current_graph_list = []
            for i, graph in enumerate(graph_list):
                graph = torch.einsum('bnd,bmd->bnm', current_dynamic_node_emb, current_dynamic_node_emb)
                if self.use_mask and continuous_time_emb is not None and self.mask_pool is not None:
                    graph_mask = self.get_mask(continuous_time_emb, current_dynamic_node_emb, self.mask_pool[:, i, :, :])
                    graph = graph_mask * graph
                current_phi = self.phi(static_node_emb, current_dynamic_node_emb, prev_dynamic_node_emb)
                graph = torch.einsum('bnm,bnm->bnm', current_phi, graph)
                if self.use_mask:
                    graph = asym_adj(self.dropout(torch.relu(graph)) + 1e-10)
                else:
                    graph = asym_adj(torch.relu(graph) + 1e-10)
                current_graph_list.append(graph)
        else:
            current_stce_p = current_dynamic_node_emb
            current_graph_list = graph_list

        node_features_for_conv = current_stce_p if current_stce_p is not None else current_dynamic_node_emb
        input_and_state = torch.cat((x, state), dim=-1)
        r_z = torch.sigmoid(self.gate(input_and_state, current_graph_list, node_features_for_conv))
        r, z = torch.split(r_z, self.hidden_dim, dim=-1)
        temp_state = r * state
        temp = torch.cat((x, temp_state), dim=-1)
        c = torch.tanh(self.update(temp, current_graph_list, node_features_for_conv))
        h = z * state + (1.0 - z) * c

        return h, z, current_dynamic_node_emb, current_stce_p, current_stce_g, current_stce_m

    def init_hidden_state(self, batch_size, device=None):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class Encoder(nn.Module):
    """
    Recurrent MetaDG Encoder.
    """
    def __init__(self, hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num=1, layer_num=1, use_mask=True, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True,
                 corr_enhance_mode='ST'):
        super(Encoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(
                hop_num_k, in_dim if i == 0 else hidden_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                node_num, graph_num, use_mask, mask_pool,
                refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode
            ))

    def forward(self, x, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None, use_mask=True):
        b, steps, n, _ = x.size()
        output_hidden = []
        current_input = x
        for cell in self.cells:
            prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
            z = None
            state = cell.init_hidden_state(b, device=x.device)
            inner_states = []
            for t in range(steps):
                if use_mask and continuous_time_emb is not None:
                    continuous_time_emb_t = continuous_time_emb[:, t, ...]
                else:
                    continuous_time_emb_t = None
                state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(
                    current_input[:, t, :, :], state, graph_list, node_emb,
                    [time_emb[:, 0, :], time_emb[:, t, :]],
                    continuous_time_emb_t,
                    prev_dynamic_node_emb,
                    prev_stce_p, prev_stce_g, prev_stce_m,
                    z=z, time_step=t
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)
        return current_input, output_hidden


class Decoder(nn.Module):
    """
    Recurrent MetaDG Decoder.
    """
    def __init__(self, hop_num_k, in_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                 node_num, graph_num=1, layer_num=1, use_mask=True, mask_pool=None, refine_dynamic_node=True, refine_dynamic_graph=True,
                 corr_enhance_mode='ST'):
        super(Decoder, self).__init__()

        self.cells = nn.ModuleList()
        for i in range(layer_num):
            self.cells.append(MetaDGCRU(
                hop_num_k, in_dim if i == 0 else hidden_dim, hidden_dim, out_dim, node_emb_dim, time_emb_dim, continuous_time_emb_dim,
                node_num, graph_num, use_mask, mask_pool,
                refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph,
                corr_enhance_mode=corr_enhance_mode
            ))

    def forward(self, x_t, init_state, graph_list, node_emb=None, time_emb=None, continuous_time_emb=None,
                prev_dynamic_node_emb=None, prev_stce_p=None, prev_stce_g=None, prev_stce_m=None, z=None, time_step=None):
        current_input = x_t
        output_hidden = []
        for i, cell in enumerate(self.cells):
            state, z, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m = cell(
                current_input, init_state[i], graph_list, node_emb,
                [time_emb[1], time_emb[0]],
                continuous_time_emb,
                prev_dynamic_node_emb,
                prev_stce_p, prev_stce_g, prev_stce_m,
                z=z, time_step=time_step
            )
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z


class MetaDG(BaseModel):
    """
    Meta-learning Dynamic Graph Neural Network (MetaDG)

    Input:
        x   : (Batch, History, Nodes, Features)
        adj : (optional) Adjacency matrix

    Output:
        (Batch, Horizon, Nodes)
    """
    def __init__(
        self,
        num_nodes,
        input_dim,
        history,
        horizon,
        hidden_dim=64,
        node_emb_dim=16,
        time_emb_dim=16,
        continuous_time_emb_dim=8,
        hop_num_k=2,
        layer_num=1,
        dropout=0.3,
        use_mask=True,
        refine_dynamic_node=True,
        refine_dynamic_graph=True,
        corr_enhance_mode='ST',
        **kwargs,
    ):
        super().__init__(
            num_nodes=num_nodes,
            input_dim=input_dim,
            history=history,
            horizon=horizon,
        )

        self.hidden_dim = hidden_dim
        self.node_emb_dim = node_emb_dim
        self.time_emb_dim = time_emb_dim
        self.continuous_time_emb_dim = continuous_time_emb_dim
        self.layer_num = layer_num
        self.use_mask = use_mask
        self.refine_dynamic_node = refine_dynamic_node
        self.refine_dynamic_graph = refine_dynamic_graph

        # Static learnable node embeddings
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_normal_(self.node_emb)

        # Stepwise / temporal position embeddings
        self.enc_time_embedding = nn.Parameter(torch.empty(history, time_emb_dim))
        nn.init.xavier_normal_(self.enc_time_embedding)
        self.dec_time_embedding = nn.Parameter(torch.empty(horizon, time_emb_dim))
        nn.init.xavier_normal_(self.dec_time_embedding)

        # Feature projections
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dec_input_proj = nn.Linear(1 + time_emb_dim, hidden_dim)

        # Continuous Fourier time embedding & mask pool
        if self.use_mask:
            self.time_frequency_emb = nn.Parameter(torch.empty(1, 1, continuous_time_emb_dim))
            nn.init.xavier_normal_(self.time_frequency_emb)
            self.scale = (1.0 / continuous_time_emb_dim) ** 0.5
            mask_emb_dim = node_emb_dim * 4
            self.mask_pool = nn.Parameter(
                torch.empty(continuous_time_emb_dim * 2, 1, node_emb_dim, mask_emb_dim)
            )
            nn.init.xavier_normal_(self.mask_pool)
        else:
            self.mask_pool = None

        self.encoder = Encoder(
            hop_num_k=hop_num_k, in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
            node_emb_dim=node_emb_dim, time_emb_dim=time_emb_dim, continuous_time_emb_dim=continuous_time_emb_dim,
            node_num=num_nodes, graph_num=1, layer_num=layer_num, use_mask=use_mask, mask_pool=self.mask_pool,
            refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode
        )

        self.decoder = Decoder(
            hop_num_k=hop_num_k, in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
            node_emb_dim=node_emb_dim, time_emb_dim=time_emb_dim, continuous_time_emb_dim=continuous_time_emb_dim,
            node_num=num_nodes, graph_num=1, layer_num=layer_num, use_mask=use_mask, mask_pool=self.mask_pool,
            refine_dynamic_node=refine_dynamic_node, refine_dynamic_graph=refine_dynamic_graph, corr_enhance_mode=corr_enhance_mode
        )

        self.fc_final = nn.Linear(hidden_dim, 1)

    def _get_continuous_time_emb(self, steps, batch_size, device):
        if not self.use_mask:
            return None
        tau = torch.linspace(0, 1, steps, device=device).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        continuous_time = 2.0 * np.pi * tau * self.time_frequency_emb
        time_cos = torch.cos(continuous_time)
        time_sin = torch.sin(continuous_time)
        emb = torch.stack([time_cos, time_sin], dim=-1).reshape(batch_size, steps, -1) * self.scale
        return emb

    def forward(self, x, adj=None, *args, **kwargs):
        """
        Forward pass for MetaDG.

        Parameters
        ----------
        x : torch.Tensor
            (Batch, History, Nodes, Features)
        adj : torch.Tensor, optional
            Predefined adjacency matrix (Nodes, Nodes), if any.

        Returns
        -------
        torch.Tensor
            (Batch, Horizon, Nodes)
        """
        self.validate_input(x)
        b, in_steps, n, f = x.size()
        device = x.device

        # Continuous time representations
        enc_continuous_time_emb = self._get_continuous_time_emb(in_steps, b, device)
        dec_continuous_time_emb = self._get_continuous_time_emb(self.horizon, b, device)

        # Stepwise temporal position embeddings
        stepwise_enc_time = self.enc_time_embedding.unsqueeze(0).expand(b, -1, -1)
        stepwise_dec_time = self.dec_time_embedding.unsqueeze(0).expand(b, -1, -1)

        # Adaptive static base graph
        node_emb = self.node_emb
        graph = torch.softmax(torch.relu(node_emb @ node_emb.T), dim=-1)
        if adj is not None:
            if adj.dim() == 2:
                graph = 0.5 * graph + 0.5 * adj.to(device)
            elif adj.dim() == 3:
                graph = 0.5 * graph.unsqueeze(0) + 0.5 * adj.to(device)

        # Spatial-temporal encoding
        x_proj = self.input_proj(x)
        h, _ = self.encoder(
            x_proj, [graph], node_emb, stepwise_enc_time,
            continuous_time_emb=enc_continuous_time_emb, use_mask=self.use_mask
        )
        h_last = h[:, -1, :, :]

        state = [h_last] * self.layer_num
        go = torch.zeros((b, self.num_nodes, 1), device=device)

        out = []
        prev_dynamic_node_emb = prev_stce_p = prev_stce_g = prev_stce_m = node_emb
        z = None

        # Autoregressive decoding across horizon steps
        for t in range(self.horizon):
            t_emb_expanded = stepwise_dec_time[:, t, :].unsqueeze(1).expand(-1, self.num_nodes, -1)
            current_input = self.dec_input_proj(torch.cat([go, t_emb_expanded], dim=-1))
            time_emb = [stepwise_dec_time[:, t, :], stepwise_enc_time[:, -1, :]]
            continuous_time_emb_t = dec_continuous_time_emb[:, t, ...] if self.use_mask else None

            h_de, state, prev_dynamic_node_emb, prev_stce_p, prev_stce_g, prev_stce_m, z = self.decoder(
                current_input, state, [graph], node_emb=node_emb,
                time_emb=time_emb,
                continuous_time_emb=continuous_time_emb_t,
                prev_dynamic_node_emb=prev_dynamic_node_emb,
                prev_stce_p=prev_stce_p,
                prev_stce_g=prev_stce_g,
                prev_stce_m=prev_stce_m,
                z=z,
                time_step=t
            )
            go = self.fc_final(h_de)
            out.append(go)

        # Output shape: (Batch, Horizon, Nodes)
        output = torch.stack(out, dim=1).squeeze(-1)
        return output