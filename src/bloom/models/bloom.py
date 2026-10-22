"""BLOOM model definitions shared by training and developmental inference.

The unidirectional and bidirectional implementations live in this module.
Configuration belongs to each model instance; importing this file never reads
data, parses command-line arguments, or starts training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn

BIO_ABLATION_LEVEL = {
    "bio0_baseline": 0,
    "bio1_actual_time": 1,
    "bio2_hierarchy": 2,
    "bio3_curated_graph": 3,
    "bio4_anatomy": 4,
    "bio5_drivers": 5,
    "bio6_grn": 6,
    "bio_grn_only": 6,
    "bio2_hierarchy_grn": 6,
    "bio7_adaptive_full": 7,
}

BIO_ABLATION_COMPONENTS: Dict[str, Set[str]] = {
    "bio0_baseline": set(),
    "bio1_actual_time": {"actual_time"},
    "bio2_hierarchy": {"actual_time", "hierarchy"},
    "bio3_curated_graph": {"actual_time", "hierarchy", "graph"},
    "bio4_anatomy": {"actual_time", "hierarchy", "graph", "anatomy"},
    "bio5_drivers": {"actual_time", "hierarchy", "graph", "anatomy", "drivers"},
    "bio6_grn": {"actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn"},
    "bio_grn_only": {"grn"},
    "bio2_hierarchy_grn": {"actual_time", "hierarchy", "grn"},
    "bio7_adaptive_full": {"actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn"},
}


@dataclass
class BLOOMConfig:
    """Model defaults matching the native training parser; recipes override them."""

    AUX_POINT_LOSS_WEIGHT: float = 0.05
    BASE_LOGVAR_MAX: float = 3.0
    BASE_LOGVAR_MIN: float = -8.0
    BIO_EMB_DIM: int = 16
    BIO_PRIOR_HIDDEN: int = 64
    CELL_ENCODER_HIDDEN: int = 256
    CURATED_GRAPH_REG_WEIGHT: float = 0.0001
    DAY_ENCODER_HIDDEN: int = 256
    FLOW_CONTEXT_DIM: int = 256
    FLOW_HIDDEN_DIM: int = 256
    FLOW_LOSS_WEIGHT: float = 1.0
    FLOW_SIGMA: float = 1.0
    FM_BIDIRECTIONAL_ADAPTER_HIDDEN: int = 64
    FM_BIDIRECTIONAL_CONDITIONING: str = "embedding_adapter"
    FM_CONTEXT_CELL_DROPOUT: float = 0.0
    FM_CONTEXT_GENE_DROPOUT: float = 0.0
    FM_POINT_BLEND_WEIGHT: float = 0.7
    FM_PREDICTION_MODE: str = "flow_sample"
    FM_PRIOR_DROPOUT: float = 0.0
    FM_PRIOR_GATE_INIT: float = 0.1
    FM_PRIOR_GATE_L1_WEIGHT: float = 0.0
    FM_PRIOR_GATE_MODE: str = "conditional"
    FM_PRIOR_RAMP_EPOCHS: int = 0
    FM_PRIOR_WARMUP_EPOCHS: int = 0
    FM_RESIDUAL_MODE: str = "none"
    FM_RESIDUAL_REQUIRE_TRAJ_HINT: bool = False
    FM_TEMPORAL_DIRECTION: str = "forward"
    FM_TYPE_EMB_INIT_STD: float = 0.0
    GRN_EXTERNAL_TF_MODE: str = "target_gate"
    SEQ_LEN: int = 1
    TEMPORAL_HIDDEN: int = 256
    TRAJ_HINT_HIDDEN: int = 128
    TRAJ_SOFT_WEIGHT: float = 0.05
    TYPE_EMB_DIM: int = 32


@dataclass(frozen=True)
class BLOOMVariant:
    """Select the Gaussian base, conditioning inputs and biological-prior group."""

    base_mode: str = "context_gaussian"
    use_history: bool = True
    celltype_mode: str = "soft"
    use_traj_soft: bool = True
    bio_ablation: str = "bio7_adaptive_full"

    def uses_bio_component(self, component: str) -> bool:
        return component in BIO_ABLATION_COMPONENTS[self.bio_ablation]

    @property
    def bio_level(self) -> int:
        return BIO_ABLATION_LEVEL[self.bio_ablation]


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.to(x.dtype)
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1.0)
    return num / den


class FMCellEncoder(nn.Module):

    def __init__(self, n_genes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(n_genes), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalVelocityField(nn.Module):

    def __init__(self, *, n_genes: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.proj_x = nn.Linear(int(n_genes), int(hidden_dim))
        self.proj_t = nn.Linear(1, int(hidden_dim))
        self.proj_c = nn.Linear(int(context_dim), int(hidden_dim))
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(n_genes)),
        )

    def forward(
        self, x_tau: torch.Tensor, tau: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        h = self.proj_x(x_tau) + self.proj_t(tau) + self.proj_c(context)
        return self.out(h)


class UnidirectionalTemporalSetEncoder(nn.Module):
    """Encode ordered stage populations and biological priors for unidirectional BLOOM."""

    def __init__(
        self,
        *,
        n_genes: int,
        cell_hidden: int,
        day_hidden: int,
        temporal_hidden: int,
        max_history_days: int,
        n_celltypes: int,
        type_emb_dim: int,
        traj_hint_hidden: int,
        flow_context_dim: int,
        bio_ablation: str = "bio0_baseline",
        args: Any = None,
        prior_data: Any = None,
    ):
        super().__init__()
        self.args = args if args is not None else BLOOMConfig()
        self.prior_data = prior_data
        self.bio_ablation = str(bio_ablation)
        self.bio_level = int(BIO_ABLATION_LEVEL[self.bio_ablation])
        self.bio_components = set(BIO_ABLATION_COMPONENTS[self.bio_ablation])
        self.adaptive_prior_fusion = self.bio_ablation == "bio7_adaptive_full"
        self.prior_component_names = [
            name
            for name in ("actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn")
            if name in self.bio_components
        ]
        self._prior_strength = 1.0
        self._last_prior_gate_tensors: Dict[str, torch.Tensor] = {}
        self._last_prior_gate_means: Dict[str, float] = {}
        self.bio_prior_hidden = int(self.args.BIO_PRIOR_HIDDEN)
        self.cell_encoder = FMCellEncoder(n_genes=int(n_genes), hidden_dim=int(cell_hidden))
        self.day_proj = nn.Sequential(
            nn.Linear(int(cell_hidden) + 1, int(day_hidden)),
            nn.GELU(),
            nn.Linear(int(day_hidden), int(day_hidden)),
            nn.GELU(),
        )
        self.day_lag_emb = nn.Embedding(int(max_history_days), int(day_hidden))
        self.temporal_gru = nn.GRU(
            input_size=int(day_hidden),
            hidden_size=int(temporal_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.type_emb = nn.Embedding(int(n_celltypes), int(type_emb_dim))
        if float(self.args.FM_TYPE_EMB_INIT_STD) <= 0.0:
            nn.init.zeros_(self.type_emb.weight)
        else:
            nn.init.normal_(
                self.type_emb.weight, mean=0.0, std=float(self.args.FM_TYPE_EMB_INIT_STD)
            )
        self.traj_hint_proj = nn.Sequential(
            nn.Linear(int(n_genes), int(traj_hint_hidden)),
            nn.GELU(),
            nn.Linear(int(traj_hint_hidden), int(traj_hint_hidden)),
            nn.GELU(),
        )
        self.actual_time_proj: Optional[nn.Module] = None
        self.actual_time_context_proj: Optional[nn.Module] = None
        self.hierarchy_emb = nn.ModuleList()
        self.hierarchy_proj: Optional[nn.Module] = None
        self.graph_node_emb: Optional[nn.Embedding] = None
        self.graph_proj: Optional[nn.Module] = None
        self.anatomy_emb = nn.ModuleList()
        self.anatomy_proj: Optional[nn.Module] = None
        self.driver_proj: Optional[nn.Module] = None
        self.grn_proj: Optional[nn.Module] = None
        extra_fused_components = 0
        if "actual_time" in self.bio_components:
            self.actual_time_proj = nn.Sequential(
                nn.Linear(3, int(day_hidden)),
                nn.GELU(),
                nn.Linear(int(day_hidden), int(day_hidden)),
            )
            if self.adaptive_prior_fusion:
                self.actual_time_context_proj = nn.Sequential(
                    nn.Linear(3, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
        if "hierarchy" in self.bio_components:
            cards = self.prior_data.hierarchy_cardinalities if self.prior_data is not None else []
            self.hierarchy_emb = nn.ModuleList(
                [
                    nn.Embedding(int(card), int(self.args.BIO_EMB_DIM), padding_idx=0)
                    for card in cards
                ]
            )
            if len(cards) > 0:
                self.hierarchy_proj = nn.Sequential(
                    nn.Linear(len(cards) * int(self.args.BIO_EMB_DIM), self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
            extra_fused_components += 1
        if "graph" in self.bio_components:
            graph_card = len(self.prior_data.graph_node_names) if self.prior_data is not None else 1
            self.graph_node_emb = nn.Embedding(
                int(graph_card), int(self.args.BIO_EMB_DIM), padding_idx=0
            )
            self.graph_proj = nn.Sequential(
                nn.Linear(int(self.args.BIO_EMB_DIM) + 6, self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            edges = (
                self.prior_data.graph_edges
                if self.prior_data is not None
                else np.zeros((0, 2), dtype=np.int64)
            )
            weights = (
                self.prior_data.graph_edge_weights
                if self.prior_data is not None
                else np.zeros((0,), dtype=np.float32)
            )
            self.register_buffer(
                "graph_edges", torch.tensor(edges, dtype=torch.long), persistent=False
            )
            self.register_buffer(
                "graph_edge_weights", torch.tensor(weights, dtype=torch.float32), persistent=False
            )
            extra_fused_components += 1
        if "anatomy" in self.bio_components:
            cards = self.prior_data.anatomy_cardinalities if self.prior_data is not None else []
            numeric_dim = (
                int(self.prior_data.numeric_context.shape[1]) if self.prior_data is not None else 0
            )
            self.anatomy_emb = nn.ModuleList(
                [
                    nn.Embedding(int(card), int(self.args.BIO_EMB_DIM), padding_idx=0)
                    for card in cards
                ]
            )
            anatomy_input_dim = len(cards) * int(self.args.BIO_EMB_DIM) + numeric_dim
            if anatomy_input_dim > 0:
                self.anatomy_proj = nn.Sequential(
                    nn.Linear(anatomy_input_dim, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
            extra_fused_components += 1
        if "drivers" in self.bio_components:
            self.driver_proj = nn.Sequential(
                nn.Linear(int(n_genes), self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            extra_fused_components += 1
        if "grn" in self.bio_components:
            self.grn_proj = nn.Sequential(
                nn.Linear(int(n_genes), self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            grn = (
                self.prior_data.grn_adjacency
                if self.prior_data is not None
                else np.zeros((n_genes, n_genes), dtype=np.float32)
            )
            grn_target_support = (
                self.prior_data.grn_target_support
                if self.prior_data is not None
                else np.zeros((n_genes,), dtype=np.float32)
            )
            self.register_buffer(
                "grn_adjacency", torch.tensor(grn, dtype=torch.float32), persistent=True
            )
            self.register_buffer(
                "grn_target_support",
                torch.tensor(grn_target_support, dtype=torch.float32),
                persistent=True,
            )
            extra_fused_components += 1
        final_extra_dim = (
            0 if self.adaptive_prior_fusion else extra_fused_components * self.bio_prior_hidden
        )
        self.final_proj = nn.Sequential(
            nn.Linear(
                int(temporal_hidden)
                + int(type_emb_dim)
                + int(traj_hint_hidden)
                + 1
                + final_extra_dim,
                int(flow_context_dim),
            ),
            nn.GELU(),
            nn.Linear(int(flow_context_dim), int(flow_context_dim)),
        )
        self.prior_adapters = nn.ModuleDict()
        self.prior_global_logits = nn.ParameterDict()
        self.prior_conditional_gates = nn.ModuleDict()
        if self.adaptive_prior_fusion:
            initial_probability = float(self.args.FM_PRIOR_GATE_INIT)
            initial_logit = math.log(initial_probability / (1.0 - initial_probability))
            for name in self.prior_component_names:
                adapter = nn.Sequential(
                    nn.Linear(self.bio_prior_hidden, int(flow_context_dim)),
                    nn.GELU(),
                    nn.Linear(int(flow_context_dim), int(flow_context_dim)),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.prior_adapters[name] = adapter
                self.prior_global_logits[name] = nn.Parameter(
                    torch.tensor(initial_logit, dtype=torch.float32)
                )
                gate = nn.Sequential(
                    nn.Linear(int(flow_context_dim) + self.bio_prior_hidden, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, 1),
                )
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, initial_logit)
                self.prior_conditional_gates[name] = gate

    def _zero_bio_feature(self, context_x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(context_x.shape[0]),
            self.bio_prior_hidden,
            device=context_x.device,
            dtype=context_x.dtype,
        )

    def set_prior_training_progress(self, epoch: int) -> float:
        """Set the residual-prior curriculum strength for one training epoch."""
        if not self.adaptive_prior_fusion:
            self._prior_strength = 1.0
            return self._prior_strength
        warmup = int(self.args.FM_PRIOR_WARMUP_EPOCHS)
        ramp = int(self.args.FM_PRIOR_RAMP_EPOCHS)
        if int(epoch) <= warmup:
            strength = 0.0
        elif ramp <= 0:
            strength = 1.0
        else:
            strength = min(1.0, max(0.0, (float(epoch) - float(warmup)) / float(ramp)))
        self._prior_strength = float(strength)
        return self._prior_strength

    def set_prior_inference_strength(self) -> None:
        self._prior_strength = 1.0

    def prior_gate_regularization(self) -> torch.Tensor:
        if not self.adaptive_prior_fusion or not self._last_prior_gate_tensors:
            return self.final_proj[0].weight.new_zeros(())
        return torch.stack([gate.mean() for gate in self._last_prior_gate_tensors.values()]).mean()

    def prior_gate_summary(self) -> Dict[str, float]:
        return dict(self._last_prior_gate_means)

    def _apply_context_dropout(
        self, context_x: torch.Tensor, context_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-only robustness augmentation; default probabilities are zero."""
        if not self.training:
            return (context_x, context_mask)
        gene_p = float(self.args.FM_CONTEXT_GENE_DROPOUT)
        cell_p = float(self.args.FM_CONTEXT_CELL_DROPOUT)
        if gene_p > 0.0:
            keep_gene = (
                torch.rand(context_x.shape[0], 1, 1, context_x.shape[-1], device=context_x.device)
                >= gene_p
            ).to(context_x.dtype)
            context_x = context_x * keep_gene
        if cell_p > 0.0:
            keep_cell = (torch.rand_like(context_mask) >= cell_p).to(context_mask.dtype)
            dropped_mask = context_mask * keep_cell
            was_valid = context_mask.sum(dim=2) > 0
            now_empty = (dropped_mask.sum(dim=2) <= 0) & was_valid
            if torch.any(now_empty):
                first_valid = torch.argmax(context_mask, dim=2)
                rows = torch.nonzero(now_empty, as_tuple=False)
                dropped_mask[rows[:, 0], rows[:, 1], first_valid[rows[:, 0], rows[:, 1]]] = 1.0
            context_mask = dropped_mask
        return (context_x, context_mask)

    def _adaptive_fuse_priors(
        self, base_context: torch.Tensor, prior_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self._last_prior_gate_tensors = {}
        self._last_prior_gate_means = {}
        if not self.adaptive_prior_fusion:
            return base_context
        fused = base_context
        residuals: List[torch.Tensor] = []
        component_dropout = float(self.args.FM_PRIOR_DROPOUT)
        for name in self.prior_component_names:
            feature = prior_features[name]
            if str(self.args.FM_PRIOR_GATE_MODE) == "conditional":
                gate = torch.sigmoid(
                    self.prior_conditional_gates[name](torch.cat([base_context, feature], dim=-1))
                )
            else:
                gate = (
                    torch.sigmoid(self.prior_global_logits[name])
                    .reshape(1, 1)
                    .expand(feature.shape[0], 1)
                )
            if self.training and component_dropout > 0.0:
                keep = (
                    torch.rand(feature.shape[0], 1, device=feature.device) >= component_dropout
                ).to(feature.dtype)
                gate_for_residual = gate * keep / (1.0 - component_dropout)
            else:
                gate_for_residual = gate
            residuals.append(gate_for_residual * self.prior_adapters[name](feature))
            self._last_prior_gate_tensors[name] = gate
            self._last_prior_gate_means[name] = float(gate.detach().mean().cpu())
        if residuals:
            residual_sum = torch.stack(residuals, dim=0).sum(dim=0) / math.sqrt(
                float(len(residuals))
            )
            fused = fused + float(self._prior_strength) * residual_sum
        return fused

    def graph_regularization(self) -> torch.Tensor:
        if (
            "graph" not in self.bio_components
            or self.graph_node_emb is None
            or self.graph_edges.numel() == 0
        ):
            return self.final_proj[0].weight.new_zeros(())
        src = self.graph_edges[:, 0]
        dst = self.graph_edges[:, 1]
        diff2 = torch.mean((self.graph_node_emb(src) - self.graph_node_emb(dst)) ** 2, dim=1)
        weights = torch.abs(self.graph_edge_weights).to(diff2.dtype)
        return torch.sum(diff2 * weights) / torch.sum(weights).clamp_min(1e-08)

    def forward(
        self,
        *,
        context_x: torch.Tensor,
        context_mask: torch.Tensor,
        day_valid: torch.Tensor,
        target_type_prop: torch.Tensor,
        target_type_id: Optional[torch.Tensor],
        traj_hint: Optional[torch.Tensor],
        traj_hint_mask: Optional[torch.Tensor],
        use_type: bool,
        use_traj_hint: bool,
        batch_context: Optional[Dict[str, torch.Tensor]] = None,
        population_only: bool = False,
    ) -> torch.Tensor:
        b, d, _, _ = context_x.shape
        context_x, context_mask = self._apply_context_dropout(context_x, context_mask)
        cell_h = self.cell_encoder(context_x)
        day_h = masked_mean(cell_h, context_mask, dim=2)
        day_h = torch.cat([day_h, target_type_prop.unsqueeze(-1)], dim=-1)
        day_h = self.day_proj(day_h)
        if (
            "actual_time" in self.bio_components
            and (not self.adaptive_prior_fusion)
            and (self.actual_time_proj is not None)
            and (batch_context is not None)
        ):
            day_h = day_h + self.actual_time_proj(
                batch_context["actual_time_features"].to(day_h.dtype)
            )
        lag_ids = torch.arange(d, device=context_x.device).unsqueeze(0).expand(b, d)
        day_h = day_h + self.day_lag_emb(lag_ids)
        day_h = day_h * day_valid.unsqueeze(-1)
        _, h_n = self.temporal_gru(day_h)
        context = h_n[-1]
        if use_type and target_type_id is not None:
            type_feat = self.type_emb(target_type_id)
        else:
            type_feat = torch.zeros(
                b, self.type_emb.embedding_dim, device=context_x.device, dtype=context_x.dtype
            )
        if use_traj_hint and traj_hint is not None and (traj_hint_mask is not None):
            traj_feat = self.traj_hint_proj(traj_hint)
            traj_feat = traj_feat * traj_hint_mask.unsqueeze(-1)
            traj_present = traj_hint_mask.unsqueeze(-1)
        else:
            traj_feat = torch.zeros(
                b,
                self.traj_hint_proj[-2].out_features,
                device=context_x.device,
                dtype=context_x.dtype,
            )
            traj_present = torch.zeros(b, 1, device=context_x.device, dtype=context_x.dtype)
        fused_parts = [context, type_feat, traj_feat, traj_present]
        prior_features: Dict[str, torch.Tensor] = {}
        if "actual_time" in self.bio_components and self.adaptive_prior_fusion:
            if self.actual_time_context_proj is not None and batch_context is not None:
                time_raw = batch_context["actual_time_features"].to(context_x.dtype)
                time_summary = masked_mean(time_raw, day_valid, dim=1)
                prior_features["actual_time"] = self.actual_time_context_proj(time_summary)
            else:
                prior_features["actual_time"] = self._zero_bio_feature(context_x)
        if "hierarchy" in self.bio_components:
            if (
                not population_only
                and self.hierarchy_proj is not None
                and (batch_context is not None)
            ):
                codes = batch_context["hierarchy_codes"]
                hierarchy_raw = torch.cat(
                    [emb(codes[:, i]) for i, emb in enumerate(self.hierarchy_emb)], dim=-1
                )
                hierarchy_feature = self.hierarchy_proj(hierarchy_raw)
            else:
                hierarchy_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["hierarchy"] = hierarchy_feature
            else:
                fused_parts.append(hierarchy_feature)
        if "graph" in self.bio_components:
            if (
                not population_only
                and self.graph_node_emb is not None
                and (self.graph_proj is not None)
                and (batch_context is not None)
            ):
                graph_raw = torch.cat(
                    [
                        self.graph_node_emb(batch_context["graph_node_codes"]),
                        batch_context["graph_node_features"].to(context_x.dtype),
                    ],
                    dim=-1,
                )
                graph_feature = self.graph_proj(graph_raw)
            else:
                graph_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["graph"] = graph_feature
            else:
                fused_parts.append(graph_feature)
        if "anatomy" in self.bio_components:
            anatomy_parts: List[torch.Tensor] = []
            if not population_only and batch_context is not None:
                codes = batch_context["anatomy_codes"]
                anatomy_parts.extend([emb(codes[:, i]) for i, emb in enumerate(self.anatomy_emb)])
                if int(batch_context["numeric_context"].shape[1]) > 0:
                    anatomy_parts.append(batch_context["numeric_context"].to(context_x.dtype))
            if self.anatomy_proj is not None and anatomy_parts:
                anatomy_feature = self.anatomy_proj(torch.cat(anatomy_parts, dim=-1))
            else:
                anatomy_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["anatomy"] = anatomy_feature
            else:
                fused_parts.append(anatomy_feature)
        if "drivers" in self.bio_components:
            if not population_only and self.driver_proj is not None and (batch_context is not None):
                driver_feature = self.driver_proj(
                    batch_context["driver_program"].to(context_x.dtype)
                )
            else:
                driver_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["drivers"] = driver_feature
            else:
                fused_parts.append(driver_feature)
        if "grn" in self.bio_components:
            if self.grn_proj is not None:
                source_gene_mean = masked_mean(context_x, context_mask, dim=2)
                source_gene_mean = masked_mean(source_gene_mean, day_valid, dim=1)
                grn_signal = torch.matmul(
                    source_gene_mean, self.grn_adjacency.to(source_gene_mean.dtype)
                )
                if str(self.args.GRN_EXTERNAL_TF_MODE) == "target_gate":
                    grn_signal = grn_signal + source_gene_mean * self.grn_target_support.to(
                        source_gene_mean.dtype
                    )
                grn_feature = self.grn_proj(grn_signal)
            else:
                grn_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["grn"] = grn_feature
            else:
                fused_parts.append(grn_feature)
        fused = torch.cat(fused_parts, dim=-1)
        fused = self.final_proj(fused)
        if self.adaptive_prior_fusion:
            for name in self.prior_component_names:
                if name not in prior_features:
                    prior_features[name] = self._zero_bio_feature(context_x)
            fused = self._adaptive_fuse_priors(fused, prior_features)
        return fused


class BidirectionalTemporalSetEncoder(nn.Module):
    """Encode populations with the shared developmental-direction embedding and adapter."""

    def __init__(
        self,
        *,
        n_genes: int,
        cell_hidden: int,
        day_hidden: int,
        temporal_hidden: int,
        max_history_days: int,
        n_celltypes: int,
        type_emb_dim: int,
        traj_hint_hidden: int,
        flow_context_dim: int,
        bio_ablation: str = "bio0_baseline",
        args: Any = None,
        prior_data: Any = None,
    ):
        super().__init__()
        self.args = args if args is not None else BLOOMConfig()
        self.prior_data = prior_data
        self.bio_ablation = str(bio_ablation)
        self.bio_level = int(BIO_ABLATION_LEVEL[self.bio_ablation])
        self.bio_components = set(BIO_ABLATION_COMPONENTS[self.bio_ablation])
        self.adaptive_prior_fusion = self.bio_ablation == "bio7_adaptive_full"
        self.prior_component_names = [
            name
            for name in ("actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn")
            if name in self.bio_components
        ]
        self._prior_strength = 1.0
        self._last_prior_gate_tensors: Dict[str, torch.Tensor] = {}
        self._last_prior_gate_means: Dict[str, float] = {}
        self.bio_prior_hidden = int(self.args.BIO_PRIOR_HIDDEN)
        self.cell_encoder = FMCellEncoder(n_genes=int(n_genes), hidden_dim=int(cell_hidden))
        self.day_proj = nn.Sequential(
            nn.Linear(int(cell_hidden) + 1, int(day_hidden)),
            nn.GELU(),
            nn.Linear(int(day_hidden), int(day_hidden)),
            nn.GELU(),
        )
        self.day_lag_emb = nn.Embedding(int(max_history_days), int(day_hidden))
        self.temporal_gru = nn.GRU(
            input_size=int(day_hidden),
            hidden_size=int(temporal_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.type_emb = nn.Embedding(int(n_celltypes), int(type_emb_dim))
        if float(self.args.FM_TYPE_EMB_INIT_STD) <= 0.0:
            nn.init.zeros_(self.type_emb.weight)
        else:
            nn.init.normal_(
                self.type_emb.weight, mean=0.0, std=float(self.args.FM_TYPE_EMB_INIT_STD)
            )
        self.traj_hint_proj = nn.Sequential(
            nn.Linear(int(n_genes), int(traj_hint_hidden)),
            nn.GELU(),
            nn.Linear(int(traj_hint_hidden), int(traj_hint_hidden)),
            nn.GELU(),
        )
        self.actual_time_proj: Optional[nn.Module] = None
        self.actual_time_context_proj: Optional[nn.Module] = None
        self.hierarchy_emb = nn.ModuleList()
        self.hierarchy_proj: Optional[nn.Module] = None
        self.graph_node_emb: Optional[nn.Embedding] = None
        self.graph_proj: Optional[nn.Module] = None
        self.anatomy_emb = nn.ModuleList()
        self.anatomy_proj: Optional[nn.Module] = None
        self.driver_proj: Optional[nn.Module] = None
        self.grn_proj: Optional[nn.Module] = None
        extra_fused_components = 0
        if "actual_time" in self.bio_components:
            self.actual_time_proj = nn.Sequential(
                nn.Linear(3, int(day_hidden)),
                nn.GELU(),
                nn.Linear(int(day_hidden), int(day_hidden)),
            )
            if self.adaptive_prior_fusion:
                self.actual_time_context_proj = nn.Sequential(
                    nn.Linear(3, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
        if "hierarchy" in self.bio_components:
            cards = self.prior_data.hierarchy_cardinalities if self.prior_data is not None else []
            self.hierarchy_emb = nn.ModuleList(
                [
                    nn.Embedding(int(card), int(self.args.BIO_EMB_DIM), padding_idx=0)
                    for card in cards
                ]
            )
            if len(cards) > 0:
                self.hierarchy_proj = nn.Sequential(
                    nn.Linear(len(cards) * int(self.args.BIO_EMB_DIM), self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
            extra_fused_components += 1
        if "graph" in self.bio_components:
            graph_card = len(self.prior_data.graph_node_names) if self.prior_data is not None else 1
            self.graph_node_emb = nn.Embedding(
                int(graph_card), int(self.args.BIO_EMB_DIM), padding_idx=0
            )
            self.graph_proj = nn.Sequential(
                nn.Linear(int(self.args.BIO_EMB_DIM) + 6, self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            edges = (
                self.prior_data.graph_edges
                if self.prior_data is not None
                else np.zeros((0, 2), dtype=np.int64)
            )
            weights = (
                self.prior_data.graph_edge_weights
                if self.prior_data is not None
                else np.zeros((0,), dtype=np.float32)
            )
            self.register_buffer(
                "graph_edges", torch.tensor(edges, dtype=torch.long), persistent=False
            )
            self.register_buffer(
                "graph_edge_weights", torch.tensor(weights, dtype=torch.float32), persistent=False
            )
            extra_fused_components += 1
        if "anatomy" in self.bio_components:
            cards = self.prior_data.anatomy_cardinalities if self.prior_data is not None else []
            numeric_dim = (
                int(self.prior_data.numeric_context.shape[1]) if self.prior_data is not None else 0
            )
            self.anatomy_emb = nn.ModuleList(
                [
                    nn.Embedding(int(card), int(self.args.BIO_EMB_DIM), padding_idx=0)
                    for card in cards
                ]
            )
            anatomy_input_dim = len(cards) * int(self.args.BIO_EMB_DIM) + numeric_dim
            if anatomy_input_dim > 0:
                self.anatomy_proj = nn.Sequential(
                    nn.Linear(anatomy_input_dim, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
                )
            extra_fused_components += 1
        if "drivers" in self.bio_components:
            self.driver_proj = nn.Sequential(
                nn.Linear(int(n_genes), self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            extra_fused_components += 1
        if "grn" in self.bio_components:
            self.grn_proj = nn.Sequential(
                nn.Linear(int(n_genes), self.bio_prior_hidden),
                nn.GELU(),
                nn.Linear(self.bio_prior_hidden, self.bio_prior_hidden),
            )
            grn = (
                self.prior_data.grn_adjacency
                if self.prior_data is not None
                else np.zeros((n_genes, n_genes), dtype=np.float32)
            )
            grn_target_support = (
                self.prior_data.grn_target_support
                if self.prior_data is not None
                else np.zeros((n_genes,), dtype=np.float32)
            )
            self.register_buffer(
                "grn_adjacency", torch.tensor(grn, dtype=torch.float32), persistent=True
            )
            self.register_buffer(
                "grn_target_support",
                torch.tensor(grn_target_support, dtype=torch.float32),
                persistent=True,
            )
            extra_fused_components += 1
        final_extra_dim = (
            0 if self.adaptive_prior_fusion else extra_fused_components * self.bio_prior_hidden
        )
        self.final_proj = nn.Sequential(
            nn.Linear(
                int(temporal_hidden)
                + int(type_emb_dim)
                + int(traj_hint_hidden)
                + 1
                + final_extra_dim,
                int(flow_context_dim),
            ),
            nn.GELU(),
            nn.Linear(int(flow_context_dim), int(flow_context_dim)),
        )
        self.bidirectional_mode = str(self.args.FM_TEMPORAL_DIRECTION) == "bidirectional"
        self.direction_embedding: Optional[nn.Embedding] = None
        self.direction_adapter: Optional[nn.Module] = None
        if self.bidirectional_mode:
            self.direction_embedding = nn.Embedding(2, int(flow_context_dim))
            nn.init.zeros_(self.direction_embedding.weight)
            if str(self.args.FM_BIDIRECTIONAL_CONDITIONING) == "embedding_adapter":
                self.direction_adapter = nn.Sequential(
                    nn.Linear(
                        2 * int(flow_context_dim), int(self.args.FM_BIDIRECTIONAL_ADAPTER_HIDDEN)
                    ),
                    nn.GELU(),
                    nn.Linear(
                        int(self.args.FM_BIDIRECTIONAL_ADAPTER_HIDDEN), int(flow_context_dim)
                    ),
                )
                nn.init.zeros_(self.direction_adapter[-1].weight)
                nn.init.zeros_(self.direction_adapter[-1].bias)
        self.prior_adapters = nn.ModuleDict()
        self.prior_global_logits = nn.ParameterDict()
        self.prior_conditional_gates = nn.ModuleDict()
        if self.adaptive_prior_fusion:
            initial_probability = float(self.args.FM_PRIOR_GATE_INIT)
            initial_logit = math.log(initial_probability / (1.0 - initial_probability))
            for name in self.prior_component_names:
                adapter = nn.Sequential(
                    nn.Linear(self.bio_prior_hidden, int(flow_context_dim)),
                    nn.GELU(),
                    nn.Linear(int(flow_context_dim), int(flow_context_dim)),
                )
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
                self.prior_adapters[name] = adapter
                self.prior_global_logits[name] = nn.Parameter(
                    torch.tensor(initial_logit, dtype=torch.float32)
                )
                gate = nn.Sequential(
                    nn.Linear(int(flow_context_dim) + self.bio_prior_hidden, self.bio_prior_hidden),
                    nn.GELU(),
                    nn.Linear(self.bio_prior_hidden, 1),
                )
                nn.init.zeros_(gate[-1].weight)
                nn.init.constant_(gate[-1].bias, initial_logit)
                self.prior_conditional_gates[name] = gate

    def _zero_bio_feature(self, context_x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            int(context_x.shape[0]),
            self.bio_prior_hidden,
            device=context_x.device,
            dtype=context_x.dtype,
        )

    def set_prior_training_progress(self, epoch: int) -> float:
        """Set the residual-prior curriculum strength for one training epoch."""
        if not self.adaptive_prior_fusion:
            self._prior_strength = 1.0
            return self._prior_strength
        warmup = int(self.args.FM_PRIOR_WARMUP_EPOCHS)
        ramp = int(self.args.FM_PRIOR_RAMP_EPOCHS)
        if int(epoch) <= warmup:
            strength = 0.0
        elif ramp <= 0:
            strength = 1.0
        else:
            strength = min(1.0, max(0.0, (float(epoch) - float(warmup)) / float(ramp)))
        self._prior_strength = float(strength)
        return self._prior_strength

    def set_prior_inference_strength(self) -> None:
        self._prior_strength = 1.0

    def prior_gate_regularization(self) -> torch.Tensor:
        if not self.adaptive_prior_fusion or not self._last_prior_gate_tensors:
            return self.final_proj[0].weight.new_zeros(())
        return torch.stack([gate.mean() for gate in self._last_prior_gate_tensors.values()]).mean()

    def prior_gate_summary(self) -> Dict[str, float]:
        return dict(self._last_prior_gate_means)

    def _apply_context_dropout(
        self, context_x: torch.Tensor, context_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-only robustness augmentation; default probabilities are zero."""
        if not self.training:
            return (context_x, context_mask)
        gene_p = float(self.args.FM_CONTEXT_GENE_DROPOUT)
        cell_p = float(self.args.FM_CONTEXT_CELL_DROPOUT)
        if gene_p > 0.0:
            keep_gene = (
                torch.rand(context_x.shape[0], 1, 1, context_x.shape[-1], device=context_x.device)
                >= gene_p
            ).to(context_x.dtype)
            context_x = context_x * keep_gene
        if cell_p > 0.0:
            keep_cell = (torch.rand_like(context_mask) >= cell_p).to(context_mask.dtype)
            dropped_mask = context_mask * keep_cell
            was_valid = context_mask.sum(dim=2) > 0
            now_empty = (dropped_mask.sum(dim=2) <= 0) & was_valid
            if torch.any(now_empty):
                first_valid = torch.argmax(context_mask, dim=2)
                rows = torch.nonzero(now_empty, as_tuple=False)
                dropped_mask[rows[:, 0], rows[:, 1], first_valid[rows[:, 0], rows[:, 1]]] = 1.0
            context_mask = dropped_mask
        return (context_x, context_mask)

    def _adaptive_fuse_priors(
        self, base_context: torch.Tensor, prior_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self._last_prior_gate_tensors = {}
        self._last_prior_gate_means = {}
        if not self.adaptive_prior_fusion:
            return base_context
        fused = base_context
        residuals: List[torch.Tensor] = []
        component_dropout = float(self.args.FM_PRIOR_DROPOUT)
        for name in self.prior_component_names:
            feature = prior_features[name]
            if str(self.args.FM_PRIOR_GATE_MODE) == "conditional":
                gate = torch.sigmoid(
                    self.prior_conditional_gates[name](torch.cat([base_context, feature], dim=-1))
                )
            else:
                gate = (
                    torch.sigmoid(self.prior_global_logits[name])
                    .reshape(1, 1)
                    .expand(feature.shape[0], 1)
                )
            if self.training and component_dropout > 0.0:
                keep = (
                    torch.rand(feature.shape[0], 1, device=feature.device) >= component_dropout
                ).to(feature.dtype)
                gate_for_residual = gate * keep / (1.0 - component_dropout)
            else:
                gate_for_residual = gate
            residuals.append(gate_for_residual * self.prior_adapters[name](feature))
            self._last_prior_gate_tensors[name] = gate
            self._last_prior_gate_means[name] = float(gate.detach().mean().cpu())
        if residuals:
            residual_sum = torch.stack(residuals, dim=0).sum(dim=0) / math.sqrt(
                float(len(residuals))
            )
            fused = fused + float(self._prior_strength) * residual_sum
        return fused

    def graph_regularization(self) -> torch.Tensor:
        if (
            "graph" not in self.bio_components
            or self.graph_node_emb is None
            or self.graph_edges.numel() == 0
        ):
            return self.final_proj[0].weight.new_zeros(())
        src = self.graph_edges[:, 0]
        dst = self.graph_edges[:, 1]
        diff2 = torch.mean((self.graph_node_emb(src) - self.graph_node_emb(dst)) ** 2, dim=1)
        weights = torch.abs(self.graph_edge_weights).to(diff2.dtype)
        return torch.sum(diff2 * weights) / torch.sum(weights).clamp_min(1e-08)

    def forward(
        self,
        *,
        context_x: torch.Tensor,
        context_mask: torch.Tensor,
        day_valid: torch.Tensor,
        target_type_prop: torch.Tensor,
        target_type_id: Optional[torch.Tensor],
        traj_hint: Optional[torch.Tensor],
        traj_hint_mask: Optional[torch.Tensor],
        use_type: bool,
        use_traj_hint: bool,
        batch_context: Optional[Dict[str, torch.Tensor]] = None,
        population_only: bool = False,
    ) -> torch.Tensor:
        b, d, _, _ = context_x.shape
        context_x, context_mask = self._apply_context_dropout(context_x, context_mask)
        cell_h = self.cell_encoder(context_x)
        day_h = masked_mean(cell_h, context_mask, dim=2)
        day_h = torch.cat([day_h, target_type_prop.unsqueeze(-1)], dim=-1)
        day_h = self.day_proj(day_h)
        if (
            "actual_time" in self.bio_components
            and (not self.adaptive_prior_fusion)
            and (self.actual_time_proj is not None)
            and (batch_context is not None)
        ):
            day_h = day_h + self.actual_time_proj(
                batch_context["actual_time_features"].to(day_h.dtype)
            )
        lag_ids = torch.arange(d, device=context_x.device).unsqueeze(0).expand(b, d)
        day_h = day_h + self.day_lag_emb(lag_ids)
        day_h = day_h * day_valid.unsqueeze(-1)
        _, h_n = self.temporal_gru(day_h)
        context = h_n[-1]
        if use_type and target_type_id is not None:
            type_feat = self.type_emb(target_type_id)
        else:
            type_feat = torch.zeros(
                b, self.type_emb.embedding_dim, device=context_x.device, dtype=context_x.dtype
            )
        if use_traj_hint and traj_hint is not None and (traj_hint_mask is not None):
            traj_feat = self.traj_hint_proj(traj_hint)
            traj_feat = traj_feat * traj_hint_mask.unsqueeze(-1)
            traj_present = traj_hint_mask.unsqueeze(-1)
        else:
            traj_feat = torch.zeros(
                b,
                self.traj_hint_proj[-2].out_features,
                device=context_x.device,
                dtype=context_x.dtype,
            )
            traj_present = torch.zeros(b, 1, device=context_x.device, dtype=context_x.dtype)
        fused_parts = [context, type_feat, traj_feat, traj_present]
        prior_features: Dict[str, torch.Tensor] = {}
        if "actual_time" in self.bio_components and self.adaptive_prior_fusion:
            if self.actual_time_context_proj is not None and batch_context is not None:
                time_raw = batch_context["actual_time_features"].to(context_x.dtype)
                time_summary = masked_mean(time_raw, day_valid, dim=1)
                prior_features["actual_time"] = self.actual_time_context_proj(time_summary)
            else:
                prior_features["actual_time"] = self._zero_bio_feature(context_x)
        if "hierarchy" in self.bio_components:
            if (
                not population_only
                and self.hierarchy_proj is not None
                and (batch_context is not None)
            ):
                codes = batch_context["hierarchy_codes"]
                hierarchy_raw = torch.cat(
                    [emb(codes[:, i]) for i, emb in enumerate(self.hierarchy_emb)], dim=-1
                )
                hierarchy_feature = self.hierarchy_proj(hierarchy_raw)
            else:
                hierarchy_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["hierarchy"] = hierarchy_feature
            else:
                fused_parts.append(hierarchy_feature)
        if "graph" in self.bio_components:
            if (
                not population_only
                and self.graph_node_emb is not None
                and (self.graph_proj is not None)
                and (batch_context is not None)
            ):
                graph_raw = torch.cat(
                    [
                        self.graph_node_emb(batch_context["graph_node_codes"]),
                        batch_context["graph_node_features"].to(context_x.dtype),
                    ],
                    dim=-1,
                )
                graph_feature = self.graph_proj(graph_raw)
            else:
                graph_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["graph"] = graph_feature
            else:
                fused_parts.append(graph_feature)
        if "anatomy" in self.bio_components:
            anatomy_parts: List[torch.Tensor] = []
            if not population_only and batch_context is not None:
                codes = batch_context["anatomy_codes"]
                anatomy_parts.extend([emb(codes[:, i]) for i, emb in enumerate(self.anatomy_emb)])
                if int(batch_context["numeric_context"].shape[1]) > 0:
                    anatomy_parts.append(batch_context["numeric_context"].to(context_x.dtype))
            if self.anatomy_proj is not None and anatomy_parts:
                anatomy_feature = self.anatomy_proj(torch.cat(anatomy_parts, dim=-1))
            else:
                anatomy_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["anatomy"] = anatomy_feature
            else:
                fused_parts.append(anatomy_feature)
        if "drivers" in self.bio_components:
            if not population_only and self.driver_proj is not None and (batch_context is not None):
                driver_feature = self.driver_proj(
                    batch_context["driver_program"].to(context_x.dtype)
                )
            else:
                driver_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["drivers"] = driver_feature
            else:
                fused_parts.append(driver_feature)
        if "grn" in self.bio_components:
            if self.grn_proj is not None:
                source_gene_mean = masked_mean(context_x, context_mask, dim=2)
                source_gene_mean = masked_mean(source_gene_mean, day_valid, dim=1)
                grn_signal = torch.matmul(
                    source_gene_mean, self.grn_adjacency.to(source_gene_mean.dtype)
                )
                if str(self.args.GRN_EXTERNAL_TF_MODE) == "target_gate":
                    grn_signal = grn_signal + source_gene_mean * self.grn_target_support.to(
                        source_gene_mean.dtype
                    )
                grn_feature = self.grn_proj(grn_signal)
            else:
                grn_feature = self._zero_bio_feature(context_x)
            if self.adaptive_prior_fusion:
                prior_features["grn"] = grn_feature
            else:
                fused_parts.append(grn_feature)
        fused = torch.cat(fused_parts, dim=-1)
        fused = self.final_proj(fused)
        if self.bidirectional_mode:
            if batch_context is None or "direction_id" not in batch_context:
                raise RuntimeError(
                    "Bidirectional FM encoding requires batch_context['direction_id']"
                )
            direction_id = batch_context["direction_id"].long().reshape(-1)
            if torch.any((direction_id < 0) | (direction_id > 1)):
                raise ValueError("direction_id must contain only 0=forward or 1=reverse")
            if self.direction_embedding is None:
                raise RuntimeError("Bidirectional direction embedding was not initialized")
            direction_feature = self.direction_embedding(direction_id).to(fused.dtype)
            base_with_direction = fused + direction_feature
            if self.direction_adapter is not None:
                base_with_direction = base_with_direction + self.direction_adapter(
                    torch.cat([fused, direction_feature], dim=-1)
                )
            fused = base_with_direction
        if self.adaptive_prior_fusion:
            for name in self.prior_component_names:
                if name not in prior_features:
                    prior_features[name] = self._zero_bio_feature(context_x)
            fused = self._adaptive_fuse_priors(fused, prior_features)
        return fused


class BLOOM(nn.Module):
    """Conditional flow matching for developmental expression populations.

    Parameters
    ----------
    n_genes, n_celltypes : int
        Sizes of the fixed model gene panel and supplied cell-state vocabulary.
    args : BLOOMConfig or argparse.Namespace, optional
        Model and loss settings. Training supplies the complete parsed namespace.
    prior_data : object, optional
        Aligned prior arrays and category cardinalities constructed in preprocessing.
    variant : BLOOMVariant or compatible native variant, optional
        Base distribution, cell-state conditioning, trajectory and prior selection.
    bidirectional : bool
        Select the bidirectional encoder; direction tokens remain controlled by
        ``args.FM_TEMPORAL_DIRECTION`` as in the original implementation.

    Use ``loss`` for training and ``predict_expression`` or
    ``sample_expression_draws`` for inference. Parameter names and tensor shapes
    are preserved for existing state-dictionary checkpoints.

    ``batch_context`` contains ``context_x`` with shape ``[B,T,C,G]``,
    ``context_mask`` with shape ``[B,T,C]``, ``day_valid`` and
    ``target_type_prop`` with shape ``[B,T]``, and the aligned trajectory hint
    ``traj_hint`` and availability mask with shapes ``[B,G]`` and ``[B]``.
    Biological feature arrays use the same batch dimension. Bidirectional
    queries additionally supply ``direction_id`` (0 forward, 1 reverse).
    """

    def __init__(
        self,
        *,
        n_genes: int,
        n_celltypes: int,
        variant: Optional[BLOOMVariant] = None,
        args: Any = None,
        prior_data: Any = None,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.args = (
            args
            if args is not None
            else BLOOMConfig(FM_TEMPORAL_DIRECTION="bidirectional" if bidirectional else "forward")
        )
        self.prior_data = prior_data
        encoder_type = (
            BidirectionalTemporalSetEncoder if bidirectional else UnidirectionalTemporalSetEncoder
        )
        self.n_genes = int(n_genes)
        self.variant_at_construction = variant
        max_history_days = max(1, int(self.args.SEQ_LEN))
        self.encoder = encoder_type(
            n_genes=int(n_genes),
            cell_hidden=int(self.args.CELL_ENCODER_HIDDEN),
            day_hidden=int(self.args.DAY_ENCODER_HIDDEN),
            temporal_hidden=int(self.args.TEMPORAL_HIDDEN),
            max_history_days=int(max_history_days),
            n_celltypes=int(n_celltypes),
            type_emb_dim=int(self.args.TYPE_EMB_DIM),
            traj_hint_hidden=int(self.args.TRAJ_HINT_HIDDEN),
            flow_context_dim=int(self.args.FLOW_CONTEXT_DIM),
            bio_ablation=str(variant.bio_ablation) if variant is not None else "bio0_baseline",
            args=self.args,
            prior_data=self.prior_data,
        )
        self.velocity_field = ConditionalVelocityField(
            n_genes=int(n_genes),
            context_dim=int(self.args.FLOW_CONTEXT_DIM),
            hidden_dim=int(self.args.FLOW_HIDDEN_DIM),
        )
        self.point_head = nn.Sequential(
            nn.Linear(int(self.args.FLOW_CONTEXT_DIM), int(self.args.FLOW_CONTEXT_DIM)),
            nn.GELU(),
            nn.Linear(int(self.args.FLOW_CONTEXT_DIM), int(n_genes)),
        )
        self.base_mean_head = nn.Linear(int(self.args.FLOW_CONTEXT_DIM), int(n_genes))
        self.base_logvar_head = nn.Linear(int(self.args.FLOW_CONTEXT_DIM), int(n_genes))
        self.composition_head = nn.Linear(int(self.args.FLOW_CONTEXT_DIM), int(n_celltypes))

    def set_prior_training_progress(self, epoch: int) -> float:
        return self.encoder.set_prior_training_progress(epoch)

    def set_prior_inference_strength(self) -> None:
        self.encoder.set_prior_inference_strength()

    @staticmethod
    def context_at_developmental_fraction(
        batch_context: Dict[str, torch.Tensor], fraction: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Change the requested biological target time within the last measured
        interval while leaving the FM solver coordinate tau untouched.

        fraction=0 is the nearest context stage and fraction=1 is the measured
        endpoint. This is usable at arbitrary fractions during downstream
        interpolation. It requires actual_time_features from bio1+.
        """
        if "actual_time_features" not in batch_context:
            raise RuntimeError(
                "Developmental-time interpolation requires a biological mode containing actual_time"
            )
        fraction = fraction.reshape(-1).to(
            device=batch_context["actual_time_features"].device,
            dtype=batch_context["actual_time_features"].dtype,
        )
        if int(fraction.shape[0]) != int(batch_context["actual_time_features"].shape[0]):
            raise ValueError("One developmental fraction is required per batch row")
        if torch.any((fraction < 0.0) | (fraction > 1.0)):
            raise ValueError("Developmental fractions must be in [0, 1]")
        out = dict(batch_context)
        time_features = batch_context["actual_time_features"].clone()
        nearest_source = time_features[:, -1, 0]
        measured_target = time_features[:, -1, 1]
        requested_target = nearest_source + fraction * (measured_target - nearest_source)
        time_features[:, :, 1] = requested_target.unsqueeze(1)
        time_features[:, :, 2] = requested_target.unsqueeze(1) - time_features[:, :, 0]
        out["actual_time_features"] = time_features
        out["developmental_fraction"] = fraction
        return out

    @torch.no_grad()
    def predict_expression_at_developmental_fraction(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        fraction: torch.Tensor,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        fraction_context = self.context_at_developmental_fraction(batch_context, fraction)
        return self.predict_expression(
            batch_context=fraction_context,
            target_type_id=target_type_id,
            variant=variant,
            num_steps=num_steps,
            num_samples=num_samples,
        )

    def encode_context(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
    ) -> torch.Tensor:
        use_type = str(variant.celltype_mode) in {"soft", "hard"}
        return self.encoder(
            context_x=batch_context["context_x"],
            context_mask=batch_context["context_mask"],
            day_valid=batch_context["day_valid"],
            target_type_prop=batch_context["target_type_prop"],
            target_type_id=target_type_id if use_type else None,
            traj_hint=batch_context["traj_hint"],
            traj_hint_mask=batch_context["traj_hint_mask"],
            use_type=use_type,
            use_traj_hint=bool(variant.use_traj_soft),
            batch_context=batch_context,
        )

    def encode_population_context(self, *, batch_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode history without target-cell labels or target-specific priors."""
        return self.encoder(
            context_x=batch_context["context_x"],
            context_mask=batch_context["context_mask"],
            day_valid=batch_context["day_valid"],
            target_type_prop=batch_context["target_type_prop"],
            target_type_id=None,
            traj_hint=batch_context["traj_hint"],
            traj_hint_mask=batch_context["traj_hint_mask"],
            use_type=False,
            use_traj_hint=False,
            batch_context=batch_context,
            population_only=True,
        )

    def predict_celltype_proportions(
        self, *, population_context: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Predict a cell-state probability vector from the supplied population history."""
        context = self.encode_population_context(batch_context=population_context)
        return torch.softmax(self.composition_head(context), dim=-1)

    def draw_x0(
        self, *, context: torch.Tensor, variant: BLOOMVariant, dtype: torch.dtype
    ) -> torch.Tensor:
        b = int(context.shape[0])
        g = int(self.n_genes)
        if str(variant.base_mode) == "standard":
            return torch.randn(b, g, device=context.device, dtype=dtype) * float(
                self.args.FLOW_SIGMA
            )
        if str(variant.base_mode) == "context_gaussian":
            mu0 = self.base_mean_head(context)
            logvar0 = torch.clamp(
                self.base_logvar_head(context),
                min=float(self.args.BASE_LOGVAR_MIN),
                max=float(self.args.BASE_LOGVAR_MAX),
            )
            std0 = torch.exp(0.5 * logvar0)
            eps = torch.randn(b, g, device=context.device, dtype=dtype)
            return mu0 + std0 * eps
        raise ValueError(variant.base_mode)

    def _trajectory_anchor(
        self, *, batch_context: Dict[str, torch.Tensor], reference_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the trajectory anchor expression and a [B, 1] anchor mask.

        If FM_RESIDUAL_MODE='trajectory', the anchor is the previous trajectory
        cell expression stored in batch_context['traj_hint'].  If no anchor is
        available for a cell, mask=0 and the code safely falls back to the old
        non-residual target for that cell.
        """
        if str(self.args.FM_RESIDUAL_MODE) != "trajectory":
            anchor = torch.zeros_like(reference_x)
            anchor_mask = torch.zeros(
                (int(reference_x.shape[0]), 1), device=reference_x.device, dtype=reference_x.dtype
            )
            return (anchor, anchor_mask)
        anchor = batch_context["traj_hint"].to(device=reference_x.device, dtype=reference_x.dtype)
        anchor_mask = (
            batch_context["traj_hint_mask"]
            .to(device=reference_x.device, dtype=reference_x.dtype)
            .unsqueeze(-1)
        )
        if bool(self.args.FM_RESIDUAL_REQUIRE_TRAJ_HINT) and torch.sum(anchor_mask).item() <= 0:
            raise RuntimeError(
                "FM_RESIDUAL_REQUIRE_TRAJ_HINT=True but this batch has no trajectory anchors."
            )
        anchor = anchor * anchor_mask
        return (anchor, anchor_mask)

    def _target_for_flow(
        self, *, batch_context: Dict[str, torch.Tensor], target_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor, anchor_mask = self._trajectory_anchor(
            batch_context=batch_context, reference_x=target_x
        )
        if str(self.args.FM_RESIDUAL_MODE) == "trajectory":
            target_for_flow = target_x - anchor
        else:
            target_for_flow = target_x
        return (target_for_flow, anchor, anchor_mask)

    def _expression_from_flow_target(
        self, *, flow_target_prediction: torch.Tensor, anchor: torch.Tensor
    ) -> torch.Tensor:
        if str(self.args.FM_RESIDUAL_MODE) == "trajectory":
            return anchor + flow_target_prediction
        return flow_target_prediction

    def point_predict_expression(
        self,
        *,
        context: torch.Tensor,
        batch_context: Dict[str, torch.Tensor],
        reference_x: torch.Tensor,
    ) -> torch.Tensor:
        point_raw = self.point_head(context)
        anchor, _ = self._trajectory_anchor(batch_context=batch_context, reference_x=reference_x)
        return self._expression_from_flow_target(flow_target_prediction=point_raw, anchor=anchor)

    def loss(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_x: torch.Tensor,
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        include_regularization: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        context = self.encode_context(
            batch_context=batch_context, target_type_id=target_type_id, variant=variant
        )
        target_for_flow, anchor, anchor_mask = self._target_for_flow(
            batch_context=batch_context, target_x=target_x
        )
        x0 = self.draw_x0(context=context, variant=variant, dtype=target_x.dtype)
        b = int(target_x.shape[0])
        tau = torch.rand(b, 1, device=target_x.device, dtype=target_x.dtype)
        x_tau = (1.0 - tau) * x0 + tau * target_for_flow
        target_velocity = target_for_flow - x0
        pred_velocity = self.velocity_field(x_tau=x_tau, tau=tau, context=context)
        flow_loss = ((pred_velocity - target_velocity) ** 2).mean()
        point_raw = self.point_head(context)
        point_pred_expression = self._expression_from_flow_target(
            flow_target_prediction=point_raw, anchor=anchor
        )
        aux_point_loss = torch.mean((point_pred_expression - target_x) ** 2)
        total = (
            float(self.args.FLOW_LOSS_WEIGHT) * flow_loss
            + float(self.args.AUX_POINT_LOSS_WEIGHT) * aux_point_loss
        )
        traj_soft_loss = torch.tensor(0.0, device=target_x.device, dtype=target_x.dtype)
        if str(self.args.FM_RESIDUAL_MODE) == "none" and bool(variant.use_traj_soft):
            traj_hint = batch_context["traj_hint"]
            traj_hint_mask = batch_context["traj_hint_mask"] > 0
            if torch.any(traj_hint_mask):
                diff = point_pred_expression[traj_hint_mask] - traj_hint[traj_hint_mask]
                traj_soft_loss = torch.mean(diff**2)
                total = total + float(self.args.TRAJ_SOFT_WEIGHT) * traj_soft_loss
        graph_reg_loss = torch.tensor(0.0, device=target_x.device, dtype=target_x.dtype)
        if (
            include_regularization
            and variant.uses_bio_component("graph")
            and (float(self.args.CURATED_GRAPH_REG_WEIGHT) > 0.0)
        ):
            graph_reg_loss = self.encoder.graph_regularization().to(target_x.dtype)
            total = total + float(self.args.CURATED_GRAPH_REG_WEIGHT) * graph_reg_loss
        prior_gate_reg_loss = torch.tensor(0.0, device=target_x.device, dtype=target_x.dtype)
        if (
            include_regularization
            and self.encoder.adaptive_prior_fusion
            and (float(self.args.FM_PRIOR_GATE_L1_WEIGHT) > 0.0)
        ):
            prior_gate_reg_loss = self.encoder.prior_gate_regularization().to(target_x.dtype)
            total = total + float(self.args.FM_PRIOR_GATE_L1_WEIGHT) * prior_gate_reg_loss
        metrics = {
            "flow_loss": float(flow_loss.detach().cpu()),
            "aux_point_loss": float(aux_point_loss.detach().cpu()),
            "traj_soft_loss": float(traj_soft_loss.detach().cpu()),
            "graph_reg_loss": float(graph_reg_loss.detach().cpu()),
            "prior_gate_reg_loss": float(prior_gate_reg_loss.detach().cpu()),
            "anchor_fraction": float(torch.mean(anchor_mask.detach()).cpu()),
        }
        return (total, metrics)

    @torch.no_grad()
    def sample_mean(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        context = self.encode_context(
            batch_context=batch_context, target_type_id=target_type_id, variant=variant
        )
        b = int(context.shape[0])
        dt = 1.0 / float(num_steps)
        samples = []
        for _ in range(int(num_samples)):
            x = self.draw_x0(context=context, variant=variant, dtype=context.dtype)
            for step in range(int(num_steps)):
                tau_value = (step + 0.5) / float(num_steps)
                tau = torch.full(
                    (b, 1), fill_value=float(tau_value), device=context.device, dtype=context.dtype
                )
                v = self.velocity_field(x_tau=x, tau=tau, context=context)
                x = x + dt * v
            samples.append(x)
        flow_target_prediction = torch.stack(samples, dim=0).mean(dim=0)
        anchor, _ = self._trajectory_anchor(
            batch_context=batch_context, reference_x=flow_target_prediction
        )
        return self._expression_from_flow_target(
            flow_target_prediction=flow_target_prediction, anchor=anchor
        )

    @torch.no_grad()
    def sample_expression_draws(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        """Return individual stochastic expression draws with shape [S, B, G].

        Keeping draws separate preserves the variability needed for
        distributional metrics and sampling-based uncertainty summaries.
        """
        if int(num_steps) <= 0 or int(num_samples) <= 0:
            raise ValueError("num_steps and num_samples must both be > 0")
        context = self.encode_context(
            batch_context=batch_context, target_type_id=target_type_id, variant=variant
        )
        b = int(context.shape[0])
        dt = 1.0 / float(num_steps)
        draws: List[torch.Tensor] = []
        for _ in range(int(num_samples)):
            x = self.draw_x0(context=context, variant=variant, dtype=context.dtype)
            for step in range(int(num_steps)):
                tau_value = (step + 0.5) / float(num_steps)
                tau = torch.full(
                    (b, 1), fill_value=float(tau_value), device=context.device, dtype=context.dtype
                )
                velocity = self.velocity_field(x_tau=x, tau=tau, context=context)
                x = x + dt * velocity
            anchor, _ = self._trajectory_anchor(batch_context=batch_context, reference_x=x)
            draws.append(self._expression_from_flow_target(flow_target_prediction=x, anchor=anchor))
        return torch.stack(draws, dim=0)

    @torch.no_grad()
    def sample_expression_draws_at_developmental_fraction(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        fraction: torch.Tensor,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        """Return [S, B, G] stochastic draws at arbitrary biological fractions."""
        fraction_context = self.context_at_developmental_fraction(batch_context, fraction)
        return self.sample_expression_draws(
            batch_context=fraction_context,
            target_type_id=target_type_id,
            variant=variant,
            num_steps=num_steps,
            num_samples=num_samples,
        )

    @torch.no_grad()
    def predict_with_history_gene_scaling(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        gene_indices: Sequence[int],
        scale: float,
        history_positions: Optional[Sequence[int]] = None,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Predict after scaling selected observed-history genes.

        scale=0 is an in-silico knockout and values in (0, 1) are knockdowns.
        By default every history day is changed. history_positions can restrict
        the intervention to selected model-facing history slots. This operation
        changes only context_x; it does not pretend to edit the GRN itself.
        """
        genes = sorted({int(x) for x in gene_indices})
        if not genes:
            raise ValueError("gene_indices must contain at least one gene")
        if not math.isfinite(float(scale)) or float(scale) < 0.0:
            raise ValueError("scale must be finite and >= 0")
        if genes[0] < 0 or genes[-1] >= int(self.n_genes):
            raise IndexError(f"gene index outside [0, {self.n_genes})")
        perturbed = dict(batch_context)
        context_x = batch_context["context_x"].clone()
        n_history = int(context_x.shape[1])
        if history_positions is None:
            positions = list(range(n_history))
        else:
            positions = sorted({int(x) for x in history_positions})
            if not positions:
                raise ValueError("history_positions cannot be empty when supplied")
            if positions[0] < 0 or positions[-1] >= n_history:
                raise IndexError(f"history position outside [0, {n_history})")
        for position in positions:
            context_x[:, position, :, genes] = context_x[:, position, :, genes] * float(scale)
        perturbed["context_x"] = context_x
        return self.predict_expression(
            batch_context=perturbed,
            target_type_id=target_type_id,
            variant=variant,
            num_steps=num_steps,
            num_samples=num_samples,
        )

    @torch.no_grad()
    def predict_with_history_gene_knockout(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        gene_indices: Sequence[int],
        history_positions: Optional[Sequence[int]] = None,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        """Convenience wrapper for scale=0 history-gene interventions."""
        return self.predict_with_history_gene_scaling(
            batch_context=batch_context,
            target_type_id=target_type_id,
            variant=variant,
            gene_indices=gene_indices,
            scale=0.0,
            history_positions=history_positions,
            num_steps=num_steps,
            num_samples=num_samples,
        )

    @torch.no_grad()
    def predict_expression(
        self,
        *,
        batch_context: Dict[str, torch.Tensor],
        target_type_id: torch.Tensor,
        variant: BLOOMVariant,
        num_steps: int,
        num_samples: int,
    ) -> torch.Tensor:
        """Generate expression using the configured prediction readout.

        ``flow_sample`` averages stochastic flow draws. ``point_head`` and
        ``residual_point`` use the deterministic endpoint head. ``blend``
        combines the two with FM_POINT_BLEND_WEIGHT. In trajectory-residual
        mode, predicted residuals are added to the supplied trajectory anchor.
        """
        mode = str(self.args.FM_PREDICTION_MODE)
        context = self.encode_context(
            batch_context=batch_context, target_type_id=target_type_id, variant=variant
        )
        reference_x = torch.zeros(
            (int(context.shape[0]), int(self.n_genes)), device=context.device, dtype=context.dtype
        )
        if mode == "flow_sample":
            return self.sample_mean(
                batch_context=batch_context,
                target_type_id=target_type_id,
                variant=variant,
                num_steps=num_steps,
                num_samples=num_samples,
            )
        if mode in {"point_head", "residual_point"}:
            return self.point_predict_expression(
                context=context, batch_context=batch_context, reference_x=reference_x
            )
        if mode == "blend":
            point_pred = self.point_predict_expression(
                context=context, batch_context=batch_context, reference_x=reference_x
            )
            flow_pred = self.sample_mean(
                batch_context=batch_context,
                target_type_id=target_type_id,
                variant=variant,
                num_steps=num_steps,
                num_samples=num_samples,
            )
            w = float(self.args.FM_POINT_BLEND_WEIGHT)
            w = min(1.0, max(0.0, w))
            return w * point_pred + (1.0 - w) * flow_pred
        raise ValueError(f"Unsupported FM_PREDICTION_MODE={mode!r}")


class UnidirectionalBLOOM(BLOOM):
    """BLOOM with the original unidirectional population encoder."""

    def __init__(self, *, n_genes: int, n_celltypes: int, variant=None, args=None, prior_data=None):
        super().__init__(
            n_genes=n_genes,
            n_celltypes=n_celltypes,
            variant=variant,
            args=args,
            prior_data=prior_data,
            bidirectional=False,
        )


class BidirectionalBLOOM(BLOOM):
    """BLOOM with the developmental-direction-aware population encoder."""

    def __init__(self, *, n_genes: int, n_celltypes: int, variant=None, args=None, prior_data=None):
        super().__init__(
            n_genes=n_genes,
            n_celltypes=n_celltypes,
            variant=variant,
            args=args,
            prior_data=prior_data,
            bidirectional=True,
        )


def bind_native_models(args, prior_data, *, direction="unidirectional"):
    """Bind legacy constructor names to one training run's configuration.

    The native training loops construct models without an explicit ``args``
    keyword. These small adapters preserve that interface while keeping the
    implementations and model-instance settings here. State-dictionary keys
    are unchanged. Return values are classes, including the encoder, so native
    inference adapters can use their existing type and static-method APIs.
    """
    if direction not in {"unidirectional", "bidirectional"}:
        raise ValueError("direction must be 'unidirectional' or 'bidirectional'")
    encoder = (
        BidirectionalTemporalSetEncoder
        if direction == "bidirectional"
        else UnidirectionalTemporalSetEncoder
    )
    model = BidirectionalBLOOM if direction == "bidirectional" else UnidirectionalBLOOM
    run_args, run_priors = args, prior_data

    class NativeTemporalSetEncoder(encoder):
        def __init__(self, **kwargs):
            kwargs.setdefault("args", run_args)
            kwargs.setdefault("prior_data", run_priors)
            super().__init__(**kwargs)

    class NativeConditionalFlowMatching(model):
        def __init__(self, **kwargs):
            kwargs.setdefault("args", run_args)
            kwargs.setdefault("prior_data", run_priors)
            super().__init__(**kwargs)

    return {
        "FMCellEncoder": FMCellEncoder,
        "ConditionalVelocityField": ConditionalVelocityField,
        "FMTemporalSetEncoder": NativeTemporalSetEncoder,
        "RawConditionalFlowMatching": NativeConditionalFlowMatching,
    }


__all__ = [
    "BLOOM",
    "BLOOMConfig",
    "BLOOMVariant",
    "UnidirectionalBLOOM",
    "BidirectionalBLOOM",
    "FMCellEncoder",
    "ConditionalVelocityField",
    "UnidirectionalTemporalSetEncoder",
    "BidirectionalTemporalSetEncoder",
    "bind_native_models",
]
