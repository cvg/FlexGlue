"""Building blocks of FlexGlue: transformer layers, match head and the PairContext."""

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from . import ops


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), -1).flatten(-2)


def apply_rotary(enc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return t * enc[0] + rotate_half(t) * enc[1]


def mlp(dims: list) -> nn.Sequential:
    layers = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i < len(dims) - 1:
            layers += [nn.LayerNorm(dims[i]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def ffn(dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * dim, 2 * dim),
        nn.LayerNorm(2 * dim),
        nn.GELU(),
        nn.Linear(2 * dim, dim),
    )


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int) -> None:
        super().__init__()
        self.Wr = nn.Linear(M, dim // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.Wr(x)
        emb = torch.stack([projected.cos(), projected.sin()], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = ffn(dim)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        q, k = apply_rotary(encoding, q), apply_rotary(encoding, k)
        context = F.scaled_dot_product_attention(q, k, v.contiguous())  # flash
        message = self.out_proj(context.transpose(1, 2).flatten(-2))
        return x + self.ffn(torch.cat([x, self.message_norm(message)], -1))


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.qk_norm = nn.LayerNorm(dim // num_heads)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = ffn(dim)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qk0, qk1, v0, v1 = (
            t.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
            for t in (self.to_qk(x0), self.to_qk(x1), self.to_v(x0), self.to_v(x1))
        )
        qk0, qk1 = self.qk_norm(qk0).to(v0.dtype), self.qk_norm(qk1).to(v1.dtype)
        m0 = F.scaled_dot_product_attention(qk0, qk1, v1)
        m1 = F.scaled_dot_product_attention(qk1, qk0, v0)
        m0, m1 = (self.to_out(m.transpose(1, 2).flatten(-2)) for m in (m0, m1))
        x0 = x0 + self.ffn(torch.cat([x0, self.message_norm(m0)], -1))
        x1 = x1 + self.ffn(torch.cat([x1, self.message_norm(m1)], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = SelfBlock(dim, num_heads)
        self.cross_attn = CrossBlock(dim, num_heads)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        encoding0: torch.Tensor,
        encoding1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cross_attn(
            self.self_attn(x0, encoding0), self.self_attn(x1, encoding1)
        )


class OffsetHead(nn.Module):
    """Sub-cell offset as a soft-argmax over bins spanning +-max_offset cells."""

    def __init__(
        self, dim: int, hidden: int = 256, bins: int = 16, max_offset: float = 1.0
    ) -> None:
        super().__init__()
        self.bins, self.max_offset = bins, max_offset
        self.query_encoder = mlp([dim, hidden, hidden])
        self.match_encoder = mlp([dim, hidden, hidden])
        self.head = mlp([2 * hidden, 2 * hidden, 2 * (bins + 2)])  # last bin unused

    def forward(
        self, f_query: torch.Tensor, f_match: torch.Tensor, hw: Tuple[int, int]
    ) -> torch.Tensor:
        x = torch.cat([self.query_encoder(f_query), self.match_encoder(f_match)], -1)
        logits = self.head(x).unflatten(-1, (2, self.bins + 2))[..., : self.bins + 1]
        bins = torch.arange(self.bins + 1, device=x.device, dtype=x.dtype)
        delta = ((logits.softmax(-1) * bins).sum(-1) / self.bins - 0.5) * 2
        delta = delta * self.max_offset
        return torch.stack([delta[..., 0] * 2 / hw[1], delta[..., 1] * 2 / hw[0]], -1)


class MatchHead(nn.Module):
    """Regress the correspondence of each point in the other image's feature map."""

    def __init__(
        self, dim: int, search_dim: int, query_chunk_size: int, cross: bool = True
    ) -> None:
        super().__init__()
        if not isinstance(query_chunk_size, int) or query_chunk_size <= 0:
            raise ValueError(
                f"query_chunk_size must be a positive int: {query_chunk_size}"
            )
        self.query_chunk_size = query_chunk_size
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.to_search = nn.Linear(dim, search_dim)
        # layer 0 sees no cross-view context: no matchability, no covisibility
        self.matchability = nn.Linear(dim, 1) if cross else None
        self.offset_head = OffsetHead(dim)
        self.covisibility = nn.Linear(dim, 1) if cross else None

    def forward(
        self, points_ni: torch.Tensor, cf_i: torch.Tensor, cf_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalized points (B, N, 2) of i -> predicted points in j and confidence."""
        search = F.normalize(self.to_search(cf_j.flatten(2).mT), dim=-1)
        chunks = [  # bounds the (B, query_chunk_size, h * w) logits
            self.match_points(points, cf_i, cf_j, search)
            for points in points_ni.split(self.query_chunk_size, dim=1)
        ]
        if len(chunks) == 1:
            return chunks[0]
        p2di_nj, confidence = zip(*chunks)
        return torch.cat(p2di_nj, 1), torch.cat(confidence, 1)

    def match_points(
        self,
        points_ni: torch.Tensor,
        cf_i: torch.Tensor,
        cf_j: torch.Tensor,
        search: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward on one chunk of points, with the normalized search features of j."""
        hw = cf_j.shape[-2:]
        f_query = ops.sample_features(cf_i, points_ni)
        query = F.normalize(self.to_search(f_query), dim=-1)
        logits = query @ search.mT * self.temperature
        grid = ops.cell_centers(*hw, scale=(hw[1], hw[0]), device=logits.device)
        p2di_nj, window_mass = ops.entmax15_window_readout(logits, grid, hw)
        f_match = ops.sample_features(cf_j, p2di_nj)
        p2di_nj = p2di_nj + self.offset_head(f_query, f_match, hw)
        if self.matchability is None:
            return p2di_nj, window_mass
        matchability = torch.sigmoid(self.matchability(f_query).squeeze(-1))
        return p2di_nj, matchability * window_mass


class PairContext(NamedTuple):
    """Per-call state from decode_pair."""

    layer: int  # last layer run, selects the head
    covisibility_score: torch.Tensor  # B
    cf: Tuple[torch.Tensor, torch.Tensor]  # cross features (cf0, cf1), B x D x h x w
    fine: Tuple[Optional[dict], Optional[dict]] = (None, None)  # refiner features

    def direction(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cf[i], self.cf[1 - i]

    def fine_direction(self, i: int) -> Tuple[Optional[dict], Optional[dict]]:
        return self.fine[i], self.fine[1 - i]
