"""
Kortex Latent Core — the gist->soft-token projector.

Maps one kortex HRR gist vector (d_gist, e.g. 1536) into `n_soft` soft-token
embeddings that live in the target model's input space (d_model, e.g. 5120 for
Qwen3.8-27B). These embeddings are prepended before the question's token
embeddings, so the model conditions on the compressed codebase with zero text.

This is the trainable heart of KLC (see ARCHITECTURE.md, component 2). The base
model stays frozen; only this projector (and optionally a small LoRA) is trained
by context distillation (distill_train.py).

Design notes:
- We normalize the incoming gist (kortex already spherically normalizes, but the
  proxy may pass a raw sum), then expand to n_soft distinct tokens via a per-slot
  head so each soft token can specialize (early tokens ~ structure, later ~ detail).
- Output is scaled to match the RMS of real Qwen input embeddings so the injected
  tokens are not out-of-distribution in magnitude (a common failure mode for soft
  prompts). The target RMS is measured once from the embedding table (see
  `calibrate_output_scale`) and frozen in.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GistProjector(nn.Module):
    def __init__(
        self,
        d_gist: int = 1536,
        d_model: int = 5120,
        n_soft: int = 8,
        hidden: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_gist = d_gist
        self.d_model = d_model
        self.n_soft = n_soft

        self.in_norm = nn.LayerNorm(d_gist)
        self.trunk = nn.Sequential(
            nn.Linear(d_gist, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # One head per soft-token slot so tokens differentiate.
        self.slot_heads = nn.ModuleList(
            [nn.Linear(hidden, d_model) for _ in range(n_soft)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        # Frozen scalar so injected embeddings match real-embedding RMS.
        self.register_buffer("output_scale", torch.tensor(1.0))

    def forward(self, gist: torch.Tensor) -> torch.Tensor:
        """gist: (B, d_gist)  ->  soft tokens: (B, n_soft, d_model)."""
        if gist.dim() == 1:
            gist = gist.unsqueeze(0)
        h = self.trunk(self.in_norm(gist))            # (B, hidden)
        toks = torch.stack([head(h) for head in self.slot_heads], dim=1)  # (B, n_soft, d_model)
        toks = self.out_norm(toks)
        return toks * self.output_scale

    @torch.no_grad()
    def calibrate_output_scale(self, embed_weight: torch.Tensor):
        """Match injected-token RMS to the model's real input-embedding RMS.

        embed_weight: (vocab, d_model) tensor = model.get_input_embeddings().weight
        Call once before training so soft tokens sit in-distribution.
        """
        target_rms = embed_weight.float().pow(2).mean().sqrt()
        # Measure our own current output RMS on a random gist batch.
        probe = torch.randn(64, self.d_gist, device=embed_weight.device)
        cur = self.forward(probe)
        cur_rms = cur.float().pow(2).mean().sqrt().clamp_min(1e-6)
        self.output_scale.copy_((target_rms / cur_rms).to(self.output_scale.dtype))
        return float(self.output_scale)


def load_gist_klc(path: str):
    """Read a repo.klc produced by `aim-index export-gist` (component 1).

    Format (little-endian):
      magic 'KLC1' (4 bytes) | d (u32) | k (u32) |
      gist:   f32 x d |
      entries: k x ( path_key f32 x d , content f32 x d )
    Returns dict: {'gist': (d,), 'path_keys': (k,d), 'contents': (k,d)}
    """
    import numpy as np
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KLC1":
            raise ValueError(f"bad magic {magic!r}, expected b'KLC1'")
        d = int.from_bytes(f.read(4), "little")
        k = int.from_bytes(f.read(4), "little")
        gist = np.frombuffer(f.read(4 * d), dtype="<f4").copy()
        rest = np.frombuffer(f.read(4 * 2 * d * k), dtype="<f4").reshape(k, 2, d)
    return {
        "gist": torch.from_numpy(gist),
        "path_keys": torch.from_numpy(rest[:, 0, :].copy()),
        "contents": torch.from_numpy(rest[:, 1, :].copy()),
    }


if __name__ == "__main__":
    # Smoke test: shapes + RMS calibration, no model needed.
    proj = GistProjector(d_gist=1536, d_model=5120, n_soft=8)
    fake_embed = torch.randn(248320, 5120) * 0.02  # Qwen-like embedding scale
    scale = proj.calibrate_output_scale(fake_embed)
    g = torch.randn(4, 1536)
    out = proj(g)
    rms = out.float().pow(2).mean().sqrt().item()
    print(f"projector OK | out shape {tuple(out.shape)} | scale {scale:.3f} | out RMS {rms:.4f} "
          f"| target RMS {fake_embed.float().pow(2).mean().sqrt().item():.4f}")
    print(f"params: {sum(p.numel() for p in proj.parameters())/1e6:.1f}M")
