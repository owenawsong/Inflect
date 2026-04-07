"""
Inflect — Paralinguistic Module (Para Module)

A small transformer that generates Mimi latents for paralinguistic sounds.
Given a speaker's voice characteristics and a tag (e.g. [laughs]), it
outputs Mimi latents that decode to the actual sound in that speaker's voice.

Architecture (non-autoregressive):
  1. Speaker encoder:   mean-pool speaker_latents [N, 512] → [256]
  2. Tag embedding:     tag_id → [128]
  3. Fusion:            concat → [384] → Linear → [256] = conditioning vector
  4. Duration pred:     conditioning → MLP → predicted T (frame count)
  5. Decoder:           positional queries [T, 256] cross-attend to conditioning
  6. Output proj:       [T, 256] → [T, 512] = Mimi latents

Total params: ~12M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inflect.data.tags import NUM_TAGS

MAX_FRAMES   = 50      # hard cap: 4 seconds @ 12.5 Hz
MIMI_DIM     = 512
COND_DIM     = 256
TAG_DIM      = 128
SPEAKER_DIM  = 512


class SpeakerEncoder(nn.Module):
    """Mean-pools variable-length speaker latents → fixed conditioning vector."""
    def __init__(self, in_dim: int = SPEAKER_DIM, out_dim: int = COND_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, speaker_latents: torch.Tensor) -> torch.Tensor:
        """
        speaker_latents: [B, N, 512] (variable N, padded)
        Returns: [B, COND_DIM]
        """
        pooled = speaker_latents.mean(dim=1)   # [B, 512]
        return self.proj(pooled)               # [B, COND_DIM]


class DurationPredictor(nn.Module):
    """Predicts the number of Mimi frames for the paralinguistic sound."""
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, COND_DIM] → [B, 1] (predicted frame count, float)"""
        return self.net(cond)


class ParaModule(nn.Module):
    def __init__(
        self,
        num_tags:     int = NUM_TAGS,
        mimi_dim:     int = MIMI_DIM,
        cond_dim:     int = COND_DIM,
        tag_dim:      int = TAG_DIM,
        speaker_dim:  int = SPEAKER_DIM,
        n_heads:      int = 4,
        n_layers:     int = 4,
        max_frames:   int = MAX_FRAMES,
    ):
        super().__init__()
        self.cond_dim   = cond_dim
        self.max_frames = max_frames

        # Speaker encoder
        self.speaker_enc = SpeakerEncoder(speaker_dim, cond_dim)

        # Tag embedding
        self.tag_emb = nn.Embedding(num_tags, tag_dim)

        # Fusion: speaker [COND_DIM] + tag [TAG_DIM] → [COND_DIM]
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim + tag_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )

        # Duration predictor
        self.duration_pred = DurationPredictor(cond_dim)

        # Positional embeddings for decoder queries (learned, up to max_frames)
        self.pos_emb = nn.Embedding(max_frames, cond_dim)

        # Transformer decoder: queries = positional, memory = conditioning
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cond_dim,
            nhead=n_heads,
            dim_feedforward=cond_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,          # pre-norm for stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Project to Mimi latent dim
        self.output_proj = nn.Linear(cond_dim, mimi_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_conditioning(
        self,
        speaker_latents: torch.Tensor,  # [B, N, 512]
        tag_ids:         torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Returns conditioning vector [B, 1, COND_DIM] (memory for transformer)."""
        spk  = self.speaker_enc(speaker_latents.float())   # [B, COND_DIM]
        tag  = self.tag_emb(tag_ids)                       # [B, TAG_DIM]
        cond = self.fusion(torch.cat([spk, tag], dim=-1))  # [B, COND_DIM]
        return cond.unsqueeze(1)                           # [B, 1, COND_DIM]

    def forward(
        self,
        speaker_latents: torch.Tensor,  # [B, N, 512]
        tag_ids:         torch.Tensor,  # [B]
        target_T:        int | None = None,  # frame count for training (use GT)
    ) -> dict:
        """
        Training: pass target_T to use ground-truth duration.
        Inference: leave target_T=None to use predicted duration.

        Returns dict with keys:
          - pred_latents:    [B, T, 512]  — predicted Mimi latents
          - pred_duration:   [B, 1]       — predicted frame count (float)
        """
        B = speaker_latents.shape[0]
        memory = self.encode_conditioning(speaker_latents, tag_ids)  # [B, 1, COND_DIM]

        # Duration prediction
        pred_dur = self.duration_pred(memory.squeeze(1))   # [B, 1]

        # Decide T
        if target_T is not None:
            T = target_T
        else:
            T = max(1, min(int(pred_dur.mean().round().item()), self.max_frames))

        # Build positional query vectors [B, T, COND_DIM]
        pos_ids = torch.arange(T, device=speaker_latents.device)  # [T]
        queries = self.pos_emb(pos_ids).unsqueeze(0).expand(B, -1, -1)  # [B, T, COND_DIM]

        # Transformer decode: queries attend to conditioning memory
        decoded = self.transformer(queries, memory)  # [B, T, COND_DIM]

        # Project to Mimi latent space
        pred_latents = self.output_proj(decoded)     # [B, T, 512]

        return {
            "pred_latents":  pred_latents,
            "pred_duration": pred_dur,
        }

    @torch.no_grad()
    def generate(
        self,
        speaker_latents: torch.Tensor,  # [1, N, 512] or [N, 512]
        tag_id: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Inference: generate para latents for a single sample.
        Returns [T, 512] Mimi latents.
        """
        self.eval()
        if speaker_latents.dim() == 2:
            speaker_latents = speaker_latents.unsqueeze(0)  # [1, N, 512]
        speaker_latents = speaker_latents.to(device)
        tag_tensor = torch.tensor([tag_id], device=device)

        out = self.forward(speaker_latents, tag_tensor, target_T=None)
        return out["pred_latents"].squeeze(0)  # [T, 512]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ParaModule()
    print(f"ParaModule params: {count_params(model):,}")

    # Smoke test
    B = 4
    speaker = torch.randn(B, 30, 512)
    tags    = torch.randint(0, NUM_TAGS, (B,))
    out     = model(speaker, tags, target_T=20)
    print(f"pred_latents:  {out['pred_latents'].shape}")   # [4, 20, 512]
    print(f"pred_duration: {out['pred_duration'].shape}")  # [4, 1]
    print("Smoke test passed.")
