"""
Inflect — Paralinguistic Module (Para Module) v2

Generates 80-dim log-mel spectrograms for paralinguistic sounds.
Given a speaker embedding and a tag (e.g. [laughs]), outputs a mel
spectrogram that a vocoder (HiFiGAN/BigVGAN) converts to audio.

Architecture (non-autoregressive):
  1. Speaker encoder:  [80] mean-mel → Linear → [256]
  2. Tag embedding:    tag_id → Embedding → [128]
  3. Fusion:           concat [384] → Linear → [256] = conditioning
  4. Duration pred:    conditioning → MLP → predicted T (frame count)
  5. Decoder:          T positional queries cross-attend to conditioning
  6. Output proj:      [T, 256] → [T, 80] = log-mel spectrogram

Total params: ~4.6M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from inflect.data.tags import NUM_TAGS

MEL_DIM      = 80
COND_DIM     = 256
TAG_DIM      = 128
SPEAKER_DIM  = 80     # mean-pooled mel embedding
MAX_FRAMES   = 250    # ~2.67s at 93.75 Hz


class SpeakerEncoder(nn.Module):
    def __init__(self, in_dim: int = SPEAKER_DIM, out_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        """speaker_emb: [B, 80] → [B, COND_DIM]"""
        return self.net(speaker_emb.float())


class DurationPredictor(nn.Module):
    def __init__(self, cond_dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),   # duration must be positive
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, COND_DIM] → [B, 1] (predicted frame count, positive)"""
        return self.net(cond)


class ParaModule(nn.Module):
    def __init__(
        self,
        num_tags:    int = NUM_TAGS,
        mel_dim:     int = MEL_DIM,
        cond_dim:    int = COND_DIM,
        tag_dim:     int = TAG_DIM,
        speaker_dim: int = SPEAKER_DIM,
        n_heads:     int = 4,
        n_layers:    int = 4,
        max_frames:  int = MAX_FRAMES,
    ):
        super().__init__()
        self.cond_dim   = cond_dim
        self.max_frames = max_frames
        self.mel_dim    = mel_dim

        self.speaker_enc = SpeakerEncoder(speaker_dim, cond_dim)
        self.tag_emb     = nn.Embedding(num_tags, tag_dim)
        self.fusion      = nn.Sequential(
            nn.Linear(cond_dim + tag_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.GELU(),
        )
        self.duration_pred = DurationPredictor(cond_dim)
        self.pos_emb       = nn.Embedding(max_frames, cond_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cond_dim,
            nhead=n_heads,
            dim_feedforward=cond_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(cond_dim, mel_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_conditioning(self, speaker_emb, tag_ids):
        """Returns conditioning [B, 1, COND_DIM]."""
        spk  = self.speaker_enc(speaker_emb)
        tag  = self.tag_emb(tag_ids)
        cond = self.fusion(torch.cat([spk, tag], dim=-1))
        return cond.unsqueeze(1)

    def forward(self, speaker_emb, tag_ids, target_T=None):
        """
        speaker_emb: [B, 80]
        tag_ids:     [B]
        target_T:    int | None (use GT length during training)

        Returns:
          pred_mel:      [B, T, 80]
          pred_duration: [B, 1]
        """
        B      = speaker_emb.shape[0]
        memory = self.encode_conditioning(speaker_emb, tag_ids)  # [B, 1, COND_DIM]
        pred_dur = self.duration_pred(memory.squeeze(1))          # [B, 1]

        T = target_T if target_T is not None else \
            max(1, min(int(pred_dur.mean().round().item()), self.max_frames))

        pos_ids = torch.arange(T, device=speaker_emb.device)
        queries = self.pos_emb(pos_ids).unsqueeze(0).expand(B, -1, -1)  # [B, T, COND_DIM]

        decoded  = self.transformer(queries, memory)  # [B, T, COND_DIM]
        pred_mel = self.output_proj(decoded)           # [B, T, 80]

        return {"pred_mel": pred_mel, "pred_duration": pred_dur}

    @torch.no_grad()
    def generate(self, speaker_emb, tag_id, device="cpu"):
        """
        Inference: generate mel for a single sample.
        speaker_emb: [80] or [1, 80]
        Returns: [T, 80] log-mel spectrogram.
        """
        self.eval()
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        speaker_emb = speaker_emb.to(device)
        tag_tensor  = torch.tensor([tag_id], device=device)
        out = self.forward(speaker_emb, tag_tensor, target_T=None)
        return out["pred_mel"].squeeze(0)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ParaModule()
    print(f"ParaModule params: {count_params(model):,}")
    B = 4
    spk  = torch.randn(B, 80)
    tags = torch.randint(0, NUM_TAGS, (B,))
    out  = model(spk, tags, target_T=100)
    print(f"pred_mel:      {out['pred_mel'].shape}")
    print(f"pred_duration: {out['pred_duration'].shape}")
    print("Smoke test passed.")
