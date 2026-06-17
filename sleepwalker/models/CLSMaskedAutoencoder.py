import torch
import torch.nn as nn

from einops import rearrange

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.WindowedSpectrogram import WindowedSpectrogram
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.utils import SinusoidalPositionalEncoding


class CLSMaskedAutoencoder(BaseModel):
    """
    CLS-Bottleneck Masked Autoencoder (CLS-BMAE).

    Differs from MaskedAutoencoder in the forward pass:
    - The encoder sees ALL tokens (no masking before encoding).
    - Only the CLS token is passed from encoder to decoder (strict bottleneck).
    - Decoder input for each target position i is: cls_z + PE_dec(i).
      No mask tokens are used; the decoder must unpack all positional detail from z.
    - The reconstruction target is a random subset of positions (controlled by
      mask_fraction), rather than all masked-out positions.
    """

    def __init__(
        self,
        *,
        token_size=300,
        window_size=100,
        window_step_size=50,
        enc_heads=8,
        enc_depth=8,
        enc_dim=256,
        enc_dropout=0.1,
        enc_mlp_ratio=4,
        dec_heads=8,
        dec_depth=4,
        dec_dropout=0.1,
        dec_mlp_ratio=4,
        mask_fraction=0.75,
        use_cls=True,  # Always True in CLS-BMAE; kept for API compatibility
    ):
        spec = [
            WindowedSpectrogram(hop_length=window_step_size, win_length=window_size, token_length=token_size),
            Normalize(stat_dims=(1, 3)),
        ]
        super().__init__(preprocessors=spec)

        self.token_size = token_size
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.mask_fraction = mask_fraction

        self.enc_heads = enc_heads
        self.enc_depth = enc_depth
        self.enc_dim = enc_dim
        self.enc_dropout = enc_dropout
        self.enc_mlp_ratio = enc_mlp_ratio
        self.dec_heads = dec_heads
        self.dec_depth = dec_depth
        self.dec_dropout = dec_dropout
        self.dec_mlp_ratio = dec_mlp_ratio

        self.use_cls = True  # structural requirement, not a toggle

        F = self.window_size // 2 + 1
        self.input_projection = nn.Linear(F, self.enc_dim)
        self.output_projection = nn.Linear(self.enc_dim, F)

        # Single shared PE: encoder and decoder operate in the same enc_dim space.
        self.positional_encoding = SinusoidalPositionalEncoding(dim=self.enc_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim,
            nhead=self.enc_heads,
            dim_feedforward=int(self.enc_dim * self.enc_mlp_ratio),
            dropout=self.enc_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.enc_depth, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(enc_dim)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim,
            nhead=self.dec_heads,
            dim_feedforward=int(self.enc_dim * self.dec_mlp_ratio),
            dropout=self.dec_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=self.dec_depth, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(self.enc_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.enc_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def get_hyperparameters(self):
        return {
            'token_size': self.token_size,
            'window_size': self.window_size,
            'window_step_size': self.window_step_size,
            'mask_fraction': self.mask_fraction,
            'enc_heads': self.enc_heads,
            'enc_depth': self.enc_depth,
            'enc_dim': self.enc_dim,
            'enc_dropout': self.enc_dropout,
            'enc_mlp_ratio': self.enc_mlp_ratio,
            'dec_heads': self.dec_heads,
            'dec_depth': self.dec_depth,
            'dec_dropout': self.dec_dropout,
            'dec_mlp_ratio': self.dec_mlp_ratio,
            'use_cls': self.use_cls,
        }

    def _random_mask(self, B, N, device):
        rand_indices = torch.stack([torch.randperm(N) for _ in range(B)], 0).to(device)
        n_reconstruct = int((1 - self.mask_fraction) * N)
        ids_reconstruct = rand_indices[:, :n_reconstruct]
        ids_skip = rand_indices[:, n_reconstruct:]
        return ids_reconstruct, ids_skip

    def feature_dim(self):
        return self.window_size // 2 + 1

    def _classifier(self, x):
        return super()._classifier(x)

    def _features(self, x):
        """Full unmasked encoder pass. Returns (B, N+1, enc_dim, D) — same shape as MaskedAutoencoder.embed()."""
        B, _, N, D = x.shape
        x = rearrange(x, 'B F N D -> (B D) N F')
        x = self.input_projection(x)
        x = x + self.positional_encoding.pe[1:N + 1].unsqueeze(0)

        pe = self.positional_encoding.pe[0]
        cls = (pe + self.cls_token).expand(B * D, -1, -1)
        x = torch.cat((cls, x), dim=1)

        z = self.encoder(x)
        z = self.encoder_norm(z)
        z = rearrange(z, '(B D) N F -> B N F D', D=D, B=B)
        return z

    def embed(self, x):
        x = self.apply_preprocessors(x, len(self.preprocessors) + 1)
        return self._features(x)

    def forward(self, x):
        x = self.apply_preprocessors(x, len(self.preprocessors) + 1)
        return self._forward(x)

    def _forward(self, x):
        B, _, N, D = x.shape
        x_enc = rearrange(x, 'B F N D -> (B D) N F')
        x_enc = self.input_projection(x_enc)
        x_enc = x_enc + self.positional_encoding.pe[1:N + 1].unsqueeze(0)

        # Encode full sequence with CLS prepended
        pe = self.positional_encoding.pe[0]
        cls = (pe + self.cls_token).expand(B * D, -1, -1)
        x_enc = torch.cat((cls, x_enc), dim=1)  # (B*D, N+1, enc_dim)

        z = self.encoder(x_enc)
        z = self.encoder_norm(z)

        # Bottleneck: discard all patch tokens, keep only CLS
        cls_z = z[:, 0:1]  # (B*D, 1, enc_dim)

        # Sample positions to reconstruct — one shared mask per batch item across all channels
        ids_reconstruct, _ = self._random_mask(B, N, x.device)  # (B, n_reconstruct)
        ids_reconstruct_bd = ids_reconstruct.repeat_interleave(D, dim=0)  # (B*D, n_reconstruct)
        n_reconstruct = ids_reconstruct.shape[1]

        # Decoder input: broadcast CLS + positional encoding of each target position
        # PE offset by 1 to match encoder convention (CLS=0, tokens=1..N)
        pe_dec = self.positional_encoding.pe[ids_reconstruct + 1]       # (B, n_reconstruct, enc_dim)
        pe_dec = pe_dec.repeat_interleave(D, dim=0)                     # (B*D, n_reconstruct, enc_dim)
        q = cls_z.expand(B * D, n_reconstruct, self.enc_dim) + pe_dec  # (B*D, n_reconstruct, enc_dim)

        o = self.decoder(q)
        o = self.decoder_norm(o)
        o = self.output_projection(o)  # (B*D, n_reconstruct, F)

        # Scatter predictions back into a full (B, F, N, D) tensor
        F_dim = self.window_size // 2 + 1
        patches = torch.zeros(B * D, N, F_dim, device=z.device)
        _BD = torch.arange(B * D, device=z.device)[:, None]
        patches[_BD, ids_reconstruct_bd] = o
        patches = rearrange(patches, '(B D) N F -> B F N D', B=B, D=D)

        # mask=1 at reconstructed positions (these contribute to the loss)
        _Bb = torch.arange(B, device=z.device)[:, None]
        mask = torch.zeros(B, N, device=z.device)
        mask[_Bb, ids_reconstruct] = 1
        mask = mask.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, D)

        return patches, mask
