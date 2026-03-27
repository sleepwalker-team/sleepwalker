import torch
import torch.nn as nn

from einops import rearrange

from sleepwalker.models.Basemodel import BaseModel
from sleepwalker.models.preprocessors.WindowedSpectrogram import WindowedSpectrogram
from sleepwalker.models.preprocessors.Normalize import Normalize
from sleepwalker.models.utils import SinusoidalPositionalEncoding

class MaskedAutoencoder(BaseModel):

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
        dec_heads=4,
        dec_depth=4,
        dec_dim=128,
        dec_dropout=0.1,
        dec_mlp_ratio=4,
        mask_fraction=0.5,
        use_cls=True,
    ):
        spec = [
            WindowedSpectrogram(hop_length=window_step_size, win_length=window_size, token_length=token_size),
            Normalize()
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
        self.dec_dim = dec_dim
        self.dec_dropout = dec_dropout
        self.dec_mlp_ratio = dec_mlp_ratio

        self.use_cls = use_cls

        # Project inputs into unified transformer embedding space (and project back, later)
        F = self.window_size//2 + 1
        self.input_projection = nn.Linear(F, self.enc_dim)
        self.output_projection = nn.Linear(self.dec_dim, F)

        # Define positional encodings
        self.encoder_positional_encoding = SinusoidalPositionalEncoding(dim=self.enc_dim)
        self.decoder_positional_encoding = SinusoidalPositionalEncoding(dim=self.dec_dim)
        
        # Define transformer encoder and decoder pipelines
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.enc_dim, 
            nhead=self.enc_heads,
            dim_feedforward=int(self.enc_dim * self.enc_mlp_ratio),
            dropout=self.enc_dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.enc_depth, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(enc_dim)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=self.dec_dim, 
            nhead=self.dec_heads,
            dim_feedforward=int(self.dec_dim * self.dec_mlp_ratio),
            dropout=self.dec_dropout, 
            batch_first=True, 
            norm_first=True 
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=self.dec_depth, enable_nested_tensor=False)
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.enc_to_dec_projection = nn.Linear(self.enc_dim, self.dec_dim)

        # Define tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1,1,self.enc_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    # TODO: At the moment, we concatenate the channel dimension to the batch dimension, meaning we cut out different timesteps per channel
    def _random_mask(self, B, N, device):
        rand_indices = torch.stack([torch.randperm(N) for _ in range(B)], 0).to(device)
        to_keep = int((1-self.mask_fraction)*N)
        ids_keep = rand_indices[:, :to_keep]
        ids_discard = rand_indices[:, to_keep:]

        return ids_keep, ids_discard

    def feature_dim(self):
        # TODO: Correct?
        return self.window_size//2 + 1

    def _classifier(self, x):
        return super()._classifier(x)

    def embed(self, x):
        x = self.apply_preprocessors(x, len(self.preprocessors)+1)
        return self._features(x)

    def _features(self, x):
        # Project into transformer space
        B, _, N, D = x.shape
        x = rearrange(x, 'B F N D -> (B D) N F')
        x = self.input_projection(x)
        x = x + self.encoder_positional_encoding.pe[1:N+1].unsqueeze(0)

        if self.use_cls:
            pe = self.encoder_positional_encoding.pe[0]
            cls = pe + self.cls_token
            cls = cls.expand(B*D, -1, -1)
            x = torch.cat((cls, x), dim=1)

        z = self.encoder(x)
        z = self.encoder_norm(z)
        z = rearrange(z, '(B D) N F -> B N F D', D=D, B=B)
        return z

    def forward(self, x):
        x = self.apply_preprocessors(x, len(self.preprocessors)+1)
        return self._forward(x)

    def _forward(self, x):
        # Project into transformer space
        B, _, N, D = x.shape
        x = rearrange(x, 'B F N D -> (B D) N F')
        x = self.input_projection(x)
        x = x + self.encoder_positional_encoding.pe[1:N+1].unsqueeze(0)

        # Apply masking
        ids_keep, ids_discard = self._random_mask(B*D, N, x.device)
        _B = torch.arange(B*D)[:, None] # Indexing helper
        x = x[_B, ids_keep]

        # Create mask (for later)
        mask = torch.zeros((B*D, N)).to(x.device)
        mask[_B, ids_discard] = 1

        if self.use_cls:
            pe = self.encoder_positional_encoding.pe[0]
            cls = pe + self.cls_token
            cls = cls.expand(B*D, -1, -1)
            x = torch.cat((cls, x), dim=1)

        z = self.encoder(x)
        z = self.encoder_norm(z)
        dec_keep = self.enc_to_dec_projection(z)

        # Create full tokens again, filling dropped-out tokens with masking token
        offset = 1 if self.use_cls else 0
        dec_tokens = torch.zeros(B*D, N+offset, self.dec_dim).to(z.device)
        dec_tokens[_B, ids_keep+offset] = dec_keep[:, 1:] if self.use_cls else dec_keep
        dec_tokens[_B, ids_discard+offset] = self.mask_token.expand(B*D, ids_discard.shape[-1], -1)
        if self.use_cls:
            dec_tokens[:, 0] = dec_keep[:, 0]

        dec_tokens = self.decoder_positional_encoding(dec_tokens)
        dec_tokens = self.decoder(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)

        # This projection can be channel-specific. For now, we have just one
        dec_tokens = rearrange(dec_tokens, '(B D) N dec_dim -> B N D dec_dim', B=B, D=D)
        if self.use_cls:
            dec_tokens = dec_tokens[:, 1:]
        patches = self.output_projection(dec_tokens)
        patches = rearrange(patches, 'B N D F -> B F N D')
        mask = rearrange(mask, '(B D) N -> B 1 N D', B=B, D=D)

        return patches, mask
