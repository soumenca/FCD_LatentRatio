from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act2(x + residual)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock3D(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        return features, self.downsample(features)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualEncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualConvBlock3D(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.block(x)
        return features, self.downsample(features)


class ResidualDecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class CompactRatioInteractionLearningModule(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 16,
        latent_channels: int = 4,
        ratio_eps: float = 1e-3,
        ratio_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("CRIL expects exactly 2 input channels: T1w and FLAIR.")
        self.ratio_eps = float(ratio_eps)
        self.ratio_clip = float(ratio_clip)

        self.local_branch = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.cross_modal_branch = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.ratio_branch = nn.Sequential(
            nn.Conv3d(2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(hidden_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden_channels, latent_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(latent_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t1 = x[:, 0:1]
        flair = x[:, 1:2]
        safe_flair = torch.where(torch.abs(flair) < self.ratio_eps, torch.full_like(flair, self.ratio_eps), flair)
        safe_t1 = torch.where(torch.abs(t1) < self.ratio_eps, torch.full_like(t1, self.ratio_eps), t1)
        ratio_features = torch.cat(
            [
                torch.clamp(t1 / safe_flair, min=-self.ratio_clip, max=self.ratio_clip),
                torch.clamp(flair / safe_t1, min=-self.ratio_clip, max=self.ratio_clip),
            ],
            dim=1,
        )
        local_features = self.local_branch(x)
        cross_modal_features = self.cross_modal_branch(x)
        ratio_like_features = self.ratio_branch(ratio_features)
        return self.fuse(torch.cat([local_features, cross_modal_features, ratio_like_features], dim=1))


class LightweightBottleneckAttention3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 16,
        num_heads: int = 4,
        pooled_size: tuple[int, int, int] | list[int] = (6, 6, 6),
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        pooled = tuple(int(value) for value in pooled_size)
        hidden_dim = max(embed_dim, int(round(embed_dim * mlp_ratio)))
        self.project_in = nn.Conv3d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool3d(pooled)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.project_out = nn.Sequential(
            nn.Conv3d(embed_dim, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        features = self.project_in(x)
        pooled = self.pool(features)
        batch_size, channels, depth, height, width = pooled.shape
        tokens = pooled.flatten(2).transpose(1, 2)
        qkv = self.norm1(tokens)
        attended_tokens, _ = self.attention(qkv, qkv, qkv)
        tokens = tokens + self.dropout(attended_tokens)
        tokens = tokens + self.dropout(self.mlp(self.norm2(tokens)))
        pooled = tokens.transpose(1, 2).reshape(batch_size, channels, depth, height, width)
        pooled = nn.functional.interpolate(pooled, size=x.shape[2:], mode="trilinear", align_corners=False)
        update = self.project_out(pooled)
        return residual + update


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 5:
            raise ValueError("This baseline expects 5 encoder stages for the U-Net-E5 setup.")

        channels = tuple(int(c) for c in encoder_channels)
        self.enc1 = EncoderStage(in_channels, channels[0])
        self.enc2 = EncoderStage(channels[0], channels[1])
        self.enc3 = EncoderStage(channels[1], channels[2])
        self.enc4 = EncoderStage(channels[2], channels[3])
        self.bottleneck = ConvBlock3D(channels[3], channels[4])
        self.dec4 = DecoderStage(channels[4], channels[3], channels[3])
        self.dec3 = DecoderStage(channels[3], channels[2], channels[2])
        self.dec2 = DecoderStage(channels[2], channels[1], channels[1])
        self.dec1 = DecoderStage(channels[1], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 5:
            raise ValueError("This residual baseline expects 5 encoder stages for the U-Net-E5 setup.")

        channels = tuple(int(c) for c in encoder_channels)
        self.enc1 = ResidualEncoderStage(in_channels, channels[0])
        self.enc2 = ResidualEncoderStage(channels[0], channels[1])
        self.enc3 = ResidualEncoderStage(channels[1], channels[2])
        self.enc4 = ResidualEncoderStage(channels[2], channels[3])
        self.bottleneck = ResidualConvBlock3D(channels[3], channels[4])
        self.dec4 = ResidualDecoderStage(channels[4], channels[3], channels[3])
        self.dec3 = ResidualDecoderStage(channels[3], channels[2], channels[2])
        self.dec2 = ResidualDecoderStage(channels[2], channels[1], channels[1])
        self.dec1 = ResidualDecoderStage(channels[1], channels[0], channels[0])
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.head(x)


class SegResNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        init_filters: int = 16,
        stage_blocks: tuple[int, int, int, int] | list[int] = (1, 2, 2, 4),
        stage_blocks_up: tuple[int, int, int] | list[int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        blocks_down = tuple(int(b) for b in stage_blocks)
        blocks_up = tuple(int(b) for b in stage_blocks_up)
        if len(blocks_down) != 4:
            raise ValueError("SegResNet3D expects 4 encoder stages in stage_blocks.")
        if len(blocks_up) != 3:
            raise ValueError("SegResNet3D expects 3 decoder stages in stage_blocks_up.")

        try:
            from monai.networks.nets import SegResNet as MonaiSegResNet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MONAI is required for SegResNet experiments. Install it with `pip install monai` "
                "or rerun `bash code/scripts/setup_env.sh` after updating requirements."
            ) from exc

        self.model = MonaiSegResNet(
            spatial_dims=3,
            init_filters=int(init_filters),
            in_channels=in_channels,
            out_channels=out_channels,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CRILUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
        cril_hidden_channels: int = 16,
        cril_latent_channels: int = 4,
        ratio_eps: float = 1e-3,
        ratio_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.cril = CompactRatioInteractionLearningModule(
            in_channels=in_channels,
            hidden_channels=cril_hidden_channels,
            latent_channels=cril_latent_channels,
            ratio_eps=ratio_eps,
            ratio_clip=ratio_clip,
        )
        self.unet = UNet3D(
            in_channels=cril_latent_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.cril(x)
        return self.unet(latent)


class CRILResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
        cril_hidden_channels: int = 16,
        cril_latent_channels: int = 4,
        ratio_eps: float = 1e-3,
        ratio_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.cril = CompactRatioInteractionLearningModule(
            in_channels=in_channels,
            hidden_channels=cril_hidden_channels,
            latent_channels=cril_latent_channels,
            ratio_eps=ratio_eps,
            ratio_clip=ratio_clip,
        )
        self.unet = ResidualUNet3D(
            in_channels=cril_latent_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.cril(x)
        return self.unet(latent)


class CRILSegResNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        init_filters: int = 16,
        stage_blocks: tuple[int, int, int, int] | list[int] = (1, 2, 2, 4),
        stage_blocks_up: tuple[int, int, int] | list[int] = (1, 1, 1),
        cril_hidden_channels: int = 16,
        cril_latent_channels: int = 4,
        ratio_eps: float = 1e-3,
        ratio_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.cril = CompactRatioInteractionLearningModule(
            in_channels=in_channels,
            hidden_channels=cril_hidden_channels,
            latent_channels=cril_latent_channels,
            ratio_eps=ratio_eps,
            ratio_clip=ratio_clip,
        )
        self.segresnet = SegResNet3D(
            in_channels=cril_latent_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            stage_blocks=stage_blocks,
            stage_blocks_up=stage_blocks_up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.cril(x)
        return self.segresnet(latent)


class AttentionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
        attention_embed_dim: int = 16,
        attention_num_heads: int = 4,
        attention_pooled_size: tuple[int, int, int] | list[int] = (6, 6, 6),
        attention_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.attention = LightweightBottleneckAttention3D(
            in_channels=in_channels,
            embed_dim=attention_embed_dim,
            num_heads=attention_num_heads,
            pooled_size=attention_pooled_size,
            mlp_ratio=attention_mlp_ratio,
        )
        self.unet = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        return self.unet(attended)


class CRILAttnUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
        cril_hidden_channels: int = 16,
        cril_latent_channels: int = 4,
        ratio_eps: float = 1e-3,
        ratio_clip: float = 10.0,
        attention_embed_dim: int = 16,
        attention_num_heads: int = 4,
        attention_pooled_size: tuple[int, int, int] | list[int] = (6, 6, 6),
        attention_mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.cril = CompactRatioInteractionLearningModule(
            in_channels=in_channels,
            hidden_channels=cril_hidden_channels,
            latent_channels=cril_latent_channels,
            ratio_eps=ratio_eps,
            ratio_clip=ratio_clip,
        )
        self.attention = LightweightBottleneckAttention3D(
            in_channels=cril_latent_channels,
            embed_dim=attention_embed_dim,
            num_heads=attention_num_heads,
            pooled_size=attention_pooled_size,
            mlp_ratio=attention_mlp_ratio,
        )
        self.unet = UNet3D(
            in_channels=cril_latent_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.cril(x)
        attended_latent = self.attention(latent)
        return self.unet(attended_latent)
