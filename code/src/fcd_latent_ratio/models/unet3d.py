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


class CompactRatioInteractionLearningModule(nn.Module):
    def __init__(self, in_channels: int = 2, hidden_channels: int = 16, latent_channels: int = 4) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("CRIL expects exactly 2 input channels: T1w and FLAIR.")

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
        ratio_features = torch.cat(
            [
                t1 / (torch.abs(flair) + 1e-6),
                flair / (torch.abs(t1) + 1e-6),
            ],
            dim=1,
        )
        local_features = self.local_branch(x)
        cross_modal_features = self.cross_modal_branch(x)
        ratio_like_features = self.ratio_branch(ratio_features)
        return self.fuse(torch.cat([local_features, cross_modal_features, ratio_like_features], dim=1))


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


class CRILUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int, int] | list[int] = (32, 64, 128, 256, 512),
        cril_hidden_channels: int = 16,
        cril_latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.cril = CompactRatioInteractionLearningModule(
            in_channels=in_channels,
            hidden_channels=cril_hidden_channels,
            latent_channels=cril_latent_channels,
        )
        self.unet = UNet3D(
            in_channels=cril_latent_channels,
            out_channels=out_channels,
            encoder_channels=encoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.cril(x)
        return self.unet(latent)
