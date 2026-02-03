import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from internal.configs.instantiate_config import InstantiatableConfig
from internal.optimizers import Adam
from internal.schedulers import ExponentialDecayScheduler


@dataclass
class OptimizationConfig:
    embedding_lr_init: float = 1e-3
    embedding_lr_final_factor: float = 0.1
    lr_init: float = 1e-3
    lr_final_factor: float = 0.1
    eps: float = 1e-8
    max_steps: Optional[int] = None
    warm_up: int = 4000


@dataclass
class DecoupledAppearanceModelConfig(InstantiatableConfig):
    enabled: bool = False
    num_output_channels: int = 3
    n_appearances: int = -1
    n_appearance_embedding_dims: int = 32
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "DecoupledAppearanceModel":
        return DecoupledAppearanceModel(self)


# modified from `decouple_appearance()`
class DecoupledAppearanceModel(nn.Module):
    def __init__(self, config: DecoupledAppearanceModelConfig):
        super().__init__()
        self.config = config

    def setup(self, stage, module=None):
        if stage == "fit":
            if module is not None:
                if self.config.n_appearances <= 0:
                    max_input_id = 0
                    appearance_group_ids = module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                    if appearance_group_ids is not None:
                        for i in appearance_group_ids.values():
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    n_appearances = max_input_id + 1
                    self.config.n_appearances = n_appearances

            if self.config.n_appearances > 0:
                self.embedding = nn.Embedding(self.config.n_appearances, self.config.n_appearance_embedding_dims)
                self.appearance_network = AppearanceNetwork(self.config.n_appearance_embedding_dims + 3, self.config.num_output_channels)
                self.embedding.weight.data.normal_(0, 1e-4)

    def forward(self, rendered: torch.Tensor, appearance_id: torch.Tensor):
        appearance_embedding = self.embedding(appearance_id)
        h, w = list(rendered.shape[-2:])
        crop_image_down = torch.nn.functional.interpolate(rendered, size=(h // 32, w // 32), mode="bilinear", align_corners=True)
        net_input = torch.cat([crop_image_down, appearance_embedding[..., None, None].repeat(1, 1, h // 32, w // 32)], dim=1)

        return self.appearance_network(net_input, h, w) * rendered

    def training_setup(self, module: lightning.LightningModule):
        if self.config.optimization.max_steps is None:
            self.config.optimization.max_steps = module.trainer.max_steps

        embedding_optimizer, embedding_scheduler = self._create_optimizer_and_scheduler(
            params=self.embedding.parameters(),
            name="appearance_embedding",
            lr_init=self.config.optimization.embedding_lr_init,
            lr_final_factor=self.config.optimization.embedding_lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )
        network_optimizer, network_scheduler = self._create_optimizer_and_scheduler(
            params=self.appearance_network.parameters(),
            name="appearance_network",
            lr_init=self.config.optimization.lr_init,
            lr_final_factor=self.config.optimization.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )

        return [embedding_optimizer, network_optimizer], [embedding_scheduler, network_scheduler]

    @staticmethod
    def _create_optimizer_and_scheduler(params, name, lr_init, lr_final_factor, max_steps, eps, warm_up):
        optimizer = torch.optim.Adam(
            params=[{"params": list(params), "name": name}],
            lr=lr_init,
            eps=eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: lr_final_factor ** min(max(iter - warm_up, 0) / max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler


# https://github.com/autonomousvision/gaussian-opacity-fields
def decouple_appearance(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = image.size(1), image.size(2)
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()
    transformed_image = mapping_image * image

    return transformed_image, mapping_image


class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()

        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)

        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    H, W = 1200 // 32, 1600 // 32
    input_channels = 3 + 64
    output_channels = 3
    input = torch.randn(1, input_channels, H, W).cuda()
    model = AppearanceNetwork(input_channels, output_channels).cuda()

    output = model(input)
    print(output.shape)
