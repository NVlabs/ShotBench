# MIT License
#
# Copyright (c) 2023 Wentao Zhu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AutoShot pytorch model.

Code adapted from https://github.com/wentaozhu/AutoShot/commit/77c82ff826a9301bb173d9be786297a49d73d081.

@inproceedings{zhuautoshot,
  title={AutoShot: A Short Video Dataset and State-of-the-Art Shot Boundary Detection},
  author={Zhu, W. and Huang, Y. and Xie, X. and Liu, W. and Deng, J. and Zhang, D. and Wang, Z. and Liu, J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2023}
}
"""

import argparse
import math
from pathlib import Path
from typing import Callable, Final, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import init


class Linear_(nn.Module):  # noqa: N801
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, act: str = "ReLU", is_folded: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.act_type = act
        self.is_folded = is_folded
        self.linear = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=self.bias)
        self.act = _act(self.act_type)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result_linear = self.linear(inputs)
        result = self.act(result_linear)
        return result

    @property
    def multiply_adds(self) -> int:
        result = self.in_features * self.out_features
        return result

    @property
    def params(self) -> int:
        params = self.in_features * self.out_features
        if self.bias is True and self.is_folded is False:
            params += self.out_features
        return params


class Identity_(nn.Module):  # noqa: N801
    """Skip connection. Maintained from reference implementation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def _act(act_type: Optional[str]) -> Union[Identity_, nn.ReLU]:
    """Activation function factory. Maintained from reference implementation."""
    if act_type is None or act_type == "Identity":
        return Identity_()
    elif act_type == "ReLU":
        result = nn.ReLU(inplace=True)
        return result
    else:
        raise NotImplementedError("Not implemented.")


class AutoShot(nn.Module):
    def __init__(
        self,
        D: int = 1024,
        use_many_hot_targets: bool = True,
        use_frame_similarity: bool = True,
        use_mean_pooling: bool = False,
        dropout_rate: float = 0.5,
        frame_similarity_on_last_layer: bool = False,
        use_color_histograms: bool = True,
    ) -> None:
        super().__init__()
        self.reprocess_layer = lambda x: x / 255.0
        self.Layer_0_3 = DilatedDCNNV2(3, 16, multiplier=1)
        self.Layer_1_8 = DilatedDCNNV2ABC(16 * 4, 16, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_2_8 = DilatedDCNNV2ABC(16 * 4, 32, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_3_8 = DilatedDCNNV2ABC(32 * 4, 32, multiplier=4, n_dilation=5, st_type="A")
        self.Layer_4_13 = DilatedDCNNV2(32 * 4, 64, multiplier=3, n_dilation=5)
        self.Layer_5_12 = DilatedDCNNV2(64 * 4, 64, multiplier=2, n_dilation=5)
        self.Layer_6_0 = Attention1D(
            dim_in=256 * 3 * 6,
            dim_out=1024,
            num_heads=4,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
            with_cls_token=False,
            n_layer=0,
        )

        self.pool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2))

        if use_frame_similarity is True and use_color_histograms is True:
            in_features = 4864
        elif use_frame_similarity is True and use_color_histograms is False:
            in_features = 4736
        elif use_frame_similarity is False and use_color_histograms is True:
            in_features = 4736
        else:
            in_features = 4608

        self.fc1_0 = Linear_(in_features=in_features, out_features=D, bias=True, act="ReLU")
        in_features += 1024
        self.fc1 = Linear_(in_features=in_features, out_features=D, bias=True, act="ReLU")
        self.cls_layer1 = Linear_(in_features=1024, out_features=1, bias=True, act="Identity")
        self.cls_layer2 = (
            Linear_(in_features=1024, out_features=1, bias=True, act="Identity") if use_many_hot_targets else None
        )
        self.frame_sim_layer = FrameSimilarity(in_channels=448, inner_channels=101) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128) if use_color_histograms else None
        self.use_mean_pooling = use_mean_pooling
        self.dropout = torch.nn.Dropout(p=1.0 - dropout_rate) if dropout_rate is not None else None
        self.frame_similarity_on_last_layer = frame_similarity_on_last_layer
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = m.weight.size(1)
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                m.weight.data.uniform_(-limit, limit)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs forward pass of AutoShot network.

        Args:
            inputs: tensor of shape [# batch, # frames, height, width, RGB]

        Returns:
            outputs tensor of shape [# batch, # frames, 1] of probabilities each frame is a shot transition.
        """
        inputs = inputs.permute((0, 4, 1, 2, 3))
        x = self.reprocess_layer(inputs)

        block_features = []
        shortcut = None
        num_layers = 6
        for layer_index in range(num_layers):
            if layer_index == 0:
                op = self.Layer_0_3
            elif layer_index == 1:
                op = self.Layer_1_8
            elif layer_index == 2:
                op = self.Layer_2_8
            elif layer_index == 3:
                op = self.Layer_3_8
            elif layer_index == 4:
                op = self.Layer_4_13
            elif layer_index == 5:
                op = self.Layer_5_12

            x = op(x)
            if layer_index in [0, 2, 4]:
                shortcut = x
            else:
                x = shortcut + x
                x = self.pool(x)
                block_features.append(x)
        transf_x = self.Layer_6_0(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])  # out is [BS, C, N]
        else:
            x = x.permute(0, 2, 3, 4, 1)
            shape = [x.shape[0], x.shape[1], np.prod(x.shape[2:])]
            x = x.reshape(shape=shape)  # out is [BS, C, N * H * W]

        if self.frame_sim_layer is not None and not self.frame_similarity_on_last_layer:
            x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], dim=2)

        if transf_x is not None:
            x = torch.cat([transf_x, x], dim=2)

            x = self.fc1(x)
        else:
            x = self.fc1_0(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.frame_sim_layer is not None and self.frame_similarity_on_last_layer:
            x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)

        one_hot = self.cls_layer1(x)

        # scale from 0 to 1
        one_hot = torch.sigmoid(one_hot)
        return one_hot


def gather_nd(params: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of
    indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = int(np.prod(orig_shape[:-1]))
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f"the last dimension of indices must less or equal to the rank of params. "
            f"Got indices:{indices.shape}, params:{params.shape}. {m} > {n}"
        )

    indices_lst = torch.reshape(indices, (num_samples, m)).transpose(0, 1).tolist()
    output = params[indices_lst]  # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


class FrameSimilarity(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        similarity_dim: int = 128,
        lookup_window: int = 101,
        output_dim: int = 128,
        stop_gradient: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.projection = Linear_(in_features=in_channels, out_features=similarity_dim, bias=use_bias, act="Identity")
        self.fc = Linear_(in_features=inner_channels, out_features=output_dim, bias=True, act="ReLU")

        self.lookup_window = lookup_window
        self.stop_gradient = stop_gradient
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"
        if torch.cuda.is_available() is True:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # pt version [BS, C, N, H, W], so [3, 4] means apply avg on spatial dim, out dim is [BS, C, N]
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)

        if self.stop_gradient:
            x = x.detach()

        x = x.permute(dims=[0, 2, 1])  # out is [BS, N ,C]
        batch_size, time_window, old_channels = x.shape
        x = x.reshape(shape=[batch_size * time_window, old_channels])  # [BS X N, C]
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)  # norm at C dim

        _, new_channels = x.shape
        x = x.reshape(shape=[batch_size, time_window, new_channels])
        y = x.permute(dims=[0, 2, 1])
        similarities = torch.matmul(x, y)  # [batch_size, time_window, time_window]
        # note that it operates on dimensions of the input tensor in a backward fashion
        # from last dimension to the first dimension
        similarities_padded = F.pad(
            similarities,
            pad=[
                (self.lookup_window - 1) // 2,
                (self.lookup_window - 1) // 2,
                0,
                0,
                0,
                0,
            ],
        )

        batch_indices = (
            torch.arange(0, batch_size, device=self.device)
            .reshape(shape=[batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=self.device)
            .reshape(shape=[1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=self.device)
            .reshape(shape=[1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        indices = torch.stack([batch_indices, time_indices, lookup_indices], dim=-1)

        similarities = gather_nd(similarities_padded, indices)
        return self.fc(similarities)


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window: int = 101, output_dim: int = 128) -> None:
        super(ColorHistograms, self).__init__()
        self.fc = (
            Linear_(in_features=101, out_features=output_dim, bias=True, act="ReLU") if output_dim is not None else None
        )
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"
        if torch.cuda.is_available() is True:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def unsorted_segment_sum(self, data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
        """https://gist.github.com/bbrighttaer/207dc03b178bbd0fef8d1c0c1390d4be"""
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long()
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"
        shape = [num_segments] + list(data.shape[1:])
        tensor = torch.zeros(*shape, device=self.device).scatter_add(0, segment_ids, data.float())
        tensor = tensor.type(data.dtype)
        return tensor

    def compute_color_histograms(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.type(dtype=torch.int32)
        # pt version [BS, C, N, H, W]  ---> tf version [BS, N, H, W, C]
        frames = frames.permute(0, 2, 3, 4, 1)

        def get_bin(frames: torch.Tensor) -> torch.Tensor:
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width = frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3]

        no_channels = frames.shape[-1]
        assert no_channels == 3 or no_channels == 6
        if no_channels == 3:
            frames_flatten = frames.reshape(shape=[batch_size * time_window, height * width, 3])
        else:
            frames_flatten = frames.reshape(shape=[batch_size * time_window, height * width * 2, 3])

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = torch.arange(0, batch_size * time_window, dtype=torch.int32, device=self.device) << 9
        frame_bin_prefix = frame_bin_prefix.unsqueeze(dim=-1)
        binned_values = binned_values + frame_bin_prefix

        ones = torch.ones_like(binned_values, dtype=torch.int32, device=self.device)
        histograms = self.unsorted_segment_sum(
            data=ones, segment_ids=binned_values.type(dtype=torch.long), num_segments=batch_size * time_window * 512
        )
        histograms = torch.sum(histograms, dim=1)
        histograms = histograms.reshape(shape=[batch_size, time_window, 512])

        histograms_normalized = histograms.type(dtype=torch.float32)
        histograms_normalized = histograms_normalized / torch.norm(histograms_normalized, dim=2, keepdim=True)
        return histograms_normalized

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        y = x.permute(dims=[0, 2, 1])
        similarities = torch.matmul(x, y)  # [batch_size, time_window, time_window]
        # note that it operates on dimensions of the input tensor in a backward fashion
        # from last dimension to the first dimension
        similarities_padded = F.pad(
            similarities,
            pad=[
                (self.lookup_window - 1) // 2,
                (self.lookup_window - 1) // 2,
                0,
                0,
                0,
                0,
            ],
        )

        batch_indices = (
            torch.arange(0, batch_size, device=self.device)
            .reshape(shape=[batch_size, 1, 1])
            .repeat([1, time_window, self.lookup_window])
        )
        time_indices = (
            torch.arange(0, time_window, device=self.device)
            .reshape(shape=[1, time_window, 1])
            .repeat([batch_size, 1, self.lookup_window])
        )
        lookup_indices = (
            torch.arange(0, self.lookup_window, device=self.device)
            .reshape(shape=[1, 1, self.lookup_window])
            .repeat([batch_size, time_window, 1])
            + time_indices
        )

        indices = torch.stack([batch_indices, time_indices, lookup_indices], dim=-1)

        similarities = gather_nd(similarities_padded, indices)

        if self.fc is not None:
            return self.fc(similarities)
        return similarities


class DilatedDCNNV2ABC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        batch_norm: int = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        octave_conv: bool = False,
        multiplier: int = 4,
        n_dilation: int = 4,
        st_type: str = "A",
    ) -> None:
        super().__init__()
        assert not (octave_conv and batch_norm)
        self.share = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=multiplier * filters,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            dilation=(1, 1, 1),
            bias=False,
        )
        init.kaiming_normal_(self.share.weight, mode="fan_in", nonlinearity="relu")

        self.conv_blocks = nn.ModuleList()

        n_in_plane = multiplier * filters
        if st_type == "B":
            n_in_plane = in_channels

        n_filter_per_module = (filters * 4) // n_dilation  # multiplier
        for dilation in range(n_dilation - 1):
            self.conv_blocks.append(
                Conv3DConfigurable(
                    n_in_plane,
                    n_filter_per_module,
                    2**dilation,
                    mid_filter=n_in_plane,
                    separable=True,
                    sharable=True,
                    use_bias=not batch_norm,
                    octave=octave_conv,
                )
            )
        self.conv_blocks.append(
            Conv3DConfigurable(
                n_in_plane,
                (filters * 4) - n_filter_per_module * (n_dilation - 1),  # multiplier
                2 ** (n_dilation - 1),
                mid_filter=n_in_plane,
                separable=True,
                sharable=True,
                use_bias=not batch_norm,
                octave=octave_conv,
            )
        )

        self.octave = octave_conv
        self.multiplier = multiplier
        self.n_dilation = n_dilation
        self.st_type = st_type

        self.batch_norm = (
            torch.nn.BatchNorm3d(num_features=filters * 4, eps=1e-3, momentum=0.1)  # multiplier,
            if batch_norm
            else None
        )
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feature = self.share(inputs)
        if self.st_type == "A":
            x = []
            for block in self.conv_blocks:
                x.append(block(feature))
            x = torch.cat(x, dim=1)
        elif self.st_type == "B":
            x = []
            for block in self.conv_blocks:
                x.append(block(inputs))
            x = torch.cat(x, dim=1)
            x = x + feature
        elif self.st_type == "C":
            x = []
            for block in self.conv_blocks:
                x.append(block(feature))
            x = torch.cat(x, dim=1)
            x = x + feature
        else:
            raise Exception("Not Implemented ST Type" + self.st_type)

        if self.octave:
            raise Exception("Position octave 1: should not be here!")

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.activation is not None:
            if self.octave:
                raise Exception("Position octave 2: should not be here!")
            else:
                x = self.activation(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        multiplier: int = 2,
        n_dilation: int = 4,
        batch_norm: bool = True,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        octave_conv: bool = False,
    ) -> None:
        super().__init__()
        assert not (octave_conv and batch_norm)

        self.n_dilation = n_dilation
        self.conv_blocks = nn.ModuleList()

        n_filter_per_module = (filters * 4) // n_dilation  # multiplier
        for dilation in range(n_dilation - 1):
            self.conv_blocks.append(
                Conv3DConfigurable(
                    in_channels,
                    n_filter_per_module,
                    mid_filter=multiplier * filters,
                    dilation_rate=2**dilation,
                    use_bias=not batch_norm,
                    octave=octave_conv,
                )
            )
        self.conv_blocks.append(
            Conv3DConfigurable(
                in_channels,
                (filters * 4) - n_filter_per_module * (n_dilation - 1),  # multiplier
                mid_filter=multiplier * filters,
                dilation_rate=2 ** (n_dilation - 1),
                use_bias=not batch_norm,
                octave=octave_conv,
            )
        )

        self.batch_norm = torch.nn.BatchNorm3d(num_features=filters * 4, eps=1e-3, momentum=0.1) if batch_norm else None
        self.activation = activation
        self.octave = octave_conv

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = []
        for block in self.conv_blocks:
            x.append(block(inputs))
        x = torch.cat(x, dim=1)

        if self.octave:
            raise Exception("Position octave 1: should not be here !!!")

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.activation is not None:
            if self.octave:
                raise Exception("Position octave 2: should not be here !!!")
            else:
                x = self.activation(x)
        return x


class Conv3DConfigurable(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int,
        dilation_rate: int,
        mid_filter: Optional[int] = None,
        separable: bool = True,
        sharable: bool = False,
        octave: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            from torch.nn import init

            self.layers = nn.ModuleList()
            if not sharable:
                conv1 = torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=2 * filters if mid_filter is None else mid_filter,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1),
                    dilation=(1, 1, 1),
                    bias=False,
                )
                init.kaiming_normal_(conv1.weight, mode="fan_in", nonlinearity="relu")
                self.layers.append(conv1)

            conv2 = torch.nn.Conv3d(
                in_channels=2 * filters if mid_filter is None else mid_filter,
                out_channels=filters,
                kernel_size=(3, 1, 1),
                padding=(1 * dilation_rate, 0, 0),
                dilation=(dilation_rate, 1, 1),
                bias=use_bias,
            )
            init.kaiming_normal_(conv2.weight, mode="fan_in", nonlinearity="relu")
            self.layers.append(conv2)
        elif octave:
            raise Exception("Positon octave 3: should not be here !!!")
        else:
            raise Exception("Positon else 1: should not be here !!!")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class Attention1D(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        with_cls_token: bool = False,
        n_layer: int = 1,
    ):
        super().__init__()
        self.dim = dim_out
        self.num_heads = num_heads
        self.n_layer = n_layer
        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token

        self.proj_q = nn.ModuleList()
        self.proj_k = nn.ModuleList()
        self.proj_v = nn.ModuleList()
        self.attn_drop = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.proj_drop = nn.ModuleList()

        for _ in range(n_layer):
            self.proj_q.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))
            self.proj_k.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))
            self.proj_v.append(nn.Linear(dim_in, dim_out, bias=qkv_bias))

            self.attn_drop.append(nn.Dropout(attn_drop))
            self.proj.append(nn.Linear(dim_out, dim_out))
            self.proj_drop.append(nn.Dropout(proj_drop))

            dim_in = dim_out

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        x = rearrange(x, "b c t H W -> b t (c H W)")
        if self.n_layer == 0:
            return None

        for idx in range(self.n_layer):
            q = rearrange(self.proj_q[idx](x), "b t (h d) -> b h t d", h=self.num_heads)
            k = rearrange(self.proj_k[idx](x), "b t (h d) -> b h t d", h=self.num_heads)
            v = rearrange(self.proj_v[idx](x), "b t (h d) -> b h t d", h=self.num_heads)

            attn_score = torch.einsum("bhlk,bhtk->bhlt", [q, k]) * self.scale
            attn = F.softmax(attn_score, dim=-1)
            attn = self.attn_drop[idx](attn)

            x = torch.einsum("bhlt,bhtv->bhlv", [attn, v])
            x = rearrange(x, "b h t d -> b t (h d)")

            x = self.proj[idx](x)
            x = self.proj_drop[idx](x)
        return x
