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

"""AutoShot shot detection.

Requires `ffmpeg` installed on host.
"""

from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch

from algo import utils
from algo.autoshot_pytorch import AutoShot


@lru_cache
def load_model() -> Callable[[torch.Tensor], torch.Tensor]:
    """Singleton of model."""

    model = AutoShot()
    state_dict = torch.load("algo/autoshot-pytorch-weights.pth", weights_only=True)["net"]
    model.load_state_dict(state_dict)
    model.eval().cuda()
    return model


def detect(video_path: Path, threshold: float = 0.4) -> list[tuple[int, int]]:
    """Shot detection with AutoShot."""

    model = load_model()
    frames = utils.get_frames(video_path, height=27, width=48)
    with torch.no_grad():
        predictions = utils.get_predictions(model, frames, threshold)
    scenes = utils.get_scenes(predictions)
    return scenes.tolist()
