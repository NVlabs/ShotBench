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

"""PySceneDetect shot detection."""

from pathlib import Path

from scenedetect import detect as _detect
from scenedetect.detectors import ContentDetector


def detect(video_path: Path) -> list[tuple[int, int]]:
    """Shot detection with PySceneDetect."""

    scene_list = _detect(
        video_path.as_posix(),
        ContentDetector(threshold=25, min_scene_len=15),
        start_in_scene=True,
    )

    end_frame_idx = [0]
    for scene in scene_list:
        new_end_frame_idx = scene[1].get_frames()
        end_frame_idx.append(new_end_frame_idx)

    shots = []
    for i in range(len(end_frame_idx) - 1):
        shots.append([end_frame_idx[i], end_frame_idx[i + 1]])
    return shots
