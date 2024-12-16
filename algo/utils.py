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

"""Some utilities for inference."""

import subprocess
from pathlib import Path
from typing import Callable, Generator, Optional

import numpy as np
import numpy.typing as npt
import torch


def get_frames(video_file: Path, width: int, height: int) -> Optional[npt.NDArray[np.uint8]]:
    """Fetch resized frames for video."""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        video_file.as_posix(),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    video_stream, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode('utf-8')}")
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, height, width, 3])
    return video


def get_batches(frames: npt.NDArray[np.uint8]) -> Generator[npt.NDArray[np.uint8], None, None]:
    """Fetch 100 frames, and pad the first and last batches accordingly with the first or last frame."""
    total_frames = len(frames)
    reminder = -total_frames % 50
    for i in range(0, total_frames + reminder, 50):
        start_idx = max(i - 25, 0)
        end_idx = min(i + 75, total_frames)
        batch = frames[start_idx:end_idx]
        # Add padding at the beginning if necessary
        if i < 25:
            padding_start = [frames[0]] * (25 - i)
            batch = np.concatenate([padding_start, batch], axis=0)
        # Add padding at the end if necessary
        if end_idx > total_frames:
            padding_end = [frames[-1]] * (end_idx - total_frames)
            batch = np.concatenate([batch, padding_end], axis=0)
        yield batch


def get_predictions(
    model: Callable[[torch.Tensor], torch.Tensor],
    frames: npt.NDArray[np.uint8],
    threshold: float,
) -> npt.NDArray[np.uint8]:
    """Get predictions from the video frame array.

    Args:
        model: shot detection model.
        frames: uint8 array of shape (# frames, height, width, 3), with RGB channels.
        threshold: probability threshold for shot detection.

    Returns:
        0/1 prediction array of shape (# frames, 1)
    """
    assert frames.ndim == 4, "Expected frames tensor to have rank 4."
    predictions = []
    for batch in get_batches(frames):
        batch_gpu = torch.from_numpy(batch.copy()).cuda()
        one_hot = model(batch_gpu.unsqueeze(0))
        predictions.append(one_hot[0, 25:75])
    predictions = torch.concatenate(predictions, 0)[: len(frames)]
    return (predictions > threshold).to(torch.uint8).cpu().numpy()


def get_scenes(predictions: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
    """Convert prediction array to scene array.

    Args:
        predictions: array of shape [# frames, 1].
            Values are 1 if frame is a shot transition, and 0 if it's not.

    Returns:
        scene array of shape [# scenes, 2], where the value at each row is the start and end frame of the shot.
    """
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append((start, i))
        t_prev = t
    if t == 0:
        scenes.append((start, i))
    if not scenes:
        scenes.append((0, len(predictions) - 1))
    return np.array(scenes, dtype=np.int32)
