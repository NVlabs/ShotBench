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

"""Metrics utilities for benchmarks."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class ShotDetectionMetrics:
    precision: float
    recall: float
    F1: float
    TP: Optional[int] = None
    FP: Optional[int] = None
    FN: Optional[int] = None
    FP_mistakes: Optional[List[int]] = None
    FN_mistakes: Optional[List[int]] = None


def evaluate_scenes(
    gt_scenes_int: npt.NDArray[np.int32],
    pred_scenes_int: npt.NDArray[np.int32],
    n_frames_miss_tolerance: int = 2,
) -> ShotDetectionMetrics:
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19

    Examples of computation with different tolerance margin:
    n_frames_miss_tolerance = 0
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.5, 5.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.5, 5.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.5, 4.5]] -> MISS
    n_frames_miss_tolerance = 1
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.0, 6.0]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.0, 6.0]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.0, 5.0]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[3.0, 4.0]] -> MISS
    n_frames_miss_tolerance = 2
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[4.5, 6.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[4.5, 6.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[3.5, 5.5]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[2.5, 4.5]] -> HIT
      gt_scenes:   [[0, 2], [3, 9]] -> gt_trans:   [[1.5, 3.5]] -> MISS
    """

    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes_int.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes_int.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0
    fp_mistakes, fn_mistakes = [], []

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans):
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        elif i == len(gt_trans):
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            fp_mistakes.append(pred_trans[j])
            j += 1
        elif pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            fn_mistakes.append(gt_trans[i])
            i += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    results = ShotDetectionMetrics(
        precision=p,
        recall=r,
        F1=f1,
        TP=tp,
        FP=fp,
        FN=fn,
        FN_mistakes=fn_mistakes,
        FP_mistakes=fp_mistakes,
    )
    return results
