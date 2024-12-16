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

"""Utility to evaluate different algorithms on different datasets."""

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluate_utils import ShotDetectionMetrics, evaluate_scenes


class Runner:
    def __init__(self, gt_path: Path, pred_path: str) -> None:
        self.gt_files = sorted(gt_path.glob("*.txt"))
        self.pred_files = sorted(pred_path.glob("*.txt"))

        if len(self.gt_files) != len(self.pred_files):
            raise ValueError("Expected same number of predicted and gt files.")

        for f1, f2 in zip(self.gt_files, self.pred_files):
            if f1.name != f2.name:
                raise ValueError(f"Expected matching names, got {f1} and {f2}")

    def evaluate(self, dataset: str, model: str) -> dict:
        overall_metrics = []
        for gt, pred in zip(self.gt_files, self.pred_files):
            gt_df = pd.read_csv(gt, sep="\t", header=None)
            pred_df = pd.read_csv(pred, sep="\t", header=None)
            video_metrics = evaluate_scenes(
                gt_scenes_int=gt_df.values,
                pred_scenes_int=pred_df.values,
            )
            overall_metrics.append(asdict(video_metrics))

        results_df = pd.DataFrame(overall_metrics)
        total = results_df[["TP", "FP", "FN"]].sum(axis=0)
        precision = total["TP"] / (total["TP"] + total["FP"])
        recall = total["TP"] / (total["TP"] + total["FN"])
        F1 = (precision * recall * 2) / (precision + recall)
        return {
            "model": model,
            "dataset": dataset,
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(F1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation CLI")
    parser.add_argument(
        "dataset",
        choices=["rai", "bbc", "shot", "clipshots-test"],
        help="Select one of the datasets: rai, bbc, shot."
    )
    parser.add_argument(
        "model",
        choices=["pyscenedetect", "transnetv2", "autoshot"],
        help="Select one of the models."
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default="data",
        help="Specify directory, will default to `data` in current path."
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        default="infer",
        help="Specify directory, will default to `infer` in current path."
    )
    args = parser.parse_args()

    gt_path = (Path(args.gt_path) / args.dataset)
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} not found! Please run `download.py` first.")

    pred_path = (Path(args.pred_path) / f"{args.dataset}/{args.model}")
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} not found! Please run `infer.py` first.")

    runner = Runner(gt_path, pred_path)
    result = runner.evaluate(dataset=args.dataset, model=args.model)
    print(result)


if __name__ == "__main__":
    main()
