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

"""Utility to run inference of different models on different datasets."""

import argparse
from pathlib import Path

import pandas as pd

from algo import autoshot, pyscenedetect, transnetv2


class Runner:
    def __init__(self, dataset: Path, model: str) -> None:
        self.mp4_files = sorted(dataset.glob("*.mp4"))
        if model == "pyscenedetect":
            self.fn = pyscenedetect.detect
        elif model == "transnetv2":
            self.fn = transnetv2.detect
        elif model == "autoshot":
            self.fn = autoshot.detect
        else:
            raise NotImplementedError(f"Detection algo {model} not implemented!")

    def write(self, pred_path: Path) -> None:
        for f in self.mp4_files:
            shots = pd.DataFrame(self.fn(f))
            print(f"Found {len(shots)} shots in {f}")
            pred_file = pred_path / (f.stem + ".txt")
            print(f"Writing result to {pred_file}")
            shots.to_csv(pred_file.as_posix(), sep="\t", index=False, header=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer CLI")
    parser.add_argument(
        "dataset",
        choices=["rai", "bbc", "shot", "clipshots-test"],
        help="Select one of the datasets: `rai`, `bbc`, `shot`."
    )
    parser.add_argument(
        "model",
        choices=["pyscenedetect", "transnetv2", "autoshot"],
        help="Select one of the models: `pyscenedetect`, `transnetv2`, `autoshot`."
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default="data",
        help="Specify directory to read from, will default to `data` in current path."
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        default="infer",
        help="Specify directory to write to, will default to `infer` in current path."
    )
    args = parser.parse_args()

    dataset = Path(args.gt_path) / args.dataset
    if not dataset.exists():
        raise FileNotFoundError(f"{dataset} does not exist!")

    output_path = (Path(args.pred_path) / f"{args.dataset}/{args.model}")
    output_path.mkdir(exist_ok=True, parents=True)

    runner = Runner(dataset, args.model)
    runner.write(output_path)


if __name__ == "__main__":
    main()
