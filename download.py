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

"""Utility to download shot detection datasets."""

import argparse
import pickle
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import gdown
import mediafiregrabber
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def download_rai(output_path: Path) -> None:
    """Download RAI Nature dataset from https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19."""

    if output_path.is_file():
        raise ValueError(f"{output_path} should not be a file!")
    output_path.mkdir(exist_ok=True, parents=True)

    url = "https://drive.google.com/uc?id=1YDQQw7SaszbIaY3cpOKltc0C30XAE2in"

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_file = (tmp_path / "rai.zip")
        gdown.download(
            url,
            zip_file.as_posix(),
            quiet=False,
        )

        pattern = r"ShotDetector/video_rai/.*?(_gt\.txt|\.mp4)"
        with zipfile.ZipFile(zip_file, "r") as fp:
            file_names = fp.namelist()
            for f in file_names:
                match = re.match(pattern, f)
                if match:
                    with fp.open(f) as src:
                        output_file = output_path / Path(f.replace("_gt", "")).name
                        with open(output_file, "wb") as dst:
                            dst.write(src.read())
                    print(f"Unzipping {f} to {output_file}...")


def download_bbc(output_path: Path) -> None:
    """Download BBC Nature dataset from https://aimagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19.

    N.B. video files are hosted on MediaFire [https://www.mediafire.com/folder/texdwptt9242j/BBCPH], not on GDrive.
    """

    if output_path.is_file():
        raise ValueError(f"{output_path} should not be a file!")
    output_path.mkdir(exist_ok=True, parents=True)

    annotations = [
        "annotations/shots/01_From_Pole_to_Pole.txt",
        "annotations/shots/02_Mountains.txt",
        "annotations/shots/03_Ice Worlds.txt",
        "annotations/shots/04_Great Plains.txt",
        "annotations/shots/05_Jungles.txt",
        "annotations/shots/06_Seasonal_Forests.txt",
        "annotations/shots/07_Fresh_Water.txt",
        "annotations/shots/08_Ocean_Deep.txt",
        "annotations/shots/09_Shallow_Seas.txt",
        "annotations/shots/10_Caves.txt",
        "annotations/shots/11_Deserts.txt"
    ]

    url = "https://drive.google.com/uc?id=1vr-bBxXFpz5-vyirVsW6-RF4EjXC7nfN"

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_file = (tmp_path / "bbc.zip")
        gdown.download(
            url,
            zip_file.as_posix(),
            quiet=False,
        )

        video_urls = [
            "https://www.mediafire.com/file/57yxmyy1hf15irw/bbc_01.mp4/file",
            "https://www.mediafire.com/file/i8sv7n278o6yjki/bbc_02.mp4/file",
            "https://www.mediafire.com/file/yofp2f98yxxs5ze/bbc_03.mp4/file",
            "https://www.mediafire.com/file/xxencq76zdlds3l/bbc_04.mp4/file",
            "https://www.mediafire.com/file/59xzdnp9z9fybq9/bbc_05.mp4/file",
            "https://www.mediafire.com/file/xklnjsuti2hcmj3/bbc_06.mp4/file",
            "https://www.mediafire.com/file/7avryrl267udnyj/bbc_07.mp4/file",
            "https://www.mediafire.com/file/8oakmj39brq96bj/bbc_08.mp4/file",
            "https://www.mediafire.com/file/hgc462lcp3m6dvg/bbc_09.mp4/file",
            "https://www.mediafire.com/file/vn28vgqo1bz851x/bbc_10.mp4/file",
            "https://www.mediafire.com/file/jcc7e29c0j6ygng/bbc_11.mp4/file",
        ]

        with zipfile.ZipFile(zip_file, "r") as fp:
            file_names = fp.namelist()
            for i, (annotation, video_url) in enumerate(zip(annotations, video_urls)):
                with fp.open(annotation) as src:
                    output_txt = output_path / ("bbc_" + str(i + 1).zfill(2) + ".txt")
                    with open(output_txt, "wb") as dst:
                        dst.write(src.read())
                print(f"Unzipping {annotation} to {output_txt}...")
                output_video = output_path / ("bbc_" + str(i + 1).zfill(2) + ".mp4")
                print(f"Please download manually {video_url} as {output_video}")


def download_shot(output_path: Path) -> None:
    """SHOT dataset from https://github.com/wentaozhu/AutoShot."""

    if output_path.is_file():
        raise ValueError(f"{output_path} should not be a file!")
    output_path.mkdir(exist_ok=True, parents=True)

    gt_url = "https://drive.google.com/uc?id=1ivfFXt958VeM6pSI3bG2y5Bjirc5UEOV"
    videos_zips = [
        "https://drive.google.com/uc?id=1dNdRse85_m_UzEW9pTImXUfO1aLwP9bF",
        "https://drive.google.com/uc?id=1S2hMlfzJt7FJun1tlmk6tjcF_Ir4GVZq",
        "https://drive.google.com/uc?id=1yXVDzCT4Pzb8ehdgEhG456FS52Maz6on",
        "https://drive.google.com/uc?id=1-Sh4V-bnQqvGSKoi2T-4Buyb9I7qCE1g",
        "https://drive.google.com/uc?id=1toySPvtDO6hLVKA3CGea-Gdi0CVHDqj2",
        "https://drive.google.com/uc?id=1Unge1xMKMl7E18SsxFpG0yxlw44egRsh",
        "https://drive.google.com/uc?id=1F1T_qt2UdIakXL2FyWi2GAQTNifZ_nBk",
        "https://drive.google.com/uc?id=1IeF3PQySgHXKDN8I8GjjcVrUdyv812iU",
    ]

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:
        tmp_path = Path(tmp_dir)
        zip_file = (tmp_path / "gt.pickle")
        gdown.download(
            gt_url,
            zip_file.as_posix(),
            quiet=False,
        )

        with zip_file.open("rb") as fp:
            gt = pickle.load(fp)
        for k, v in gt.items():
            output_txt = output_path / f"{k}.txt"
            df = pd.DataFrame(v)
            df.to_csv(output_txt, index=False, header=None, sep="\t")

        for i, url in enumerate(videos_zips):
            zip_file = (output_path / f"videos_{i}.zip")
            try:
                if not zip_file.exists():
                    gdown.download(
                        url,
                        zip_file.as_posix(),
                        quiet=False,
                    )
            except Exception as e:
                print(e)
                continue
            pattern = r"^(?!.*__MACOSX).*\.mp4$"
            with zipfile.ZipFile(zip_file, "r") as fp:
                file_names = fp.namelist()
                for f in file_names:
                    match = re.match(pattern, f)
                    if match and Path(f).stem in gt.keys():
                        with fp.open(f) as src:
                            output_file = output_path / Path(f).name
                            with open(output_file, "wb") as dst:
                                dst.write(src.read())
                        print(f"Unzipping {f} to {output_file}...")
                    else:
                        print(f"Skipping {f}")


def download_all(output_path: Path) -> None:
    """Convenience function to download all datasets."""
    download_rai(output_path / "rai")
    download_bbc(output_path / "bbc")
    download_shot(output_path / "shot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CLI")
    parser.add_argument(
        "option",
        choices=["rai", "bbc", "shot", "all"],
        help="Select one of the datasets: rai, bbc, shot or all."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Specify directory, will default to `data` in current path."
    )
    args = parser.parse_args()

    default = Path("data") / ("" if args.option == "all" else args.option)
    path = Path(args.path) if args.path is not None else default
    if args.option == "all":
        download_all(path)
    elif args.option == "bbc":
        download_bbc(path)
    elif args.option == "rai":
        download_rai(path)
    elif args.option == "shot":
        download_shot(path)
    else:
        raise ValueError(f"Dataset `{args.option}` not available.")


if __name__ == "__main__":
    main()
