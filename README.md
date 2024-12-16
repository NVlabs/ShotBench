# Shot Bench

This repo contains code for evaluation of shot detection models tested in NVIDIA's Cosmos video curation.

## Setup

Use the `requirements.txt` to install the required dependencies.

```bash
conda create -n shot-bench python=3.10
conda activate shot-bench
pip install -r requirements.txt
```

Some models rely on `ffmpeg` binary to be present on the host. For this, run:
```bash
sudo apt-get install ffmpeg -y
```

## Prepare datasets

Run the download CLI, specifying the dataset you want to download, for example:

```bash
python3 download.py rai
```
You have additional datasets as options, such as `bbc`, `shot`.

This will download to a `data` directory the various datasets.
Each dataset consists of a list of video and text pairs. Each txt contains the start and end frame IDs of each shot in the corresponding video, e.g.:
```
0       81
106     289
317     341
342     485
...
```

## Run inference

Run the inference CLI, specifying the dataset and model, for example:

```bash
python3 infer.py rai pyscenedetect
```

You have additional models as options, such as `transnetv2` and `autoshot`.

## Run evaluation

Run the inference CLI, specifying the dataset and model, for example:
```bash
python3 evaluate.py rai pyscenedetect
```

The outputs should be the following:
```
{'model': 'pyscenedetect', 'dataset': 'bbc', 'precision': 0.893, 'recall': 0.884, 'F1': 0.889}
{'model': 'transnetv2',    'dataset': 'bbc', 'precision': 0.983, 'recall': 0.951, 'F1': 0.967}
{'model': 'autoshot',      'dataset': 'bbc', 'precision': 0.984, 'recall': 0.922, 'F1': 0.952}

{'model': 'pyscenedetect', 'dataset': 'rai', 'precision': 0.856, 'recall': 0.807, 'F1': 0.831}
{'model': 'transnetv2',    'dataset': 'rai', 'precision': 0.918, 'recall': 0.921, 'F1': 0.919}
{'model': 'autoshot',      'dataset': 'rai', 'precision': 0.889, 'recall': 0.923, 'F1': 0.906}

{'model': 'pyscenedetect', 'dataset': 'shot', 'precision': 0.769, 'recall': 0.673, 'F1': 0.718}
{'model': 'transnetv2',    'dataset': 'shot', 'precision': 0.884, 'recall': 0.767, 'F1': 0.821}
{'model': 'autoshot',      'dataset': 'shot', 'precision': 0.866, 'recall': 0.806, 'F1': 0.835}

{'model': 'pyscenedetect', 'dataset': 'clipshots-test', 'precision': 0.395, 'recall': 0.602, 'F1': 0.477}
{'model': 'transnetv2',    'dataset': 'clipshots-test', 'precision': 0.685, 'recall': 0.772, 'F1': 0.726}
{'model': 'autoshot',      'dataset': 'clipshots-test', 'precision': 0.653, 'recall': 0.781, 'F1': 0.711}
```

# Acknowledgements

We would like to acknowledge the following projects where parts of the codes in the [algo](algo) folder is derived from:
- [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)
- [wentaozhu/AutoShot](https://github.com/wentaozhu/AutoShot)
