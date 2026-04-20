# Setup Guide

This guide walks you through setting up the EgoVizML environment from scratch.

---

## 1. Prerequisites

- Python 3.11 or 3.12 (Python 3.13 is **not** supported)
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Git
- A CUDA-capable GPU is strongly recommended for running Detic and SHAN inference

---

## 2. Clone the repository

```bash
git clone https://github.com/adeshkadambi/EgoVizML.git
cd EgoVizML
```

---

## 3. Install Python dependencies

```bash
pip install poetry
poetry install
poetry shell
```

Verify the install worked:

```bash
pytest tests/
```

All 11 tests should pass.

---

## 4. Clone external model repositories

Two external repositories must be cloned into the project root. They are **not** git submodules — clone them manually:

### Detectron2

```bash
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
```

### Detic

```bash
git clone https://github.com/facebookresearch/Detic.git
cd Detic
pip install -r requirements.txt
cd ..
```

Detic depends on Detectron2, so clone Detectron2 first.

### SHAN (hand-object detector)

The `shan_model/` folder is already included in this repository (vendored from https://github.com/ddshan/hand_object_detector). You will need to download the pretrained model weights separately:

```bash
# Download weights into shan_model/models/res101_handobj_100K/pascal_voc/
# See shan_model/README.md for the download link
```

---

## 5. Pre-built wheel files

Two `.whl` files are included in the repository root for cases where standard pip installs fail:

| File                                            | Platform                         | When needed                                |
| ----------------------------------------------- | -------------------------------- | ------------------------------------------ |
| `fasttext-0.9.2-cp311-cp311-win_amd64.whl`      | Windows, Python 3.11             | If `pip install fasttext` fails on Windows |
| `numpy-2.1.3-cp312-cp312-macosx_14_0_arm64.whl` | macOS Apple Silicon, Python 3.12 | If numpy install fails on M-series Mac     |

To install manually:

```bash
pip install <filename>.whl
```

If you are on a different platform/Python version you may need to build from source or find a compatible wheel at https://pypi.org.

---

## 6. Running the pipeline

See the pipeline steps in [README.md](README.md). Scripts are in `scripts/` and should be run from the project root with the Poetry environment active:

```bash
poetry shell
python scripts/video_to_subclips_and_frames.py --help
```

---

## 7. Common issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for known errors and fixes.
