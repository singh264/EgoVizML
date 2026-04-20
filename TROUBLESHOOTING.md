# Troubleshooting

Known errors and fixes encountered during development. If you hit a new issue, add it here.

---

### AssertionError: Torch not compiled with CUDA enabled

This usually occurs after running `poetry install`, which installs a CPU-only version of PyTorch.

**Fix:** Uninstall and reinstall PyTorch with CUDA support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Adjust the CUDA version (`cu118`) to match what's installed on your machine (`nvidia-smi` will show your CUDA version).

---

### ModuleNotFoundError: No module named 'egoviz'

This means the package isn't installed in your current environment.

**Fix:**

```bash
poetry install
poetry shell
```

If you're running notebooks, make sure the kernel is set to the Poetry virtualenv.

---

### Poetry can't find a compatible Python version

The project requires Python 3.11 or 3.12. Python 3.13 is **not** supported.

**Fix:** Install Python 3.12 and point Poetry to it:

```bash
poetry env use python3.12
poetry install
```

---

### fasttext or numpy install fails

Use the pre-built `.whl` files included in the repository root. See [SETUP.md](SETUP.md#5-pre-built-wheel-files) for instructions.
