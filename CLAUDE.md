# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.13 (`.python-version`), managed with [uv](https://docs.astral.sh/uv/).
- Install / sync deps: `uv sync`
- Run a notebook kernel against the project env: `uv run jupyter lab` (or use the `.venv` produced by `uv sync` as the kernel in VS Code).
- There is no test suite, lint config, or build step — this is a Jupyter-driven training project.

### Building fishsense-core with CUDA

`fishsense-core` is a maturin-built Rust extension pinned to tag `fishsense_core-v2.1.0` in [pyproject.toml](pyproject.toml). The CUDA execution provider is gated behind the Cargo feature `cuda` and is OFF by default. To build with CUDA acceleration:

```bash
uv sync --reinstall-package fishsense-core --config-setting 'build-args=--features cuda'
```

This downloads CUDA-enabled ORT binaries (~250MB) and bundles `libonnxruntime_providers_cuda.so` and friends into the wheel under `fishsense_core/`. Without `--config-setting`, the package builds CPU-only.

At runtime the CUDA libs are dynamically loaded via the `nvidia-*` pip packages that come in transitively through torch/ultralytics. **`torch` must be imported before `fishsense_core`** so it preloads `libcudart`/`libcublas`/`libcudnn` with `RTLD_GLOBAL`; otherwise ORT silently falls back to CPU. The notebook loader handles this — see [deepfish_yolo_segmentation_eval.ipynb](deepfish_yolo_segmentation_eval.ipynb) cell `cuda-probe`, which reports inference latency so you can tell at a glance whether CUDA is active (~50–500ms/image vs ~4–5s/image on CPU at 1080p).

### SAM 3 setup (for both Group A and Group B in the eval notebook)

The eval notebook [deepfish_yolo_segmentation_eval.ipynb](deepfish_yolo_segmentation_eval.ipynb) uses Meta's [SAM 3.1](https://github.com/facebookresearch/sam3) in two slots:
- **Group A**: `sam3` standalone with open-vocabulary text prompt `"fish"`.
- **Group B**: `psam3` pipeline (YOLO26 → SAM 3 with native bbox prompts on the full image, no cropping).

Both go through the standalone `sam3` Python package (not Ultralytics' SAM wrapper, which only exposes geometric prompts). The `sam3` package's stale `numpy<2` pin would normally conflict with `fishsense-core`'s `numpy>=2.3.5`; pyproject.toml works around that with `tool.uv.override-dependencies`. SAM 3 inference is wrapped in scoped `torch.amp.autocast("cuda", dtype=torch.bfloat16)` + `torch.inference_mode()` to satisfy its internal bf16 fast-path; the scope keeps it from bleeding into other models.

To enable:

1. Request access at [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) on HuggingFace.
2. Download `sam3.1_multiplex.pt` (~3.3 GB) and place it at the **repo root**:
   ```bash
   huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
       --local-dir . --local-dir-use-symlinks False
   ```
3. `uv sync` will pull in `sam3` and its transitive deps (einops, hydra-core, iopath, ftfy, pycocotools, regex, tabulate, setuptools<80 — pinned because sam3 still uses `pkg_resources`).

**Hardware note**: SAM 3.1 needs ~4 GB of VRAM in bfloat16. With the other models loaded, full-eval VRAM peak is ~5–6 GB. The 6 GB RTX 3060 will be borderline; **8 GB+ recommended** for headroom. The simplest fall-back if you OOM is to comment out the SAM 3 block in the loader cell — the rest of the eval continues without it (downstream cells gate on `HAS_SAM3`).

If `sam3.1_multiplex.pt` is missing or the import fails, the notebook gracefully degrades to the four-way comparison without SAM 3.

## What this repo does

Trains YOLO (Ultralytics) models on the [DeepFish](https://github.com/alzayats/DeepFish) dataset for two tasks:

- [deepfish_yolo_detection.ipynb](deepfish_yolo_detection.ipynb) — fish bounding-box detection (`yolo26n.pt`).
- [deepfish_yolo_segmentation.ipynb](deepfish_yolo_segmentation.ipynb) — fish instance segmentation (`yolo26n-seg.pt`).

Both notebooks share the same shape: load DeepFish split CSVs → convert the `Segmentation/masks/` PNGs into YOLO label files → write a `deepfish_*.yaml` dataset descriptor → call `YOLO(...).train(...)`.

## Data layout (important — paths cross the repo boundary)

Source data lives **inside** the repo at [data/DeepFish/](data/) (`Segmentation/{images,masks}/<video_id>/*.{jpg,png}`, plus `Segmentation/{train,val,test}.csv` for splits). The `data/` directory is gitignored.

YOLO-formatted training data is written **one level above the repo** at `../datasets/DeepFish/{Detection,Segmentation}/{images,labels}/{train,val,test}/`. The notebooks symlink images from `data/DeepFish/...` into that external `datasets/` tree and write `.txt` label files alongside. The generated `deepfish_detect.yaml` / `deepfish_segmentation.yaml` (also gitignored) point Ultralytics at this external root via an absolute path.

If `../datasets/DeepFish` does not exist, the data-prep cells will create it. Do not "fix" the `../datasets` path to a repo-local one without understanding why it is external — the layout is intentional so multiple sibling repos can share one converted dataset.

## Label generation conventions

- **Detection labels** (`deepfish_yolo_detection.ipynb`): masks → connected components via `skimage.measure.label` + `regionprops` → one bbox per region → normalized `class x_center y_center w h` lines, class `0 = fish`.
- **Segmentation labels** (`deepfish_yolo_segmentation.ipynb`): masks binarized at threshold 127 → external contours via `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` → polygons with ≥3 points → normalized `class x1 y1 x2 y2 ...` lines. Images with zero polygons get their label file *deleted* (Ultralytics treats missing label files as background).
- Split assignment for each image is decided by stem-matching against the IDs in `Segmentation/{train,val,test}.csv`. Images not present in any split raise `ValueError`.

## Outputs

Ultralytics writes training runs to [runs/detect/train/](runs/) and [runs/segment/train/](runs/) (gitignored). Trained weights land at `runs/<task>/train/weights/best.pt`. The `yolo26n.pt` / `yolo26n-seg.pt` files at the repo root are the pretrained starting checkpoints.
