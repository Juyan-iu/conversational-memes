# 02_classifier — MemeTector v4

CLIP-based meme image classifier used to identify meme replies in the Bluesky firehose.

## Overview

MemeTector v4 is a binary classifier (MEME / NOT_MEME) built on top of `CLIP ViT-L/14@336px`. It uses a Gated Fusion architecture combining visual and OCR-text embeddings, trained with 5-Fold cross-validation and ensemble inference.

## Key Improvements over v3

| Component | v3 | v4 |
|-----------|----|----|
| Backbone | CLIP ViT-L/14 (224px) | **CLIP ViT-L/14@336px** |
| Training data | MEME 500 / NOT_MEME 500 (sampled) | **ALL 502 / 736** (full dataset) |
| Training strategy | Single split | **5-Fold CV + ensemble** |
| Fusion | concat + cosine | **Gated Fusion** (learnable weighted sum) |
| Augmentation | MixUp | **CutMix + MixUp + RandomErasing** |
| Precision | float16 | **bfloat16** (A100 native, no overflow) |
| Batch size | 32 | **64** |
| TTA | 10-view | **12-view** |

**v3 gap:** Val 91.4% vs Test 86.0% (5.4% overfit)  
**v4 solution:** 5-Fold ensemble + full data + stronger regularization

## Training Data

- `MEME/`: 502 meme images
- `NOT_MEME/`: 736 non-meme images (602 NOT_MEME + 134 untitled)
- Total: 1,238 images

Data directory structure expected in Google Drive:
```
MyDrive/Meme_26SP/content/
├── MEME/
├── NOT_MEME/
└── untitled folder/
```

## Pretrained Checkpoints

5 fold checkpoints (~1.6 GB total) available on Google Drive:

📁 **[Download Checkpoints](https://drive.google.com/drive/folders/1k5aMXj3_Ooyvc5aHVj6MyQHg6xA5qVkV?usp=drive_link)**

```
memetector_v4/
├── fold1_best.pth
├── fold2_best.pth
├── fold3_best.pth
├── fold4_best.pth
└── fold5_best.pth
```

Place downloaded checkpoints in `02_classifier/checkpoints/` before running inference in `01_collection/meme_pipeline.py`.

## Setup

This notebook is designed to run on **Google Colab with A100 GPU** (80GB VRAM recommended).

```
Runtime → Change runtime type → A100 GPU
```

Install dependencies (handled in notebook Cell 1):
```python
!pip install transformers accelerate scikit-learn tqdm Pillow easyocr opencv-python-headless
```

## Usage

Open `memetector_v4_a100_optimized.ipynb` in Google Colab:

1. Mount Google Drive (Cell: `cell_drive`)
2. Set `DATA_DIR` and `OUT_DIR` paths
3. Run all cells sequentially
4. Trained fold checkpoints saved to `OUT_DIR` in Google Drive

## Architecture

```
Image → CLIP ViT-L/14@336px → Visual embedding (768-dim)
Text (OCR) → CLIP text encoder → Text embedding (768-dim)
                    ↓
             Gated Fusion
                    ↓
            MLP classifier → MEME / NOT_MEME
```

**Training:**
- Phase 1: Frozen CLIP backbone, train fusion + classifier head
- Phase 2: Unfreeze top CLIP layers, fine-tune end-to-end
- Optimizer: AdamW with warmup + cosine decay
- Augmentation: CutMix + MixUp + RandomErasing

**Inference:**
- 5-fold ensemble (average logits)
- 12-view Test-Time Augmentation (TTA)
- Threshold: 0.5

## Requirements

```
torch>=2.2.0
torchvision>=0.17.0
transformers>=4.49.0
easyocr>=1.7.1
scikit-learn>=1.3.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.13.0
```
