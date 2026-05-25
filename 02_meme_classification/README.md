# 02_meme_classification - MemeTector Training

This folder contains the MemeTector v4 training notebook used to prepare the
binary meme classifier for the collection pipeline in
[`../01_collection`](../01_collection/README.md).

## What This Stage Does

`memetector_v4_a100_optimized (2).ipynb` trains a binary image classifier for
MEME / NOT_MEME prediction. The trained checkpoints are then loaded by
`../01_collection/meme_pipeline.py` to identify meme replies in Bluesky firehose
archives.

The classifier uses CLIP ViT-L/14@336px image and text encoders, OCR text from
the image, a gated image/text fusion module, and a binary MLP classifier. The
released training run uses 5-fold cross-validation and ensemble inference.

## Files

```text
02_meme_classification/
├── memetector_v4_a100_optimized (2).ipynb  # Training and evaluation notebook
├── results_v4.png                          # Saved evaluation figure
├── summary_v4.json                         # Saved evaluation metrics
└── README.md
```

Generated checkpoints are not tracked in git:

```text
memetector_v4/
├── fold1_best.pth
├── fold2_best.pth
├── fold3_best.pth
├── fold4_best.pth
└── fold5_best.pth
```

## Model

The notebook trains `MemeDetectorV4`, matching the architecture loaded by
`01_collection/meme_pipeline.py`:

```text
Image -> CLIP ViT-L/14@336px -> visual embedding
OCR text -> CLIP text encoder -> text embedding
visual + text embeddings -> gated fusion
image/text/fused/cosine features -> MLP classifier -> MEME / NOT_MEME
```

Training details:

```text
Backbone:        openai/clip-vit-large-patch14-336
Input size:      336 x 336
Splits:          15% held-out test split, 5-fold CV on the remaining pool
Training:        Phase 1 frozen CLIP head training, Phase 2 full fine-tuning
Loss:            Focal loss with class weights and label smoothing
Augmentation:    CutMix, MixUp, RandomErasing, RandAugment
Precision:       bfloat16 on A100
Inference:       5-fold ensemble, 12-view TTA in the notebook
Threshold:       0.5
```

Note that `01_collection/meme_pipeline.py` can run the same fold checkpoints
with faster inference settings. Its default collection configuration disables
TTA for throughput.

## Training Data

The notebook expects a local or Google Drive directory with two binary classes:

```text
Meme_26SP/content/
├── MEME/              # 502 meme images
├── NOT_MEME/          # 602 non-meme images
└── untitled folder/   # 134 additional non-meme images
```

The notebook combines `NOT_MEME/` and `untitled folder/` as the NOT_MEME class,
for a total of 1,238 images:

```text
MEME:      502
NOT_MEME:  736
Total:   1,238
```

## Pretrained Checkpoints

Instead of retraining, you can download the released 5-fold checkpoints:

[Download checkpoints](https://drive.google.com/drive/folders/1k5aMXj3_Ooyvc5aHVj6MyQHg6xA5qVkV?usp=drive_link)

After downloading or training, pass the checkpoint directory to stage 01:

```bash
python ../01_collection/meme_pipeline.py --run --date 2024-06-01 \
  --archive-base /path/to/firehose_archives \
  --model-dir /path/to/memetector_v4 \
  --output-dir ../01_collection/meme_dataset
```

## Requirements

- Google Colab or another CUDA environment
- A100 GPU recommended for the released training configuration
- Python 3.10+
- Training images arranged as shown above

The notebook installs its Python dependencies in the first cell:

```python
!pip install transformers accelerate scikit-learn tqdm Pillow easyocr opencv-python-headless
```

Core packages:

```text
torch
torchvision
transformers
accelerate
scikit-learn
Pillow
easyocr
opencv-python-headless
tqdm
matplotlib
seaborn
```

## Configure

In the notebook, update these paths before training:

```python
DATA_DIR = "/content/drive/MyDrive/Meme_26SP/content/"
OUT_DIR = "/content/drive/MyDrive/Meme_26SP/memetector_v4"
```

`DATA_DIR` should contain the training folders. `OUT_DIR` is where the fold
checkpoints, metric summary, and result figure will be saved.

## Run

Open the notebook in Google Colab, select an A100 runtime, and run the cells in
order:

```text
1. Install dependencies.
2. Mount Google Drive.
3. Set DATA_DIR and OUT_DIR.
4. Load MEME and NOT_MEME images.
5. Extract OCR text and visual-part variants.
6. Train 5 folds.
7. Evaluate the ensemble on the held-out test split.
8. Save fold checkpoints, results_v4.png, and summary_v4.json.
```

## Results

The included `summary_v4.json` reports:

```text
Mean best validation accuracy: 0.8973
Held-out test accuracy:        0.9032
Held-out test F1:              0.8889
Held-out test ROC-AUC:         0.9767
```

These metrics document the classifier run used to prepare the checkpoints. Stage
01 uses the checkpoints for candidate meme-reply collection rather than for
benchmark evaluation directly.

## Notes

- This stage is required only if you want to train or inspect the classifier.
  For reproducing collection, downloading the released fold checkpoints is
  enough.
- Keep the checkpoint filenames as `fold1_best.pth` through `fold5_best.pth`;
  `01_collection/meme_pipeline.py` loads that naming pattern directly.
- The training images and trained checkpoints are not tracked in git because of
  size and data-release constraints.
