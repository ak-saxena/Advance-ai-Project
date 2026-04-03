# Multimodal Fake News Detection on Fakeddit

This repository contains our Advanced AI project for multimodal fake news detection on the Fakeddit dataset.

The project implements and compares three multimodal architectures that jointly use news headlines and images:

- **BERT + ResNet-50** (text + CNN image encoder)
- **BERT + ViT** (text + Vision Transformer)
- **CLIP-based multimodal classifier** (frozen CLIP backbone + classification head)

A Streamlit demo app lets you load trained weights and classify new (headline, image) pairs into six Fakeddit-style labels.

---

## 1. Project Structure

The key files in this repo are:

- `dataset_downloader.ipynb` — Download and prepare the Fakeddit subset and images, then save a cleaned CSV `clean_df.csv`.
- `bertandrestnet.ipynb` — Train the **BERT + ResNet-50** multimodal classifier on Fakeddit.
- `bert_ViT_v2.ipynb` — Train the **BERT + ViT** multimodal classifier (v2 configuration) and save best and final checkpoints.
- `CLIPv1.ipynb` — First CLIP experiments and evaluation (prediction tables, per-class probabilities, diagnostics).
- `CLIPv2_1.ipynb` — Final CLIP-based classifier training and evaluation (saves the best CLIP model used in the app).
- `app.py` (ensemble version) — Streamlit app that loads all three models (BERT+ResNet, BERT+ViT, CLIP) and shows individual predictions plus a majority-vote result.
- `app_clip_only.py` (single-CLIP version) — Alternative Streamlit app that only loads the CLIP classifier.
- `Model.py` — Python module defining `BERTResNetClassifier`, `BERTViTClassifierV22`, and `CLIPMultimodalClassifier` classes used by notebooks and apps.
- `README.md` — This documentation file.

> **Note:** In the repository, keep **one** main `app.py` (we recommend the ensemble version) and optionally rename the single-CLIP variant to `app_clip_only.py` for clarity.

---

## 2. Task and Dataset

We work on the **Fakeddit** dataset: Reddit posts with text and images annotated into multiple fake-news-related categories.

### 2.1 Label Space

We use the **6-way label setting** (`6_way_label` column):

| Label ID | Label Name           | Description |
|----------|----------------------|-------------|
| 0        | TRUE                 | Factually accurate content based on verified information |
| 1        | SATIRE               | Humorous or satirical content, not meant to be taken literally |
| 2        | FALSE CONNECTION     | Headlines or visuals that do not accurately reflect the article content |
| 3        | IMPOSTER CONTENT     | Content impersonating genuine sources (e.g. fake news mimicking legitimate outlets) |
| 4        | MANIPULATED CONTENT  | Content that has been altered or edited to mislead |
| 5        | MISLEADING CONTENT   | Uses selective information or framing to mislead the audience |

### 2.2 Data Format

Each example contains:

- A textual **headline/title** in column `clean_title`.
- An associated **image** given by `image_url`, downloaded and saved as `<id>.jpg`.
- Metadata: `author`, `domain`, `subreddit`, `score`, `upvote_ratio`, `created_utc`.

The cleaned dataframe `clean_df.csv` has 13 columns:

```
author, clean_title, created_utc, domain, hasImage, id,
image_url, linked_submission_id, num_comments, score,
subreddit, upvote_ratio, 6_way_label
```

All training notebooks assume (on Colab):

```python
CLEANDFPATH = "/content/drive/MyDrive/Fake-news-detector/multimodal_only_samples/working/clean_df.csv"
IMAGEDIR    = "/content/drive/MyDrive/Fake-news-detector/multimodal_only_samples/working/images"
```

If you place the data elsewhere, update these constants at the top of each notebook.

### 2.3 Dataset Creation Notebook

`dataset_downloader.ipynb` performs:

1. Mounts Google Drive: `drive.mount('/content/drive')`.
2. Loads the raw Fakeddit CSV.
3. Filters to the subset used in this project (posts with images, 6-way labels).
4. Downloads all images to `IMAGEDIR` using `image_url` and `id`.
5. Verifies that expected images exist and logs any download failures.
6. Saves the cleaned dataframe as `clean_df.csv` (shape `(18470, 13)`).

---

## 3. Environment and Installation

### 3.1 Recommended Environment

All notebooks are written and tested in **Google Colab** with GPU acceleration (A100 in the saved runs; any modern GPU with ≥12–16 GB VRAM works well).

For the installation challenge on your own machine:

- Python **3.10+**
- CUDA-enabled GPU (optional but strongly recommended for training)
- CPU is sufficient for running the Streamlit demo with pre-trained weights

### 3.2 Python Dependencies (pip)

```bash
pip install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
  transformers \
  accelerate \
  scikit-learn \
  pandas \
  numpy \
  pillow \
  matplotlib \
  tqdm \
  streamlit
```

> **Notes:**
> - Adjust the `--index-url` CUDA version (`cu121`) to match your system, or drop it entirely for a CPU-only install.
> - `transformers` automatically downloads the required pretrained models (BERT, ViT, CLIP) from Hugging Face on first use — ensure you have an internet connection.

### 3.3 Using `uv` (optional, faster installs)

If you have [`uv`](https://github.com/astral-sh/uv) installed:

```bash
uv pip install torch torchvision torchaudio transformers accelerate scikit-learn pandas numpy pillow matplotlib tqdm streamlit
```

### 3.4 Colab-Specific Notes

On Colab, also available are:

- `google.colab` and `drive.mount` for Google Drive access (pre-installed).
- `tqdm` and `matplotlib` (pre-installed).

No additional steps are needed on Colab beyond running the notebook cells.

---

## 4. Training Notebooks

All training logic is in Jupyter notebooks so you can read and run it step by step.

### 4.1 BERT + ResNet-50 (`bertandrestnet.ipynb`)

This notebook combines BERT text features with ResNet-50 image features.

**Pipeline overview:**

1. **Data loading** — Mount Drive, load `clean_df.csv`, verify data shape `(18470, 13)` and image files.
2. **Dataset and splits** — Map `6_way_label` → integers `0..5`; split into train/val/test.
3. **Preprocessing**
   - Text: `BertTokenizer` (`bert-base-uncased`), `max_length=80`.
   - Images: resize to `(256, 256)`, convert to tensor, normalize with ImageNet stats.
4. **Model: `BERTResNetClassifier`**
   - *Image branch:* `resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)` → dropout → `Linear(1000 → 6)`.
   - *Text branch:* `BertModel.from_pretrained("bert-base-uncased")` → `[CLS]` pooled output → dropout → `Linear(768 → 6)`.
   - *Fusion:* element-wise `torch.max` of text and image logits → CrossEntropy loss.
5. **Training loop** — Standard PyTorch training with AdamW optimizer and optional LR scheduler; tracks accuracy and F1 on validation.
6. **Checkpoint** — Saves best model as `m1.pth` (used by the ensemble `app.py`).

### 4.2 BERT + ViT (`bert_ViT_v2.ipynb`)

This notebook uses a Vision Transformer instead of ResNet for the image branch.

**Key configuration (set at the top of the notebook):**

```python
CLEANDFPATH    = "..."                    # path to clean_df.csv
IMAGEDIR       = "..."                    # path to images/
SAVE_DIR       = ".../bert_vit_v2_2"     # where to save checkpoints & reports
BERT_NAME      = "bert-base-uncased"
VIT_NAME       = "google/vit-base-patch16-224-in21k"
MAX_LEN        = 80
BATCH_SIZE     = 16
NUM_EPOCHS     = 12
HEAD_LR        = 1e-4
BACKBONE_LR    = 5e-6
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.4
EARLY_STOPPING_PATIENCE = 4
```

**Steps:**

1. Load and inspect `clean_df.csv`.
2. Create train/val/test splits; save to `SAVE_DIR` as `train_split_v2_2.csv`, `val_split_v2_2.csv`, `test_split_v2_2.csv`.
3. Compute class weights to handle label imbalance; save as JSON.
4. Build dataset returning `input_ids`, `attention_mask` (BERT) and `pixel_values` (`ViTImageProcessor`).
5. Train `BERTViTClassifierV22` with mixed precision (`autocast` + `GradScaler`), cosine LR scheduler, and early stopping.
6. Save:
   - `bert_vit_v2_2_best_score.pth` — best validation checkpoint (used by `app.py`).
   - `bert_vit_v2_2_final.pth` — final epoch checkpoint.
   - Prediction CSVs and a text `classification_report`.

### 4.3 CLIP-Based Classifier (`CLIPv1.ipynb` and `CLIPv2_1.ipynb`)

The CLIP notebooks explore using a frozen CLIP backbone with a classification head.

**`CLIPv1.ipynb`** — Early experiments and evaluation tables, including prediction DataFrames with `image_id`, `text`, `true_label`, `pred_label`, and per-class probabilities.

**`CLIPv2_1.ipynb`** — Main training notebook:

```python
CLIP_NAME    = "openai/clip-vit-base-patch32"
NUM_CLASSES  = 6
```

- Builds dataloaders returning CLIP-compatible `input_ids`, `attention_mask`, and `pixel_values`.
- Defines `CLIPMultimodalClassifier`:
  - Loads `CLIPModel` from `transformers`.
  - Optionally freezes CLIP backbone weights.
  - Adds dropout (`dropout=0.2`) and a `Linear` classification head for 6 classes.
- Trains and evaluates using a shared `evaluate_model` function computing: loss, accuracy, weighted/macro precision, recall, F1, and a `classification_report`.
- Saves the best checkpoint as `clip_multimodal_bestv1.pth` (or `clip_multimodal_best.pth`).

**Final CLIP performance (from notebook output):**

| Metric              | Value  |
|---------------------|--------|
| Test Loss           | 1.2882 |
| Test Accuracy       | 0.8322 |
| Weighted Precision  | 0.8304 |
| Weighted Recall     | 0.8322 |
| Weighted F1         | 0.8300 |
| Macro F1            | 0.7386 |

---

## 5. Inference Demo (Streamlit Apps)

Both apps let you enter a headline and upload an image, then classify the pair into one of the six labels with confidence scores.

### 5.1 Ensemble App (`app.py` — Three Models)

Loads BERT+ResNet, BERT+ViT, and CLIP in parallel, shows individual predictions, and computes a majority vote.

**Required files in the working directory:**

```
Model.py
m1.pth                          # BERT + ResNet-50 checkpoint
bert_vit_v2_2_best_score.pth    # BERT + ViT checkpoint
clip_multimodal_bestv1.pth      # CLIP checkpoint
app.py
```

**Run:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**UI flow:**

1. Enter an **Article Headline** in the text field.
2. Upload a **JPEG or PNG image**.
3. Click **Predict**.
4. The app (using `@st.cache_resource` to load models only once):
   - Runs `predict_bert_resnet`, `predict_bert_vit`, and `predict_clip`.
   - Shows three result columns with the predicted label, confidence, and per-class probability expander.
   - Computes a majority vote; if ≥2/3 models agree, shows the final consensus label and its description.

If a checkpoint is missing or incompatible, the app catches the error and renders a readable error message in the UI.

### 5.2 Single-CLIP App (`app_clip_only.py`)

A minimal CLIP-only demo for simpler setups.

**Required files:**

```
Model.py
clip_multimodal_best.pth    # configure MODEL_PATH in the script if different
app_clip_only.py
```

**Run:**

```bash
streamlit run app_clip_only.py
```

The UI shows the predicted label, confidence percentage, a class description, and a per-class probability expander.

---

## 6. How to Reproduce Training

### 6.1 Data Preparation

1. Open `dataset_downloader.ipynb` in Colab.
2. Run `drive.mount('/content/drive')`.
3. Set your base path in the notebook (default: `Fake-news-detector/multimodal_only_samples/working`).
4. Run all cells to download images, verify them, and save `clean_df.csv`.

### 6.2 BERT + ResNet Training

1. Open `bertandrestnet.ipynb`.
2. Update `CLEANDFPATH` and `IMAGEDIR` if needed.
3. Switch to a GPU runtime in Colab.
4. Run all cells — saves `m1.pth` upon completion.

### 6.3 BERT + ViT Training

1. Open `bert_ViT_v2.ipynb`.
2. Update `CLEANDFPATH`, `IMAGEDIR`, and `SAVE_DIR`.
3. Run all cells — saves `bert_vit_v2_2_best_score.pth` and `bert_vit_v2_2_final.pth`.

### 6.4 CLIP Training

1. Open `CLIPv2_1.ipynb`.
2. Verify data paths and `CLIP_NAME`.
3. Run all cells — saves `clip_multimodal_bestv1.pth`.

After training, copy all three checkpoints into the same folder as `app.py`.

---

## 7. Installation Challenge — Quick Start

For another group installing this project during the lab, follow this condensed checklist:

```bash
# 1. Clone the repo
git clone <your-public-repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    transformers accelerate scikit-learn pandas numpy pillow matplotlib tqdm streamlit

# 3. Place the provided checkpoints in the repo root:
#    m1.pth
#    bert_vit_v2_2_best_score.pth
#    clip_multimodal_bestv1.pth

# 4. Run the demo
streamlit run app.py
```

Test with any headline + image — verify that all three models predict a label and the majority vote appears. If anything fails, contact the authors; we can fix issues on the fly during the session.

---

## 8. Configuration and Paths Reference

All hard-coded paths are grouped near the top of each file:

| File                   | Key Variables |
|------------------------|---------------|
| All training notebooks | `CLEANDFPATH`, `IMAGEDIR`, `SAVE_DIR`, `BEST_MODEL_PATH` |
| `bert_ViT_v2.ipynb`    | Also `BERT_NAME`, `VIT_NAME`, `BATCH_SIZE`, `NUM_EPOCHS`, `DROPOUT`, etc. |
| `CLIPv2_1.ipynb`       | `CLIP_NAME`, `NUM_CLASSES` |
| `app.py` (ensemble)    | Checkpoint filenames passed to `load_checkpoint_weights(...)` |
| `app_clip_only.py`     | `MODEL_PATH = "clip_multimodal_best.pth"` |

When moving to a new machine:

1. Place checkpoints next to `app.py` (or update the paths).
2. Update `CLEANDFPATH` and `IMAGEDIR` in the notebooks.
3. Run a quick cell to confirm data loads before starting full training.

---

## 9. License

This project is released under the **AGPL-3.0** license.

Make sure your public GitHub repository includes:

- A `LICENSE` file with the full AGPL-3.0 license text.
- This `README.md`.

Update this section if you choose a different OSI-approved license.

---

## 10. Authors

| Name       | Student ID |
|------------|------------|
| Author 1   | XXXXXXX    |
| Author 2   | XXXXXXX    |
| Author 3   | XXXXXXX    |

Replace with actual names and student IDs. Feel free to add GitHub profile links or contact emails if required by the course.
