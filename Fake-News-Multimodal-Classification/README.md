# Multimodal Fake News Detection on Fakeddit

This repository contains our Advanced AI project for multimodal fake news detection on the Fakeddit dataset.

The project implements and compares three multimodal architectures that jointly use news headlines and images:

- **BERT + ResNet-50** (text + CNN image encoder)
- **BERT + ViT** (text + Vision Transformer)
- **CLIP-based multimodal classifier** (frozen CLIP backbone + classification head)

A Streamlit demo app lets you load trained weights and classify new (headline, image) pairs into six Fakeddit-style labels.

---

## Task and labels

Each sample has a headline (`clean_title`) and an image. The target is one of **six** Fakeddit categories:


| Class ID | Label Name           | Description |
|----------|----------------------|-------------|
| 0        | **TRUE**                  | Factually accurate content based on verified information |
| 1        | **SATIRE**              | Humorous or satirical content, not meant to be taken literally |
| 2        | **FALSE CONNECTION**     | Headlines or visuals that do not accurately reflect the article content |
| 3        | **IMPOSTER CONTENT**    | Content impersonating genuine sources |
| 4        | **MANIPULATED CONTENT**  | Content that has been altered or edited to mislead |
| 5        | **MISLEADING CONTENT**   | Selective framing or missing context |

**Decision rule (all models):** predict the class with the **highest logit** among the six outputs.
### 1.2 Data Format

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

### 1.3 Dataset Creation Notebook

`dataset_downloader.ipynb` performs:

1. Mounts Google Drive: `drive.mount('/content/drive')`.
2. Loads the raw Fakeddit CSV.
3. Filters to the subset used in this project (posts with images, 6-way labels).
4. Downloads all images to `IMAGEDIR` using `image_url` and `id`.
5. Verifies that expected images exist and logs any download failures.
6. Saves the cleaned dataframe as `clean_df.csv` (shape `(18470, 13)`).
---

## Why three model families?

The goal was not only raw accuracy but understanding **how fusion behaves** under **class imbalance**, **long training**, and **limited Colab GPU** memory:

- **Late fusion on logits** is modular and easy to interpret.
- **Feature-level concatenation** is more expressive but heavier and more sensitive to optimization.
- **CLIP** starts from a **pretrained joint vision–language space**, so alignment is not learned from scratch; richer fusion over embeddings can outperform simpler stacks when training is tuned.

---

## Model 1: BERT + ResNet-50 (`bertandrestnet.ipynb`)

| Component | Choice |
|-----------|--------|
| Text | `bert-base-uncased`, `[CLS]` pooling → linear layer to 6 logits |
| Image | ImageNet-pretrained **ResNet-50** → dropout → linear to 6 logits |
| Fusion | **Element-wise `max`** over the two 6-dimensional logit vectors (per class, keep the stronger modality) |
| Loss | Weighted `CrossEntropyLoss` (class weights for imbalance) |

**Reported test performance (notebook output):** ~**76.34%** accuracy; precision/recall ~0.763.

**Properties:** Simple, interpretable late fusion; each branch can be reasoned about separately.

---

## Model 2: BERT + ViT (`bert_ViT_v2.ipynb`)


| Component | Choice |
|-----------|--------|
| Text | `bert-base-uncased`, `[CLS]` embedding |
| Image | `google/vit-base-patch16-224-in21k` **ViT**, `[CLS]` token embedding |
| Projection | Each modality → **512-d** (GELU, LayerNorm, dropout) |
| Fusion | **Concatenate** text + image → **1024-d** → MLP classifier → 6 logits |

**Reported test performance (notebook output):** ~**76.18%** accuracy; weighted F1 ~**0.761** (see saved test report in the notebook run).

**Properties:** Stronger image representation than a classic CNN; **higher GPU use** and tighter batch-size limits; learning rate and schedule matter more. The notebook uses a combined validation score (e.g. weighted + macro F1) for model selection and logs per-class metrics for rare classes.

---

## Model 3: CLIP multimodal classifier v1 (`CLIPv1.ipynb`)

| Component | Choice |
|-----------|--------|
| Backbone | `openai/clip-vit-base-patch32` (`CLIPModel` in Hugging Face) |
| Embeddings | L2-normalized **image** and **text** projection vectors |
| Fusion | Concatenate **`[img, text, |img−text|, img⊙text]`** → MLP (GELU, dropout) → 6 logits |

**Training schedule (notebook constants):**

1. **Stage 1 — frozen CLIP:** train **classifier head only** for **2** epochs (higher LR on the head).
2. **Stage 2 — full fine-tune:** **unfreeze entire CLIP** for up to **6** more epochs with smaller learning rate (early stopping may cut this short).

**Reported test performance (notebook output):** **76.72%** accuracy; **weighted F1 ~0.7826** (best among the three baseline variants in this project narrative).

**Conceptual progression:** ResNet uses **late logit fusion**; ViT uses **early feature concatenation**; CLIP v1 adds **interaction-aware** features (difference and product) on top of a **shared pretrained** embedding space.

---

## Model 4: CLIP multimodal classifier v2 (`CLIPv2_1.ipynb`)

CLIP v2 refines the setup for **imbalance** and **generalization**. Highlights from the notebook:

- **Offline augmentation** for underrepresented classes (e.g. 3, 4, 5): `RandomResizedCrop`, horizontal flip, `ColorJitter`, small `RandomRotation`; augmented images saved and merged into training metadata (`is_augmented` flag).
- **Balanced class weights** via `sklearn.utils.class_weight.compute_class_weight`.
- **`CrossEntropyLoss` with label smoothing** (0.1 in the notebook).
- **`WeightedRandomSampler`** for training batches.
- **Staged training:** **2** epochs **head-only**, then **unfreeze top transformer blocks** (last **2** vision + **2** text layers by default) for **top-layer** fine-tuning with reduced LRs—not full-model unfreeze from step one like v1’s second stage.
- Classifier adds **LayerNorm** on the fused 4×projection-dim vector (see `Fakeddit-WebApp/CLIPv2/Model.py`).

**Reported test performance (notebook output):** **83.22%** accuracy; weighted F1 ~**0.83**; macro F1 ~**0.74**.

---

## Results summary (from notebook runs)

| Notebook | Fusion idea | Test accuracy (approx.) | Notes |
|----------|-------------|-------------------------|--------|
| `bertandrestnet.ipynb` | Max over logits | **76.34%** | Baseline dual-encoder |
| `bert_ViT_v2.ipynb` | Concat 512+512 | **76.18%** | Weighted F1 ~0.761 |
| `CLIPv1.ipynb` | CLIP + diff/product | **76.72%** | Weighted F1 ~0.783 |
| `CLIPv2_1.ipynb` | Same fusion + aug / weights / staged top layers | **83.22%** | Strongest reported |

*Exact splits and seeds are defined inside each notebook; reproduce by running the full pipeline.*

---

## Repository layout (research + demos)

```
Fake-News-Multimodal-Classification/
├── bertandrestnet.ipynb      # BERT + ResNet, max fusion
├── bert_ViT_v2.ipynb         # BERT + ViT, concat fusion
├── CLIPv1.ipynb              # CLIP fusion v1, two-stage full FT
├── CLIPv2_1.ipynb            # CLIP fusion v2, augmentation + class weights + staged top layers
├── dataset_downloader.ipynb  # Helpers to prepare Fakeddit TSV + images (Colab-oriented)
├── data/                     # Local data layout (TSV splits, images) when present
├── requirements.txt          # Minimal pinned deps (expand locally, see below)
└── Fakeddit-WebApp/
    ├── RESVITCLIPv1/         # One Streamlit app: all three baselines + majority vote
    │   ├── app.py
    │   ├── Model.py          
    │   └── *.pth             # Place checkpoints here (not always in git)
    └── CLIPv2/               # Streamlit app: CLIP v2 only
        ├── app.py
        ├── Model.py          # CLIP V2 (data aug + freeze backbones)
        └── clip_multimodal_best.pth
```

---

## Dependencies

Training notebooks use **PyTorch**, **transformers**, **torchvision**, **Pillow**, **pandas**, **scikit-learn**, and (in `dataset_downloader.ipynb`) Colab-specific paths. The root `requirements.txt` currently lists `scikit-learn>=1.3`; for local runs, install at least:

```bash
pip install torch torchvision transformers streamlit pillow pandas scikit-learn accelerate
```

Match **CUDA** builds of PyTorch to your GPU if applicable.

---

## Data

- **Paper / dataset:** [Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://aclanthology.org/2020.lrec-1.755/).
- Use **`dataset_downloader.ipynb`** to download or organize multimodal TSVs and images (paths in the notebook target Google Drive when run in Colab; adjust for your machine).
- This repo may contain **`data/working/splits/`** CSVs when you have created train/val/test splits locally.

---

## Streamlit apps

### 1. Three-model demo + majority vote — `Fakeddit-WebApp/RESVITCLIPv1/`

Runs **BERT+ResNet**, **BERT+ViT**, and **CLIP v1** side by side, then applies a **majority vote** over the three predicted class indices (if all disagree, the app reports no majority).

**Expected checkpoint filenames** (same directory as `app.py`):

| File | Model |
|------|--------|
| `m1.pth` | BERT + ResNet |
| `bert_vit_v2_2_best_score.pth` | BERT + ViT |
| `clip_multimodal_bestv1.pth` | CLIP v1-style multimodal classifier |

```bash
cd Fakeddit-WebApp/RESVITCLIPv1
streamlit run app.py
```

### 2. CLIP v2 only — `Fakeddit-WebApp/CLIPv2/`

Single-model UI using **`clip_multimodal_best.pth`** and `CLIPMultimodalClassifier` from this folder’s `Model.py`.

```bash
cd Fakeddit-WebApp/CLIPv2
streamlit run app.py
```

**Weights:** Pretrained `.pth` files are often **large** and may be excluded from Git; copy your trained checkpoints into the folders above or use [Git LFS](https://git-lfs.com/) if you publish them.

First run will download Hugging Face models (**BERT**, **ViT**, **CLIP**) unless already cached.

---

## Reproducibility notes

- Notebooks assume **GPU** (Colab or local CUDA) for reasonable runtime; ViT and CLIP are the most memory-hungry.
- **Splits:** `bert_ViT_v2.ipynb` and `CLIPv2_1.ipynb` use stratified `train_test_split` (e.g. 80% train, then half of the remainder for val/test); align with the notebook you treat as canonical if you compare numbers across experiments.
- **Class imbalance:** BERT+ResNet uses weighted loss; CLIP v2 adds sampling, augmentation, and label smoothing—compare confusions and per-class F1 in the notebooks, not only accuracy.

---

## License and attribution

Use of the **Fakeddit** dataset is subject to its original license and terms. Cite the Fakeddit paper when using this benchmark. Model checkpoints and code here are for research and education unless you add your own license file.

---

## Acknowledgments

- **Fakeddit** authors for the multimodal benchmark.
- **Hugging Face** `transformers` for BERT, ViT, and CLIP implementations.
- Earlier README iterations referenced a single-streamlit layout; this document reflects the **current** multi-notebook and **dual Streamlit** structure.
