# DSE-Prompt

Official PyTorch implementation of **DSE-Prompt**, a dual-state evidence-decoupled prompt-learning framework for zero-shot anomaly detection.

> The codebase is developed on top of [FAPrompt](https://github.com/guojiajeremy/FAPrompt) and [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP).  
> The Python environment follows the official FAPrompt implementation.

---

## Overview

DSE-Prompt extends fine-grained abnormality prompt learning with two state-specific evidence aggregation branches:

- a **normal-state branch**, which absorbs evidence from patches with low abnormal similarity;
- an **abnormal-state branch**, which selects high-confidence abnormal evidence using a dynamic threshold.

Each branch contains learnable visual anchors, positional embeddings, spatial attention, a prompt-dependent gate, and a residual modulation coefficient. The normal and abnormal prompts are therefore enhanced through different evidence-admission rules before image-level anomaly discrimination and pixel-level anomaly localization.

The current implementation includes:

- learnable normal and abnormal evidence anchors;
- spatial attention over CLIP patch features;
- gated residual prompt enhancement;
- dynamic abnormal-evidence threshold scheduling;
- normal-region exclusion based on abnormal similarity;
- fallback evidence selection with a penalty coefficient;
- abnormal-prompt diversity regularization;
- image-level global/local score fusion;
- pixel-level AUROC and AUPRO evaluation;
- image-level AUROC and AP evaluation.

---

## Method Components

### Dual Evidence Aggregation Modules

The implementation uses two independent `SCA_Module` instances:

```python
sca_module_norm = SCA_Module(
    embed_dim=768,
    num_anchors=num_norm_anchors,
    num_patches=1369
)

sca_module_abn = SCA_Module(
    embed_dim=768,
    num_anchors=num_abn_anchors,
    num_patches=1369
)
```

For each branch, learnable anchor queries attend to normalized CLIP patch features. The aggregated evidence is injected into the text features through a gated residual connection:

```text
enhanced_text = normalize(text_features + res_scale × gated_evidence)
```

### Normal-State Evidence Selection

The normal branch excludes patches whose similarity to abnormal prompts is higher than `norm_exclusion_thresh`. This reduces contamination of the normal prompt by suspicious regions.

### Abnormal-State Evidence Selection

The abnormal branch retains patches whose cosine similarity to an abnormal anchor is higher than the current threshold. During training, the threshold increases linearly from `thresh_start` to `thresh_end`.

When no patch satisfies the threshold, the most similar patch is selected as fallback evidence and its contribution is multiplied by `fallback_penalty`.

### Prompt Diversity Regularization

Pairwise cosine similarity among enhanced abnormal prompts is constrained to the interval:

```text
[div_sim_lower, div_sim_upper]
```

The regularization weight is gradually increased during the warm-up stage.

---

## Repository Structure

Before uploading the project to GitHub, organize the files as follows:

```text
DSE-Prompt/
├── AnomalyCLIP_lib/        # AnomalyCLIP/OpenCLIP implementation
├── dataset.py              # Dataset loader
├── FAPrompt.py             # Fine-grained prompt learner
├── train.py                # DSE-Prompt training
├── test.py                 # Evaluation
├── train.sh                # Training example
├── test.sh                 # Testing example
├── loss.py                 # Focal and Dice losses
├── metrics.py              # AUROC, AP, F1, and AUPRO
├── logger.py               # Logging utility
├── utils.py                # Image and mask transformations
├── visualization.py        # Heatmap visualization
├── requirements.txt
└── README.md
```

The uploaded files should be renamed as follows:

| Current filename | Recommended filename |
|---|---|
| `FAPrompt(1).py` | `FAPrompt.py` |
| `train(13).py` | `train.py` |
| `test(3).py` | `test.py` |
| `train(2).sh` | `train.sh` |
| `test(2).sh` | `test.sh` |
| `metrics(1).py` | `metrics.py` |
| `README(1).md` | `README.md` |

The repository must also contain `dataset.py` and the `AnomalyCLIP_lib/` directory because they are imported by the training and testing scripts.

---

## Environment

The runtime environment follows the official FAPrompt implementation.

### Hardware

- Single NVIDIA GeForce RTX 3090

### Main Dependencies

```text
Python
tqdm == 4.67.1
timm == 0.6.12
scikit-image == 0.19.2
scikit-learn == 1.0.2
scipy == 1.7.3
seaborn == 0.11.2
torch == 2.4.1
torchvision == 0.19.1
transformers == 4.31.0
```

The current code additionally uses:

```text
numpy
opencv-python
tabulate
setuptools
```

A corresponding installation command is:

```bash
pip install \
    tqdm==4.67.1 \
    timm==0.6.12 \
    scikit-image==0.19.2 \
    scikit-learn==1.0.2 \
    scipy==1.7.3 \
    seaborn==0.11.2 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    transformers==4.31.0 \
    numpy \
    opencv-python \
    tabulate \
    setuptools
```

Install the PyTorch build that matches the CUDA environment of your machine when necessary.

---

## Data Preparation

The dataset preparation procedure is the same as FAPrompt and AnomalyCLIP.

### Step 1: Download the datasets

Examples of supported anomaly-detection datasets include:

#### Industrial datasets

- MVTec AD
- VisA
- ELPV
- SDD
- AITEX
- BTAD
- DAGM
- DTD-Synthetic
- MPDD

#### Medical datasets

- BrainMRI
- HeadCT
- LAG
- Br35H
- CVC-ColonDB
- CVC-ClinicDB
- Kvasir
- Endo
- ISIC
- TN3K

### Step 2: Generate dataset JSON files

Generate the JSON annotation files using the same format adopted by AnomalyCLIP.

### Step 3: Set dataset paths

Do not keep private server paths in the public repository. Replace paths such as:

```text
/home/ubuntu/username/data/...
```

with your own local or server path when running the code.

Example directory:

```text
data/
├── mvtec_anomaly_detection/
├── visa_anomaly_detection/
├── DAGM_anomaly_detection/
├── br35_anomaly_detection/
└── ...
```

---

## Training

The main training script is `train.py`.

### Recommended Configuration

The paper configuration uses:

| Argument | Value |
|---|---:|
| Backbone | ViT-L/14@336px |
| Image size | 518 |
| Feature layers | 6, 12, 18, 24 |
| DPAM layer | 20 |
| Prompt depth | 9 |
| Normal context length | 12 |
| Deep text context length | 4 |
| Batch size | 8 |
| Epochs | 25 |
| Learning rate | 1e-4 |
| Random seed | 111 |
| Normal anchors | 1 |
| Abnormal anchors | 10 |
| Threshold start | 0.4 |
| Threshold end | 0.7 |
| Threshold warm-up | 5 epochs |
| Normal exclusion threshold | 0.6 |
| Fallback penalty | 0.1 |
| Global score weight | 0.5 |
| Diversity interval | [0.0, 0.2] |
| Gradient clipping | 1.0 |

### Command

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset mvtecad \
    --train_data_path /path/to/mvtec_anomaly_detection \
    --save_path ./checkpoints/trained_on_mvtecad \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --batch_size 8 \
    --epoch 25 \
    --learning_rate 0.0001 \
    --print_freq 1 \
    --save_freq 1 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --warmup_epochs 5 \
    --div_sim_lower 0.0 \
    --div_sim_upper 0.2 \
    --thresh_start 0.4 \
    --thresh_end 0.7 \
    --thresh_warmup_epochs 5 \
    --fallback_penalty 0.1 \
    --global_score_weight 0.5 \
    --norm_exclusion_thresh 0.6 \
    --num_norm_anchors 1 \
    --num_abn_anchors 10 \
    --seed 111
```

Alternatively:

```bash
bash train.sh
```

Before running `train.sh`, replace the dataset path, output path, and Python filename with the paths used on your machine.

### Checkpoints

A checkpoint is saved after every `save_freq` epochs and contains:

```text
prompt_learner
sca_module_norm
sca_module_abn
```

Example:

```text
checkpoints/
└── trained_on_mvtecad/
    ├── epoch_1.pth
    ├── epoch_2.pth
    └── epoch_25.pth
```

---

## Evaluation

The evaluation script reports:

- Pixel AUROC
- Pixel AUPRO
- Image AUROC
- Image AP

### Command

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset br35 \
    --data_path /path/to/Br35H \
    --save_path ./results/trained_on_mvtecad/br35 \
    --checkpoint_path ./checkpoints/trained_on_mvtecad/epoch_25.pth \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4 \
    --sigma 10 \
    --sca_threshold 0.7 \
    --fallback_penalty 0.1 \
    --global_score_weight 0.5 \
    --norm_exclusion_thresh 0.6 \
    --num_norm_anchors 1 \
    --num_abn_anchors 10 \
    --seed 111
```

Alternatively:

```bash
bash test.sh
```

Before running `test.sh`, update:

- the testing dataset name;
- the dataset path;
- the checkpoint path;
- the output directory;
- the Python filename.

The anchor numbers and prompt configuration used during testing must match those stored in the checkpoint.

For the final checkpoint trained with a threshold schedule ending at `0.7`, use:

```bash
--sca_threshold 0.7
```

---

## Output

The logger creates:

```text
save_path/
└── log.txt
```

The evaluation results are printed and saved as a Markdown-style table:

```text
| Objects | Pixel_AUROC | Pixel_AUPRO | Image_AUROC | Image_AP |
|---------|-------------|-------------|-------------|----------|
| ...     | ...         | ...         | ...         | ...      |
| Mean    | ...         | ...         | ...         | ...      |
```

---

## Visualization

`visualization.py` overlays the anomaly score map on the original image using the JET color map.

The visualization utility expects:

- input image paths;
- predicted anomaly maps;
- image size;
- output directory;
- class names.

The current output path is constructed inside the function. Before public release, replace the hard-coded subdirectory:

```python
'imgs/btad2'
```

with a general path such as:

```python
'visualizations'
```

---

## Important Reproducibility Notes

1. `train.py` defines `t_n_ctx=10` by default, while the provided training shell script explicitly uses `t_n_ctx=4`. The paper configuration should therefore be run with:

   ```bash
   --t_n_ctx 4
   ```

2. `train.py` defines `epoch=15` by default, while the provided shell script uses 25 epochs. To reproduce the paper setting, explicitly use:

   ```bash
   --epoch 25
   ```

3. The final training threshold is `0.7`. Evaluation should use the same final threshold unless a different checkpoint-specific value is intended.

4. The testing script currently defines `global_score_weight=0.4` by default, whereas the training script and paper setting use `0.5`. For consistent reproduction, explicitly pass:

   ```bash
   --global_score_weight 0.5
   ```

5. The number of normal and abnormal anchors used during evaluation must match the corresponding checkpoint.

6. Do not upload datasets, checkpoints, private server paths, API keys, or user credentials to GitHub.

---

## Acknowledgements

This implementation is based on:

- **FAPrompt**: Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection
- **AnomalyCLIP**: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection
- **CLIP/OpenCLIP**

We thank the authors for releasing their code.

---



## License

Please check the licenses of FAPrompt, AnomalyCLIP, and all other upstream dependencies before assigning a license to this repository.

Until a compatible license is added, the repository should not be treated as granting unrestricted permission to copy, modify, or redistribute the code.
