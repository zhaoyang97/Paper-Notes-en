---
title: >-
  [Paper Note] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training
description: >-
  [AAAI 2026][Model Compression][Dataset Distillation] This paper proposes EEG-DLite, a dataset distillation framework that combines self-supervised encoding, outlier filtering, and diversity sampling to compress a 2,500-hour EEG dataset to just 5% of its original size, achieving performance comparable to or exceeding full-data pretraining while reducing GPU pretraining time from 30 hours to 2 hours.
tags:
  - AAAI 2026
  - Model Compression
  - Dataset Distillation
  - EEG Foundation Model
  - Self-Supervised Learning
  - Coreset Selection
  - Pretraining Efficiency
date: 2026-05-08
content_hash: 2ab1c868c3eff5da
---

# EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training

**Conference**: AAAI 2026
**arXiv**: [2512.12210](https://arxiv.org/abs/2512.12210)
**Code**: [github](https://github.com/t170815518/EEG-DLite)
**Area**: Model Compression
**Keywords**: Dataset Distillation, EEG Foundation Model, Self-Supervised Learning, Coreset Selection, Pretraining Efficiency

## TL;DR
This paper proposes EEG-DLite, a dataset distillation framework that combines self-supervised encoding, outlier filtering, and diversity sampling to compress a 2,500-hour EEG dataset to just 5% of its original size, achieving performance comparable to or exceeding full-data pretraining while reducing GPU pretraining time from 30 hours to 2 hours.

## Background & Motivation

### State of the Field
Large-scale EEG foundation models (e.g., LaBraM) have demonstrated strong generalization across diverse downstream tasks—including emotion recognition, motor imagery, and clinical classification—through self-supervised pretraining on large volumes of unlabeled EEG data. Such models are typically built on Transformer architectures with parameter counts exceeding 400 million, pretrained on datasets of 2,500+ hours.

### Limitations of Prior Work

**Prohibitive training cost**: Pretraining requires substantial GPU time (30 hours on 4×RTX 4090) and storage resources, limiting the feasibility of hyperparameter tuning and architecture search.

**EEG signal characteristics**: EEG signals have extremely low signal-to-noise ratios (SNR) and are highly susceptible to artifacts from eye movements and muscle activity; temporally adjacent segments are also highly redundant.

**Data quality overlooked**: Existing work focuses on architectural innovation and transfer learning, with little investigation into how the composition and quality of pretraining data affect model generalization.

### Root Cause
Large-scale EEG datasets contain substantial amounts of noisy and redundant samples, yet existing foundation models still require pretraining on the full dataset, leading to significant resource waste.

### Starting Point
Inspired by dataset distillation methods in computer vision, the paper designs a distillation strategy tailored to the low-SNR and high-redundancy characteristics of EEG signals. The core idea is to first compress high-dimensional EEG data into a low-dimensional latent space using a self-supervised autoencoder, then perform efficient denoising and redundancy removal within that space.

## Method

### Overall Architecture
EEG-DLite is a three-stage dataset distillation pipeline decoupled from any specific foundation model architecture:
1. **Multi-view Neural Compressor**: A self-supervised autoencoder that encodes EEG segments into compact low-dimensional representations.
2. **Outlier Removal**: Noise/artifact samples are filtered in the latent space using HBOS.
3. **Diversity Sampling**: A k-center greedy algorithm selects the most representative subset.

### Key Designs

1. **Multi-view Neural Compressor**:

    - **Function**: Compresses high-dimensional EEG signals $X \in \mathbb{R}^{C \times T}$ into 64-dimensional latent representations $\mathbf{z}$.
    - **Mechanism**: For each EEG segment, three views are simultaneously computed—raw signal, FFT magnitude, and FFT phase—concatenated and processed through a CNN to extract patch-level tokens, which are then fed into a Transformer encoder to capture global dependencies.
    - **Encoder**: 6 self-attention layers, 8 attention heads; **Decoder**: 2-layer Transformer + MLP.
    - **Design Motivation**: Direct sample selection in the raw EEG space is overly sensitive to noise and computationally expensive due to high dimensionality; spectral views provide complementary signal quality information.

2. **Self-Supervised Training Objective**:

    - **Reconstruction Loss**: $\mathcal{L}_{Rec} = \sum_{i=1}^{L}(\mathbf{x}'_i - \mathbf{x}_i)^2$, ensuring accurate encoding of neural signal content.
    - **Inter-sample Discriminative Contrastive Loss (IDC)**: $\mathcal{L}_{IDC}$ penalizes high cosine similarity between tokens from different samples within a batch, encouraging feature diversity.
    - **Joint Objective**: $\mathcal{L} = \mathcal{L}_{Rec} + \beta \cdot \mathcal{L}_{IDC}$, where $\beta=0.0001$.
    - **Design Motivation**: The IDC loss encourages the encoder to learn more discriminative representations, facilitating subsequent outlier detection and diversity sampling.

3. **Outlier Sample Removal**:

    - **Function**: Computes an OOD score for each sample using HBOS and removes the top $\tau\%$ highest-scoring samples.
    - **OOD Score**: $\text{OOD}(X) = \sum_{i=1}^{d} \log \frac{1}{p_i(x_i) + \alpha}$, where $p_i$ is the histogram-based probability of the $i$-th feature dimension.
    - **Design Motivation**: EEG data frequently contains artifact-contaminated low-quality segments that degrade subsequent diversity sampling; anomaly detection in the latent space is more robust than in the raw signal space.

4. **Diversity Sampling**:

    - **Function**: Selects $\eta\%$ most representative samples from the denoised data.
    - **Core Formulation**: $\min_{\boldsymbol{\mu} \subset \mathcal{Z}'} \max_{\mathbf{z} \in \mathcal{Z}'} \min_{k \in \mathcal{K}} \|\mathbf{z} - \boldsymbol{\mu}_k\|_2^2$ (k-center problem).
    - **Implementation**: A greedy approximation algorithm iteratively selects the sample farthest from the current set, with time complexity $\mathcal{O}(k \times N \times d)$.
    - **Design Motivation**: Ensures the selected subset covers the distributional diversity of the original data, as opposed to naive random sampling.

### Loss & Training
- Compressor training: Adam optimizer, learning rate 0.001, 50 epochs, gradient clipping with max norm 5.0.
- Learning rate schedule: decay by 0.5 every 10 epochs.
- EEG segments are split into 20 non-overlapping patches.

## Key Experimental Results

### Main Results

Experiments are conducted based on the LaBraM-base architecture, evaluated on 4 downstream tasks across various distillation methods and ratios:

| Dataset | Metric | EEG-DLite (5%) | Random (5%) | Full (100%) | Notes |
|--------|------|----------------|-------------|-------------|------|
| SEED-V | Accuracy | **38.6** | 34.6 | 41.0 | 5-class emotion recognition |
| SEED-V | F1 | **38.9** | 34.9 | 41.2 | EEG-DLite greatly outperforms Random |
| MoBI | PCC | **0.550** | 0.530 | 0.538 | Regression task, surpasses full data |
| MoBI | R² | **0.283** | 0.260 | 0.288 | Near full-data performance |
| TUEV | Balanced Acc | **62.9** | 62.3 | 64.1 | 6-class EEG event detection |
| TUEV | F1 | **80.7** | 79.3 | 83.1 | Using only 5% of data |
| TUAB | Balanced Acc | **80.7** | 80.7 | 81.4 | Binary normal/abnormal EEG |
| TUAB | AUROC | **90.3** | 90.0 | 90.2 | Surpasses full dataset |

**Core Finding**: Pretraining on just 5% of the distilled data achieves performance close to or exceeding full-data training.

### Ablation Study

| Configuration | SEED-V Acc (η=5%) | MoBI PCC (η=5%) | Notes |
|------|-------------------|-----------------|------|
| Random baseline | 34.6 | 0.530 | Lower bound |
| PCA + Diversity Sampling | 31.0 | 0.534 | PCA inferior to SSL |
| M3D (generative) | 26.9 | 0.465 | Generative approach fails |
| EEG-DLite (τ=0, no OOD removal) | — | — | OOD removal is beneficial |
| **EEG-DLite (full)** | **38.6** | **0.550** | All components optimal |

**OOD Removal Ablation** (SEED dataset, EEGNet supervised learning):

| Configuration | η(%) | Acc | F1 | κ |
|------|-------|-----|-----|---|
| Full method τ=0 | 25 | 54.6 | 55.1 | 29.1 |
| Full method τ=1% | 25 | **55.3** | **55.7** | **31.3** |
| Full Data | 100 | 54.6 | 55.4 | 30.8 |

### Key Findings
1. **5% is sufficient**: Using only 5% of the data achieves full-dataset performance levels, indicating substantial redundancy in large-scale EEG corpora.
2. **Generative methods fail**: M3D-generated synthetic EEG samples exhibit severe quality issues (unnatural plateaus and blocky patterns), underperforming random sampling at all ratios.
3. **SSL outperforms PCA**: Self-supervised representations are more stable and discriminative than PCA-reduced features.
4. **Distillation can surpass full data**: On TUEV, MoBI, and TUAB, models pretrained on the distilled subset **exceed** those trained on the full dataset.
5. **Increased subject-level variance**: After diversity sampling, the per-subject contribution ratios vary significantly, reflecting individual differences in EEG signals.
6. **Substantial time reduction**: GPU pretraining time decreases from 30 hours to 2 hours (4×RTX 4090).

## Highlights & Insights
1. **First dataset distillation work for EEG foundation models**: Addresses a gap in research on pretraining data efficiency for physiological signals.
2. **Elegant framework design**: The three-stage pipeline (compress → denoise → dereplicate) is logically coherent and decoupled from downstream foundation model architectures.
3. **Multi-view encoding**: Jointly leverages time-domain and frequency-domain information, fully exploiting the spectral characteristics of EEG signals.
4. **High practical value**: 5% data → 15× training speedup, with significant implications for resource-constrained scenarios.
5. **Counter-intuitive finding**: A carefully selected small dataset can yield a better-performing model than training on the full dataset.

## Limitations & Future Work
1. Validation is limited to a single foundation model (LaBraM); extension to other EEG architectures is needed.
2. The distillation process itself requires training an autoencoder on the full dataset, incurring non-trivial upfront overhead.
3. Subject-aware sampling strategies remain unexplored and may further improve cross-subject generalization.
4. The effect of different pretraining objectives (e.g., contrastive learning vs. masked prediction) on distilled subset selection has not been investigated.
5. The failure of generative methods on EEG warrants deeper analysis and purpose-built designs.

## Related Work & Insights
- **Coreset Selection** (Sener & Savarese 2018): The k-center greedy algorithm is a classical method, adapted here to the EEG latent space.
- **M3D** (Zhang et al. 2024): A lightweight generative distillation method from computer vision that completely fails on EEG, highlighting fundamental differences between physiological and visual data.
- **LaBraM** (Jiang et al. 2024): The baseline foundation model used in this work, which first demonstrated the effectiveness of large-scale EEG pretraining.
- **Inspiration**: The proposed framework may generalize to pretraining data optimization for other physiological signals (EMG, ECG).

## Rating
- Novelty: ⭐⭐⭐⭐ (First EEG dataset distillation work, though individual components are not entirely novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 downstream tasks, multiple distillation ratios, comprehensive ablations and comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (High practical impact: 15× training speedup with no performance degradation)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)
- [\[NeurIPS 2025\] BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](../../NeurIPS2025/model_compression/barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](../../ICLR2026/model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)

<!-- RELATED:END -->
