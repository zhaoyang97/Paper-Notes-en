---
title: >-
  [Paper Note] MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications
description: >-
  [CVPR 2026][Self-Supervised Learning][Mars remote sensing] MOMO is the first foundation model for Mars remote sensing. It independently pre-trains MAE on three Mars sensors (HiRISE/CTX/THEMIS) and proposes an Equal Valid…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Mars remote sensing"
  - "foundation model"
  - "model merging"
  - "checkpoint selection strategy"
  - "multi-sensor"
date: 2026-05-08
content_hash: 53117340c645ffa2
---

# MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications

**Conference**: CVPR 2026
**arXiv**: [2604.02719](https://arxiv.org/abs/2604.02719)
**Code**: [https://github.com/kerner-lab/MOMO](https://github.com/kerner-lab/MOMO)
**Area**: Self-Supervised Learning
**Keywords**: Mars remote sensing, foundation model, model merging, checkpoint selection strategy, multi-sensor

## TL;DR

MOMO is the first foundation model for Mars remote sensing. It independently pre-trains MAE on three Mars sensors (HiRISE/CTX/THEMIS) and proposes an Equal Validation Loss (EVL) checkpoint selection strategy for model merging, outperforming ImageNet pre-training and Earth observation foundation models across 9 downstream tasks in Mars-Bench.

## Background & Motivation

**Background**: Foundation models in Earth observation (EO) have exceeded 150 models (SatMAE, CROMA, Prithvi, etc.), achieving broad success in applications such as food security and disaster response. Mars orbital satellites similarly collect systematic observations of planetary surfaces, yet no foundation model specifically designed for Mars remote sensing has existed. Researchers continue to rely on ImageNet pre-trained models for Mars-related tasks.

**Limitations of Prior Work**: (1) EO foundation models cannot be directly transferred to Mars: differences in atmospheric conditions, illumination, surface materials, and sensor characteristics lead to large distribution gaps. (2) Mars sensors vary enormously: HiRISE at 0.25 m/pixel, CTX at 5 m/pixel, and THEMIS at 100 m/pixel; spatial coverage also differs significantly (CTX covers ~100%, HiRISE less than 3%). (3) Conventional multi-sensor fusion approaches (e.g., channel stacking, heterogeneous joint training) are infeasible or non-scalable for Mars data.

**Key Challenge**: Mars multi-sensor data spans a 400× range in spatial resolution (0.25 m to 100 m), exhibits vast differences in coverage, and lacks spatiotemporally aligned co-observations. Conventional channel stacking or joint training either requires data alignment or necessitates full retraining when new sensors are added.

**Goal**: How to efficiently build a unified Mars multi-sensor foundation model that surpasses ImageNet and EO pre-training across diverse Mars remote sensing tasks.

**Key Insight**: Rather than mixing heterogeneous data directly, each sensor's MAE is trained independently, followed by parameter merging via task arithmetic. The key insight is that merging at the final checkpoint may be unstable due to differing training trajectories across sensors. The EVL strategy is proposed to align training stages prior to merging.

**Core Idea**: Independent MAE pre-training per sensor + validation-loss-aligned checkpoint selection strategy + task arithmetic model merging = the first foundation model for Mars remote sensing.

## Method

### Overall Architecture

MOMO is constructed in three steps: (1) preparing approximately 12 million high-quality Mars image samples (~4 million per sensor), (2) independently pre-training an MAE model for each sensor, and (3) selecting the optimal checkpoint combination via the EVL strategy and merging into a unified model via task arithmetic. This unified model is then fine-tuned and evaluated on 9 downstream tasks in Mars-Bench.

### Key Designs

1. **Enhanced Reconstruction Loss**:

    - **Function**: Enables MAE pre-training to learn richer, structured representations.
    - **Mechanism**: Standard MAE uses MSE as the reconstruction objective, which only focuses on pixel-level intensity matching and is insensitive to higher-order spatial features (shapes, boundary continuity), causing the model to recover color textures but fail to reconstruct critical geomorphic features (e.g., precise crater shapes). MOMO introduces a composite loss $\mathcal{L}_\text{total} = \lambda_1\mathcal{L}_\text{MSE} + \lambda_2\mathcal{L}_\text{SSIM} + \lambda_3\mathcal{L}_\text{LPIPS} + \lambda_4\mathcal{L}_\text{grad}$, where LPIPS provides perceptual-level constraints, SSIM ensures structural consistency, and the gradient loss $\mathcal{L}_\text{grad}$ penalizes differences between predicted and ground-truth image gradients in the horizontal and vertical directions to enhance spatial smoothness.
    - **Design Motivation**: Mars remote sensing tasks (especially segmentation) require precise boundary and shape recognition; representations learned with pure MSE are limited in these downstream tasks.

2. **Equal Validation Loss (EVL) Checkpoint Selection Strategy**:

    - **Function**: Selects the most compatible checkpoint combination across sensor models prior to merging.
    - **Mechanism**: During training, validation loss $\mathcal{L}_i^{(e)}$ is recorded and checkpoints are saved approximately every 100K samples. EVL first identifies all "loss-aligned" epoch combinations $\mathbf{t_c} = (e^1, ..., e^n)$ such that the validation loss difference between any two sensors satisfies $\Delta_{ij} = |\mathcal{L}_i^{(e_a^i)} - \mathcal{L}_j^{(e_b^j)}| \leq \epsilon$. Among all qualifying combinations, the one minimizing the average distance to each sensor's early-stopping epoch is selected: $\mathbf{t_c}^\star = \min_{\mathbf{t_c} \in \mathcal{E}_\text{EVL}} \bar{D}(\mathbf{t_c})$, where $\bar{D} = \frac{1}{n}\sum_i |e^i - s_{es}^i|$.
    - **Design Motivation**: Large distributional differences across sensors lead to divergent training trajectories. Merging at the final checkpoint or at each sensor's individual early-stopping point may cause some models to be overfitted and others underfitted. EVL ensures that all models are at a comparable convergence level and close to their respective optimal generalization points at the time of merging, reducing the risk of unstable fusion.

3. **Task Arithmetic Model Merging**:

    - **Function**: Combines multiple sensor-specific models into a single unified model.
    - **Mechanism**: After selecting optimal checkpoints, the sensor-specific model parameters $\{\theta_i^{(e_\star^i)}\}$ are extracted and directly merged via task arithmetic addition: $\text{MOMO} = \mathcal{T}(\theta_1^{(e_\star^1)}, ..., \theta_n^{(e_\star^n)})$.
    - **Design Motivation**: Compared to data-merge training, model merging is more scalable — adding a new sensor requires only training that sensor's model and merging it, without full retraining. Loss landscape visualization shows that EVL-selected models are closer in weight space, residing within the same loss basin.

### Loss & Training

Each sensor is trained independently for 5 epochs over approximately 4 million samples. A HEALPix-based train/validation split is used to prevent data leakage. Pre-training data undergoes quality filtering via SSIM and noise estimation (threshold 0.4), with stratified sampling by Martian geological units to ensure geographic representativeness.

## Key Experimental Results

### Main Results

| Model | AtmosDust | DoMars16k | Frost | Landmark | Boulder | ConeQuest | Crater Binary | Crater Multi | MMLS | Avg. Rank |
|-------|-----------|-----------|-------|----------|---------|-----------|---------------|-------------|------|-----------|
| Scratch | 0.94 | 0.73 | 0.95 | 0.79 | 0.07 | 0.52 | 0.37 | 0.05 | 0.50 | 4.11 |
| ImageNet | 0.92 | **0.91** | 0.97 | **0.92** | 0.16 | 0.70 | 0.55 | 0.11 | 0.57 | 2.33 |
| SatMAE | 0.96 | 0.93 | 0.97 | 0.92 | 0.05 | 0.68 | 0.46 | 0.04 | 0.32 | 3.00 |
| **MOMO** | **0.96** | 0.92 | **0.98** | 0.91 | **0.20** | **0.71** | **0.54** | **0.12** | **0.57** | **1.67** |

Classification tasks report average F1-score; segmentation tasks report mIoU. MOMO achieves an average rank of 1.67, significantly outperforming all baselines.

### Ablation Study

| Checkpoint Strategy | Boulder | ConeQuest | Crater Binary |
|--------------------|---------|-----------|---------------|
| Early Stopping (ES) | 0.12 | 0.68 | 0.50 |
| Last Epoch (LE) | 0.18 | 0.70 | 0.50 |
| **EVL (Ours)** | **0.20** | **0.71** | **0.54** |

EVL consistently outperforms ES and LE across all three segmentation tasks, with an average gain of approximately 2.5% mIoU.

### Key Findings

- **Segmentation vs. Classification**: MOMO demonstrates a clear advantage on segmentation tasks (~4% average mIoU gain over ImageNet) but only marginal improvement on classification (~1.25%), indicating that in-domain pre-training benefits fine-grained spatial understanding more significantly.
- **EO Models Underperform on Mars**: EO foundation models such as CROMA and Prithvi perform far worse than ImageNet on segmentation tasks, confirming a fundamental distribution gap between Earth and Mars data.
- **MOMO vs. Data Merge (DM)**: The DM approach collapses severely on the Frost task (F1=0.44 vs. MOMO's 0.98), demonstrating the instability of directly mixing heterogeneous data during training.
- **MOMO vs. Single-Sensor Pre-training**: Single-sensor pre-training performs reasonably well on its corresponding sensor tasks, but requires maintaining multiple models. MOMO handles all sensors with a single unified model while achieving on average 4.2% higher segmentation performance.
- **Loss Landscape Analysis**: EVL-selected checkpoints are more tightly clustered in weight space and reside in flatter loss basins, explaining their superior stability and generalization.

## Highlights & Insights

- **Validation-Loss-Aligned Checkpoint Selection**: This is a simple yet effective technical contribution that can be broadly applied to any scenario requiring the merging of models trained on heterogeneous data. The core idea is to "merge at comparable convergence levels" rather than naively merging at final or individually optimal checkpoints.
- **Model Merging as an Alternative to Data Merging**: This approach avoids the instability of heterogeneous joint training and naturally supports incremental extension (new sensors only require training and merging a new model), a design principle applicable to multi-sensor EO foundation models as well.
- **Enhanced MAE Loss**: In scenarios where structural features are critical — such as Martian geomorphology — a composite loss combining LPIPS, SSIM, and gradient terms substantially outperforms pure MSE, a finding applicable to any MAE pre-training context where spatial structure is important.

## Limitations & Future Work

- Computational constraints precluded comparison with additional model merging baselines (e.g., Git Re-Basin, alignment-based methods); more effective merging strategies may exist.
- The approach assumes linear mode connectivity between models, which may not hold under extremely large distributional differences.
- Only three sensors are used; the effectiveness of extending to additional sensors (e.g., the CRISM spectrometer) remains unknown.
- Improvements on classification tasks are limited (~1%), suggesting that ImageNet pre-training may already be sufficient for simpler tasks.

## Related Work & Insights

- **vs. SatMAE**: SatMAE performs reasonably on Mars classification tasks but collapses on segmentation (Boulder mIoU of only 0.05), demonstrating poor transferability from EO to Mars.
- **vs. Data Merge (DM)**: DM suffers severe collapse on certain tasks (Frost F1=0.44), whereas MOMO avoids the instability of heterogeneous data training through model merging.
- **vs. Purohit et al. (2023)**: Prior work pre-trained only on a single sensor (CTX) and evaluated on 2 tasks; MOMO provides comprehensive evaluation across three sensors and 9 tasks, and introduces a model merging framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First foundation model for Mars remote sensing; the EVL checkpoint selection strategy is a novel technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 9 Mars-Bench tasks with multiple baselines, checkpoint strategy ablation, and loss landscape analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and the methodology is systematically described, though some mathematical notation is somewhat redundant.
- **Value**: ⭐⭐⭐⭐ Pioneering introduction of foundation models to planetary science; open-sourced weights, code, and data represent a significant contribution to the planetary remote sensing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoBridge: A Semantic-Anchored Multi-View Foundation Model for Geo-Localization](geobridge_semantic-anchored_multi-view_foundation_model_for_geo-localization.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/self_supervised/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)

</div>

<!-- RELATED:END -->
