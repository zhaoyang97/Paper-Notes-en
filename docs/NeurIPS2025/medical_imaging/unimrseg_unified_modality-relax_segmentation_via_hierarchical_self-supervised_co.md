---
title: >-
  [Paper Note] UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation
description: >-
  [NeurIPS 2025][Medical Imaging][Missing modality segmentation] This paper proposes UniMRSeg, a unified missing-modality segmentation framework that employs a Hierarchical Self-Supervised Compensation (HSSC) mechanism—spanning input-level modality reconstruction, feature-level contrastive learning, and output-level consistency regularization—to achieve optimal average performance and minimal performance variance across all possible modality combinations using 100% shared parameters.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Missing modality segmentation
  - self-supervised compensation
  - contrastive learning
  - reverse attention adapter
  - unified parameters
date: 2026-05-08
content_hash: 513a36a41091c78f
---

# UniMRSeg: Unified Modality-Relax Segmentation via Hierarchical Self-Supervised Compensation

**Conference**: NeurIPS 2025
**arXiv**: [2509.16170](https://arxiv.org/abs/2509.16170)
**Code**: [GitHub](https://github.com/Xiaoqi-Zhao-DLUT/UniMRSeg)
**Area**: Medical Imaging / Multimodal Segmentation
**Keywords**: Missing modality segmentation, self-supervised compensation, contrastive learning, reverse attention adapter, unified parameters

## TL;DR

This paper proposes UniMRSeg, a unified missing-modality segmentation framework that employs a Hierarchical Self-Supervised Compensation (HSSC) mechanism—spanning input-level modality reconstruction, feature-level contrastive learning, and output-level consistency regularization—to achieve optimal average performance and minimal performance variance across all possible modality combinations using 100% shared parameters.

## Background & Motivation

Multimodal image segmentation is critical in applications such as autonomous driving, medical diagnosis, and robotics, yet real-world scenarios frequently involve incomplete modalities due to sensor failures, low data quality, or clinical constraints. For instance, brain tumor diagnosis ideally requires four MRI modalities (Flair, T1ce, T1, T2), but complete acquisition is often infeasible in clinical practice.

Core limitations of existing methods:

**High deployment cost**: Most methods train separate models or independent encoder parameters for different modality combinations, requiring exhaustive model subsets and additional modality-classification preprocessing steps. Four MRI modalities yield 15 valid missing-modality combinations, necessitating a large number of independent models.

**Limitations of reconstruction-based methods**: Methods based on modality reconstruction (e.g., M3AE, SSLSOD) attempt to predict missing modalities to align features, but suffer from: (a) pretraining objectives that prioritize global feature compression, producing representations insufficiently fine-grained for segmentation tasks requiring precise spatial information; and (b) cascading low-quality reconstructions into segmentation networks, which amplifies error propagation.

**Isolated use of self-supervised techniques**: Existing work typically develops masked reconstruction, contrastive learning, or knowledge distillation in isolation, without effectively exploiting the synergy among all three paradigms.

The goal of UniMRSeg is to approximate complete-modality representation quality across all modality combinations using a single set of shared parameters.

## Method

### Overall Architecture

A three-stage progressive learning framework built upon a unified 3D U-Net-style encoder–decoder architecture (with embedded 3D ASPP, dilation rates [1, 6, 12, 18]):
- Stage 1: Multi-granularity modality reconstruction (input-level compensation)
- Stage 2: Modality-invariant contrastive learning (feature-level compensation)
- Stage 3: Incomplete-modality adaptive fine-tuning (output-level compensation)

### Key Designs

1. **Multi-Granularity Modality Reconstruction (Stage 1)**: Three data perturbation strategies are combined: (a) **Random modality dropout**: each modality is dropped with 50% probability, with at least one retained; (b) **Modality order shuffling**: the order of remaining modalities is randomly permuted to remove dependence on fixed modality ordering and decouple modality-agnostic representations; (c) **Spatial masking**: random spatial regions of the input are masked. Perturbed samples are fed into a 3D U-Net reconstruction network, with normalized slices from the complete original modalities as reconstruction targets; the loss is L1 + SSIM. **Design Motivation**: multi-granularity perturbation compels the model to simultaneously learn fine-grained local patterns and holistic semantic representations.

2. **Modality-Invariant Contrastive Learning (Stage 2)**: Positive and negative pairs are constructed as follows—the complete-modality sample $I_k$ and its randomly missing counterpart $\hat{I}_k$ form a positive pair, while samples from different instances form negative pairs. The NT-Xent loss is applied across all five encoder levels:

$$l^i(u,v) = -\log \frac{\exp(\text{sim}(\mathbf{f}_u^i, \mathbf{f}_v^i) / \tau)}{\sum_{k=1}^{2B} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(\mathbf{f}_u^i, \mathbf{f}_k^i) / \tau)}$$

Crucially, a segmentation constraint (Dice loss) is jointly optimized to guide the direction of feature clustering, ensuring that the learned contrastive space is aligned with the segmentation objective rather than yielding generic representations.

3. **Reverse Attention Adapter (Stage 3)**: The encoder is frozen; only the decoder and lightweight adapters are trained. The adapter operates as follows:

    - Incomplete-modality features $\hat{F}^i_{cp}$ are extracted through the frozen encoder.
    - Initial adaptive features $\hat{F}^i_{ada-in}$ are generated via 3D convolution.
    - The fused features $\hat{F}^i_h$ are processed by a 3D Swin Transformer to capture global cross-modal correlations.
    - A mutual attention map is computed and then **inverted**: this highlights semantically difficult regions that the encoder fails to perceive.
    - The reverse attention map is multiplied with $\hat{F}^i_h$ to yield compensated features $\hat{F}^i_{ada}$.

Mathematical rationale: $f_{\text{inc}} + \mathcal{A}(f_{\text{inc}}) \approx f_{\text{com}}$, where the adapter $\mathcal{A}$ acts as a residual correction. Freezing the encoder is an intentional design choice to preserve the task-guided contrastive representations established in Stage 2.

### Loss & Training

- **Stage 1**: L1 + SSIM reconstruction loss
- **Stage 2**: $L_{\text{NT-Xent}}$ (5-level contrastive loss) + $L_{\text{Dice}}$ (segmentation constraint), jointly optimized
- **Stage 3**: $L_{\text{fc}} = \frac{1}{B} \sum_{k=1}^B \sum_m^M \frac{1}{5} \sum_{i=1}^5 \|F_k^i - \hat{F}_{k,m}^i\|_1$ (5-level feature consistency) + $L_{\text{pc}} = \frac{1}{B} \sum_{k=1}^B \sum_{m=1}^M l_{\text{Dice}}(P_k, \hat{P}_{k,m})$ (prediction consistency) + $L_{\text{Dice}}$ (complete-modality supervision), with $M=14$ incomplete modality combinations
- AdamW optimizer, learning rate 0.0001, weight decay 0.00001, 300 epochs, warmup scheduling

## Key Experimental Results

### Main Results (BraTS2020 Brain Tumor Segmentation, Average Dice% over 15 Modality Combinations)

| Method | Whole ↑ | Core ↑ | Enhancing ↑ | Std Dev ↓ |
|------|--------|--------|------------|---------|
| NestedFormer | 52.01 | 39.59 | 40.78 | 23.09/19.53/24.20 |
| SFusion | 73.23 | 60.90 | 48.14 | 10.47/17.07/21.80 |
| ShaSpec | 74.81 | 65.16 | 55.90 | 10.08/15.45/21.62 |
| PASSION | 76.39 | 66.06 | 58.53 | 10.07/15.34/21.84 |
| **UniMRSeg** | **80.64** | **73.33** | **63.10** | **8.43/13.04/19.86** |

### Ablation Study (Contribution of Each Stage and Component)

| Configuration | Whole | Core | Enhancing | Notes |
|------|-------|------|-----------|------|
| Baseline (3D-UNet) | 63.31 | 51.60 | 38.40 | — |
| + Random modality dropout | 66.98 | 55.47 | 42.25 | +3.7/+3.9/+3.9 |
| + Modality order shuffling | 67.78 | 56.85 | 44.17 | Further decoupling |
| + Spatial masking (Stage 1 complete) | 69.35 | 59.89 | 47.12 | +14.7% vs. baseline |
| + Contrastive learning (encoder) | 72.45 | 64.02 | 51.45 | Significant feature-level gain |
| + Segmentation constraint (Stage 2 complete) | 74.53 | 65.25 | 53.97 | Task guidance is critical |
| + Feature consistency (adapter) | 78.12 | 69.38 | 59.25 | Reverse attention adapter is effective |
| **+ Prediction consistency (Stage 3 complete)** | **80.64** | **73.33** | **63.10** | **+44.6% vs. baseline** |
| Single-stage joint training | 20.32 | 13.67 | 10.03 | Complete convergence failure |

### Key Findings

- **Superiority of unified parameters**: UniMRSeg achieves optimal average performance and minimal standard deviation across all modality combinations with 100% shared parameters, without requiring a modality-classification preprocessing step.
- **Synergistic effect of hierarchical compensation**: The three levels of compensation are not simply additive—input-level alone yields +6.0%, feature-level alone +9.2%, and output-level alone +6.2%, but their combination achieves +17.3%, exceeding the sum of individual contributions.
- **Necessity of three-stage design**: Single-stage joint training completely fails to converge (Dice drops to 10–20%), as six competing loss terms cause optimization instability.
- **Necessity of freezing the encoder**: Fine-tuning the encoder during Stage 3 leads to a 6.4% performance drop, disrupting the synergy between the adapter and the encoder.
- **Cross-task generalization**: The method achieves state-of-the-art performance on four tasks: brain tumor segmentation (MRI, 15 combinations), RGB-D salient object detection (3 combinations), RGB-T salient object detection (3 combinations), and RGB-D semantic segmentation (3 combinations).

## Highlights & Insights

- **Unified parameters with full combination coverage**: A single model handles 21 modality combinations (MRI-15 + RGB-D-3 + RGB-T-3), substantially simplifying clinical deployment.
- **Organic integration of self-supervised techniques**: This work is the first to effectively combine masked reconstruction, contrastive learning, and knowledge distillation within the same task, demonstrating their complementarity.
- **Elegant design of reverse attention**: Rather than directly compensating missing information, the adapter identifies regions that the encoder "fails to see" and targets compensation accordingly—an intuitive and effective reverse-thinking approach.
- **Thorough ablation analysis**: The contrast between single-stage failure and three-stage success is highly convincing.

## Limitations & Future Work

- Self-supervised pretraining is conducted only on each task's own training set, without leveraging large-scale external data.
- The 3D U-Net backbone is relatively simple; adopting stronger architectures (e.g., nnU-Net V2, SwinUNETR) may yield further improvements.
- Stage 3 requires parallel forward passes for all modality combinations (14 incomplete combinations for MRI), resulting in relatively high training cost.
- Strategies for dynamically determining the minimum number of required modalities have not been explored.

## Related Work & Insights

- The work inherits the masked reconstruction paradigm from MAE, the contrastive learning framework from SimCLR, and the teacher–student paradigm from knowledge distillation; its innovation lies in the organic integration of all three.
- The reverse attention mechanism shares conceptual similarities with reverse attention in salient object detection.
- The framework provides direct reference value for designing robust AI systems that handle missing modalities in clinical settings.

## Rating

- Novelty: ⭐⭐⭐⭐ The hierarchical self-supervised compensation integration is novel, and the reverse attention adapter is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four distinct tasks, 21 modality combinations, and extremely detailed ablation studies provide highly compelling evidence.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and complete; the three-stage pipeline is intuitively visualized.
- Value: ⭐⭐⭐⭐⭐ The work addresses a practical pain point in multimodal segmentation; the unified-parameter design has significant implications for clinical deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)

<!-- RELATED:END -->
