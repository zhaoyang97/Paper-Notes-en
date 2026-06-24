---
title: >-
  [Paper Note] MotionMap: Representing Multimodality in Human Pose Forecasting
description: >-
  [CVPR 2025][Human Understanding][human pose forecasting] MotionMap is proposed, introducing a new paradigm that represents the spatial distribution of motion using heatmaps. By combining t-SNE dimensionality reduction with a codebook, it achieves variable-mode forecasting and confidence quantization, yielding optimal mode coverage with minimal sampling.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "human pose forecasting"
  - "multimodality"
  - "heatmap"
  - "codebook"
  - "t-SNE"
  - "uncertainty estimation"
  - "mode coverage"
date: 2026-05-08
content_hash: 241d5febf54aeabb
---

# MotionMap: Representing Multimodality in Human Pose Forecasting

**Conference**: CVPR 2025  
**arXiv**: [2412.18883](https://arxiv.org/abs/2412.18883)  
**Code**: [https://github.com/vita-epfl/MotionMap](https://github.com/vita-epfl/MotionMap)  
**Area**: Human Understanding  
**Keywords**: human pose forecasting, multimodality, heatmap, codebook, t-SNE, uncertainty estimation, mode coverage

## TL;DR

MotionMap is proposed, introducing a new paradigm that represents the spatial distribution of motion using heatmaps. By combining t-SNE dimensionality reduction with a codebook, it achieves variable-mode forecasting and confidence quantization, yielding optimal mode coverage with minimal sampling.

## Background & Motivation

**Background**: Human pose forecasting is inherently a multimodal problem, where an infinite number of possible future motions can stem from the same observed sequence. Existing methods attempt to cover these futures with a limited number of predictions, but they can never cover all modes, making the problem inherently ill-posed.

**Limitations of Prior Work**:
1. **Diversity $\neq$ Realism**: While diversity-driven methods like DLow and DivSamp sample diversely, their predictions are often incoherent with the observed sequence.
2. **Implicit distribution + massive sampling**: Generative models like GAN, VAE, and Diffusion learn implicit distributions and require extensive random sampling to cover more modes, which is highly inefficient.
3. **Inability to determine sampling count**: Different observed sequences require different numbers of predictions, but existing methods use a fixed number.
4. **Equal weighting of all samples**: Existing approaches fail to distinguish which predictions are more likely and which represent rare modes.
5. **Averaged-out rare modes**: Rare yet plausible future motions are easily suppressed in implicit modeling.

**Key Insight**: Rather than attempting to learn an unbounded set of future motions, one should explicitly learn the different transitions present in the training set. This reformulates the problem into a well-posed one, as the number of possible futures for each input sequence is upper-bounded by the constraint of the training set size.

## Method

### Overall Architecture

A two-module system is trained in two stages:
1. **Stage 1 - Autoencoder**: GRU encoders encode the input sequence $X$ and the future sequence $Y$ separately, and after concatenating their latents, decoder predicts the complete sequence.
2. **Stage 2 - MotionMap Module**: The model learns to predict the heatmap (motion spatial distribution) from the observed sequence $X$. During inference, the local maxima of the heatmap along with the codebook are used to replace the missing future latents.

### Key Designs

#### Module 1: MotionMap Heatmap Representation

MotionMap maps all possible future motions to a distribution in a 2D space:
- **Dimensionality Reduction**: The encoding $z_y$ of all future sequences in the training set is projected to a 2D space using t-SNE and quantized into integer coordinates $h_y$.
- **Heatmap Construction**: For each sample's $M$ multimodal ground truths (GTs), a Gaussian peak is placed at its corresponding 2D position.
- **Codebook**: A mapping $h_y \to \overline{z_y}$ is established (taking the mean when multiple $z_y$ map to the same $h_y$).
- **Core Property**: Variable number of modes (different heatmaps have varying numbers of peaks for different samples), ensuring rare modes are not suppressed by averaging.

#### Module 2: Improved Definition of Multimodal GT

The method of finding multimodal GTs in existing literature is improved:
- **Limitation**: The original method measures similarity using only the distance of the last frame, thereby losing motion dynamics; furthermore, individuals with different body shapes fail to match even if they perform the exact same motion.
- **Solution**: (1) The similarity is calculated using the last three frames instead of just the last frame; (2) skeleton scaling (Motion Transfer) is performed via Cartesian-to-spherical coordinate transformation to eliminate body shape differences.

#### Module 3: Dual Uncertainty Estimation

Uncertainty is decomposed into two sources:
- **Mode Uncertainty**: The height of each peak in the MotionMap indicates the confidence of the corresponding mode (higher peak = higher confidence).
- **Prediction Uncertainty**: The uncertainty module $\mathcal{U}$ of the autoencoder predicts the conditional variance of each joint (heteroscedastic regression).
- For example, the nose joint uncertainty in prediction 4 (sharp turn) is higher than that in prediction 6 (smooth motion), as directional changes are more challenging.

### Loss & Training

**Autoencoder Training**: Negative log-likelihood loss
$$\mathcal{L} = \frac{\text{error}}{\sigma^2} + \log\sigma^2$$
Jointly optimizing the mean and variance enables heteroscedastic uncertainty estimation.

**MotionMap Training**: Pixel-wise weighted binary cross-entropy loss (penalizing false negatives more than false positives) to prevent rare modes from being ignored.

**Fine-tuning**: The decoder is fine-tuned again by replacing the true $z_y$ with the average latent $\overline{z_y}$ from the codebook to close the training-inference gap.

## Key Experimental Results

### Main Results

**Human3.6M & AMASS Datasets** (all methods restricted to 7 predictions):

| Method | Diversity↑ | ADE↓ | FDE↓ | MMADE↓ | MMFDE↓ |
|------|-----------|------|------|--------|--------|
| DLow | 11.77 | 0.445 | 0.730 | 0.576 | 0.715 |
| DivSamp | 15.73 | 0.480 | 0.685 | 0.542 | 0.671 |
| BeLFusion | 7.11 | 0.441 | 0.597 | 0.491 | 0.586 |
| CoMusion | 7.32 | 0.426 | 0.613 | 0.531 | 0.623 |
| **MotionMap** | **7.84** | **0.474** | **0.598** | **0.466** | **0.532** |

AMASS dataset:

| Method | MMADE↓ | MMFDE↓ |
|------|--------|--------|
| BeLFusion | 0.488 | 0.564 |
| CoMusion | 0.526 | 0.602 |
| **MotionMap** | **0.450** | **0.514** |

MotionMap consistently outperforms baseline methods on multimodal metrics MMADE/MMFDE.

### Ablation Study

The paper demonstrates the contribution of each component through qualitative visualization analysis:
- **Sampling Efficiency Comparison**: Under the same number of predictions, the coverage of MotionMap is far superior to DLow (which is anchor-based but predicts unlikely transitions) and BeLFusion (which lacks diversity and misses rare modes).
- **Ablation of Motion Transfer**: After applying skeleton scaling, actions across different body shapes are correctly identified as the same mode.
- **Heteroscedastic vs. Homoscedastic**: Conditional uncertainty is semantically richer (showing high uncertainty in sharp turning regions and low uncertainty during smooth motions).

### Key Findings

1. **Highest Sampling Efficiency**: Only 7 predictions are required to reach optimal mode coverage, whereas DivSamp/DLow require more samples only to cover "unlikely" regions.
2. **MotionMap vs. BeLFusion**: Both share the same encoder/decoder; the difference lies solely in the way latents are obtained. The heatmap + codebook paradigm significantly outperforms diffusion-based repetitive sampling.
3. **Variable Number of Modes**: The number of predicted modes varies naturally across different test samples (depending on the number of peaks in the heatmap) rather than being fixed.
4. **Ranking Capability**: Predictions corresponding to high-confidence peaks are generally closer to the actual ground truth, whereas low-confidence peaks represent rare but plausible transitions.
5. **Controllability**: The spatial distribution of the MotionMap corresponds to the action label space, enabling the selective forecasting of specific motion types using action labels.

## Highlights & Insights

1. **Problem Re-formulation**: Converting the ill-posed pose forecasting problem into a well-posed one by explicitly learning the transition patterns present in the training set represents a valuable paradigm shift.
2. **Heatmaps as Motion Distribution Representations**: Intuitive, interpretable, and adaptable to a variable number of modes—this approach is far more transparent than implicit latent distributions.
3. **Sampling Efficiency**: Deterministic peak extraction without relying on random sampling is a critical advantage for real-world applications (e.g., where robots require rapid decision-making).
4. **Dual Uncertainty**: Separating the uncertainty of "what to do" (mode) from "how to do it" (execution) is highly valuable for safety-critical applications.

## Limitations & Future Work

1. **Lack of Fine-Grained Intra-Mode Details**: Tiny variations within the same mode (e.g., walking at different step frequencies) are aggregated into a single prediction, resulting in a loss of intra-mode diversity.
2. **Irreversible t-SNE Dimensionality Reduction**: The pipeline of dimensionality reduction, quantization, and codebook representation introduces information loss, and replacing projections with a codebook mean may blur fine details.
3. **Codebook Storage**: A $128 \times 128$ heatmap + a $128$-dimensional embedding adds up to 64MB, which is a non-trivial footprint.
4. **Multimodal GT Definition**: The pipeline still relies on distance thresholds (0.5 for Human3.6M and 0.4 for AMASS), introducing threshold sensitivity.

## Related Work & Insights

- **BeLFusion**: Conditional latent diffusion is used for pose forecasting. MotionMap shares its architecture but improves upon how latents are acquired.
- **DLow**: A pioneer in multi-distribution sampling strategies, though its latent anchors do not account for input-dependent likelihoods.
- **STARS**: An anchor-based sampling method; the "peaks" in MotionMap are, in a sense, data-driven adaptive anchors.
- **Inspirations**: The heatmap + codebook paradigm of MotionMap can be generalized to other sequence forecasting problems (e.g., trajectory forecasting, gesture generation), rendering implicit distributions explicit.

## Rating

⭐⭐⭐⭐ — The theoretical reformulation of the problem is profound (well-posed reformulation), and the heatmap representation is intuitive and elegant. The sampling efficiency makes it highly practical for real-world applications. However, the t-SNE + codebook approach feels somewhat engineered compared to end-to-end methods, and the lack of fine-grained intra-mode details remains a notable shortcoming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HUMOF: Human Motion Forecasting in Interactive Social Scenes](../../ICLR2026/human_understanding/humof_human_motion_forecasting_in_interactive_social_scenes.md)
- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[CVPR 2025\] PoseBH: Prototypical Multi-Dataset Training Beyond Human Pose Estimation](posebh_prototypical_multi-dataset_training_beyond_human_pose_estimation.md)
- [\[CVPR 2025\] UniPose: A Unified Multimodal Framework for Human Pose Comprehension, Generation and Editing](unipose_a_unified_multimodal_framework_for_human_pose_comprehension_generation_a.md)
- [\[CVPR 2026\] Forecasting 3D Scanpaths in Egocentric Video](../../CVPR2026/human_understanding/forecasting_3d_scanpaths_in_egocentric_video.md)

</div>

<!-- RELATED:END -->
