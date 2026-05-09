---
title: >-
  [Paper Note] Semantic Discrepancy-aware Detector for Image Forgery Identification
description: >-
  [ICCV 2025][Image Generation][Image forgery detection] This paper proposes the Semantic Discrepancy-aware Detector (SDD), which leverages three modules — Semantic Token Sampling (STS), Concept-level Forgery Discrepancy Learning (CFDL), and a Low-level Forgery Feature Enhancer — to align CLIP's visual semantic concept space with the forgery space via reconstruction learning. SDD achieves state-of-the-art performance on the UnivFD and SynRIS benchmarks ($ap_m$ 98.51%, AUROC 95.1%).
tags:
  - ICCV 2025
  - Image Generation
  - Image forgery detection
  - semantic concept space
  - reconstruction learning
  - CLIP
  - vision-language models
date: 2026-05-08
content_hash: f411dd97d38f4190
---

# Semantic Discrepancy-aware Detector for Image Forgery Identification

**Conference**: ICCV 2025
**arXiv**: [2508.12341](https://arxiv.org/abs/2508.12341)
**Code**: [GitHub](https://github.com/wzy1111111/SSD)
**Area**: Image Generation
**Keywords**: Image forgery detection, semantic concept space, reconstruction learning, CLIP, vision-language models

## TL;DR

This paper proposes the Semantic Discrepancy-aware Detector (SDD), which leverages three modules — Semantic Token Sampling (STS), Concept-level Forgery Discrepancy Learning (CFDL), and a Low-level Forgery Feature Enhancer — to align CLIP's visual semantic concept space with the forgery space via reconstruction learning. SDD achieves state-of-the-art performance on the UnivFD and SynRIS benchmarks ($ap_m$ 98.51%, AUROC 95.1%).

## Background & Motivation

With the rapid advancement of GANs and diffusion models, generated images have become increasingly photorealistic, making forgery detection critically important. Existing methods suffer from the following limitations:

**Poor generalization of single-feature approaches**: Early methods (e.g., CNNSpot) exploit single-type cues such as noise patterns, texture statistics, or frequency signals, which tend to overfit specific generative models and degrade sharply on unseen ones.

**Misalignment between semantic concept space and forgery space**:
- **Limitations of frozen pretrained models** (e.g., UnivFD): Directly applying frozen CLIP features with linear probing preserves semantic priors but ignores fine-grained forgery details. Because the semantic concept space is frozen, real and fake samples sharing similar semantics are easily misclassified.
- **Limitations of prompt tuning** (e.g., FatFormer): Although forgery-aware adapters construct a forgery-adaptive space effectively, soft prompts based on simple [CLASS] embeddings are inherently limited in semantic descriptive granularity — narrow concept coverage may bias detection toward incorrect semantic dimensions.

**Key Insight**: Through statistical analysis, the authors find that different semantic concepts may guide the model to discover different forgery traces, revealing a nuanced relationship between semantic concepts and forgery features. Neither purely freezing nor purely fine-tuning the semantic space is optimal.

## Method

### Overall Architecture

SDD consists of three key modules forming a reconstruction learning-based forgery detection framework:

1. **Semantic Token Sampling (STS)**: Samples representative semantic patch tokens from real images.
2. **Concept-level Forgery Discrepancy Learning (CFDL)**: Uses these tokens to capture forgery discrepancies via reconstruction learning.
3. **Low-level Forgery Feature Enhancer**: Fuses reconstruction discrepancy maps into low-level features to extract highly generalizable forgery features.

The CLS token from LoRA-CLIP and the enhanced low-level features are concatenated and fed into a linear classifier for real/fake binary classification.

### Key Designs

1. **Semantic Token Sampling (STS)**:

    - **Function**: Uniformly samples a set of representative patch tokens $f_s \in \mathbb{R}^{M \times D}$ from real images, serving as visual anchors to smoothly bridge the semantic concept space and the forgery space.
    - **Mechanism**: Jensen-Shannon (JS) divergence is used to measure distributional differences among tokens. Given an initial token $\tilde{r}$, the JS divergence range $[0,1]$ is divided into $M$ equal intervals, and one token is selected from each interval: $f_s = \mathcal{S}(\mathbb{R}^{N \times D}, \delta)$, where $\delta$ is the sampling rate and $M = 1/\delta$.
    - **Design Motivation**: Incorporating all real patch tokens into the reconstruction module is computationally expensive and introduces redundancy. JS divergence-based uniform sampling ensures that tokens are evenly distributed in the unified CLIP space, representing the semantic distribution of real images without biasing toward any specific non-forgery-related distribution. A key advantage is that it bypasses the semantic bias introduced by text prompts.

2. **Concept-level Forgery Discrepancy Learning (CFDL)**:

    - **Function**: Employs a Transformer encoder-decoder to perform reconstruction learning, capturing forgery discrepancies at the semantic concept level.
    - **Mechanism**:
        - LoRA fine-tuning of CLIP-ViT is used to obtain high-level visual features $V_H = \mathcal{F}_{LoRA}(\mathcal{I})$.
        - **Encoder**: Sampled semantic tokens $f_s$ serve as queries, and $V_H$ serves as keys/values: $R_1 = \text{LN}(\text{MHA}(f_s, V_H, V_H))$, $R_2 = \text{LN}(\text{MHA}(R_1, V_H, V_H))$.
        - **Decoder**: $R_3 = \text{LN}(\text{MHA}(R_2, R_2, R_2))$, $R_e = \text{LN}(\text{MHA}(R_1, R_3, R_3))$.
        - Reconstruction loss is computed on real samples only: $\mathcal{L}_r = \frac{1}{B}\sum_{i=0}^B \text{MSE}(R_e, V_H)$.
        - Reconstruction discrepancy map: $\mathcal{D}_s = |R_f - f_r|$.
    - **Design Motivation**: By minimizing the reconstruction gap only for real samples, the reconstruction discrepancy for forged samples is naturally amplified. Unlike FatFormer's use of text prompts, CFDL relies purely on visual information — the sampled semantic tokens provide richer details than coarse-grained text prompts, revealing more hidden forgery traces.

3. **Low-level Forgery Feature Enhancer**:

    - **Function**: Uses the reconstruction discrepancy map to guide multi-scale low-level forgery feature extraction, compensating for the tendency of high-level semantic features to overlook weakly semantic forgery cues.
    - **Mechanism**:
        - A multi-stage convolutional network extracts $F(n)$ ($n=1,2,3$).
        - The discrepancy map $\mathcal{D}_s$ is deconvolved and element-wise multiplied with $F(n)$: $F'(n) = F(n) \otimes \text{deconv}(\mathcal{D}_s)$.
        - Adaptive weight coefficient: $\frac{1}{e_n} = \frac{1}{e^{|F'(n) - F(n)|}}$. When the discrepancy is large, the weight is small (fast regime: focuses on semantically strongly correlated features); when the discrepancy is small, the weight approaches 1 (slow regime: preserves features weakly correlated with semantics but strongly correlated with forgery).
        - Final output: $F_{low}(n) = F(n) + \frac{F(n)}{e_n}$.
    - **Design Motivation**: Relying solely on high-level semantic concepts misses pixel-level forgery traces. The inverse exponential adaptive weight elegantly balances two types of features: forgery features strongly correlated with semantic concepts (lower weight when discrepancy is large, avoiding over-reliance on semantics) and forgery-discriminative features weakly correlated with semantics (higher weight when discrepancy is small, preserving them).

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{bce} + \lambda_1 \mathcal{L}_{tri} + \lambda_2 \mathcal{L}_r$

- $\mathcal{L}_{bce}$: Binary cross-entropy for real/fake classification.
- $\mathcal{L}_{tri}$: Triplet loss to pull intra-class features closer and push inter-class features apart: $\mathcal{L}_{tri} = \max(0, d(f_p, f_a) - d(f_n, f_a) + \alpha)$.
- $\mathcal{L}_r$: Reconstruction loss (MSE), computed on real samples only.

Training settings: learning rate $1 \times 10^{-5}$, batch size 32, LoRA parameters $r=6, \alpha=6$, dropout 0.8. ProGAN images are used as training data.

## Key Experimental Results

### Main Results

| Dataset | Metric | SDD (Ours) | FatFormer | UnivFD | NPR | Gain |
|---------|--------|-----------|-----------|--------|-----|------|
| UnivFD | $ap_m$ | **98.51** | 98.18 | 93.38 | 92.19 | +0.33 vs. FatFormer |
| UnivFD | $acc_m$ | **93.61** | 90.86 | 81.38 | 86.20 | +2.75 vs. FatFormer |
| SynRIS | AUROC (avg.) | **95.1** | 78.0 | 62.3 | 90.3 | +4.8 vs. NPR |

Notable advantage on SynRIS: on images generated by high-fidelity text-to-image models, SDD (95.1%) substantially outperforms FatFormer (78.0%) and UnivFD (62.3%).

### Ablation Study

| # | STS | CFDL | Enhancer | $ap_m$ | $acc_m$ | Notes |
|---|:---:|:---:|:---:|--------|---------|-------|
| 1 | - | ✓ | - | 97.37 | 81.64 | CFDL only |
| 2 | - | ✓ | ✓ | 97.41 | 90.17 | Enhancer contributes significantly (+8.53 acc) |
| 3 | ✓ | ✓ | - | 97.39 | 89.98 | STS is beneficial |
| 4 | ✓ | ✓ | ✓ | **98.52** | **93.61** | Full model, best performance |

Comparison of adaptive weight functions:

| Function | $ap_m$ | $acc_m$ | Notes |
|----------|--------|---------|-------|
| $f(x) = \|x\|$ | 97.77 | 92.12 | Linear weight |
| $f(x) = x^2$ | 97.93 | 92.34 | Quadratic weight |
| $1/e^x$ (Ours) | **98.52** | **93.61** | Inverse exponential is optimal |

### Key Findings

- **Visual information outperforms text prompts**: SDD uses no text prompts, relying solely on visual semantic tokens and reconstruction learning, yet surpasses prompt-based FatFormer on UnivFD. This demonstrates that fine-grained visual information is more suitable than coarse-grained text descriptions for capturing forgery traces.
- **Largest advantage on SynRIS** (AUROC 95.1% vs. NPR 90.3%), as high-fidelity generative models capture global semantics well but fail to faithfully reproduce local pixel details. SDD captures both semantic concepts and low-level forgery features simultaneously.
- **Attention map visualizations** show that SDD attends to different regions for different forged images (background, local objects, edge details), while real images exhibit almost no forgery discrepancy regions — validating the effectiveness of the reconstruction loss.
- In t-SNE visualizations, the real/fake decision boundary for ProGAN is more ambiguous than for other models, reflecting a more complex and nuanced boundary under semantic concept supervision.

## Highlights & Insights

- **First purely vision-based pretrained VLM forgery detection paradigm**: No text prompts are used, bypassing the semantic bias caused by coarse-grained textual descriptions.
- **JS divergence-based uniform sampling** is a simple yet effective token selection strategy that ensures sampled tokens are uniformly distributed in the semantic space.
- **Elegant application of reconstruction learning**: Computing reconstruction loss only on real samples causes the reconstruction discrepancy for forged samples to be naturally amplified — a more elegant approach than explicitly constructing contrastive samples.
- **Dual-regime property of the inverse exponential adaptive weight** $1/e^x$: the fast regime captures features strongly correlated with semantic concepts, while the slow regime preserves forgery-discriminative features with weak semantic correlation, achieving automatic balance between the two.

## Limitations & Future Work

- Training data is limited to ProGAN (or Stable Diffusion v1); generalization to emerging generative models relies on the method's intrinsic robustness rather than data coverage.
- The sampling rate $\delta$ in the STS module is a user-defined hyperparameter whose optimal value may vary across datasets.
- The reconstruction module introduces additional inference latency, which must be considered in practical deployment.
- Performance on DALL-E 3 is relatively weaker (AUROC 85.9%), indicating room for improvement in generalizing to certain high-end commercial models.
- The sensitivity of LoRA parameter settings ($r=6, \alpha=6$) is not thoroughly discussed.

## Related Work & Insights

- UnivFD (CVPR 2023) demonstrated the value of VLM feature spaces for forgery detection; this work addresses the limitations of frozen features in that paradigm.
- FatFormer (CVPR 2024) constructs a forgery-adaptive space via prompt tuning, but this paper identifies insufficient semantic granularity as a key limitation of text prompts.
- Reconstruction learning is widely used in unsupervised representation learning (e.g., MAE); this work innovatively applies it to forgery detection by exploiting the asymmetry in reconstructability between real and forged images.
- NPR focuses on pixel-level neighbor relationships, which is complementary to SDD's combination of semantic concepts and low-level features.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The purely vision-based paradigm, the idea of aligning semantic concept space with forgery space, and the inverse exponential adaptive weight all represent significant innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on two major benchmarks with in-depth ablation studies; extended evaluations such as video forgery detection are lacking.
- Writing Quality: ⭐⭐⭐⭐ Motivation is well-argued and the method is clearly described, though some notation is slightly inconsistent.
- Value: ⭐⭐⭐⭐⭐ With the proliferation of AI-generated imagery, generalizable forgery detection methods hold substantial practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)

</div>

<!-- RELATED:END -->
