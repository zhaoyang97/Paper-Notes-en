---
title: >-
  [Paper Note] Bridging the Perception Gap in Image Super-Resolution Evaluation
description: >-
  [CVPR 2026][Image Restoration][Super-resolution evaluation] Through a large-scale user study, this paper reveals a severe misalignment between existing SR evaluation metrics (PSNR, SSIM, LPIPS…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Super-resolution evaluation"
  - "image quality metrics"
  - "perception gap"
  - "relative quality index"
  - "user study"
date: 2026-05-08
content_hash: 7ee54b8e0c61cfdc
---

# Bridging the Perception Gap in Image Super-Resolution Evaluation

**Conference**: CVPR 2026
**arXiv**: [2503.13074](https://arxiv.org/abs/2503.13074)
**Code**: [Project Page](https://color.cvc.uab.cat/rqi/)
**Area**: Image Super-Resolution / Image Quality Assessment
**Keywords**: Super-resolution evaluation, image quality metrics, perception gap, relative quality index, user study

## TL;DR
Through a large-scale user study, this paper reveals a severe misalignment between existing SR evaluation metrics (PSNR, SSIM, LPIPS, etc.) and human perception. After analyzing their inherent deficiencies, the paper proposes a minimalist yet effective Relative Quality Index (RQI) framework that learns relative quality differences between image pairs to enable more reliable SR evaluation, and can also serve as a loss function to guide SR model training.

## Background & Motivation
**Background**: SR techniques have advanced rapidly (RealESRGAN → SwinIR → StableSR → SeeSR), producing increasingly high-quality outputs, while evaluation metrics have remained largely unchanged.

**Limitations of Prior Work**: Researchers have grown increasingly skeptical of evaluation metrics — models achieving higher metric scores do not necessarily produce better visual results. Many works are compelled to conduct user studies or combine multiple metrics for validation.

**Key Challenge**: SR models evolve rapidly while evaluation standards stagnate. Three inherent categories of challenges exist between metrics and human perception:
   - (a) **Distortion-based FR metrics** (PSNR, SSIM) favor over-smoothed average solutions, which is contrary to perceptual preference.
   - (b) **Perceptual FR metrics** (LPIPS, DISTS) fail when GT quality is poor.
   - (c) **No-reference metrics** (NIQE, CLIP-IQA) cannot assess fidelity.
   - (d) Subtle differences between high-quality SR outputs cannot be distinguished by existing metrics.

**Goal**: Design an SR evaluation framework capable of simultaneously addressing all four challenges above.

**Key Insight**: Replace absolute quality scores with relative quality differences — allowing any image (including degraded ones) to serve as a reference, and learning the quality gap between the target and the reference.

**Core Idea**: Since GT may be imperfect and SR outputs may surpass GT, the framework abandons the assumption of a perfect reference and instead learns **relative** quality relationships.

## Method

### Overall Architecture
**Training**: Dense image pairs $\{I_i, I_j\}$ are constructed from IQA datasets → relative quality labels $q_i - q_j$ (MOS differences) are computed → an FR-IQA model is trained to predict these differences.
**Evaluation**: Given SR output $I_{HR}$ and GT $I_{GT}$ → the model outputs $s = f_{RQI}(I_{HR}, I_{GT})$, where a positive value indicates that the SR output surpasses GT in quality.

### Key Designs

1. **Three Key Properties of the RQI Training Framework**:

    - **(a) Asymmetry**: Swapping the input order produces the opposite result, $f_{RQI}(I_i, I_j) = -f_{RQI}(I_j, I_i)$, in contrast to the symmetry of conventional FR metrics.
    - **(b) Relative Difference**: Rather than predicting absolute quality scores, the model learns the perceptual quality gap between two images. This allows the reference image itself to be degraded, addressing the problem of imperfect GT.
    - **(c) Dense Pairwise Comparison**: Conventional methods construct only $\{I_0, I_i\}$ pairs (reference vs. degraded); RQI constructs arbitrary $\{I_i, I_j\}$ pairs (degraded vs. degraded), substantially increasing training samples and naturally encompassing subtle quality differences.
    - **Design Motivation**: The three properties correspond respectively to Goal 1 (fidelity assessment), Goal 2 (robustness to imperfect GT), and Goal 3 (fine-grained discrimination).

2. **Training Details**:

    - Training objective: Huber loss regression on relative differences:
    $L = \begin{cases} \frac{1}{2}(\hat{y}_{ij} - (q_i - q_j))^2, & \text{if } |\hat{y}_{ij} - ..| \leq \delta \\ \delta(|\hat{y}_{ij} - ..| - \frac{1}{2}\delta), & \text{otherwise} \end{cases}$
    - Labels are normalized to $[-1, 1]$; the activation function of the final regression layer is removed to support negative outputs.
    - **Design Motivation**: Huber loss provides smooth gradients for small differences, yielding more stable training on subtle quality variations.

3. **General Framework Design**:

    - Compatible with arbitrary FR-IQA model backbones (AHIQ, MANIQA, TOPIQ).
    - Trainable on arbitrary IQA datasets (Kadid-10K, PieAPP, PIPAL).
    - No SR-specific data collection is required; zero-shot transfer to SR evaluation is achieved.
    - **Design Motivation**: Generality is the key contribution — rather than designing a new metric, the framework elevates the paradigm of existing metrics.

### Loss & Training
- Huber loss regression with $\delta$ as the smoothing threshold.
- 8:2 train/validation split with no scene overlap.
- The best model on the validation set is directly transferred for zero-shot evaluation.

## Key Experimental Results

### Main Results (Alignment with Human Perception, SRCC)

| Metric | DIV2K | RealSR | DRealSR | Set5&14 |
|--------|-------|--------|---------|---------|
| SSIM | -0.348 | -0.220 | -0.354 | -0.321 |
| PSNR | -0.079 | -0.116 | -0.355 | -0.204 |
| LPIPS | 0.415 | 0.008 | -0.141 | 0.282 |
| CLIP-IQA | 0.593 | 0.377 | 0.268 | 0.642 |
| AFINE | 0.581 | 0.449 | 0.484 | 0.578 |
| DeQA-Score | 0.613 | 0.452 | 0.437 | 0.699 |
| **RQI** | **0.744** | **0.504** | **0.529** | 0.664 |

### Ablation Study (Effectiveness of the RQI Framework)

| Training Set / Model | Conventional FR | RQI | Gain |
|----------------------|----------------|-----|------|
| PIPAL / MANIQA (DIV2K) | 0.624 | **0.744** | +0.120 |
| PIPAL / TOPIQ (DRealSR) | 0.042 | **0.357** | +0.315 |
| Kadid / AHIQ (Set5&14) | 0.292 | **0.426** | +0.134 |

### Key Findings
- PSNR and SSIM exhibit **negative correlation** with human perception across all datasets — a serious challenge to established evaluation conventions in the SR community.
- LPIPS shows near-zero correlation on real-world SR datasets (RealSR, DRealSR).
- No-reference metrics (NIQE, CLIP-IQA) generally outperform FR metrics, but cannot assess fidelity.
- The RQI framework consistently improves all model backbones across all datasets.
- Using RQI as a loss function for SR training simultaneously improves perceptual quality and structural fidelity.

## Highlights & Insights
- A large-scale user study (7 SR models × 5 benchmarks × 15 participants per comparison) provides authoritative human preference data.
- The finding that PSNR/SSIM negatively correlates with human perception is a striking wake-up call for the SR community.
- The elegance of RQI lies in its simplicity — only the training data construction strategy and objective definition are changed, with no architectural modifications.
- The dual utility as both an evaluation metric and a training loss function enhances its practical value.

## Limitations & Future Work
- The number and diversity of user study participants may affect the generalizability of the conclusions.
- Validation is currently limited to ×4 SR; other scale factors and degradation types remain to be explored.
- RQI still requires a GT image as reference and is not applicable in fully no-reference scenarios.
- Using MOS differences as a linear approximation may be insufficiently accurate under extreme quality disparities.

## Related Work & Insights
- AFINE also considers the imperfect GT assumption but requires SR-specific training data, a constraint that RQI does not share.
- LLM-based metrics such as DeQA-Score achieve strong performance but at high computational cost; RQI reaches comparable performance with conventional architectures.
- Insight: Paradigm-level innovation in evaluation metrics (i.e., redefining what "good" means) may be more impactful than model-level innovation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The relative quality framework is conceptually concise and insightful, though the core idea is not highly complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale user study + systematic analysis + multiple models and datasets + application as a loss function.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem analysis is thorough; the abstraction of three goals is precise.
- **Value**: ⭐⭐⭐⭐⭐ Fundamentally advances SR evaluation; the "PSNR/SSIM negative correlation" finding is poised to reshape community practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)

</div>

<!-- RELATED:END -->
