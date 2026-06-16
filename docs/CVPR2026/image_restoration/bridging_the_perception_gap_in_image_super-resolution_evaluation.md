---
title: >-
  [Paper Note] Bridging the Perception Gap in Image Super-Resolution Evaluation
description: >-
  [CVPR 2026][Image Restoration][Paper Note] A large-scale user study reveals that existing SR evaluation metrics (PSNR, SSIM, LPIPS, etc.) are severely inconsistent with human perception. After analyzing these inherent defects, a minimalist yet effective framework called Relative Quality Index (RQI) is proposed. By learning the relative quality difference betwee
tags:
  - CVPR 2026
  - Image Restoration
date: 2026-05-08
content_hash: b045e26ed33f5d7b
---
# Bridging the Perception Gap in Image Super-Resolution Evaluation

**Conference**: CVPR 2026  
**arXiv**: [2503.13074](https://arxiv.org/abs/2503.13074)  
**Code**: [Project Page](https://color.cvc.uab.cat/rqi/)  
**Area**: Image Super-Resolution / Image Quality Assessment  
**Keywords**: Super-resolution evaluation, image quality metrics, perception gap, relative quality index, user study

## TL;DR
A large-scale user study reveals that existing SR evaluation metrics (PSNR, SSIM, LPIPS, etc.) are severely inconsistent with human perception. After analyzing these inherent defects, a minimalist yet effective framework called Relative Quality Index (RQI) is proposed. By learning the relative quality difference between image pairs, RQI achieves more reliable SR evaluation and can serve as a loss function to guide SR training.

## Background & Motivation
**Background**: As SR technology advances rapidly (RealESRGAN → SwinIR → StableSR → SeeSR), the quality of model outputs continues to improve; however, evaluation metrics have remained unchanged for a long time.

**Limitations of Prior Work**: Researchers increasingly distrust evaluation metrics, as models with high scores do not necessarily yield better visual effects. Numerous studies have to rely on user studies or multiple stacked metrics for verification.

**Key Challenge**: While SR models evolve quickly, evaluation standards have stagnated. There are four inherent challenges between metrics and human perception:
   - (a) **Distortion-based FR metrics** (PSNR, SSIM) favor smooth averaged solutions, which is contrary to perceptual preference.
   - (b) **Perceptual FR metrics** (LPIPS, DISTS) fail when the Ground Truth (GT) quality is poor.
   - (c) **No-reference metrics** (NIQE, CLIP-IQA) cannot evaluate fidelity.
   - (d) Differences between high-quality SR outputs are subtle and indistinguishable by existing metrics.

**Goal**: To design an SR evaluation framework that can simultaneously address the four challenges mentioned above.

**Key Insight**: Replace absolute quality scores with relative quality differences—allowing any image (including those with degradation) to be used as a reference to learn the quality drop between the target and reference.

**Core Idea**: Since GT may be imperfect and SR outputs may surpass GT, the assumption that the reference is perfect should be discarded in favor of learning **relative** quality relationships.

## Method

### Overall Architecture

The core shift of RQI is: since GT may be imperfect or SR outputs may even surpass GT, the assumption that the reference image is perfect and the prediction of an "absolute quality score" are abandoned. Instead, the framework learns the relative quality drop between two images. During training, dense image pairs $\{I_i, I_j\}$ are constructed from IQA datasets, using their MOS difference $q_i - q_j$ as the label. An FR-IQA model is trained to predict this difference. During evaluation, both the SR output $I_{HR}$ and the GT $I_{GT}$ are fed into the model, outputting $s = f_{RQI}(I_{HR}, I_{GT})$. A positive value indicates that the SR quality is superior to the GT.

### Key Designs

**1. Three Properties of the Relative Quality Framework: Addressing Three Evaluation Difficulties Simultaneously**

Existing metrics struggle because distortion-based FR metrics favor smooth solutions, perceptual FR metrics fail with poor GT, and subtle differences in high-quality SR are hard to distinguish. RQI addresses these with three interlocking properties: asymmetry $f_{RQI}(I_i, I_j) = -f_{RQI}(I_j, I_i)$, where swapping inputs negates the output (unlike traditional symmetric FR metrics), corresponding to fidelity evaluation; relative difference, which learns the perceptual drop between images rather than predicting absolute scores, allowing for robust evaluation with degraded references; and dense pairing comparison, where RQI constructs arbitrary "degradation vs. degradation" pairs $\{I_i, I_j\}$ instead of just "reference vs. degradation" pairs $\{I_0, I_i\}$. This significantly expands training samples and naturally captures subtle quality gradients for fine-grained distinction.

**2. Huber Loss for Relative Difference Regression: Stabilizing Training on Subtle Differences**

Learning relative drops requires stable gradients for very small quality differences. RQI employs a Huber loss to regress the MOS difference: when the residual $|\hat{y}_{ij} - (q_i - q_j)| \le \delta$, it takes $\tfrac{1}{2}(\hat{y}_{ij} - (q_i - q_j))^2$; otherwise, it takes $\delta(|\hat{y}_{ij} - (q_i-q_j)| - \tfrac{1}{2}\delta)$. Labels are normalized to $[-1, 1]$, and the activation of the final regression layer is removed to support negative outputs. The Huber loss provides smooth gradients for small differences while remaining insensitive to large outliers, resulting in more stable training on subtle quality variances.

**3. A Universal Framework Rather Than a New Metric: Enhancing Existing FR-IQA Paradigms**

RQI does not aim to create a specific new metric but rather to transform the training paradigm of existing ones. It can be applied to any FR-IQA model (AHIQ, MANIQA, TOPIQ) and trained on any IQA dataset (Kadid-10K, PieAPP, PIPAL), allowing for zero-shot transfer to SR evaluation without collecting SR-specific data. By modifying only the training pairing and target definition without altering the architecture, it universally improves the human alignment of various off-the-shelf metrics.

### Loss & Training

Huber loss is used to regress relative differences, where $\delta$ is the smoothing threshold. The dataset is split 8:2 for training/validation with non-overlapping scenes. The best model from the validation set is selected for zero-shot transfer evaluation.

## Key Experimental Results

### Main Results (Consistency with Human Perception, SRCC Metric)

| Metric | DIV2K | RealSR | DRealSR | Set5&14 |
|------|-------|--------|---------|---------|
| SSIM | -0.348 | -0.220 | -0.354 | -0.321 |
| PSNR | -0.079 | -0.116 | -0.355 | -0.204 |
| LPIPS | 0.415 | 0.008 | -0.141 | 0.282 |
| CLIP-IQA | 0.593 | 0.377 | 0.268 | 0.642 |
| AFINE | 0.581 | 0.449 | 0.484 | 0.578 |
| DeQA-Score | 0.613 | 0.452 | 0.437 | 0.699 |
| **RQI** | **0.744** | **0.504** | **0.529** | 0.664 |

### Ablation Study (Effectiveness of RQI Framework)

| Training Set / Model | Traditional FR | RQI | Gain |
|--------------|--------|-----|------|
| PIPAL / MANIQA (DIV2K) | 0.624 | **0.744** | +0.120 |
| PIPAL / TOPIQ (DRealSR) | 0.042 | **0.357** | +0.315 |
| Kadid / AHIQ (Set5&14) | 0.292 | **0.426** | +0.134 |

### Key Findings
- PSNR and SSIM are **negatively correlated** with human perception across all datasets! This seriously challenges the evaluation conventions in the SR field.
- LPIPS shows near-zero correlation on real-world SR datasets (RealSR, DRealSR).
- NR metrics (NIQE, CLIP-IQA) generally outperform FR metrics but cannot evaluate fidelity.
- The RQI framework consistently improves the performance of all models across all datasets.
- Training SR models with RQI as a loss function simultaneously improves perceptual quality and structural fidelity.

## Highlights & Insights
- The large-scale user study (7 SR models × 5 benchmarks × 15 participants/comparison) provides authoritative human preference data.
- The discovery that "PSNR/SSIM are negatively correlated with human perception" serves as a wake-up call for the SR community.
- The ingenuity of the RQI framework lies in its extreme simplicity—altering only the training data construction and target definition without changing architectures.
- The dual-purpose utility as a loss function adds significant practical value.

## Limitations & Future Work
- The number and diversity of participants in the user study might affect the universality of the conclusions.
- Currently, verification is limited to $\times 4$ SR tasks; other magnification scales and degradation types remain to be tested.
- RQI still requires a GT image as a reference, making it inapplicable in completely no-reference scenarios.
- Using MOS difference as a linear approximation may be inaccurate under extreme quality variances.

## Related Work & Insights
- AFINE also considers the imperfect GT assumption but requires SR-specific data for training, a restriction RQI does not have.
- LLM-based metrics like DeQA-Score perform well but have high computational costs, whereas RQI achieves similar levels using traditional architectures.
- Insight: Paradigm innovation in evaluation metrics (defining "what is good") might be more important than model innovation.

## Rating
- Novelty: ⭐⭐⭐⭐ The RQI relative quality framework is simple yet profound, though the core idea is not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale user study + systematic analysis + multi-model multi-dataset + utility as a loss function.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough problem analysis and accurate abstraction of the three goals.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental advancement to the SR evaluation field; the "PSNR/SSIM negative correlation" discovery will likely change community conventions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bridging Human Evaluation to Infrared and Visible Image Fusion](bridging_human_evaluation_to_infrared_and_visible_image_fusion.md)
- [\[CVPR 2026\] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution](bridging_fidelity-reality_with_controllable_one-step_diffusion_for_image_super-r.md)
- [\[CVPR 2026\] Task-Aware Image Signal Processor for Advanced Visual Perception](task-aware_image_signal_processor_for_advanced_visual_perception.md)
- [\[CVPR 2026\] AceTone: Bridging Words and Colors for Conditional Image Grading](acetone_bridging_words_and_colors_for_conditional_image_grading.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)

</div>

<!-- RELATED:END -->
