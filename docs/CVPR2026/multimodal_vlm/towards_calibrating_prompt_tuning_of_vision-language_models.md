---
title: >-
  [Paper Note] Towards Calibrating Prompt Tuning of Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][prompt tuning] Addressing the "dual miscalibration" issue (base class under-confidence + novel class over-confidence) in prompt-tuned CLIP, this work proposes mean-variance margin regularization and text moment matching loss. These complementary regularization terms serve as plug-and-play modules that significantly reduce ECE across 7 prompt tuning methods and 11 datasets.
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "prompt tuning"
  - "calibration"
  - "CLIP"
  - "confidence estimation"
  - "pretrained semantic preservation"
date: 2026-05-08
content_hash: c2a7e20a29f8e0c8
---

# Towards Calibrating Prompt Tuning of Vision-Language Models

**Conference**: CVPR 2026  
**arXiv**: [2602.19024](https://arxiv.org/abs/2602.19024)  
**Code**: [https://github.com/ashshaksharifdeen/TCPT](https://github.com/ashshaksharifdeen/TCPT)  
**Area**: Multimodal VLM  
**Keywords**: prompt tuning, calibration, CLIP, confidence estimation, pretrained semantic preservation

## TL;DR
Addressing the "dual miscalibration" issue (base class under-confidence + novel class over-confidence) in prompt-tuned CLIP, this work proposes mean-variance margin regularization and text moment matching loss. These complementary regularization terms serve as plug-and-play modules that significantly reduce ECE across 7 prompt tuning methods and 11 datasets.

## Background & Motivation
**Background**: Prompt tuning is the dominant approach for adapting CLIP to downstream tasks. It achieves parameter-efficient fine-tuning by learning a few prompt tokens, improving accuracy on base classes while maintaining zero-shot generalization to novel classes.

**Limitations of Prior Work**: Existing prompt tuning methods focus almost exclusively on accuracy, neglecting confidence calibration. Mismatches between predicted confidence and actual accuracy lead to unreliable decision-making, which is particularly hazardous in safety-critical scenarios like autonomous driving and medical imaging.

**Key Challenge**: Prompt tuning causes "dual miscalibration"—the contraction of logit margins for base classes leads to under-confidence, while the inflation of margins for novel classes leads to over-confidence. Existing post-processing calibration methods (e.g., temperature scaling in DAC) cannot constrain how prompt tuning alters the embedding space, potentially leading to embedding collapse or clustering issues.

**Goal**: To simultaneously address base class under-confidence and novel class over-confidence during training without sacrificing accuracy.

**Key Insight**: Analysis reveals a correlation pattern between margin variability and ECE: a negative correlation for base classes and a positive correlation for novel classes.

**Core Idea**: Stabilize the logit distribution by maximizing the average margin and minimizing margin variance, while preserving CLIP's semantic geometry by matching the first and second moments of tuned versus frozen text embeddings.

## Method

### Overall Architecture
The paper addresses the "dual miscalibration" introduced by prompt tuning in CLIP. The authors observe that the statistical properties of per-sample logit margins (the difference between the correct class and the strongest competitor) follow stable correlation patterns with ECE. Thus, calibration is framed as "controlling margin distribution" and "preserving text embedding geometry." In practice, two regularization terms are appended to the standard cross-entropy loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{Margin}}\mathcal{L}_{\text{Margin}} + \lambda_{\text{mom}}\mathcal{L}_{\text{mom}}$. Both terms are active only during training and introduce zero computational overhead during inference.

### Key Designs

**1. Mean-Variance Margin Regularization: Addressing Under-and-Over-confidence in Logit Space**

Base class under-confidence stems from margins that are too small. While increasing the mean margin is an intuitive fix, simply maximizing the mean can push misclassified novel samples toward even more severe over-confidence. The authors constrain both the mean and variance. For each sample, the margin is defined as $m_i = z_{i,y_i} - \max_{j\neq y_i} z_{i,j}$, and the loss is:

$$\mathcal{L}_{\text{Margin}} = -\alpha \cdot \frac{1}{B}\sum_i m_i + \beta \cdot \text{Var}(m_1,\dots,m_B)$$

The mean term (weighted by $\alpha$) ensures base classes are sufficiently separated to recover confidence, while the variance term (weighted by $\beta$) supresses margin dispersion within a batch, preventing individual samples from becoming overly confident and thus mitigating novel class over-confidence.

**2. Text Moment Matching Loss: Preserving CLIP's Semantic Geometry in Embedding Space**

Margin regularization only affects logits and does not directly constrain the embedding space. If prompt learning is too unconstrained, text embeddings may shift globally, destroying relative inter-class relationships. The authors use a moment matching term to anchor the global statistics of tuned text embeddings to those of the frozen CLIP, aligning the first moment (mean) and second moment (covariance):

$$\mathcal{L}_{\text{mom}} = \|\mu_{\tilde{c}} - \mu_{c^0}\|_2^2 + \|\Sigma_{\tilde{c}} - \Sigma_{c^0}\|_F^2$$

Unlike direct $L_2$ alignment per class—which might hinder task adaptation—moment matching only constrains the center and spread of the batch embeddings. This preserves the semantic geometry while allowing sufficient flexibility for downstream adaptation.

### Loss & Training
The total loss combines cross-entropy with the two regularization terms: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{Margin}}\mathcal{L}_{\text{Margin}} + \lambda_{\text{mom}}\mathcal{L}_{\text{mom}}$, where $\lambda_{\text{Margin}}$ and $\lambda_{\text{mom}}$ control the respective strengths.

These terms complement each other. The margin term enhances inter-class discriminability in the logit space but carries a failure mode: when the top-1 prediction is wrong, it may exacerbate over-confidence. The moment matching loss stabilizes the embedding space and maintains inter-class structure, neutralizing this side effect. Ablations confirm that using the margin term alone may increase novel class ECE, whereas the combination yields consistent improvements across both base and novel sets.

## Key Experimental Results

### Main Results (CoOp, Mean over 11 Datasets)

| Method | Base Acc | Base ECE↓ | Novel Acc | Novel ECE↓ |
|------|----------|-----------|-----------|------------|
| Zero-Shot CLIP | 69.50 | 3.58 | - | - |
| CoOp | 81.00 | 6.35 | 71.64 | 6.56 |
| + Temp. Scaling | 83.06 | 2.96 | 72.10 | 5.84 |
| + DAC | - | - | - | 5.21 |
| + ZS-Norm | 80.50 | 3.44 | 71.80 | 4.85 |
| + **Ours** | **81.00** | **2.30** | **71.64** | **3.98** |

### Generalization across Prompt Tuning Methods

| Prompt Tuning Method | Base ECE (Orig) | Base ECE (Ours) | Novel ECE (Orig) | Novel ECE (Ours) |
|------------------|----------------|-----------------|-----------------|-----------------|
| CoOp | 6.35 | 2.30 | 6.56 | 3.98 |
| CoCoOp | 5.89 | 2.45 | 5.32 | 3.67 |
| MaPLe | 4.78 | 1.98 | 4.85 | 3.21 |
| KgCoOp | 5.12 | 2.15 | 5.01 | 3.45 |

### Key Findings
- Consistent ECE reduction across all 7 prompt tuning methods and 11 datasets.
- Accuracy remains largely unaffected (within ±0.5%), indicating that calibration improvements do not come at the cost of performance.
- Moment matching loss contributes most significantly to novel class ECE, validating the importance of preserving embedding geometry for generalization.
- Improvements are particularly notable on challenging datasets like DTD and EuroSat (with ECE reductions exceeding 5 points).

## Highlights & Insights
- **Plug-and-Play**: As a training-time regularizer, it introduces no inference overhead and is compatible with any prompt tuning method.
- **Analysis-Driven Design**: The design stems from an analysis of margin variability and ECE correlation, targeting the specific causes of base under-confidence and novel over-confidence.
- **Moment Matching vs. Direct Alignment**: Matching global statistics rather than per-sample embeddings successfully balances the trade-off between semantic preservation and task-specific adaptation.

## Limitations & Future Work
- Hyperparameters such as $\alpha, \beta, \lambda$ require tuning on a validation set and may vary across datasets.
- Validation is limited to classification tasks; applicability to structured output tasks like detection or segmentation remains unknown.
- Moment matching assumes an approximately Gaussian distribution for category embeddings; performance may degrade for highly asymmetric distributions.
- Domain shift scenarios (e.g., prompt transfer from natural to medical images) were not considered.

## Related Work & Insights
- **vs. DAC**: DAC utilizes post-hoc temperature scaling for novel classes but cannot constrain embedding space deformation during training; Ours preserves embedding structure during training.
- **vs. ZS-Norm**: ZS-Norm matches global statistics of logit distributions; Ours performs moment matching in the embedding space to more fundamentally preserve inter-class relationships.

## Rating
- Novelty: ⭐⭐⭐⭐ The analysis of dual miscalibration and the dual-regularization design are both innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive validation across 7 methods and 11 datasets with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow, though formula notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Calibration is a critical issue for VLM deployment; this method is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](../../ICML2025/multimodal_vlm/understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[CVPR 2026\] Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining](improving_calibration_in_test-time_prompt_tuning_for_vision-language_models_via_.md)
- [\[CVPR 2026\] Cluster-Aware Neural Collapse Prompt Tuning for Long-Tailed Generalization of Vision-Language Models](cluster-aware_neural_collapse_prompt_tuning_for_long-tailed_generalization_of_vi.md)
- [\[CVPR 2026\] CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment](capt_confusion-aware_prompt_tuning_for_reducing_vision-language_misalignment.md)
- [\[CVPR 2026\] FedMPT: Federated Multi-Label Prompt Tuning of Vision-Language Models](fedmpt_federated_multi-label_prompt_tuning_of_vision-language_models.md)

</div>

<!-- RELATED:END -->
