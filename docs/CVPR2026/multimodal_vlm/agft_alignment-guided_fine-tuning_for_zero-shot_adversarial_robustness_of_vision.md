---
title: >-
  [Paper Note] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][adversarial robustness] AGFT proposes an alignment-guided fine-tuning framework that enhances zero-shot adversarial robustness of VLMs while preserving the pre-trained cross-modal semantic structure, through text-guided adversarial training and distribution consistency calibration. The method achieves an average robust accuracy of 46.57% across 15 zero-shot benchmarks, surpassing the previous state of the art by 3.1 percentage points.
tags:
  - CVPR 2026
  - Multimodal VLM
  - adversarial robustness
  - vision-language models
  - zero-shot generalization
  - alignment guidance
  - distribution consistency calibration
date: 2026-05-08
content_hash: 23f31106b1abe21e
---

# AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.29410](https://arxiv.org/abs/2603.29410)
**Code**: [GitHub](https://github.com/YuboCui/AGFT)
**Area**: Multimodal VLM / Adversarial Robustness
**Keywords**: adversarial robustness, vision-language models, zero-shot generalization, alignment guidance, distribution consistency calibration

## TL;DR
AGFT proposes an alignment-guided fine-tuning framework that enhances zero-shot adversarial robustness of VLMs while preserving the pre-trained cross-modal semantic structure, through text-guided adversarial training and distribution consistency calibration. The method achieves an average robust accuracy of 46.57% across 15 zero-shot benchmarks, surpassing the previous state of the art by 3.1 percentage points.

## Background & Motivation
**State of the Field**: VLMs such as CLIP exhibit strong zero-shot capabilities but are highly vulnerable to adversarial perturbations (CLIP's zero-shot robust accuracy is only 6.24%).

**Limitations of Prior Work**:
- Existing methods (TeCoA, GLADIATOR) adopt classification-guided adversarial fine-tuning, using hard-label supervision to push features toward target class clusters.
- This paradigm **disrupts the pre-trained cross-modal alignment structure**, distorting fine-grained semantic correspondences between images and text, thereby degrading zero-shot generalization.

**Root Cause**: Enhancing adversarial robustness requires modifying the visual feature space, yet such modifications destroy the cross-modal semantic structure upon which CLIP's generalization depends. How can a balance between "robustness" and "alignment" be achieved?

**Starting Point**: Rather than fine-tuning the VLM as a classifier, the paper preserves its nature as a cross-modal alignment model — using the original model's probabilistic predictions as soft supervision to guide adversarial features toward alignment with text embeddings.

**Core Idea**: Replacing hard labels with soft alignment distributions, combined with temperature calibration to eliminate confidence-scale mismatch, yields adversarial training that preserves cross-modal structure.

## Method

### Overall Architecture
Input: ImageNet training set → PGD generates adversarial examples $\mathbf{x}_{adv}$ → frozen pre-trained CLIP computes soft prediction distribution $\mathbf{p}_{rob}$ (after temperature calibration) → fine-tune the image encoder so that predictions on adversarial examples match $\mathbf{p}_{rob}$ → evaluate on 15 zero-shot datasets.

### Key Designs
1. **Text-Guided Adversarial Training**:

    - **Function**: Uses soft probability distributions from the pre-trained CLIP (rather than hard labels) as the adversarial training target.
    - **Mechanism**: $p_{orig}^{i,j} = \frac{\exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^j)) / \tau)}{\sum_k \exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^k)) / \tau)}$, with the adversarial training loss expressed in KL-divergence form: $L = -\mathbb{E}_{i,j}[p_{rob}^{i,j} \log \frac{\exp(\cos(f_\theta(x_{adv}^i), f_\phi(t^j))/\tau)}{\sum_k ...}]$
    - **Design Motivation**: Hard labels attend only to the correct class, ignoring the relative similarity relationships between an image and other texts. Soft distributions preserve these relationships, enabling the fine-tuned feature space to maintain the same semantic structure as the original CLIP.

2. **Distribution Consistency Calibration**:

    - **Function**: Adjusts the target distribution via temperature scaling to disentangle confidence scale from semantic structure.
    - **Mechanism**: $p_{rob}^{i,j} = \frac{\exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^j)) / (\tau/\gamma))}{\sum_k \exp(\cos(f_{\theta_{orig}}(x^i), f_\phi(t^k)) / (\tau/\gamma))}$, where $\gamma \in (0,1]$; increasing the effective temperature smooths the distribution.
    - **Design Motivation**: Directly using $p_{orig}$ as the target would force the robust model to inherit the pre-trained model's confidence scale (absolute logit magnitude), which may be misaligned with the robust feature space. Temperature scaling decouples "relative semantic relationships" from "confidence scale," retaining only the former as the supervision signal.

3. **Final Objective**:

    $$\min \mathbb{E}_{\mathbf{x} \in \mathcal{D}}[\max_{\mathbf{x}_{adv} \in B(\mathbf{x}, \epsilon)} L(\mathbf{x}_{adv}, \mathbf{t}, \mathbf{p}_{rob}, \tau)]$$

    - Inner maximization: PGD generates adversarial examples.
    - Outer minimization: aligns adversarial predictions with the calibrated soft distribution.

### Loss & Training
- Only the image encoder is fine-tuned (full parameters); the text encoder is frozen.
- SGD, lr=$4 \times 10^{-4}$, cosine decay, 10 epochs.
- Adversarial training uses 2-step PGD, $\epsilon \in \{1/255, 2/255, 4/255\}$.
- Hyperparameters: $\gamma = 0.4$, $\tau = 1/180$.

## Key Experimental Results

### Main Results (PGD-20, $\epsilon=1/255$, Zero-Shot Robust Accuracy)

| Method | Caltech101 | CIFAR10 | Food101 | ImageNet | STL10 | Avg. (15 datasets) |
|--------|-----------|---------|---------|----------|-------|-------------------|
| CLIP (no defense) | 21.27 | 10.31 | 4.06 | 1.13 | 33.10 | 6.24 |
| TeCoA | 71.83 | 59.85 | 29.01 | 41.29 | 83.33 | 38.51 |
| GLADIATOR | 73.34 | 67.89 | 34.92 | 44.53 | 86.53 | 43.46 |
| **AGFT** | **82.23** | **71.72** | **44.76** | 44.95 | **88.52** | **46.57** |

### Zero-Shot Clean Accuracy

| Method | Avg. Clean (15 datasets) | Avg. Robust (15 datasets) | Notes |
|--------|--------------------------|--------------------------|-------|
| CLIP | 66.20 | 6.24 | Strong clean accuracy but highly non-robust |
| TeCoA | 56.93 | 38.51 | Severe clean accuracy drop |
| GLADIATOR | 60.34 | 43.46 | Better balance |
| **AGFT** | **61.35** | **46.57** | Best on both clean and robust |

### Key Findings
- AGFT simultaneously outperforms all baselines on both robustness and clean accuracy, demonstrating that preserving the alignment structure yields a win-win outcome.
- The largest gains appear on fine-grained datasets such as StanfordCars (+12.6%) and Food101 (+9.8%).
- AGFT maintains its advantage under stronger attacks including C&W and AutoAttack.

## Highlights & Insights
- **Deep core insight**: The paper identifies that classification-guided fine-tuning destroys cross-modal alignment as the performance bottleneck for zero-shot adversarial robustness.
- The temperature calibration analysis offers a novel perspective — decoupling "semantic structure" from "confidence scale."
- The method is remarkably simple: it essentially replaces only the target distribution in adversarial training.

## Limitations & Future Work
- Validation is limited to ViT-B/32; effectiveness on larger architectures (e.g., ViT-L) remains to be verified.
- The temperature parameter $\gamma$ requires tuning and may need adjustment for different domains.
- Robust accuracy on domain-specific datasets such as EuroSAT remains relatively low (16.25%).

## Related Work & Insights
- The approach shares conceptual similarity with knowledge distillation, but targets structural preservation rather than model compression.
- The temperature calibration technique is inspired by temperature tricks in label smoothing and knowledge distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The alignment-guided paradigm as a replacement for classification-guided fine-tuning is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 15 datasets × multiple attack types × multiple baselines; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation derivation and method exposition follow rigorous logic.
- Value: ⭐⭐⭐⭐ — Provides important insights for research on adversarial robustness of VLMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/multimodal_vlm/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[CVPR 2026\] AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors](anomalyvfm_--_transforming_vision_foundation_models_into_zero-shot_anomaly_detec.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)

<!-- RELATED:END -->
