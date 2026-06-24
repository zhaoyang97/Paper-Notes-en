---
title: >-
  [Paper Note] Improving Sustainability of Adversarial Examples in Class-Incremental Learning
description: >-
  [AAAI 2026][Self-Supervised Learning][Adversarial examples] This paper proposes the SAE framework to address the degradation of adversarial examples (AEs) caused by domain drift in class-incremental learning (CIL). Through a semantic correction module (jointly guided by CLIP and the CIL model) and a filtering-and-augmentation module (removing semantically confusing samples), SAE maintains attack effectiveness even after a 9× increase in the number of classes…
tags:
  - "AAAI 2026"
  - "Self-Supervised Learning"
  - "Adversarial examples"
  - "incremental learning"
  - "continual learning"
  - "robustness preservation"
  - "CLIP semantics"
date: 2026-05-08
content_hash: 44f8138dff1df406
---

# Improving Sustainability of Adversarial Examples in Class-Incremental Learning

**Conference**: AAAI 2026
**arXiv**: [2511.09088](https://arxiv.org/abs/2511.09088)  
**Code**: None  
**Area**: LLM NLP / Adversarial Robustness
**Keywords**: Adversarial examples, incremental learning, continual learning, robustness preservation, CLIP semantics

## TL;DR

This paper proposes the SAE framework to address the degradation of adversarial examples (AEs) caused by domain drift in class-incremental learning (CIL). Through a semantic correction module (jointly guided by CLIP and the CIL model) and a filtering-and-augmentation module (removing semantically confusing samples), SAE maintains attack effectiveness even after a 9× increase in the number of classes, achieving an average attack success rate improvement of 31.28%.

## Background & Motivation

- **Background**: Current AEs are typically crafted against static models. However, with the growing adoption of CIL, models are no longer static—the continual introduction of new class data causes significant domain drift in the decision boundaries of old classes.
- **Limitations of Prior Work**: Experiments show that adding as few as 30 new classes (ResNet-32 on CIFAR-100) causes the success rate of SOTA attacks to drop substantially. Semantic-level attacks fall below 20% success rate after more than 30 incremental classes.
- **Key Challenge**: Domain drift alters the direction and magnitude of perturbations required to move inputs toward the target domain, causing old AEs to either misclassify into unintended classes or degrade into benign noise. Optimizing AEs solely on gradients from the initial CIL model leads to overfitting.
- **Key Insight**: CLIP is leveraged to provide universal semantic "anchors" for target classes, combined with CIL model gradients for directional correction, while semantically confusing samples are filtered out.

## Method

### Overall Architecture

SAE consists of two core modules: (1) a **Semantic Correction Module**—jointly guided by CLIP's universal semantics and CIL model gradients to optimize perturbations; and (2) a **Filtering-and-Augmentation Module**—detecting and removing samples containing confounding target-class semantics while enhancing the diversity of the remaining samples. The resulting universal perturbation $\delta$ can be applied to any updated black-box CIL model during the incremental process.

### Key Designs

1. **CLIP Semantic Enhancement (Core of the Semantic Correction Module)**

    - Utilizes a POOD dataset (a publicly available out-of-distribution dataset whose labels overlap neither with target classes nor the CIL training set).
    - Employs CLIP text/image encoders to compute the target direction $D_t$, non-target direction $D_{nt}$, and adversarial direction $D_{adv}$.
    - Optimizes $L_{\text{CLIP}}$ to pull AEs closer to the target class semantics and push them away from non-target class semantics.
    - As CLIP is trained on billions of image-text pairs, its semantic representations possess cross-domain generalizability, serving as an anchor resistant to domain drift.

2. **CIL Model Gradient Correction**

    - CLIP's static semantics alone cannot fully counteract semantic drift; gradients from the initial CIL model $f_1$ are required for correction.
    - A BCE loss is computed as: $-\log(p_{y_t}) - \sum \log(1 - p_{y_{nt}})$.
    - The theoretical basis is that knowledge distillation or orthogonal projection in CIL preserves the effectiveness of gradients from the initial model.

3. **Filtering-and-Augmentation Module**

    - **Filtering**: Certain non-target-class samples inadvertently contain target-class semantics (e.g., a "bicycle" image containing "road" features).
    - Cosine similarity is computed using penultimate-layer features of $f_1$; samples exceeding threshold $\sigma$ are removed.
    - **Augmentation**: Retained samples undergo random rotation, scaling, translation, and patch operations to prevent semantic overfitting.

4. **Attacker Capability Assumptions**

    - Access to the initial CIL model and the full CIL label set is assumed.
    - No access to CIL training data or the training process is required.
    - Publicly available POOD datasets and pretrained CLIP models may be used.

### Loss & Training

- Total loss: $L = L_{\text{CLIP}} + L_{\text{Surr}}$, with $\delta$ updated iteratively via gradient descent and clipped to within $\epsilon$ at each step.
- For each POOD sample category $y_p$, similarity is computed and optimization is performed independently.
- An $l_\infty$ norm constraint ensures perturbation imperceptibility.

## Key Experimental Results

### Main Results (CIFAR-100, average SASR across 10 target classes)

| Attack Method | Finetune | Replay | MEMO | DER | iCaRL | AVG |
|---|---|---|---|---|---|---|
| MIFGSM | 7.29 | 13.41 | 21.71 | 36.85 | 17.94 | 21.64 |
| GAKer | 0.04 | 0.02 | 0.01 | 0.00 | 0.00 | 0.62 |
| SAE (Ours) | Significant gain | Significant gain | Significant gain | Significant gain | Significant gain | +31.28% |

### Ablation Study

| Component | Contribution |
|---|---|
| $L_{\text{CLIP}}$ only | Provides basic semantic guidance, but unstable under large domain drift |
| + $L_{\text{Surr}}$ (CIL correction) | Significantly improves sustainability; complementary to CLIP |
| + Filtering module | Removes confounding semantics, reducing variance |
| + Augmentation module | Further improves generalization and prevents overfitting |

### Key Findings

- **SAE remains effective after a 9× increase in class count**: Average attack success rate improves by 31.28% over baselines, while the strongest baseline nearly collapses after a 3× increase.
- **GradCAM visualization**: As CIL updates proceed, the target-class activation regions of baseline methods progressively shrink, while SAE maintains stable activation.
- **Critical role of CLIP semantic anchors**: Removing the CLIP component leads to a substantial drop in sustainability.
- **Differential impact of CIL methods**: Knowledge-distillation-based methods (e.g., iCaRL, PodNet) retain more information from the old model, making SAE most effective against them.

## Highlights & Insights

- **Novelty of problem formulation**: This is the first systematic study of the "sustainability" of adversarial examples in the context of incremental learning, representing an important intersection of adversarial attack and continual learning research.
- **Clever use of CLIP as a semantic anchor**: CLIP's cross-domain universality naturally serves as a stable reference for resisting domain drift.
- **Realistic attacker assumptions**: Only the initial model and publicly available data are required; no access to CIL training data or the training process is needed.

## Limitations & Future Work

- Only targeted attacks are considered; the sustainability of untargeted attacks under CIL remains unexplored.
- If the target domain deviates significantly from CLIP's training distribution, the quality of the semantic anchors may degrade.
- Experiments are primarily conducted on CIFAR-100 and ImageNet-100; scalability to larger benchmarks has not been verified.
- A defender aware that the attack exploits CLIP semantics could potentially mount targeted defenses.

## Related Work & Insights

- **vs. Traditional transfer attacks (MIFGSM, CleanSheet, etc.)**: Traditional methods pursue cross-architecture transferability without considering the domain drift inherent to CIL; SAE leverages universal semantics to resist domain drift along the temporal dimension.
- **vs. Semantic-level attacks (AIM, CGNC, GAKer)**: These methods bind semantics to a static model; SAE uses an external model (CLIP) to provide stable semantics independent of the CIL process.

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Novel problem formulation at the intersection of incremental learning and adversarial sustainability |
| Technical Depth | ⭐⭐⭐⭐ | Complementary dual-module design is well-motivated with sufficient theoretical support |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers 9 CIL methods × multiple attack baselines with thorough ablation |
| Value | ⭐⭐⭐ | Scenario is relatively specific, but carries important implications for safety-critical applications |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](../../CVPR2026/self_supervised/exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[CVPR 2026\] Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models](../../CVPR2026/self_supervised/representation-steered_incremental_adapter-tuning_for_class-incremental_learning.md)
- [\[CVPR 2026\] Beyond Myopic Alignment: Lookahead Optimization for Online Class-Incremental Learning](../../CVPR2026/self_supervised/beyond_myopic_alignment_lookahead_optimization_for_online_class-incremental_lear.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](../../CVPR2026/self_supervised/hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)

</div>

<!-- RELATED:END -->
