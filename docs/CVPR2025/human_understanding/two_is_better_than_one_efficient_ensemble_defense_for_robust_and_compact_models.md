---
title: >-
  [Paper Note] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models
description: >-
  [CVPR 2025][Human Understanding][Ensemble Defense] Proposes EED (Efficient Ensemble Defense), which generates multiple sub-models from a single base model using different pruning strategies (NIS/ERM/ASE/BNSF) and dynamically ensembles them. At 80% sparsity, it achieves 55.71% PGD robust accuracy on CIFAR-10 (close to the uncompressed baseline) with a $1.86\times$ inference speedup.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Ensemble Defense"
  - "Model Pruning"
  - "Adversarial Robustness"
  - "Robust Diversity"
  - "Compression"
date: 2026-05-08
content_hash: 59315d93c44dbfaf
---

# Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models

**Conference**: CVPR 2025  
**arXiv**: [2504.04747](https://arxiv.org/abs/2504.04747)  
**Code**: None  
**Area**: Human Understanding / Adversarial Robustness  
**Keywords**: Ensemble Defense, Model Pruning, Adversarial Robustness, Robust Diversity, Compression

## TL;DR

Proposes EED (Efficient Ensemble Defense), which generates multiple sub-models from a single base model using different pruning strategies (NIS/ERM/ASE/BNSF) and dynamically ensembles them. At 80% sparsity, it achieves 55.71% PGD robust accuracy on CIFAR-10 (close to the uncompressed baseline) with a $1.86\times$ inference speedup.

## Background & Motivation

### Background

**Background**: Adversarial robust models (such as adversarially trained ResNets) are inherently computationally intensive. Model compression (pruning/quantization) is a key mechanism for deploying robust models, but compression typically degrades robustness significantly.

**Limitations of Prior Work**: (1) A severe trade-off exists between robustness and compression—an 80% sparsity level typically leads to a 5-10% drop in PGD accuracy; (2) Single-model pruning discards diverse robust features, whereas adversarial robustness requires models to defend against various attack directions.

**Key Challenge**: A single pruning run inevitably discards certain robust feature directions, but different pruning strategies preserve distinct directions—enabling complementary ensembles.

**Key Insight**: Leverage 4 different importance metrics to prune the same pre-trained model, generating 12 sparse sub-models (4 metrics $\times$ different random seeds), and use the Robust Diversity (RD) metric to select complementary subsets during dynamic ensembling.

**Core Idea**: Diverse sub-model generation via multiple pruning strategies + RD-guided dynamic ensembling = compressed yet robust models.

### Proposed Solution

**Goal**: ### Key Designs

1. **Multi-Pruning Strategy Sub-model Pool**: NIS (Neural Importance) / ERM (Empirical Risk Minimization) / ASE (Adversarial Saliency) / BNSF (BN Scaling Factor) each preserve distinct "robust directions".

2. **Robust Diversity (RD) Metric**: Measures the prediction diversity among sub-models on adversarial examples, prioritizing subsets with high RD for ensembling.

3. **Three Loss Functions**: $\mathcal{L}_{E}$ (ensemble classification) + $\mathcal{。

## Method

### Key Designs

1. **Multi-Pruning Strategy Sub-model Pool**: NIS (Neural Importance) / ERM (Empirical Risk) / ASE (Adversarial Saliency) / BNSF (BN Scaling Factor) each preserve distinct "robust directions".

2. **Robust Diversity (RD) Metric**: Measures the prediction diversity among sub-models on adversarial examples, prioritizing subsets with high RD for ensembling.

3. **Three Loss Functions**: $\mathcal{L}_{E}$ (ensemble classification) + $\mathcal{L}_{R}$ (misclassification regularization) + $\mathcal{L}_{C}$ (sparsity constraint).

### Loss & Training

$\mathcal{L}_{EED} = \mathcal{L}_E + \omega\mathcal{L}_R + \gamma\mathcal{L}_C$. $N=12$ sub-models, dynamic inference selects the optimal subset.

## Key Experimental Results

| Method | CIFAR-10 PGD | AA | Sparsity | Speed |
|------|-------------|-----|--------|------|
| Uncompressed Baseline | ~56% | ~49% | 0% | $1\times$ |
| Single-Model Pruning | ~45-52% | ~40% | 80% | — |
| **EED** | **55.71%** | **48.13%** | **80%** | **$1.86\times$** |

### Ablation Study
- $\mathcal{L}_C$ is critical for robustness: without $\mathcal{L}_C$, PGD accuracy decreases from 55.71% to 39.10%.
- Dynamic inference is more effective at 50% sparsity (+1.81% PGD).
- The RD metric is particularly important in small ensembles.

### Key Findings
- **Diversity of the 4 pruning strategies is key**—4 copies of a single strategy are far inferior to 4 different strategies.
- **Almost no loss of robustness at 80% sparsity**—55.71% vs. ~56% for the uncompressed baseline.
- **$1.86\times$ acceleration**—sparse ensembles are faster than dense single models.

## Highlights & Insights
- **A new paradigm of compression + robustness**—not "restoring robustness after compression," but "achieving robustness beyond a single model through ensembling."
- **Different pruning strategies preserve distinct robust directions**—this observation provides valuable insights for understanding adversarial robustness.

## Limitations & Future Work
- Performance degrades significantly at extreme sparsity (95%).
- Overhead of training 12 sub-models.
- Dynamic subset selection introduces inference logic overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-pruning ensembling + RD metric are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ CIFAR-10/SVHN/multiple attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Provides a practical solution for deploying robust models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation](two_by_two_learning_multi-task_pairwise_objects_assembly_for_generalizable_robot.md)
- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[CVPR 2025\] Pose Priors from Language Models](pose_priors_from_language_models.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[CVPR 2025\] Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency](efficient_video_face_enhancement_with_enhanced_spatial-temporal_consistency.md)

</div>

<!-- RELATED:END -->
