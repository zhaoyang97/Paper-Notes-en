---
title: >-
  [Paper Note] Designing to Forget: Deep Semi-parametric Models for Unlearning
description: >-
  [CVPR 2026][LLM Safety][machine unlearning] This paper proposes the "Designing to Forget" paradigm and introduces a family of deep semi-parametric models (SPMs) that achieve unlearning at inference time by simply removing training samples—without modifying model parameters. On ImageNet classification, SPMs reduce the prediction gap relative to the retrain baseline by 11% and achieve over 10× faster unlearning.
tags:
  - CVPR 2026
  - LLM Safety
  - machine unlearning
  - semi-parametric models
  - test-time deletion
  - data privacy
  - diffusion models
date: 2026-05-08
content_hash: 18dee15fbdb79659
---

# Designing to Forget: Deep Semi-parametric Models for Unlearning

**Conference**: CVPR 2026
**arXiv**: [2603.22870](https://arxiv.org/abs/2603.22870)
**Code**: [github.com/amberyzheng/spm_unlearning](https://github.com/amberyzheng/spm_unlearning)
**Area**: Others (Machine Unlearning / AI Safety)
**Keywords**: machine unlearning, semi-parametric models, test-time deletion, data privacy, diffusion models

## TL;DR
This paper proposes the "Designing to Forget" paradigm and introduces a family of deep semi-parametric models (SPMs) that achieve unlearning at inference time by simply removing training samples—without modifying model parameters. On ImageNet classification, SPMs reduce the prediction gap relative to the retrain baseline by 11% and achieve over 10× faster unlearning.

## Background & Motivation
**Background**: Machine unlearning (MU) is driven by privacy regulations such as GDPR, which require removing the influence of specific samples from trained models. Existing methods primarily approximate the effect of "never having trained on that data" by fine-tuning model parameters.

**Limitations of Prior Work**: The black-box nature of deep learning makes it difficult to disentangle the contribution of individual training samples to model parameters; existing MU algorithms require additional fine-tuning steps, incurring significant overhead in frequent-unlearning scenarios.

**Key Challenge**: Parametric models compress all training data into parameters, making unlearning necessarily parameter-modifying; non-parametric models (e.g., KNN) support deletion natively but lack sufficient performance.

**Goal**: To design a neural network architecture that is inherently suited for unlearning, rather than designing unlearning algorithms for existing architectures.

**Key Insight**: Drawing from the "delete-to-forget" property of KNN, the paper designs semi-parametric models that simultaneously achieve the performance of parametric models and the unlearning convenience of non-parametric models.

**Core Idea**: A semi-parametric model $\hat{y} = G_{\theta^*}(x, \mathcal{T})$ explicitly depends on the training set $\mathcal{T}$ at inference time; unlearning reduces to $G_{\theta^*}(x, \mathcal{T} \setminus \mathcal{U})$—removing data without modifying parameters.

## Method

### Overall Architecture
An SPM consists of three types of modules: (1) a **Fusion** module $g$ that merges the parametric and non-parametric branches; (2) a **Non-parametric** module $h$ that converts the training set into instance embeddings; and (3) a **Parametric** module $f$ consisting of standard deep network layers. The two branches operate alternately: the parametric branch processes input features, the non-parametric branch maintains instance representations of the training set, and the fusion module integrates both at each layer.

### Key Designs
1. **Fusion Module (Weighted Aggregation)**: $g(z, \mathcal{S}) = \sum_{s_i \in \mathcal{S} \setminus \{s_z\}} \alpha(z, s_i) \cdot s_i$, where $\alpha$ denotes attention weights, and crucially, the instance embedding $s_z$ of the current sample itself is excluded. The design motivation is that excluding the self-embedding forces the model to learn relative relationships from other data points, following the spirit of non-parametric methods. Without this exclusion, the model may degenerate into a parametric model by directly using its own embedding.

2. **Non-parametric Module (Permutation Equivariance)**: $\mathcal{S}^{(l)} = \{[h^{(l)}(s_i^{(l)}), y_i]\}_{i=1}^{|\mathcal{T}|}$, applying a shared instance-level transformation to each training sample, preserving permutation equivariance. The design motivation is that permutation equivariance ensures model behavior is independent of data insertion order, and enables clustering/retrieval to reduce set size.

3. **Label-Permutation Augmentation**: During training, the indices of one-hot label vectors are randomly shuffled. The design motivation is to prevent the model from ignoring $x$ and using one-hot labels as a "bias term" to bypass the non-parametric branch. This is a critical trick for ensuring the model genuinely depends on the content of training data.

4. **Efficiency Optimization**: The set $\mathcal{S}$ is reduced to a fixed size via (R)etrieval (nearest-neighbor retrieval) or (C)lustering (per-class averaging). For generative tasks, the SPM is built on a UNet architecture with the mid block replaced by the fusion module.

### Loss & Training
- Classification: cross-entropy loss + label-permutation augmentation
- Generation: standard DDPM diffusion loss; fusion operates at the patch level
- Pre-trained ResNets can be adapted into SPMs by adding a non-parametric branch

## Key Experimental Results

### Main Results (Classification Performance)

| Model | CIFAR-10 Acc↑ | ImageNet Acc↑ |
|-------|--------------|--------------|
| ResNet18 | 94.9 | 68.93 |
| ResNet18-KNN (100%) | 94.5 | 66.9 |
| SPM-C (100%) | 94.5 | 67.1 |
| SPM-R (100%) | 94.1 | 59.9 |

**Generation Performance (CIFAR-10 FID↓)**

| Method | FID | Runtime |
|--------|-----|---------|
| DDPM | 7.28 | 42s |
| SPM ($|\mathcal{T}|$=100) | 7.09 | 173s |
| SPM ($|\mathcal{T}|$=1024) | 7.04 | 1486s |

### Ablation Study (CIFAR-10 Classification Unlearning)

| Method | PG_H↓ | PG_S↓ | Unlearning Time↓ |
|--------|-------|-------|-----------------|
| Retrain (Oracle) | 0.00 | 0.00 | 2317.6s |
| GA | 18.48 | 0.99 | 8.9s |
| FT | 13.11 | 0.48 | 148.7s |
| **SPM-C (Ours)** | **0.43** | **0.08** | **0.7s** |

### Key Findings
- SPM-C achieves near-parity with ResNet18 on classification (CIFAR-10: 94.5 vs. 94.9; ImageNet: 67.1 vs. 68.93).
- **Near-perfect unlearning**: the prediction gap relative to the retrain baseline (PG_H) is only 0.43, compared to 18.48 for the best MU algorithm GA.
- **Extremely fast unlearning**: 0.7s vs. 2317.6s for retraining (3300× speedup) and 8.9s for GA (12.7× speedup).
- The generative SPM (based on DDPM) achieves FID close to standard DDPM (7.04 vs. 7.28), though inference time increases substantially due to set maintenance.
- On ImageNet, SPMs reduce the unlearning prediction gap relative to parametric models by 11%.

## Highlights & Insights
- **Paradigm shift in design philosophy**: from "how to unlearn" (algorithm-centric) to "how to design models that are easy to unlearn" (architecture-centric), representing a paradigm innovation in the MU field.
- **KNN-inspired fusion design**: excluding the self-embedding combined with attention-weighted aggregation elegantly realizes non-parametric behavior within deep networks.
- **Label-permutation augmentation** is critical for preventing the model from bypassing the non-parametric branch, reflecting careful consideration in SPM design.
- Pre-trained parametric models (e.g., ResNet) can be retrofitted as SPMs, reducing the cost of training from scratch.

## Limitations & Future Work
- **Increased inference time**: SPMs require maintaining and retrieving the training set at inference time, adding approximately 20% overhead on ImageNet in clustering mode.
- **ImageNet accuracy gap**: SPM-C (67.1) vs. ResNet18 (68.93) still shows a ~2% gap.
- **Runtime cost for generative SPMs**: at $|\mathcal{T}|=1024$, inference time increases 35×, limiting practical scalability.
- Validation on larger-scale models (e.g., ViT, DiT) has not yet been conducted.
- Only classification and unconditional generation have been verified; more complex generative tasks such as text-to-image generation remain to be explored.

## Related Work & Insights
- **Distinction from SISA** (Bourtoule et al.): SISA trains multiple models on data shards and achieves unlearning by discarding entire models; SPMs maintain a single model and remove data at inference time.
- Semi-parametric models have prior applications in NLP (retrieval-augmented generation) and vision (retrieval-augmented generation), but this work is the first to apply them to unlearning.
- **Complementarity with differential privacy**: DP provides training-time privacy guarantees, while SPMs provide post-deployment sample deletion capability.
- Label-permutation augmentation is analogous to dropout-style regularization—forcing the model not to take shortcuts.

## Technical Details
- **Fusion attention**: $\alpha(z, s_i) = \frac{\exp((W_q z)^\top (W_k s_i))}{\sum_j \exp((W_q z)^\top (W_k s_j))}$
- **Patch-level generative fusion**: the mid block of the UNet is replaced with a fusion module using Bahdanau attention.
- **SPM-C (clustering mode)**: instance embeddings are averaged per class → set size equals number of classes → runtime is comparable to parametric models.
- **GNN enhancement**: a class-aware GNN with multi-head graph attention improves CIFAR-10 from 94.1% to 94.4%.
- **PG_H / PG_S**: hard/soft prediction gap, measuring distance from the retrain oracle.
- **5-class unlearning (50%)**: SPM-C achieves PG_H = 0.02 vs. GA's 32.62, demonstrating near-perfect unlearning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — addresses unlearning from an architectural design perspective; paradigm innovation
- Experimental Thoroughness: ⭐⭐⭐⭐ — dual-task validation on classification and generation, with comparison against multiple MU algorithms
- Writing Quality: ⭐⭐⭐⭐ — concepts are clearly explained with intuitive illustrations
- Value: ⭐⭐⭐⭐ — significant implications for privacy and regulatory compliance scenarios, though inference overhead limits applicability

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[AAAI 2026\] Designing Truthful Mechanisms for Asymptotic Fair Division](../../AAAI2026/llm_safety/designing_truthful_mechanisms_for_asymptotic_fair_division.md)

<!-- RELATED:END -->
