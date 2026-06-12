---
title: >-
  [Paper Note] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency
description: >-
  [Multimodal VLM] This paper proposes SCAN, a dynamic bootstrapping dataset pruning method that iteratively identifies pruning candidates and applies dataset mutation operations…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 79b651e8da25f154
---

# SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2411.09126](https://arxiv.org/abs/2411.09126)
- **Code**: [https://github.com/guoyang9/SCAN](https://github.com/guoyang9/SCAN)
- **Area**: Multimodal VLM
- **Keywords**: Contrastive Pre-training, Data Efficiency, Dataset Pruning, Dynamic Sparse Training, CLIP, MoCo

## TL;DR

This paper proposes SCAN, a dynamic bootstrapping dataset pruning method that iteratively identifies pruning candidates and applies dataset mutation operations, achieving an average performance drop of less than 1% at a 30–35% pruning rate in CLIP and MoCo contrastive pre-training.

## Background & Motivation

Contrastive pre-training (e.g., CLIP, MoCo) is a central paradigm for learning general-purpose representations, yet its **data efficiency** has long been overlooked. The main challenges are:

**Absence of reliable labels**: Self-supervised learning objectives provide no explicit labels, making it infeasible to estimate per-sample class probabilities as in supervised methods (e.g., EL2N).

**Massive data scale**: Pre-training datasets typically contain millions to billions of samples, rendering per-sample gradient or Hessian computation impractical.

Existing approaches primarily rely on **static coreset selection**, filtering important data prior to training. Drawing an analogy to the advantage of dynamic sparse training (DST) over static sparse training, the authors argue that static pruning cannot dynamically track changes in data utility throughout pre-training.

**Core Insight**: Dataset pruning can be decomposed into two sub-problems: (1) metric identification — what proxy metric to use; and (2) pruning strategy design — how to decide which data to prune. SCAN addresses both through a dynamic bootstrapping solution.

## Method

### Metric Selection

**InfoNCE loss** is adopted as the proxy metric, as it satisfies:
- Dynamic adaptability: updated continuously during training
- Low computational overhead: requires no additional computation
- Reflection of learning state: low loss = sufficiently learned; high loss = poor alignment

### Pruning Candidate Identification

Two categories of data are identified for pruning:

**Redundant data**: The $\rho$ fraction of samples with the lowest loss, already well memorized by the model:

$$\mathcal{D}_t^{red} = \mathcal{D}_{t:i}, \quad i \in {}_{\prec\rho}\bar{\mathcal{L}}_{f \rightarrow g}$$

**Ill-matched data**: The $\rho$ fraction of samples with the highest loss, exhibiting poor image-text semantic alignment:

$$\mathcal{D}_t^{ill} = \mathcal{D}_{t:j}, \quad j \in {}_{\succ\rho}\bar{\mathcal{L}}_{f \rightarrow g}$$

The final candidate set is $\mathcal{D}' = \mathcal{D}^{red} \cup \mathcal{D}^{ill}$ (intersection from both ends).

**Warm-up strategy**: Pruning begins when the relative loss reduction between adjacent epochs satisfies $(\mathcal{L}'_{pre} - \mathcal{L}'_{cur})/(\mathcal{L}'_{pre} + \epsilon) \geq T_{td}$, where $T_{td}$ is a predefined threshold.

### Dataset Mutation (Bootstrapping)

Rather than using a fixed pruning ratio, a **cosine annealing strategy** is employed to dynamically adjust the current pruning ratio:

$$\rho_{cur} = \frac{1}{2}\left(1 + \cos\left((\tau_{cos} - (\tau_{cur} \bmod (\tau_{cos}+1))) \frac{\pi}{\tau_{cos}}\right)\right)$$

The pruning ratio increases periodically throughout training; $\rho_{cur}|\mathcal{D}'|$ samples are randomly selected from the candidate set $\mathcal{D}'$ for pruning. Every $(\tau_{cos}+1)$ epochs, the candidate set is regenerated from the full dataset and the process resets for a new iteration.

## Key Experimental Results

### CLIP Pre-training Results (CC12M+, 30% Pruning Rate)

| Architecture | Method | IN Zero-Shot Top-1 | CIFAR10 | CIFAR100 | IN Top-1 | IN-V2 | IN-R |
|------|------|------|---------|----------|----------|-------|------|
| RN101 | CLIP (Full Data) | 18.78 | 95.96 | 82.13 | 75.76 | 64.31 | 40.57 |
| RN101 | Random | 14.05 | 95.02 | 78.34 | 73.99 | 60.27 | 36.13 |
| RN101 | SemDeDup | 13.26 | 95.07 | 78.77 | 74.24 | 62.16 | 37.65 |
| RN101 | D-Pruning | 12.59 | 94.94 | 78.89 | 74.07 | 61.30 | 37.07 |
| RN101 | **SCAN** | **23.10** | **96.08** | **82.28** | **75.66** | **63.75** | **40.10** |
| ViT-B/32 | CLIP (Full Data) | 24.62 | 95.62 | 82.11 | 63.40 | 49.97 | 31.09 |
| ViT-B/32 | Random | 9.12 | - | - | - | - | - |
| ViT-B/32 | **SCAN** | **23.10+** | **95.5+** | **82.0+** | **63.0+** | **49.5+** | **31.0+** |

SCAN achieves near-lossless or even slightly improved performance at 30% pruning (RN101 zero-shot +4.3 points), substantially outperforming static methods.

### MoCo Pre-training Results (ImageNet, 35% Pruning Rate)

| Method | IN Linear Probing | IN Fine-tuning |
|------|-------------------|----------------|
| MoCo-v3 (Full Data) | 76.2 | 83.2 |
| Random | 74.8 | 82.5 |
| **SCAN** | **75.9** | **83.0** |

SCAN remains effective in the unimodal self-supervised setting, with only 0.2–0.3% performance degradation.

### Ablation Study

| Strategy | IN Zero-Shot (RN101) |
|------|---------------------|
| Static pruning (fixed ratio) | 16.52 |
| Pruning redundant data only | 19.84 |
| Pruning ill-matched data only | 18.73 |
| SCAN (dynamic + both types) | **23.10** |

Dynamic bootstrapping outperforms static pruning by 6.6 points; jointly pruning both data types surpasses either type alone.

### Key Findings

1. A substantial proportion of data in contrastive pre-training is either redundant or ill-matched, permitting safe pruning of 30–35%.
2. Dynamic pruning significantly outperforms static methods, as data importance evolves throughout training.
3. The coreset produced as a by-product of SCAN also outperforms other static coreset selection methods when used as a fixed dataset.
4. The method generalizes across both CLIP and MoCo paradigms and seven architectures.

## Highlights & Insights

1. **DST-inspired framing of dataset pruning** — the insight of identifying "which weights matter" in sparse training is transferred to "which data matters."
2. **Dual-end pruning** simultaneously removes redundant (low-loss) and noisy (high-loss) samples.
3. **Cosine annealing bootstrapping** achieves a balance between pruning stability and efficiency.
4. Strong generalizability — applicable to both multimodal (CLIP) and unimodal (MoCo) contrastive pre-training.

## Limitations & Future Work

1. Several warm-up epochs on the full dataset are still required before pruning can begin.
2. Within-batch loss comparisons have limited comparability across different batches.
3. Storing and updating pruning candidates at large scale (>10M) introduces non-trivial overhead.
4. More aggressive pruning rates (e.g., 50%+) remain unexplored.

## Related Work & Insights

- **Dataset Pruning**: EL2N (gradients), Forgetting (forgetting events), Influence Functions
- **Contrastive Pre-training**: CLIP, MoCo, SimCLR
- **Dynamic Sparse Training**: RigL, SET, DST
- **VLP Data Filtering**: SemDeDup, D-Pruning, DataComp

## Rating

- **Novelty**: ★★★★☆ — Cross-cutting innovation at the intersection of dynamic dataset pruning and contrastive pre-training
- **Value**: ★★★★★ — 30% computational savings with near-zero performance loss, directly reducing carbon footprint
- **Experimental Thoroughness**: ★★★★★ — Validated across 16 pre-trained models, two paradigms, and multiple datasets
- **Writing Quality**: ★★★★☆ — Method described clearly with well-chosen analogies

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](../../ICLR2026/multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)

</div>

<!-- RELATED:END -->
