---
title: >-
  [Paper Note] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution
description: >-
  [AAAI 2026][Model Compression][Dataset Distillation] This paper proposes TGDD, which reframes static distribution matching as a dynamic alignment process along training trajectories. It captures evolving semantics via St…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Distribution Matching"
  - "Expert Trajectory"
  - "Distribution Constraint"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 2d07ef25e6f06a24
---

# TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution

**Conference**: AAAI 2026
**arXiv**: [2512.02469](https://arxiv.org/abs/2512.02469)
**Code**: [github.com/FlyFinley/TGDD](https://github.com/FlyFinley/TGDD)
**Area**: Model Compression
**Keywords**: Dataset Distillation, Distribution Matching, Expert Trajectory, Distribution Constraint, Synthetic Data

## TL;DR

This paper proposes TGDD, which reframes static distribution matching as a dynamic alignment process along training trajectories. It captures evolving semantics via Stage-wise Distribution Matching and reduces inter-class overlap via Stage-wise Distribution Constraint, achieving SOTA on 10 datasets with a 5.0% accuracy gain on high-resolution benchmarks.

## Background & Motivation

Dataset distillation aims to compress large-scale datasets into compact synthetic datasets while preserving the training effectiveness of the original data. Existing methods fall into two categories:

**Optimization-oriented (OO-based) methods** (e.g., MTT, FTD): align the learning dynamics of synthetic and real data throughout training via gradient or trajectory matching. These methods achieve strong performance but require bi-level optimization over both networks and data, resulting in **extremely high computational costs** and poor scalability.

**Distribution matching (DM-based) methods** (e.g., DM, IDM, M3D): directly align the feature distributions of synthetic and real data in embedding space. These are efficient but suffer from two critical limitations:
   - **Neglect of representational evolution**: most methods use randomly initialized networks as feature extractors, capturing only early-stage feature distributions and failing to reflect the semantic evolution from low-level to high-level features during training.
   - **Poor inter-class separation**: MMD only constrains distributional means, offering insufficient discriminability at class boundaries, which leads to heavily overlapping inter-class features in the synthetic data.

The authors use t-SNE visualizations to illustrate this problem: class separation in DM-generated synthetic data varies drastically depending on the model's training stage — only highly optimized models can distinguish between classes. This makes the synthetic data difficult to learn from, limiting downstream performance.

**Core Motivation**: Since expert trajectories (model snapshots saved during training) already encode feature representations at every stage of training, why not leverage them directly for distribution matching? This preserves the efficiency of DM methods while introducing the training-dynamics awareness characteristic of OO methods.

## Method

### Overall Architecture

TGDD proceeds in three steps:
1. **Pre-train expert trajectories**: train $N$ expert trajectories on the real dataset, each consisting of $M$ model snapshots (this step is performed only once and the trajectories can be reused).
2. **Stage-wise Distribution Matching**: randomly sample model snapshots from the expert trajectories as feature extractors to align the feature distributions of synthetic and real data across different training stages.
3. **Stage-wise Distribution Constraint**: sample another model from the "expert region" of the same trajectory and impose a classification constraint on the synthetic data to enforce inter-class separation.

### Key Designs

1. **Expert Trajectory Construction**

   $N$ randomly initialized neural networks are trained to convergence over $M$ epochs, with all intermediate snapshots saved:
   $\mathbf{P} = \{p_{i,j} \mid 0 \leq i \leq N, \ 0 \leq j \leq M\}$

   Unlike MTT, which requires 200 trajectories, TGDD achieves competitive performance with only **5 trajectories** (validated in Figure 6(c)). A key advantage is that trajectory pre-training uses only real data and **does not require network training during distillation**, enabling offline pre-training and reuse.

2. **Stage-wise Distribution Matching**

   At each distillation iteration, a trajectory $\mathbf{P}_i$ is randomly selected, and a model snapshot $\theta_{ext} = p_{i,j}$ from a randomly chosen training stage serves as the feature extractor. MMD is used to align the per-class feature distributions between synthetic and real data:

   $L_{MMD} = \sum_{c=1}^{C} \left\| \frac{1}{|B^T_c|} \sum_{i=1}^{|B^T_c|} \psi_{\theta_{ext}}(x_i) - \frac{1}{|B^S_c|} \sum_{i=1}^{|B^S_c|} \psi_{\theta_{ext}}(s_i) \right\|^2$

   By sampling feature extractors from different training stages, the synthetic data is compelled to align with the real data **across all stages**, thereby enriching semantic diversity.

3. **Stage-wise Distribution Constraint**

   Aligning only distributional means via MMD is insufficient to guarantee inter-class separation. Given a feature extractor $\theta_{ext} = p_{i,j}$, another expert model $\theta_{exp}$ is randomly sampled from the "expert region" $P_{er} = \{p_{i,j}, ..., p_{i,j+L-1}\}$ of the same trajectory, and a classification loss is applied to the synthetic data:

   $L_{SDC} = \frac{1}{B^S_c} \sum_{c=1}^{C} \sum_{i=1}^{|B^S_c|} l(\phi_{exp}(s_i), y_i)$

   This loss directly encourages high **intra-class compactness** of synthetic data from the perspective of the expert model, thereby improving inter-class separation. Using different experts across iterations achieves an **ensemble effect** without additional training cost.

### Loss & Training

The total loss is a weighted combination of distribution matching and distribution constraint:
$$L_{overall} = L_{MMD} + \alpha L_{SDC}$$

- $\alpha$ is set to 2.5 for IPC=1,10 and 0.5 for IPC=50
- Expert region length $L = 7$
- Learning rate: 0.1 for ImageNet subsets, 0.01 for others
- 5 expert trajectories, 60 epochs (low-resolution) / 80 epochs (high-resolution)
- Differentiable augmentation (color jitter, random crop, Cutout, etc.) and multi-formation parameterization are applied

## Key Experimental Results

### Main Results

**Low/Medium-resolution datasets (ConvNet-3, CIFAR-10/100, TinyImageNet):**

| Method | Type | CIFAR-10 IPC10 | CIFAR-10 IPC50 | CIFAR-100 IPC10 | CIFAR-100 IPC50 | TinyIm IPC10 | TinyIm IPC50 |
|--------|------|------|------|------|------|------|------|
| DM | DM | 48.9 | 63.0 | 29.7 | 43.6 | 12.9 | 24.1 |
| MTT | OO | 65.3 | 71.6 | 40.1 | 47.7 | 23.2 | 28.0 |
| FTD | OO | 66.6 | 73.8 | 43.4 | 50.7 | 24.5 | — |
| M3D | DM | 63.5 | 69.9 | 42.4 | 50.9 | — | — |
| DANCE | DM | 70.8 | 76.1 | 49.8 | 52.8 | 26.4 | 28.9 |
| **TGDD** | **DM** | **71.9** | **76.5** | **51.3** | **54.6** | **29.3** | **30.9** |

**High-resolution datasets (ImageNet subsets, IPC=10):**

| Method | ImageNette | ImageWoof | ImageFruit | ImageMeow | ImageSquawk | ImageYellow |
|--------|-----------|-----------|-----------|-----------|------------|------------|
| MTT | 63.0 | 35.8 | 40.3 | 40.4 | 52.3 | 60.0 |
| FTD | 67.7 | 38.8 | 44.9 | 43.3 | — | — |
| DANCE | 80.2 | 57.8 | 52.8 | 60.4 | 77.2 | 78.8 |
| **TGDD** | **82.0** | **58.4** | **57.8** | **62.8** | **78.0** | **76.6** |

### Ablation Study

| Aug | $L_{MMD}$ | $L_{SDC}$ | CIFAR-10 IPC10 | CIFAR-10 IPC50 | CIFAR-100 IPC10 | CIFAR-100 IPC50 |
|-----|-----------|-----------|--------|--------|---------|---------|
| ✗ | ✗ | ✗ | 55.2 | 65.3 | 33.7 | 44.5 |
| ✓ | ✗ | ✗ | 63.2 | 69.5 | 40.5 | 47.2 |
| ✓ | ✓ | ✗ | 65.8 | 75.2 | 47.0 | 53.0 |
| ✓ | ✓ | ✓ | **71.9** | **76.5** | **51.3** | **54.6** |

**Cross-architecture generalization (CIFAR-10, IPC50, distilled on ConvNet-3 → evaluated on other architectures):**

| Method | ConvNet-3 | ResNet-10 | DenseNet-121 |
|--------|----------|----------|-------------|
| DM | 63.0 | 58.6 | 57.4 |
| DANCE | 76.1 | 68.0 | 64.8 |
| **TGDD** | **76.5** | **74.9** | **74.3** |

### Key Findings

- **Stage-wise distribution matching is the primary source of improvement**: adding $L_{MMD}$ improves CIFAR-10 IPC50 from 69.5→75.2 (+5.7%).
- **Distribution constraint further improves class separation**: adding $L_{SDC}$ improves CIFAR-100 IPC10 from 47.0→51.3 (+4.3%).
- **Strong cross-architecture generalization**: TGDD substantially outperforms DANCE on ResNet-10 (74.9 vs. 68.0) and DenseNet-121 (74.3 vs. 64.8).
- **Pronounced advantage on high-resolution datasets**: TGDD surpasses DANCE by 5.0% on ImageFruit IPC10.
- Performance is moderately sensitive to $\alpha$ (within a 2.8% range) and robust to trajectory count and length.
- A single trajectory already achieves near-5-trajectory performance, far below MTT's requirement of 200 trajectories.

## Highlights & Insights

- **Elegant fusion of two paradigms**: trajectory information from OO methods is used to enhance feature extraction in DM methods, retaining DM's efficiency while acquiring OO's expressiveness.
- **Clever expert region design**: feature matching and distribution constraint use different snapshots from the same trajectory, naturally injecting knowledge from different training stages.
- **Minimal storage requirements**: only 5 trajectories are needed (vs. 200 for MTT), and they can be pre-trained and reused.
- **Pareto-optimal performance-efficiency trade-off** (Figure 1): highest accuracy achieved under equivalent GPU memory and distillation time constraints.

## Limitations & Future Work

- Distillation is performed only on ConvNet architectures; using ViT or ResNet as the distillation backbone remains unexplored.
- Cross-architecture generalization, while superior to baselines, still degrades — particularly in the ConvNet→DenseNet setting.
- Overall performance on high-resolution datasets remains substantially below full-dataset training (e.g., 62.8% vs. 66.7%).
- Only classification tasks are supported; dataset distillation for downstream tasks such as detection and segmentation has not been explored.
- The distribution constraint employs standard cross-entropy loss; more refined objectives (e.g., contrastive loss) could be considered.
- Continual learning experiments use small step sizes (5 and 10); long-term forgetting effects warrant further investigation.

## Related Work & Insights

- **MTT / FTD**: representative trajectory matching methods with strong performance but prohibitive computational overhead; TGDD borrows the concept of trajectories but uses them solely for feature extraction rather than gradient matching.
- **DM / M3D / DANCE**: the evolutionary line of DM-based methods; TGDD is a natural extension of this line, addressing the fundamental problem of static feature extraction.
- **IDM** employs a dynamic model queue for richer representations but incurs additional training cost; TGDD achieves the same effect via pre-trained trajectories without added overhead.
- Continual learning experiments broaden the application scope of dataset distillation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Trajectory-guided distribution matching is a natural yet effective idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 datasets, cross-architecture evaluation, ablation, hyperparameter analysis, and continual learning.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear exposition with rich visualizations.
- **Value**: ⭐⭐⭐⭐ — A significant improvement to the DM line, balancing performance and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DP-GenG: Differentially Private Dataset Distillation Guided by DP-Generated Data](dp-geng_differentially_private_dataset_distillation_guided_by_dp-generated_data.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](../../NeurIPS2025/model_compression/hyperbolic_dataset_distillation.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)

</div>

<!-- RELATED:END -->
