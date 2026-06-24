---
title: >-
  [Paper Note] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion
description: >-
  [ECCV 2024][Local learning] HPFF is proposed to resolve block-to-block information loss and high GPU memory footprint in local learning through Hierarchical Locally Supervised Learning (HiLo, partitioning the network into independent and cascade levels of local modules) and Patch Feature Fusion (PFF, dividing auxiliary network inputs into patches for individual computation and average fusion). HPFF significantly outperforms existing local learning methods across multiple data…
tags:
  - "ECCV 2024"
  - "Local learning"
  - "gradient isolation"
  - "hierarchical supervision"
  - "Patch Feature Fusion"
  - "biological plausibility"
date: 2026-05-08
content_hash: c5f19c79c6258bfe
---

# HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion

**Conference**: ECCV 2024  
**arXiv**: [2407.05638](https://arxiv.org/abs/2407.05638)  
**Code**: [https://github.com/Zeudfish/HPFF](https://github.com/Zeudfish/HPFF)  
**Area**: Other  
**Keywords**: Local learning, gradient isolation, hierarchical supervision, Patch Feature Fusion, biological plausibility

## TL;DR

HPFF is proposed to resolve block-to-block information loss and high GPU memory footprint in local learning through Hierarchical Locally Supervised Learning (HiLo, partitioning the network into independent and cascade levels of local modules) and Patch Feature Fusion (PFF, dividing auxiliary network inputs into patches for individual computation and average fusion). HPFF significantly outperforms existing local learning methods across multiple datasets and approaches or even surpasses BP.

## Background & Motivation

Traditional deep learning relies on end-to-end backpropagation (BP), which faces two major issues:
- **Biological implausibility**: BP relies on global error signals propagating backward layer by layer, which does not align with the learning mechanisms of biological neural networks.
- **Update lock problem**: Parameters of hidden layers cannot be updated before both forward and backward passes are completed, hindering efficient parallel training.

Local Learning divides the network into gradient-isolated local modules, each optimized independently via an auxiliary network, alleviating the update lock problem and reducing memory footprint. However, existing methods suffer from critical drawbacks:
- **Shortsightedness**: Each module independently optimizes local objectives while lacking global interactions, easily falling into local optima.
- **Large performance gap**: When the network is divided into a large number of local modules (e.g., one module per layer), the performance gap between local learning and BP widens significantly.
- **Memory overhead of auxiliary networks**: The design of auxiliary networks itself consumes substantial GPU memory, diminishing the memory advantage of local learning.

Core Problem: How to facilitate information exchange between modules while maintaining gradient isolation, and reduce the memory consumption of auxiliary networks?

## Method

### Overall Architecture

HPFF consists of two complementary components:
1. **HiLo (Hierarchical Locally Supervised Learning)**: Divides the network into two levels of local modules—independent level and cascade level—achieving information exchange between modules through weight sharing and multi-level supervision.
2. **PFF (Patch Feature Fusion)**: Splits the input features of auxiliary networks into patches, computes them individually, and fuses them by averaging, thereby reducing GPU memory usage and enhancing the capture of generic patterns.

### Key Designs

1. **Hierarchical Local Modules (HiLo)**: The network is divided into $K$ basic local modules. Each module simultaneously belongs to a two-level structure:

    - **Independent Level (IL)**: Each module $f_{\theta_j}$ is equipped with an independent auxiliary network $g_{\gamma_j}$ to generate a local supervision signal $\hat{y_j} = g_{\gamma_j}(x_{j+1})$.
    - **Cascade Level (CL)**: Every $k$ adjacent modules form a cascade module, sharing a cascade auxiliary network $h_{\beta_i}$. The cascade auxiliary network receives the output of the last submodule of this cascade module: $\hat{y_i} = h_{\beta_i}(x_{i+k})$.

   Key property: Adjacent cascade modules overlap in a sliding-window manner, so each local module receives $k+1$ supervisions (1 from the independent level + $k$ from the cascade levels). The parameter update rule is:

    $\theta_j \leftarrow \theta_j - \eta_d \nabla_{\theta_j} \mathcal{L}(\hat{y_j}, y) - \sum_{n=i}^{i+k-1} \eta_c \nabla_{\theta_j} \mathcal{L}(\hat{y_n}, y)$

   In experiments, $k$ is set to 2, meaning each cascade module contains two basic modules. An excessively large $k$ would degenerate into BP and increase memory footprint.

   **Design Intuition**: The independent level excels at learning local features (tight intra-class clustering), while the cascade level excels at learning global features (well-separated inter-class boundaries); the two are complementary. This is clearly validated by t-SNE visualization.

2. **Patch Feature Fusion (PFF)**: Divides the output feature $x_{j+1}$ of the local module into $n \times n$ patches, feeds them into the auxiliary network separately, and then averages them:

    $$\hat{y_j} = \frac{\sum_{k=1}^{n} \sum_{l=1}^{n} g_{\gamma_j}(x_{j+1}^{(k,l)})}{n^2}$$

   Memory analysis: The memory footprint of the auxiliary network in the original method is $O(D + P + L \times D)$; under PFF, only one patch is processed at a time, reducing the memory to $O(D/n^2 + P + L \times D/n^2)$. In experiments, $n=2$, theoretically reducing the auxiliary network-related memory to $1/4$.

   **Extra Gain**: Patch-level average fusion allows the network to focus on patterns prevalent across multiple patches, learning more generalizable feature representations. Feature visualization shows more and finer-grained activation regions after PFF.

3. **Plug-and-Play Compatibility**: HPFF can be directly integrated into existing local learning methods (such as PredSim, DGL, and InfoPro) without modifying the backbone network structure.

### Loss & Training

- Each local module uses cross-entropy loss, with weighted aggregation of multi-route supervision from both independent and cascade auxiliary networks.
- The final module is directly connected to a global pooling layer and a fully-connected layer to output classification results.
- CIFAR-10/SVHN: SGD with Nesterov, lr=0.8, batch=1024, 400 epochs, cosine annealing.
- STL-10: lr=0.1, batch=128.
- ImageNet: VGG13 lr=0.025, ResNet-101/152 lr=0.05, 90 epochs.

## Key Experimental Results

### Main Results

**CIFAR-10 (Test Error ↓):**

| Method | ResNet-32 K=16 | ResNet-110 K=55 | Gain |
|------|---------------|----------------|------|
| DGL | 14.08 | 14.45 | Baseline |
| DGL + HPFF | **8.94** | **8.74** | ↓5.14 / ↓5.71 |
| InfoPro | 12.93 | 13.22 | Baseline |
| InfoPro + HPFF | **8.99** | **8.96** | ↓3.94 / ↓4.26 |
| BP (End-to-End) | 6.37 | 5.42 | Traditional Upper Bound |

**ImageNet (Top-1 Error ↓):**

| Network | Method | Top-1 Error | Top-5 Error |
|------|------|-------------|-------------|
| ResNet-101 (K=4) | InfoPro | 22.81 | 6.54 |
| ResNet-101 (K=4) | InfoPro + HPFF | **21.14** (↓1.67) | **5.49** (↓1.05) |
| ResNet-152 (K=4) | BP | 21.60 | 5.92 |
| ResNet-152 (K=4) | InfoPro + HPFF | **20.99** (↓1.94) | **5.29** (↓1.42) |
| ResNeXt-101 32×8d (K=4) | BP | 20.64 | 5.40 |
| ResNeXt-101 32×8d (K=4) | InfoPro + HPFF | **19.94** (↓1.75) | **5.09** (↓1.02) |

On ImageNet, HPFF makes it possible for local learning to **surpass BP** (e.g., on ResNet-152, ResNeXt-101).

### Ablation Study

**Component Contributions (DGL + ResNet-32 K=16, CIFAR-10):**

| IL | CL | PFF | Test Error | Gain |
|----|----|----|------------|------|
| ✓ | ✗ | ✗ | 14.08 | Baseline |
| ✗ | ✓ | ✗ | 10.51 | ↓3.57 |
| ✓ | ✓ | ✗ | 9.44 | ↓4.64 |
| ✓ | ✓ | ✓ | **8.94** | ↓5.14 |

**GPU Memory Comparison:**

| Network | Method | GPU Memory (GB) | Relative to BP |
|------|------|-------------|---------|
| ResNet-110 K=55 | BP | 9.26 | — |
| ResNet-110 K=55 | DGL + HPFF | 2.44 | ↓73.7% |
| ResNet-110 K=55 | InfoPro + HPFF | 2.38 | ↓74.3% |
| ResNet-110 K=55 | PredSim + HPFF | 1.90 | ↓79.5% |

### Key Findings

- **Complementarity of Independent and Cascade Levels**: Using CL alone (10.51) already outperforms IL (14.08), while HiLo combining both (9.44) yields the best performance. t-SNE confirms that IL learns local features and CL learns global features.
- **PFF Simultaneously Improves Performance and Reduces Memory**: Adding PFF drops the test error from 9.44 to 8.94, while lowering the GPU memory from 3.13GB to 2.31GB.
- **Lower Accuracy in Early Layers is Beneficial**: CKA analysis and layered linear classifier experiments show that HPFF structures early layers not to over-optimize local objectives, thereby retaining more features beneficial for global performance.
- **The More Modules Divided, the Larger the Gain from HPFF**: The improvement is most significant when $K=55$ (one module per layer), where information loss between modules is most severe.
- **High Generalizability**: HPFF consistently achieves significant improvements across three different local learning methods, four datasets, and various network architectures.

## Highlights & Insights

1. **Elegant Hierarchical Design of Independent and Cascade Levels**: It enables cross-module information transfer through sliding-window-style cascade modules while maintaining gradient isolation, avoiding new global gradients.
2. **PFF Kills Two Birds with One Stone**: It reduces memory consumption (by processing small patches) and improves performance (via multi-patch averaging focusing on general patterns), representing a simple and effective design.
3. **Local Learning Surpassing BP Becomes Reality**: On ImageNet with ResNet-152 and ResNeXt-101, HPFF enables local learning to outperform end-to-end BP in accuracy for the first time.
4. **CKA and Hierarchical Analysis Offer Deep Insights**: Deteriorated classification accuracy in early layers leads to retaining more global features, which subsequently significantly boosts accuracy in deeper layers. This finding challenges the intuition that 'each layer must achieve high classification accuracy'.

## Limitations & Future Work

- It still relies on backpropagation to compute gradients inside each local module, failing to completely eliminate BP.
- The extra auxiliary networks introduced by the cascade level increase model parameters and computational cost (though partially mitigated by PFF in terms of memory).
- The setting of $k=2$ is fixed, and adaptive methods to determine cascading range remain unexplored.
- Undergoing validation only on image classification tasks, its efficacy on other modalities such as NLP and speech remains unknown.
- The patch split of $n=2$ in PFF is suitable for larger feature maps and might not apply to small feature maps in deeper layers.

## Related Work & Insights

- **DGL** (Belilovsky et al., ICML 2019): Greedy layer-wise learning, upon which HPFF achieves the most remarkable improvement (error drops from 14.08 to 8.94).
- **InfoPro** (Wang et al.): A local learning method that preserves global features through mutual information constraints.
- **DNI** (Jaderberg et al., ICML 2017): Decouples modules using synthetic gradients, but focuses on a different direction.
- The hierarchical design concept can inspire other optimization problems that need to balance local and global trade-offs (e.g., client-to-client information sharing in federated learning).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hierarchical independent-cascade design and PFF are both novel contributions and highly complementary.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely thorough experiments spanning three methods, four datasets, multiple networks, along with ablation, CKA, t-SNE, and layer-wise analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear method description, thorough experimental analysis, and convincing visualizations.
- **Value**: ⭐⭐⭐⭐ — Enables local learning to surpass BP on large-scale datasets for the first time, carrying significant importance for parallel training and energy-efficient computing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Momentum Auxiliary Network for Supervised Local Learning](momentum_auxiliary_network_for_supervised_local_learning.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[ECCV 2024\] Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
