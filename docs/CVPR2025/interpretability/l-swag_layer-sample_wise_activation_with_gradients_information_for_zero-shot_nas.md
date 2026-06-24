---
title: >-
  [Paper Note] L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers
description: >-
  [CVPR 2025][Interpretability][Zero-Cost Proxy] This paper proposes L-SWAG (Layer-Sample Wise Activation with Gradients), a new general zero-cost proxy that evaluates network architecture quality by combining layer- and sample-wise activation and gradient information. It is the first to systematically extend zero-cost NAS to the Vision Transformer search space and establishes a new benchmark across 6 tasks in the Autoformer search space.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Zero-Cost Proxy"
  - "Neural Architecture Search"
  - "ViT"
  - "Activation"
  - "Gradient Information"
date: 2026-05-08
content_hash: a530480065269ce3
---

# L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Zero-Cost NAS  
**Keywords**: Zero-Cost Proxy, Neural Architecture Search, ViT, Activation, Gradient Information

## TL;DR

This paper proposes L-SWAG (Layer-Sample Wise Activation with Gradients), a new general zero-cost proxy that evaluates network architecture quality by combining layer- and sample-wise activation and gradient information. It is the first to systematically extend zero-cost NAS to the Vision Transformer search space and establishes a new benchmark across 6 tasks in the Autoformer search space.

## Background & Motivation

### Background

**Background**: Neural Architecture Search (NAS) aims to automatically identify optimal network architectures, but traditional NAS methods (multi-trial/one-shot training) incur immense computational overhead. Zero-Cost NAS (ZC-NAS) greatly improves search efficiency by designing zero-cost proxies to predict architecture performance without training.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Existing SOTA zero-cost proxies (such as NASWOT, SynFlow, ZenNAS, etc.) are primarily designed and verified for CNN search spaces (e.g., NAS-Bench-201), and their performance in ViT search spaces remains unknown. (2) With LLMs driving Transformer architectures to become mainstream, ViT architecture search is increasingly important, yet a systematic ViT zero-cost search benchmark is lacking. (3) Existing proxies either utilize only activations (e.g., NASWOT's kernel function) or only gradients (e.g., SynFlow), failing to effectively combine their complementary information.

### Key Challenge

**Key Challenge**: Zero-cost proxies need to accurately estimate the potential of an architecture in an extremely short time (a single forward/backward propagation). However, the complexity of attention mechanisms in ViT architectures renders the assumptions of traditional proxies (such as ReLU activation and convolutional hierarchical structures) invalid.

### Goal

**Goal**: How to design a general zero-cost proxy that performs exceptionally well in both CNN and ViT search spaces?

### Key Insight

**Key Insight**: From an information-theoretic perspective, simultaneously capture the diversity of activations across network layers and the quality of gradient signals, evaluating architectures using layer-wise and sample-wise statistics.

### Core Idea

**Core Idea**: Compute interactive statistics between activations and gradients at each layer, and aggregate sample-wise and layer-wise information to form a comprehensive architecture rating score.

## Method

### Overall Architecture

The computation workflow of L-SWAG: (1) Perform a single forward and backward propagation using a small batch of data on a candidate architecture with randomly initialized parameters. (2) Extract activation and gradient tensors at each layer to calculate layer-wise statistics. (3) Aggregate statistics across samples and layers to obtain a single scalar score. (4) Sort candidate architectures by score to select the optimal architecture.

### Key Designs

1. **Layer-Sample Wise Activation-Gradient Statistics**:
    - Function: Capture the discrimination ability and gradient flow quality of each layer on the input data.
    - Mechanism: For the $l$-th layer of the network, collect the activation matrix $A^l \in \mathbb{R}^{B \times D_l}$ (with B samples and $D_l$-dimensional features) and the corresponding gradient matrix $G^l$. Compute the sample-wise correlation matrix of activations $K_A^l = A^l (A^l)^T$ (similar to NASWOT's kernel matrix), and concurrently compute the sample-wise correlation matrix of gradients $K_G^l = G^l (G^l)^T$. L-SWAG combines the two, e.g., by computing $\text{score}^l = f(K_A^l, K_G^l)$, where $f$ can be a statistic of the element-wise product of the matrices.
    - Design Motivation: Activations reflect the diversity of features extracted by the network (a good architecture should maximize representation differences among different inputs), and gradients reflect the effective propagation of training signals (a good architecture should have smooth and discriminative gradient flow).

2. **Cross-Layer Aggregation Strategy**:
    - Function: Synthesize scores from individual layers into a global architecture quality assessment.
    - Mechanism: Weighted aggregation is performed on the scores of all searchable layers, where weights can be uniform or based on increasing/decreasing weights according to layer depth. The final score is $S = \sum_l w_l \cdot \text{score}^l$. Additionally, different aggregation strategies are considered for different layer types (attention layers vs. FFN layers).
    - Design Motivation: Functional differences between different layers in ViTs are significant—shallow layers focus on local features while deep layers target global semantics—warranting targeted weighting.

3. **Autoformer Search Space Benchmark**:
    - Function: The first systematic zero-cost NAS evaluation platform for ViTs.
    - Mechanism: Based on the Autoformer search space (searching embed_dim, depth, num_heads, mlp_ratio, etc.), evaluate the ranking correlation (Spearman/Kendall $\tau$) of various zero-cost proxies across 6 downstream tasks, including ImageNet classification and COCO detection, establishing a complete benchmark.
    - Design Motivation: The lack of standard evaluations for ViT zero-cost NAS has hindered development in this field.

### Loss & Training
The model is trained end-to-end, with the optimization objective comprehensively considering task loss and regularization terms.


## Key Experimental Results

### Key Findings

- L-SWAG achieves the best or near-best ranking correlation across all 6 tasks in the Autoformer search space.
- It also performs excellently on traditional CNN search spaces (NAS-Bench-201, NAS-Bench-101), proving its generalizability.
- Existing SOTA proxies (NASWOT, SynFlow, etc.) experience a significant drop in performance in the ViT search space.
- Simultaneously leveraging activations and gradients improves ranking correlation by approximately 15-20% compared to using either signal in isolation.
- The search is highly efficient, with a single architecture evaluation taking <0.5 seconds and the entire search process taking <10 minutes.

## Highlights & Insights

- **Filling the Gap in ViT Zero-Cost NAS**: Systematically extends ZC-NAS to the ViT domain and provides a benchmark.
- **Generality-Oriented Design**: Performs well across both CNN and ViT spaces without relying on specific architectural assumptions.
- **Complementary Information**: Combining activations + gradients offers a more comprehensive representation than a single signal alone.

## Limitations & Future Work

- The evaluation still relies on a small batch of data (~64 samples), and data selection may introduce noise.
- Extensibility in ultra-large-scale search spaces (e.g., combinatorial spaces with >10^10 architectures).
- Hyperparameters (aggregation weights, choice of statistics) need to be tuned on a validation set.
- Future work can explore deep integration with NAS search strategies (evolutionary algorithms, reinforcement learning).


## Related Work & Insights
- **vs Representative Methods in the Same Field**: This paper makes unique contributions to method design, complementing existing methods.
- **vs Traditional Methods**: Compared to traditional solutions, the proposed method achieves significant improvements in key metrics.
- **Insights**: The technical route of this paper provides important reference value for subsequent related work.


## Rating
- Novelty: ⭐⭐⭐⭐ Unique contribution to method design
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets
- Writing Quality: ⭐⭐⭐⭐ Clear and well-organized
- Value: ⭐⭐⭐⭐ Facilitates development in the field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Sample- and Parameter-Efficient Auto-Regressive Image Models](sample-_and_parameter-efficient_auto-regressive_image_models.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
