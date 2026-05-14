---
title: >-
  [Paper Note] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding
description: >-
  [NeurIPS 2025][Social Computing][vision backbone] Inspired by the depth-first search (DFS) algorithm, DeepTraverse is a visual backbone network that achieves highly competitive image classification performance with very…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "vision backbone"
  - "DFS inspired"
  - "recursive exploration"
  - "channel recalibration"
  - "parameter efficiency"
date: 2026-05-08
content_hash: 69f5b88b63561f4a
---

# DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2506.10084](https://arxiv.org/abs/2506.10084)
**Code**: None
**Area**: Social Computing
**Keywords**: vision backbone, DFS inspired, recursive exploration, channel recalibration, parameter efficiency

## TL;DR
Inspired by the depth-first search (DFS) algorithm, DeepTraverse is a visual backbone network that achieves highly competitive image classification performance with very few parameters, through a parameter-sharing recursive exploration module and an adaptive channel recalibration module.

## Background & Motivation

**Background**: Visual backbone networks (CNN/ViT) typically adopt uniformly cascaded layer-by-layer structures that implicitly abstract features stage by stage. Lightweight networks such as MobileNet, ShuffleNet, and GhostNet reduce FLOPs by optimizing convolutional operators.

**Limitations of Prior Work**: The feature abstraction pathways of conventional networks lack an explicit iterative refinement mechanism — increasing depth linearly increases parameter count, while adding attention introduces additional computation. A paradigm for "strategically exploring the feature space" is absent.

**Key Challenge**: Deeper feature refinement requires more parameters, yet resource-constrained scenarios cannot afford this. A fundamental trade-off exists between parameter efficiency and representational depth.

**Goal**: To achieve deep feature refinement under an extremely low parameter budget, breaking the linear relationship between depth and parameter count.

**Key Insight**: Inspiration is drawn from the classical DFS algorithm — DFS systematically explores along a path in depth before backtracking to evaluate and collect information. This "explore-evaluate-adjust" paradigm is embedded into the network architecture.

**Core Idea**: A parameter-sharing recursive module simulates DFS's deep exploration (multiple iterations without increasing parameters), while channel attention simulates DFS's backtracking evaluation (dynamically re-weighting feature channels).

## Method

### Overall Architecture
DeepTraverse adopts a multi-stage hierarchical structure, with the DFSBlock as its basic unit. Each DFSBlock contains two core sub-modules: DFS-EB (Exploration Block) for recursive feature refinement, and DFS-BB (Backtrack Block) for global context calibration. Multiple DFSBlocks are stacked hierarchically with residual connections.

### Key Designs

1. **DFS-Inspired Exploration Block (DFS-EB)**:

    - Function: Performs $R$ rounds of recursive refinement on the input features, using the same parameters in each round.
    - Mechanism: Initial feature extraction $F_0 = \Phi_{\text{extract}}(X)$, where $\Phi_{\text{extract}}$ comprises depthwise convolution + BN + ReLU + pointwise convolution. Recursion is then applied for $R$ steps: $F_i = F_{i-1} + \Phi_{\text{recursive}}(F_{i-1})$, where $\Phi_{\text{recursive}}$ shares the same architecture as $\Phi_{\text{extract}}$ but with weights shared across all $R$ steps.
    - Final output: $Y_{EB} = F_R = \Phi_{\text{extract}}(X) + \sum_{j=1}^{R} \Phi_{\text{recursive}}(F_{j-1})$
    - Design Motivation: Parameter sharing enables the network to perform computation equivalent to $R$ times the depth without increasing independent parameters, analogous to DFS continuously exploring deeper along the same path. Depthwise separable convolutions further compress parameter count.

2. **DFS-Inspired Backtrack Block (DFS-BB)**:

    - Function: Performs globally context-aware channel recalibration on the explored features.
    - Mechanism: Global average pooling yields $z = \text{AdaptiveAvgPool}(F)$; a bottleneck FC network then computes channel attention $s = \sigma(W_2 \delta(W_1 z))$; finally $F' = F \odot s$.
    - Design Motivation: Simulates the evaluation step during DFS backtracking — after exploration, global context is used to reassess which feature channels are more important, suppressing irrelevant channels and enhancing discriminative ones. This is essentially similar to the channel attention of SE-Net but is embedded within the DFS framework with algorithmic motivation.

3. **Integrated DFSBlock**:

    - Function: Connects DFS-EB and DFS-BB in series, with a residual shortcut connection, forming a complete unit.
    - Formula: $X_{\text{out}} = \delta(F_{\text{recalibrated}} + S(X_{\text{in}}))$, where $S$ is a dimension-aligning projection shortcut.
    - Design Motivation: The residual connection ensures gradient flow; DFS-EB provides multi-level feature exploration; DFS-BB provides global evaluation. The three components collaborate to complete a full "explore-evaluate-integrate" cycle.

### Loss & Training
- Trained from scratch for 100 epochs with an initial learning rate of 0.1
- Trained on an NVIDIA RTX 2080 Ti
- Parameter count and FLOPs evaluated using the `timm` and `thop` libraries

## Key Experimental Results

### Main Results

| Method | Params | FLOPs | CIFAR-100 | CIFAR-10 |
|------|--------|-------|-----------|----------|
| **DeepTraverse** | **0.26M** | **0.03G** | **73.84** | **93.25** |
| DenseNet | 0.60M | 0.20G | 73.02 | 92.75 |
| EfficientNet | 4.14M | 0.12G | 73.14 | 90.20 |
| StarNet | 2.70M | 0.28G | 72.27 | 92.13 |
| GhostNet | 2.76M | 0.04G | 72.46 | 91.79 |
| ResNet20 | 0.28M | 0.08G | 69.53 | 92.36 |

DeepTraverse surpasses DenseNet (0.6M), EfficientNet (4.14M), and others with only 0.26M parameters.

| Method | Params | FLOPs | ImageNet-1k Top-1 | Top-5 |
|------|--------|-------|--------------------|-------|
| **DeepTraverse** | **5.04M** | **0.84G** | **83.16** | **96.54** |
| DenseNet | 7.05M | 2.89G | 81.44 | 95.74 |
| GhostNet | 5.30M | 0.28G | 80.34 | 95.16 |
| EfficientNet | 7.28M | 0.42G | 81.18 | 95.24 |
| ResNet50 | 25.56M | 4.13G | 78.76 | 94.18 |

### Ablation Study (Wide Variant Comparison)

| Configuration | Params | FLOPs | CIFAR-100 |
|------|--------|-------|-----------|
| DeepTraverse (Wide) | 14.26M | 1.78G | 82.20 |
| WideResNet | 36.54M | 5.25G | 80.81 |

The Wide variant surpasses WideResNet by 1.39 points using only 39% of its parameters and 34% of its FLOPs.

### Key Findings
- Exceptional parameter efficiency: 0.26M parameters on CIFAR-100 outperforms EfficientNet with 4M+ parameters.
- Good scalability: performance advantages are maintained from small models to the Wide variant.
- On ImageNet-64, 0.59M parameters achieves 71.50% Top-1, surpassing GhostNet (2.74M) by 0.76 points.
- Parameter sharing in the recursive exploration module is the key source of efficiency.

## Highlights & Insights
- **Parameter-sharing recursion** is the most central technique: iterating multiple rounds with shared weights is equivalent to deepening the network without adding parameters, analogous to the application of Universal Transformer ideas within CNNs.
- The **algorithm-inspired design** narrative is worth learning from: although the core components (depthwise conv + SE attention + parameter sharing) are all established techniques, the DFS algorithmic framework provides a clear and intuitive motivation for their combination.
- The combination paradigm of channel attention and recursive refinement is transferable to lightweight backbone design for downstream tasks such as detection and segmentation.

## Limitations & Future Work
- Validation is limited to classification tasks, with no experiments on detection or segmentation — the authors acknowledge that computational constraints precluded testing on ImageNet-21k and downstream tasks.
- The DFS algorithm analogy is relatively loose — the combination of recursive convolution and SE attention is not a strict implementation of DFS, but rather borrows the high-level intuition of "deep exploration + backtracking evaluation."
- Training costs are not compared in detail with other lightweight models — fewer parameters do not imply faster training, as $R$ recursive steps actually increase forward pass time.
- ImageNet-1k evaluation appears to be conducted only on a 100-class subset (the results in Table 3 appear to reflect a subset rather than the full 1000 classes).

## Related Work & Insights
- **vs. MobileNet/ShuffleNet**: These methods optimize convolutional operator structures to reduce the cost of individual operations, whereas DeepTraverse deepens computation without increasing parameters through parameter-sharing recursion.
- **vs. SE-Net**: DFS-BB is essentially an SE module; the distinction lies in the stronger design motivation provided by its synergistic use with recursive exploration.
- **vs. Universal Transformer**: The idea is analogous but applied to CNNs — weight-sharing iterative refinement is a general strategy applicable across modalities.

## Rating
- Novelty: ⭐⭐⭐ The DFS analogy is creative, but the core components are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐ Multi-dataset validation, but lacking downstream task evaluation and full-scale ImageNet experiments.
- Writing Quality: ⭐⭐⭐⭐ Narrative structure is clear, and the DFS analogy is presented in an engaging manner.
- Value: ⭐⭐⭐ Parameter efficiency is genuinely impressive, but validation in real deployment scenarios and comparisons of latency/throughput are absent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in LLMs](auto-search_and_refinement_an_automated_framework_for_gender_bias_mitigation_in_.md)
- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](../../ICLR2026/social_computing/sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[CVPR 2026\] As Language Models Scale, Low-order Linear Depth Dynamics Emerge](../../CVPR2026/social_computing/as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)

</div>

<!-- RELATED:END -->
