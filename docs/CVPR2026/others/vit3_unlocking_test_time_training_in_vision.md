---
title: >-
  [Paper Note] ViT3: Unlocking Test-Time Training in Vision
description: >-
  [CVPR 2026][Test-Time Training] This paper systematically explores the design space of Test-Time Training (TTT) for vision tasks, distills six practical design insights, and proposes ViT3—a purely TTT-based vision architecture with linear complexity—that matches or surpasses Mamba and linear attention methods on classification, generation, detection, and segmentation tasks.
tags:
  - CVPR 2026
  - Test-Time Training
  - Linear Complexity
  - Inner Model
  - Vision Transformer
  - Convolution
date: 2026-05-08
content_hash: 8637377a71ac6348
---

# ViT3: Unlocking Test-Time Training in Vision

**Conference**: CVPR 2026
**arXiv**: [2512.01643](https://arxiv.org/abs/2512.01643)
**Code**: [GitHub](https://github.com/LeapLabTHU/ViTTT)
**Area**: Efficient Architecture / Visual Sequence Modeling
**Keywords**: Test-Time Training, Linear Complexity, Inner Model, Vision Transformer, Convolution

## TL;DR

This paper systematically explores the design space of Test-Time Training (TTT) for vision tasks, distills six practical design insights, and proposes ViT3—a purely TTT-based vision architecture with linear complexity—that matches or surpasses Mamba and linear attention methods on classification, generation, detection, and segmentation tasks.

## Background & Motivation

- **Background**: The quadratic complexity $O(N^2)$ of Vision Transformers limits their applicability to long visual sequences. TTT models offer a linear-complexity alternative by reformulating the attention operation as an online learning problem: at test time, Key-Value pairs serve as a "mini-dataset" to train a compact inner model, which is then used to process the Query.
- **Limitations of Prior Work**: The TTT design space is vast and underexplored. The choices regarding inner training (loss function, learning rate, batch size, number of epochs) and inner model (architecture, capacity) lack systematic understanding, leaving the performance of visual TTT models unrealized.
- **Key Challenge**: The absence of systematic guidelines locks visual TTT models below their potential.
- **Goal**: To provide a comprehensive study of the TTT design space for vision, derive actionable insights, and instantiate them in a competitive visual architecture.

## Method

### Overall Architecture

Input token sequence → projected into Q/K/V → K/V used as training data to train inner model $F_W$ → trained model $F_{W^*}$ processes Q to produce output → macro-architecture identical to Transformers (attention replaced by TTT layers at each stage).

### Key Designs (Six Insights)

1. **Loss Function Selection (Insight 1)**:
   - Losses for which the mixed second-order derivative $\partial^2 L / \partial \hat{V} \partial V = 0$ (e.g., MAE/L1) are unsuitable for TTT, as the outer-loop gradient signal vanishes when backpropagating through the inner update.
   - Recommended: Dot Product Loss, MSE Loss.

2. **Inner Training Configuration (Insights 2 & 3)**:
   - Vision tasks favor single-epoch full-batch gradient descent ($B = N$), unlike the mini-batch setting preferred for language tasks.
   - Causal mini-batches are suboptimal for non-causal visual data.
   - A larger inner learning rate ($\eta = 1.0$) yields the best performance.

3. **Inner Model Design (Insights 4, 5 & 6)**:
   - Increasing inner model capacity (width scaling) consistently improves performance.
   - Deep inner models suffer from optimization difficulties (higher training loss, i.e., underfitting); depth scaling is ineffective under current TTT settings.
   - Convolutional architectures—especially depthwise separable convolutions (DWConv)—are particularly well-suited as inner models: **80.1% Top-1** vs. MLP at 78.9%.

### Loss & Training

- **Outer loop**: Standard ImageNet training for 300 epochs (DeiT-S recipe).
- **Inner loop**: Dot Product Loss, $\eta = 1.0$, single epoch, full-batch.
- **Inner model**: DWConv (depthwise separable convolution), parallelizable.
- **Hierarchical architecture (H-ViT3)**: Combines local window attention with global TTT layers.

## Key Experimental Results

### Main Results — Image Classification (ImageNet-1K)

| Method | Type | Params | Top-1 |
|--------|------|--------|-------|
| DeiT-S | Transformer | 22M | 79.8 |
| Vim-S | Mamba | 26M | 80.3 |
| Agent-DeiT-S | Linear | 23M | 80.5 |
| ViT3-S | TTT | 24M | 81.6 |
| H-ViT3-S‡ | TTT | 54M | 84.9 |
| H-ViT3-B‡ | TTT | 94M | 85.5 |

### Ablation Study — Inner Model Architecture

| Inner Model | Top-1 | Note |
|-------------|-------|------|
| FC(x) linear layer | 79.1 | Equivalent to linear attention |
| MLP r1 2-layer | 78.9 | Baseline TTT |
| MLP r4 2-layer | 79.6 | Width scaling effective |
| SiLU(FC(x)) | 79.4 | Constrained design outperforms full MLP |
| DWConv(x) | 80.1 | Convolution is optimal |

### Key Findings

- TTT is stronger than linear attention, as it can employ more expressive nonlinear inner models.
- Full-batch training outperforms mini-batch training in vision (due to the non-causal nature of visual data), opposite to the conclusion for language tasks.
- Deeper inner models degrade performance (3-layer MLP: 77.5% < 2-layer MLP: 78.9%), indicating an optimization problem rather than a capacity limitation.
- Residual connections and initialization strategies cannot fully resolve the optimization difficulties of deep inner models.

## Highlights & Insights

- First systematic exploration of the visual TTT design space; six distilled insights provide clear guidance for future research.
- Identifies the optimization difficulty of deep inner models as an important open problem.
- Discovery of DWConv as an inner model leverages the locality inductive bias of convolutions.
- ViT3, as a purely TTT-based architecture, competes with heavily optimized Transformers across multiple tasks.

## Limitations & Future Work

- The optimization difficulty of deep inner models remains the central unresolved issue, capping the potential of TTT.
- Each inner update incurs roughly $4\times$ the compute of a standard forward pass, leaving room for efficiency improvement.
- Mini-batch training underperforms in vision, though designing vision-specific scan orders may alleviate this.
- The potential of TTT for long-sequence visual tasks such as video remains unexplored.

## Related Work & Insights

- **vs. Mamba**: SSM scan paths introduce a causal bias; ViT3's full-batch scheme is more naturally aligned with vision.
- **vs. Linear Attention**: Linear attention is equivalent to a $d \times d$ linear layer; TTT can be any nonlinear model, yielding greater expressive power.
- **vs. Softmax Attention**: Softmax attention can be viewed as a two-layer MLP of width $N$; TTT replaces it with a more compact yet trainable model.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The systematic exploration and six-insight summarization framework are novel for the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full coverage across classification, generation, detection, and segmentation, with highly detailed inner-design ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear structure; the insight–experiment–remark organization is textbook quality.
- **Value**: ⭐⭐⭐⭐ — Establishes a systematic foundation for visual TTT research and identifies multiple promising future directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](neural_collapse_in_test-time_adaptation.md)
- [\[AAAI 2026\] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](../../AAAI2026/others/bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)
- [\[CVPR 2026\] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?](do_vision_models_perceive_illusory_motion_in_static_images_like_humans.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/others/redundancy-aware_test-time_graph_out-of-distribution_detection.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)

</div>

<!-- RELATED:END -->
