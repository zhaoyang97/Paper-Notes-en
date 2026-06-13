---
title: >-
  [Paper Note] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging
description: >-
  [ICLR 2026][Model Compression][Model Merging] AdaRank is proposed to adaptively select singular components of task vectors via learnable binary masks (replacing heuristic top-k selection)…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Model Merging"
  - "SVD"
  - "Task Vector"
  - "Test-Time Adaptation"
  - "Multi-Task Learning"
date: 2026-05-08
content_hash: f21ba4713c9d3905
---

# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging

**Conference**: ICLR 2026
**arXiv**: [2503.22178](https://arxiv.org/abs/2503.22178)  
**Code**: To be confirmed  
**Area**: Object Detection (Model Merging / Multi-Task Learning)
**Keywords**: Model Merging, SVD, Task Vector, Test-Time Adaptation, Multi-Task Learning

## TL;DR

AdaRank is proposed to adaptively select singular components of task vectors via learnable binary masks (replacing heuristic top-k selection), combined with test-time entropy minimization, substantially alleviating inter-task interference in multi-task model merging and achieving 89.4% accuracy on ViT-B/32.

## Background & Motivation

**Background**: Model merging integrates multiple independently fine-tuned models into a unified framework, avoiding the high computational overhead of multi-model deployment. Task Arithmetic realizes merging by weighted summation of task vectors (the difference between fine-tuned and pre-trained weights), but suffers from severe inter-task interference.

**Limitations of Prior Work**: Recent SVD-based methods have advanced by truncating task vectors using low-rank structure, yet they rely on heuristic fixed top-k selection, which introduces two fundamental problems:
- **Counter-intuitive phenomenon**: Although top singular components reduce the loss most for the target task, they may impose greater net loss increases on other tasks. Experiments on ViT-B/32 show that incorporating the top singular components of MNIST benefits the semantically similar SVHN, but substantially increases the loss of the dissimilar DTD (texture classification).
- **Large variance in rank requirements**: The intrinsic rank varies widely across tasks and layers — SUN397 (397 classes) requires higher rank, while MNIST/SVHN require lower rank; early layers (task-agnostic features) have high rank with low variance, whereas later layers (task-specific representations) have low rank with high variance.

**Key Challenge**: Fixed top-k truncation may simultaneously discard critical components for certain tasks and retain components that introduce interference.

**Goal**: Adaptively select the optimal subset of singular components independently for each task and each layer.

## Method

### Overall Architecture

For each task $i$ and layer $l$, the task vector is decomposed via SVD as $\tau_i^l = U_i^l \Sigma_i^l V_i^{l\top}$. A binary mask $B_i^l \in \{0,1\}^{1 \times m}$ is introduced to determine whether each singular component is retained or pruned. The merging formula is:

$$\theta_m^l = \theta_0^l + \lambda^l \sum_{i=1}^T U_i^l (\text{diag}(B_i^l) \odot \Sigma_i^l) V_i^{l\top}$$

### Key Designs

1. **Adaptive Binary Masks**: Unlike fixed top-k selection, each singular component is subject to an independent binary decision. $B_{ir}=1$ retains the component; $B_{ir}=0$ prunes it. Setting all entries to 1 reduces to standard Task Arithmetic; setting entries $r \le k$ to 1 and the rest to 0 reduces to top-k truncation.
2. **Test-Time Entropy Minimization**: Shannon entropy minimization is used as an unsupervised surrogate objective to optimize masks on unlabeled test data. Entropy is shown to be highly correlated with multi-task supervised loss.
3. **STE Optimization**: The forward pass applies binary masks (rounded to $\{0,1\}$), while the backward pass propagates gradients through the continuous relaxation via the Straight-Through Estimator (STE).
4. **Plug-and-Play Compatibility**: AdaRank can be jointly optimized with layer-wise coefficients $\lambda^l$ and is compatible with multiple baselines including Task Arithmetic, CART, TSV-M, and Iso-CTS.

### Loss & Training

$$\arg\min_B \sum_{i=1}^T \sum_{x_i \in \mathcal{D}_i} H_i(f(\theta(B), x_i))$$

where $H_i$ denotes the Shannon entropy of task $i$'s output, and $\mathcal{D}_i$ is the unlabeled test data for task $i$.

## Key Experimental Results

### Main Results (ViT-B/32, 8 Tasks)

| Type | Method | Avg. Accuracy |
|------|--------|--------------|
| Static Merging | CART | 84.7 |
| Static Merging | Iso-CTS | 84.9 |
| Adaptive | TA+AdaMerging | 80.1 |
| Adaptive | **TA+AdaRank** | **87.9** |
| Adaptive | **CART+AdaRank** | **89.2** |
| Adaptive | **Iso-CTS+AdaRank** | **89.4** |
| Routing | WEMoE | 89.5 |

### Ablation Study

| Configuration | ViT-B/32 (8 Tasks) | Notes |
|--------------|-------------------|-------|
| Fixed top-k (k=50) | 84.7 | CART baseline |
| Random mask | ~82.0 | Inferior to top-k |
| $\lambda$-only optimization (AdaMerging) | 80.1 | Layer-wise coefficient optimization insufficient |
| **AdaRank (Joint B+λ)** | **89.2** | Joint mask+coefficient optimization is best |

### Key Findings

- **NLP Tasks**: CART+AdaRank achieves 0.7547 on RoBERTa and 0.6587 on GPT-2, significantly outperforming AdaMerging.
- **20-Task Setting**: Advantages are more pronounced — TSV-M+AdaRank achieves 86.9% on ViT-B/32, far exceeding WEMoE's 80.2%.
- Additional parameters account for only 0.032% of the total, and TTA runtime is comparable to AdaMerging.
- Model parameter count remains constant regardless of the number of tasks, outperforming routing methods that scale linearly.

## Highlights & Insights

- The paper reveals the counter-intuitive finding that top-k singular components are suboptimal in multi-task settings — an analysis with independent value in its own right.
- The method is highly general and can be seamlessly integrated into various static and adaptive model merging frameworks.
- Advantages become more pronounced in the large-scale 20-task setting, indicating that inter-task interference intensifies as the number of tasks grows.
- The approach is effective across vision and NLP domains and across architectures (both bidirectional and autoregressive Transformers).

## Limitations & Future Work

- Unlabeled test data is required for test-time adaptation, making the method inapplicable in fully data-free scenarios.
- SVD decomposition itself incurs additional preprocessing overhead of $O(d^2 d')$.
- Entropy minimization as a surrogate objective is not always perfectly correlated with multi-task loss and may fail in certain settings.
- Validation is limited to classification tasks; effectiveness on dense prediction tasks such as detection and segmentation remains unexplored.

## Related Work & Insights

- **Task Arithmetic / TIES-Merging / DARE**: Element-wise sparsification of task vectors without preserving low-rank structure.
- **CART / TSV-M / STAR**: SVD low-rank truncation with fixed top-k selection.
- **AdaMerging**: Test-time adaptation of layer-wise coefficients $\lambda$; AdaRank performs adaptation at a finer granularity (singular component level).
- **WEMoE / Twin-Merging**: Routing methods whose parameters scale linearly with the number of tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Adaptive singular component selection replaces heuristic top-k, with in-depth analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Vision + NLP, multiple backbones, 8/20-task settings, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear analysis with intuitive motivational figures.
- Value: ⭐⭐⭐⭐ A practical and general method for the model merging community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](../../ICML2026/model_compression/saliency-aware_model_merging.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](../../NeurIPS2025/model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/model_compression/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)

</div>

<!-- RELATED:END -->
