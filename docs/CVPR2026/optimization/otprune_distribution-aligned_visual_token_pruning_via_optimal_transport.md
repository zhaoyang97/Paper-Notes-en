---
title: >-
  [Paper Note] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport
description: >-
  [CVPR 2026][Optimization][visual token pruning] This work formulates visual token pruning as a distribution alignment problem under optimal transport (OT)…
tags:
  - "CVPR 2026"
  - "Optimization"
  - "visual token pruning"
  - "optimal transport"
  - "Wasserstein distance"
  - "submodular optimization"
  - "training-free"
  - "MLLM efficiency"
date: 2026-05-08
content_hash: 4bef1f3fb7343966
---

# OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport

**Conference**: CVPR 2026
**arXiv**: [2602.20205](https://arxiv.org/abs/2602.20205)  
**Code**: [xiwenc1/OTPrune](https://github.com/xiwenc1/OTPrune)  
**Area**: Optimization
**Keywords**: visual token pruning, optimal transport, Wasserstein distance, submodular optimization, training-free, MLLM efficiency

## TL;DR

This work formulates visual token pruning as a distribution alignment problem under optimal transport (OT), minimizing the 2-Wasserstein distance between the full and pruned token sets. It achieves training-free, $O(mk^2)$-complexity pruning via Gaussian surrogates, a log-det submodular objective, and greedy Cholesky selection, attaining state-of-the-art accuracy–efficiency trade-offs across 11 multimodal benchmarks.

## Background & Motivation

**Severe visual token redundancy in MLLMs**: A single image can produce hundreds of patch-level tokens, and the quadratic complexity of Transformer self-attention with respect to sequence length incurs prohibitively high inference costs.

**Empirical evidence that 70–90% of visual tokens can be pruned**: Prior work has demonstrated that the majority of visual tokens are redundant, with minimal accuracy degradation upon removal, providing strong motivation for token pruning.

**Limitations of existing pruning methods**: Attention-based methods (FastV, VTW) fail when attention scores do not accurately reflect token importance; calibration-based methods (FitPrune) rely on external datasets and generalize poorly; fine-tuning-based methods (M³) incur high computational costs.

**DivPrune focuses solely on diversity, neglecting global representativeness**: DivPrune reduces redundancy by selecting mutually dissimilar tokens, but diversity is not equivalent to representativeness—the selected subset may fail to preserve the covariance structure and semantic coverage of the original token distribution.

**Empirical support for the distribution alignment hypothesis**: The authors verify on LLaVA 1.5-7B that OT distance exhibits strong Spearman rank correlation with downstream task performance across multiple subset selection strategies, confirming that smaller OT distance corresponds to better downstream performance.

**Need for a principled, training-free, task-agnostic pruning framework**: One that preserves local diversity while ensuring global distribution alignment, while remaining computationally feasible and theoretically grounded.

## Method

### Overall Architecture

Given $m$ tokens $\bm{V} \in \mathbb{R}^{m \times d}$ output by the vision encoder, OTPrune selects a subset $\mathcal{C}$ of $k$ tokens such that the pruned token distribution $Q$ approximates the full token distribution $P$ as closely as possible. The core pipeline:

1. Approximate $P$ and $Q$ with Gaussian surrogates to obtain a closed-form 2-Wasserstein distance.
2. Derive a log-det submodular surrogate objective.
3. Solve efficiently via greedy Cholesky decomposition.

### Key Design 1: OT Distribution Alignment Formulation

- **Function**: Formulates token pruning as a distribution approximation problem, minimizing the 2-Wasserstein distance between the full token set $P$ and the pruned subset $Q$.
- **Mechanism**: $\min_Q \mathcal{W}_2^2(P, Q) = \inf_{\pi \in \Pi(P,Q)} \mathbb{E}_{(x,y) \sim \pi} [\|x - y\|_2^2]$, directly measuring the geometric discrepancy between the two distributions in the embedding space.
- **Design Motivation**: Unlike DivPrune, which pursues diversity alone, the Wasserstein distance encodes both local diversity and global representativeness—the retained subset is not only internally dissimilar but faithfully approximates the geometric structure of the original feature manifold.

### Key Design 2: Gaussian Surrogate + Log-Det Submodular Objective

- **Function**: Approximates token distributions with zero-mean Gaussians, $P \approx \mathcal{N}(0, \Sigma)$ and $Q \approx \mathcal{N}(0, \Sigma_\mathcal{C})$, transforming the Wasserstein distance into a trace objective over covariance matrices, then lower-bounded via the $\gamma$-log-det operator.
- **Mechanism**: Under the Gaussian assumption, $\mathcal{W}_2^2$ admits the closed form $\text{tr}(\Sigma) + \text{tr}(\Sigma_\mathcal{C}) - 2\text{tr}((\Sigma^{1/2} \Sigma_\mathcal{C} \Sigma^{1/2})^{1/2})$. After unit-variance normalization of features, the trace objective is maximized. Using the lower-bound relation $\log\det(\bm{I} + \gamma \bm{X}) \leq \gamma \text{tr}(\bm{X})$, the surrogate objective becomes $\max_\mathcal{C} \log\det(\bm{I} + \gamma \Sigma^{1/2} \Sigma_\mathcal{C} \Sigma^{1/2})$.
- **Design Motivation**: The log-det function is a provably monotone submodular function, guaranteeing a $(1 - 1/e)$ approximation ratio for greedy algorithms while avoiding the high cost of directly computing matrix square roots.

### Key Design 3: Sylvester Transform + Cholesky Greedy Selection

- **Function**: Applies the Sylvester determinant identity to reduce $d \times d$ matrix operations to $k \times k$ matrix operations, then greedily selects tokens via incremental Cholesky decomposition.
- **Mechanism**: Precomputes $\tilde{\bm{V}} = \bm{V}\bm{V}^\top$, reducing the objective to $\max_\mathcal{C} \log\det(\bm{I} + \tilde{\gamma} \tilde{\bm{V}}_\mathcal{C} \tilde{\bm{V}}_\mathcal{C}^\top)$. At each greedy iteration, the token maximizing the marginal gain is selected as $j = \arg\max_{i \notin \mathcal{C}} d_i^2$, with the gains of remaining tokens updated incrementally via Cholesky coefficients.
- **Design Motivation**: Total complexity is $O(mk^2)$, far below the brute-force $\binom{m}{k}$; submodularity provides a theoretical quality guarantee for the greedy solution; empirically, the actual approximation ratio far exceeds $(1 - 1/e) \approx 0.632$.

### Key Design 4: Robustness of Hyperparameter $\gamma$

- **Function**: $\gamma$ balances the OT loss against token selection regularization.
- **Mechanism**: Experiments show that $\gamma = 0.01$ performs consistently well across LLaVA 1.5-7B/13B and LLaVA 1.6-7B, with performance being insensitive to $\gamma$.
- **Design Motivation**: The method requires minimal hyperparameter tuning for deployment, enhancing practical usability.

## Key Experimental Results

### Table 1: Main Comparison on 11 Multimodal Benchmarks (LLaVA 1.5-7B, ~9.8% tokens retained)

| Method | TFLOP (ratio%) | GQA (EM) | MMB (Acc) | MME (P-score) | POPE (F1) | SQA (EM) | Avg. Rank ↓ | One-shot Rank ↓ |
|--------|----------------|----------|-----------|---------------|-----------|----------|-------------|-----------------|
| Original | 3.228 (100%) | 61.96 | 64.09 | 1506 | 85.84 | 69.41 | – | – |
| VTW | 0.603 (18.5%) | 38.94 | 21.31 | 681 | 25.35 | 65.29 | 8.23 | 3.41 |
| FastV | 0.514 (15.7%) | 38.73 | 20.62 | 696 | 32.84 | 65.15 | 8.41 | 3.59 |
| DivPrune | 0.512 (15.6%) | 56.85 | 59.19 | 1328 | 86.02 | 68.27 | 3.23 | 1.64 |
| **OTPrune** | **0.512 (15.6%)** | **56.62** | **60.40** | **1368** | 79.59 | **68.42** | **2.36** | **1.36** |
| M³ (finetune) | 0.512 (15.6%) | 60.81 | 65.81 | 1391 | 86.33 | 64.65 | 2.68 | – |

**Key takeaway**: OTPrune achieves the best average rank among training-free/calibration-free one-shot methods (Avg. Rank 2.36 vs. DivPrune 3.23), approaching even the fine-tuning-based M³.

### Table 2: Win Rate and Optimality Ratio of OTPrune vs. DivPrune on Synthetic Data

| Config (m, d, k) | Method | Win Rate (%) | Optimality Ratio (%) |
|------------------|--------|-------------|---------------------|
| (20, 10, 5) | DivPrune | 60.14 | 74.53 |
| | **OTPrune** | **94.56** | **87.81** |
| (30, 20, 10) | DivPrune | 84.62 | 76.48 |
| | **OTPrune** | **99.49** | **93.98** |
| (100, 50, 30) | DivPrune | 88.77 | 92.20 |
| | **OTPrune** | **99.99** | **100.00** |
| (1000, 256, 100) | DivPrune | 97.59 | 97.71 |
| | **OTPrune** | **100.00** | **100.00** |

**Key takeaway**: OTPrune substantially outperforms DivPrune across all dimensional configurations; in high-dimensional settings, both Win Rate and Optimality Ratio reach 100%, far exceeding the theoretical lower bound of $(1-1/e) \approx 63.2\%$.

### Supplementary: LLaVA 1.5-13B and LLaVA 1.6-7B Results

- **LLaVA 1.5-13B**: OTPrune Avg. Rank 1.27 (DivPrune 2.14), One-shot Rank 1.18.
- **LLaVA 1.6-7B**: OTPrune Avg. Rank 1.23 (DivPrune 1.77), maintaining best performance at only ~11% of original computation.
- **Wilcoxon signed-rank test**: OTPrune vs. DivPrune, p-value = 0.028, statistically significant.

## Highlights & Insights

1. **Novel perspective**: This is the first work to rigorously formulate visual token pruning as an OT distribution alignment problem, elevating the paradigm from "selecting diverse tokens" to "selecting tokens that represent the overall distribution," with clear theoretical motivation.
2. **Theoretical completeness**: The derivation chain from Wasserstein distance → Gaussian surrogate → log-det submodular objective → Cholesky greedy algorithm is rigorous at every step; the submodularity guarantee and $(1-1/e)$ approximation bound elevate pruning from a heuristic to a principled optimization.
3. **Training-free and plug-and-play**: Requires no fine-tuning or calibration datasets; directly applicable at inference time across multiple architectures including LLaVA 1.5-7B/13B and 1.6-7B.
4. **Thorough hypothesis validation**: The core assumption—"smaller OT distance → better downstream performance"—is explicitly verified via Spearman rank correlation analysis.
5. **Computationally efficient**: $O(mk^2)$ complexity; retaining ~10–15% of tokens requires only ~15% of the original FLOPs.

## Limitations & Future Work

1. **Limitations of the Gaussian approximation**: Token distributions in high-dimensional spaces may not follow a Gaussian; for semantically rich images, multimodal distributions (e.g., Gaussian mixtures) may be more appropriate.
2. **Strong zero-mean assumption**: Actual token embeddings are typically not zero-centered; ignoring mean matching may introduce bias in certain settings.
3. **Evaluation limited to the LLaVA family**: Generalizability to other MLLM architectures (e.g., Qwen-VL, InternVL) has not been verified.
4. **Overhead of precomputing $\tilde{\bm{V}} = \bm{V}\bm{V}^\top$**: For large $m$ (e.g., 2000+ tokens in LLaVA 1.6), storing and computing the $m \times m$ matrix may become a bottleneck.
5. **No text–vision interaction**: Pruning relies entirely on the internal distributional structure of visual tokens without leveraging textual query information for conditional pruning, potentially underperforming attention-based methods on specific tasks.
6. **Fixed pruning ratio**: In the one-shot setting, all images use the same pruning ratio without adaptive adjustment based on image complexity.

## Related Work & Insights

- **DivPrune** (CVPR 2025): A diversity-driven pruning method based on DPP, the direct predecessor and primary baseline for OTPrune. The core improvement of OTPrune lies in advancing from "diversity" to "distribution alignment."
- **FastV / VTW**: Attention-score-based pruning methods that suffer sharp performance degradation at high pruning ratios, demonstrating that attention is not a reliable indicator of token importance.
- **FitPrune**: A calibration-set-based method that depends on external data and generalizes poorly.
- **M³**: A fine-tuning-based method with high accuracy but substantial computational cost.
- **Optimal Transport in ML**: OT has been widely applied in generative models (WGAN) and domain adaptation; its introduction to token pruning in this paper represents a natural and elegant transfer.
- **Submodular optimization**: The log-det submodular function has a mature theoretical foundation in sensor placement, feature selection, and related fields; this paper leverages this machinery to achieve theoretically grounded greedy selection.

**Insights**: The distribution alignment paradigm can be generalized to other token/patch selection settings (e.g., token distillation in ViT, video frame selection, point cloud downsampling). The Wasserstein distance is a more principled pruning quality metric than attention scores.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to formulate token pruning as an OT problem; the derivation chain from Wasserstein distance → Gaussian surrogate → log-det submodular objective is complete and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 11 benchmarks, 3 LLaVA variants, synthetic data validation, ablation studies, and statistical significance tests; lacks validation on non-LLaVA architectures.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain from motivational hypothesis validation to theoretical derivation to experimental analysis is clear and rigorous, with professionally designed figures and tables.
- **Value**: ⭐⭐⭐⭐ — A training-free, plug-and-play token pruning method with theoretical guarantees, offering direct practical value for efficient MLLM inference; the OT perspective opens promising new directions for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/optimization/estimation_of_stochastic_optimal_transport_maps.md)
- [\[CVPR 2026\] Enhancing Visual Representation with Textual Semantics: Textual Semantics-Powered Prototypes for Heterogeneous Federated Learning](enhancing_visual_representation_with_textual_semantics_textual_semantics_powered_p.md)
- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](../../ICLR2026/optimization/cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)
- [\[NeurIPS 2025\] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling](../../NeurIPS2025/optimization/fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)

</div>

<!-- RELATED:END -->
