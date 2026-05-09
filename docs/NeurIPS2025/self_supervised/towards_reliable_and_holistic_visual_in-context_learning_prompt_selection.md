---
title: >-
  [Paper Note] Towards Reliable and Holistic Visual In-Context Learning Prompt Selection
description: >-
  [NeurIPS 2025][Self-Supervised Learning][visual in-context learning] This paper proposes RH-Partial2Global, which for the first time employs Spearman rank correlation tests to demonstrate that the "similarity-first hypothesis" in VICL is statistically significant yet exhibits extremely weak correlation strength ($\bar{\rho} \approx 0.03\text{-}0.05$). By constructing reliable candidate sets via Jackknife conformal prediction and achieving comprehensive uniform pairwise preference sampling through covering designs, the method consistently outperforms state-of-the-art approaches across three visual tasks: segmentation, detection, and colorization.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - visual in-context learning
  - conformal prediction
  - covering design
  - prompt selection
  - global ranking
date: 2026-05-08
content_hash: 812ad73dd1202ac2
---

# Towards Reliable and Holistic Visual In-Context Learning Prompt Selection

**Conference**: NeurIPS 2025
**arXiv**: [2509.25989](https://arxiv.org/abs/2509.25989)
**Code**: [github.com/Wu-Wenxiao/RH-Partial2Global](https://github.com/Wu-Wenxiao/RH-Partial2Global)
**Area**: Visual In-Context Learning / Prompt Selection
**Keywords**: visual in-context learning, conformal prediction, covering design, prompt selection, global ranking

## TL;DR
This paper proposes RH-Partial2Global, which for the first time employs Spearman rank correlation tests to demonstrate that the "similarity-first hypothesis" in VICL is statistically significant yet exhibits extremely weak correlation strength ($\bar{\rho} \approx 0.03\text{-}0.05$). By constructing reliable candidate sets via Jackknife conformal prediction and achieving comprehensive uniform pairwise preference sampling through covering designs, the method consistently outperforms state-of-the-art approaches across three visual tasks: segmentation, detection, and colorization.

## Background & Motivation
**Background**: Visual in-context learning (VICL) adapts visual foundation models to new tasks by providing a small number of in-context examples. The central challenge lies in selecting optimal in-context examples for each query. Existing methods (VPR, Partial2Global) formulate this as a global ranking problem.

**Limitations of Prior Work**: (1) **The similarity-first hypothesis lacks rigorous justification** — both VPR and Partial2Global rely on the assumption that images with higher visual similarity to the query make better in-context examples, yet this assumption has never been formally validated; (2) **Partial2Global's random sampling strategy is flawed** — it shuffles the candidate set randomly to generate subsequences for local ranking, providing no guarantee that all pairwise relationships are covered ($K=50, k=5$ requires at least 130 subsequences to cover all pairs, whereas the original method uses only 50), and may produce redundant comparisons.

**Key Challenge**: How to construct a reliable and comprehensive in-context example selection pipeline without over-relying on the similarity assumption?

**Key Insight**: Apply conformal prediction theory to filter unreliable candidates (R), and apply covering design theory to guarantee the completeness of pairwise comparisons (H).

## Method

### Overall Architecture
RH-Partial2Global introduces two orthogonal enhancements on top of Partial2Global: (1) Jackknife conformal prediction to screen a reliable candidate set $\mathcal{Y}_\alpha$, which is intersected with the similarity-based candidate set $\mathcal{Y}_q$ to obtain the refined set $\mathcal{Y}_q^* = \mathcal{Y}_\alpha \cap \mathcal{Y}_q$; (2) replacement of random shuffling with a $(K', k, 2)$ covering design to guide subsequence sampling for local ranking, ensuring all pairwise preferences are covered by at least one subsequence. Neither module requires additional model training.

### Key Designs
1. **Jackknife Conformal Prediction-Guided Candidate Selection**:
    - **Function**: Filters training samples whose quality and similarity are mutually consistent, removing candidates that are visually similar but of low quality.
    - **Mechanism**: For each training sample $x_i^{trn}$, the quality score set $\mathcal{Q}(x_i) = \{\mathfrak{q}(\mathcal{F}(x_j, x_i), x_i)\}$ and similarity score set $\mathcal{S}(x_i) = \{\mathfrak{s}(x_j, x_i)\}$ are computed over all other samples. A conformity score $\ell(x_i) = f(\mathcal{Q}(x_i), \mathcal{S}(x_i))$ is derived (where $f$ is the negative KL divergence). The $(1-\alpha)$ quantile $q_{1-\alpha}$ serves as the threshold, yielding the reliable set $\mathcal{Y}_\alpha = \{x_i | \ell(x_i) > q_{1-\alpha}\}$. For query $x_q$: $\mathcal{Y}_q^* = \mathcal{Y}_\alpha \cap \text{top-K}(\mathfrak{s}(x_q, \cdot))$.
    - **Design Motivation**: Spearman tests reveal an extremely low correlation coefficient between similarity and quality ($\bar{\rho} \approx 0.03\text{-}0.05$), indicating that similarity alone is insufficient for reliable candidate selection. Conformal prediction provides distribution-free coverage guarantees, and the Jackknife formulation makes full use of training data.

2. **Covering Design-Guided Holistic Sampling**:
    - **Function**: Replaces random shuffling with combinatorial covering designs to ensure all candidate pairs are covered by at least one local ranking subsequence.
    - **Mechanism**: A $(K, k, 2)$ covering design requires that every 2-element subset of a $K$-element set appears in at least one $k$-element block. The Schönheim lower bound $C(K,k,t) \geq \lceil\frac{K}{k}\lceil\frac{K-1}{k-1}...\rceil\rceil$ gives the minimum number of blocks ($C(50,5,2) \geq 130$). A precomputed optimal covering design $C^*(K',k,2)$ guides sampling: after randomly permuting the candidate set, $k$-length subsequences are extracted according to the covering design structure.
    - **Design Motivation**: Partial2Global's 50 random subsequences fail to cover all $\binom{50}{2} = 1225$ pairwise relations, and redundant comparisons lead to uneven preference weights. The covering design guarantees exhaustiveness while minimizing the number of subsequences.

3. **Statistical Validation of the Similarity-First Hypothesis**:
    - **Function**: Provides the first rigorous statistical test of a foundational assumption in VICL.
    - **Mechanism**: On the Pascal-5i training set, for each query sample, Spearman rank correlation tests are performed between the sequences of (IoU scores, visual similarity scores) across all candidates. Results show that 78–88% of samples reject the null hypothesis ($p < 0.05$), confirming a statistically significant monotonic relationship; however, $\bar{\rho} \approx 0.03\text{-}0.05$ indicates extremely weak correlation strength.
    - **Design Motivation**: Partial2Global questioned this assumption without providing statistical evidence. The quantitative analysis in this paper supplies the theoretical motivation for introducing conformal prediction.

### Loss & Training
The meta-learning training phase is identical to Partial2Global — training a transformer-based list-wise ranker $\phi_k$ (lengths 5 and 10) with DINOv2 features, AdamW optimizer with lr $= 5\times10^{-5}$, and batch size 64. At inference: $\alpha=0.85$ (85% confidence), negative KL divergence as the conformity function, and a CLIP visual encoder for similarity. RH-Partial2Global modifies only the candidate selection and sampling strategy at inference time and requires no additional training.

## Key Experimental Results

### Main Results: Cross-Task Comparison

| Method | Segmentation Avg (mIoU)↑ | Detection (mIoU)↑ | Colorization (MSE)↓ |
|---|---|---|---|
| MAE-VQGAN (NeurIPS'22) | 27.56 | 25.45 | 0.67 |
| SupPR (NeurIPS'23) | 35.56 | 28.22 | 0.63 |
| Partial2Global (NeurIPS'24) | 38.40 | 30.66 | 0.58 |
| **RH-Partial2Global** | **39.02** | **30.94** | **0.56** |
| Partial2Global+voting | 42.69 | 32.52 | — |
| **RH-Partial2Global+voting** | **43.08** | **33.28** | — |

### Ablation Study: Component Contributions (Segmentation Task, Average over 4 Folds)

| Configuration | Fold-0 | Fold-1 | Fold-2 | Fold-3 | Avg |
|---|---|---|---|---|---|
| Partial2Global baseline | 38.81 | 41.54 | 37.25 | 36.01 | 38.40 |
| + Conformal Prediction (R) | 39.05 | 41.80 | 37.72 | 36.35 | 38.73 |
| + Covering Design (H) | 39.10 | 41.95 | 37.85 | 36.42 | 38.83 |
| + R + H (Full) | **39.25** | **42.15** | **38.06** | **36.60** | **39.02** |

### Key Findings
- The similarity-first hypothesis is statistically significant (78–88% of samples with $p<0.05$) but exhibits extremely weak correlation strength ($\bar{\rho} \approx 0.03\text{-}0.05$).
- Conformal prediction filters approximately 15% of candidates while the performance upper bound of high-quality examples remains virtually unchanged (top-5 IoU drops by only 0.26).
- RH-Partial2Global achieves **consistent** improvements across all 3 tasks and 4 folds without any additional model training.
- Visualizations show that examples selected by RH are more aligned with the query in fine-grained attributes such as pose and scene context.

## Highlights & Insights
- This is the first work to provide rigorous statistical evidence that the similarity-first assumption in VICL is insufficiently robust.
- Conformal prediction offers theoretically grounded, distribution-free coverage guarantees for reliable candidate selection, independent of the specific task.
- Covering design is an elegant mathematical tool for systematically sampling pairwise relationships, introducing combinatorial optimization into ranking aggregation.
- Both enhancement modules are inference-time modifications, incurring no additional training cost and enabling plug-and-play deployment.

## Limitations & Future Work
- Improvements are consistent but modest (~0.6% on average), with small folds constrained by calibration set size.
- $\alpha=0.85$ is uniformly set across all tasks; adaptive $\alpha$ selection may yield further gains.
- Precomputation of covering designs may introduce non-trivial computational overhead for very large candidate sets ($K > 100$).
- Validation is limited to MAE-VQGAN as the VICL backbone; generalizability to other visual foundation models remains unexplored.
- The choice of conformity function (negative KL divergence) lacks systematic comparison against alternatives.

## Related Work & Insights
- **vs. Partial2Global (NeurIPS'24)**: A direct improvement; the core contributions lie in challenging the foundational assumption and correcting the sampling deficiency. Segmentation average mIoU improves from 38.40 → 39.02; detection from 30.66 → 30.94.
- **vs. VPR (NeurIPS'23)**: VPR relies entirely on similarity; this paper demonstrates that similarity with $\bar{\rho}<0.06$ is insufficiently reliable.
- **vs. prompt-SelF**: prompt-SelF employs voting-based ensemble (an orthogonal direction); RH-Partial2Global+voting = 43.08 > prompt-SelF = 41.02.
- **Insight**: The application of conformal prediction in machine learning is a growing trend (uncertainty quantification, LLM benchmarking, etc.); its use for in-context example selection in this paper represents a novel instantiation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The hypothesis-testing-based problem analysis is novel; the combination of conformal prediction and covering design is distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three tasks, four folds, and complete ablations, though improvement margins are limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is rigorous, mathematical derivations are clear, and the motivation-to-method logical thread is strong.
- **Value**: ⭐⭐⭐⭐ — Offers a methodological contribution to VICL prompt selection, though practical impact is constrained by the magnitude of improvements.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](../../ICLR2026/self_supervised/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[NeurIPS 2025\] T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)

<!-- RELATED:END -->
