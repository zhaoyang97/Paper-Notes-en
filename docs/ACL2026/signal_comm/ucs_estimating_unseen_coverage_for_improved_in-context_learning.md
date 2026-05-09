---
title: >-
  [Paper Note] UCS: Estimating Unseen Coverage for Improved In-Context Learning
description: >-
  [ACL 2026][Signal & Communication][In-Context Learning] This paper proposes UCS (Unseen Coverage Selection), a training-free subset-level coverage prior based on the Smoothed Good-Turing estimator that regularizes existing ICL example selection methods by estimating the number of unobserved latent clusters in candidate example sets, improving accuracy by 2-6% on intent classification and reasoning tasks.
tags:
  - ACL 2026
  - "Signal & Communication"
  - In-Context Learning
  - Example Selection
  - Coverage Estimation
  - Good-Turing Estimator
  - Clustering
date: 2025-04-17
content_hash: 43a1ccd08136c173
---

# UCS: Estimating Unseen Coverage for Improved In-Context Learning

**Conference**: ACL 2026  
**arXiv**: [2604.12015](https://arxiv.org/abs/2604.12015)  
**Code**: [https://github.com/Raina-Xin/UCS](https://github.com/Raina-Xin/UCS)  
**Area**: In-Context Learning  
**Keywords**: In-Context Learning, Example Selection, Coverage Estimation, Good-Turing Estimator, Clustering

## TL;DR

This paper proposes UCS (Unseen Coverage Selection), a training-free subset-level coverage prior based on the Smoothed Good-Turing estimator that regularizes existing ICL example selection methods by estimating the number of unobserved latent clusters in candidate example sets, improving accuracy by 2-6% on intent classification and reasoning tasks.

## Background & Motivation

**Background**: In-context learning (ICL) performance is highly dependent on which examples are placed in the prompt. Existing methods select examples based on similarity (e.g., semantic proximity to the query), diversity (e.g., DPP), or information-theoretic criteria (e.g., MDL).

**Limitations of Prior Work**: Existing methods all operate at the instance level — evaluating individual example relevance or pairwise diversity — but lack a subset-level coverage perspective. A good example set should cover diverse latent patterns (clusters) underlying the task, but no method can quantify how many latent patterns remain uncovered by the current selection.

**Key Challenge**: The latent pattern distribution in ICL example pools exhibits a severe long tail — a few patterns dominate many samples while numerous patterns have only a few samples. Similarity- or diversity-based methods tend to select from frequent patterns, causing rare patterns to be systematically neglected.

**Goal**: Propose a subset-level coverage prior that serves as a lightweight plugin to enhance existing ICL selection methods, encouraging selection of example sets that cover more latent patterns.

**Key Insight**: Borrow from ecology's classic method for estimating "unseen species count" — the Smoothed Good-Turing estimator — drawing an analogy between "uncovered latent clusters" in ICL and "unobserved species."

**Core Idea**: Define latent patterns using clusters in model-consistent embedding space, estimate how many clusters remain uncovered from frequency spectra using the Good-Turing estimator, and add this estimate as a regularization term to existing selection objectives.

## Method

### Overall Architecture

UCS has three steps: (1) Represent all candidate examples using the LLM's own embeddings (model-consistent representation); (2) Discretize continuous embeddings into cluster IDs through dictionary learning + DBSCAN (discretization); (3) Estimate total cluster count from the selected subset's frequency spectrum using the Smoothed Good-Turing estimator (coverage estimation), combined with existing selection objectives via weighted sum.

### Key Designs

1. **Model-Consistent Embedding and Cluster Discretization**:

    - Function: Convert continuous LLM embeddings into discrete latent cluster labels
    - Mechanism: Extract hidden states from the inference-time LLM for candidate examples (input portion only, excluding labels), obtain fixed-length vectors via masked mean pooling. Then perform dictionary learning (ridge coding) to get each example's encoding vector over K atoms, followed by DBSCAN (cosine distance) clustering in the normalized encoding space. Noise points are assigned as independent singleton clusters.
    - Design Motivation: Using argmax atom assignment would over-concentrate on high-frequency atoms, ignoring multi-atom combination structures. Dictionary learning + clustering captures recurring latent pattern combinations while preserving long-tail fine-grained units.

2. **Smoothed Good-Turing Coverage Estimation**:

    - Function: Estimate total (including unobserved) cluster count from the selected subset's frequency spectrum
    - Mechanism: Construct frequency spectrum $f_s(S)$ (number of clusters appearing $s$ times) from cluster labels of candidate subset $S$, then use the SGT estimator $\hat{U}_t^{SGT}(S) = -\sum_{s=1}^{M} (-t)^s w_s(t,\alpha) f_s(S)$ to predict how many new clusters would be observed if $m$ more samples were drawn. The UCS coverage function is $\Phi_{UCS}(S) = K_{seen}(S) + \hat{U}_t(S)$, simultaneously considering observed cluster count and predicted unobserved cluster count.
    - Design Motivation: Good-Turing is a classic statistical method for estimating "unseen species count," with a long history in ecology and NLP. Key insight: the numbers of singletons (appearing once) and doubletons (appearing twice) in the frequency spectrum contain rich information about unobserved categories.

3. **UCS Regularized Selection**:

    - Function: Seamlessly integrate the coverage prior into existing ICL selection methods
    - Mechanism: Selection objective is $S^* = \arg\max_{|S|=B} (U_{base}(S; x_{test}) + \lambda \Phi_{UCS}(S))$, where $U_{base}$ is the original utility of DPP/MDL/VoteK, and $\lambda$ controls coverage regularization strength. Inverse frequency weighting is used for VoteK, marginal coverage gain for DPP, and direct scoring at the candidate set level for MDL. $\lambda=0$ degenerates to the original method.
    - Design Motivation: UCS is a subset-level function (not decomposable into instance-level scores), used as a prior regularizer rather than an independent selector, maximally preserving the strengths of the original method.

### Loss & Training

UCS is entirely training-free. Offline preprocessing (embedding + clustering) takes 38-57 seconds per dataset; online inference overhead is approximately 0-3 seconds. All hyperparameters have explicit default values (dictionary atom count K, SGT truncation order M=20, expansion factor t, etc.).

## Key Experimental Results

### Main Results

| Method | Banking77 (Qwen) | CLINC150 (Qwen) | HWU64 (Qwen) |
|--------|------|------|------|
| VoteK | 0.518 | 0.703 | 0.609 |
| UCS+VoteK | 0.543 (+2.5%) | 0.744 (+4.1%) | 0.671 (+6.2%) |
| DPP | 0.831 | 0.755 | 0.791 |
| UCS+DPP | 0.831 | 0.775 (+2.0%) | 0.794 |
| MDL | 0.764 | 0.748 | 0.785 |
| UCS+MDL | 0.771 | 0.752 | 0.801 (+1.6%) |

### Ablation Study

| Config | Metric | Note |
|------|---------|------|
| UCS+VoteK | Unique clusters 10.0, cluster size 1.0 | Completely eliminates redundancy |
| VoteK original | Unique clusters 9.67, cluster size 8.50 | Significant redundancy |
| Cross-model joint dictionary | Degraded | Forcing alignment of different embedding spaces loses information |

### Key Findings

- **Query-independent methods benefit most**: VoteK + UCS improves by 6.2% on HWU64 (Qwen) and 4.1% (Llama), as VoteK originally selects the most redundant examples.
- **Effective on reasoning tasks**: On BBEH reasoning tasks, UCS+DPP improves by 12.5 pp on Shuffled Objects, and UCS+MDL improves by 8.4 pp on Causal Understanding.
- **Cluster distributions exhibit severe long tails**: Across all dataset-model combinations, cluster size distributions are extremely skewed — many singletons and a few dominant clusters — validating the necessity of the coverage prior.
- **Model-consistent embeddings outperform cross-model joint**: Joint dictionary learning impairs high-capability models' fine-grained discrimination.
- **Minimal computational overhead**: Offline preprocessing 38-57s, online additional 0-3s.

## Highlights & Insights

- **Elegant connection between statistics and NLP**: Applying ecology's "unseen species count" estimation (Good-Turing) to "uncovered latent clusters" estimation in ICL selection — the analogy is natural and methodologically rigorous.
- **Plug-and-play design**: UCS can seamlessly stack as a regularization term on any existing selection method without modifying the underlying retrieval pipeline; $\lambda=0$ degenerates to the original method, making it highly practical.
- **Interpretable cluster analysis**: UCS-generated clusters are semantically interpretable (e.g., identity verification, ATM withdrawal micro-topics in Banking77), providing task structure insights.

## Limitations & Future Work

- For already strong query-dependent methods (DPP on some datasets is near saturation), UCS gains are limited.
- Cluster quality depends on DBSCAN hyperparameter selection (eps requires adaptive heuristics).
- SGT estimator has limited statistical reliability under small sample selection budgets (B=10).
- Only evaluated at a fixed budget of B=10; performance under different budgets is unknown.

## Related Work & Insights

- **vs DPP**: DPP encourages diversity through determinant maximization but does not explicitly quantify coverage. UCS provides a complementary subset-level coverage signal, yielding better results when combined with DPP.
- **vs VoteK**: VoteK selects global example sets based on voting without diversity guarantees. UCS substantially eliminates redundancy through inverse frequency weighting.
- **vs MDL**: MDL selects informative examples using minimum description length; UCS provides an orthogonal optimization signal from the coverage perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ Good-Turing application in ICL is novel; subset-level coverage perspective is valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, three classification tasks, three reasoning tasks, though fixed budget limits analysis depth
- Writing Quality: ⭐⭐⭐⭐⭐ Methodology is clear and rigorous; theory and experiments are tightly connected
- Value: ⭐⭐⭐⭐ Practical plug-and-play tool directly applicable to ICL deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](../../ICLR2026/signal_comm/spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/signal_comm/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](../../ICLR2026/signal_comm/learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[CVPR 2026\] Dual-Imbalance Continual Learning for Real-World Food Recognition](../../CVPR2026/signal_comm/dual-imbalance_continual_learning_for_real-world_food_recognition.md)

</div>

<!-- RELATED:END -->
