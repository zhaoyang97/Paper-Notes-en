---
title: >-
  [Paper Note] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series
description: >-
  [NeurIPS 2025 Spotlight][Time Series][Multimodal time series] This paper proposes MAESTRO, a framework that addresses modality heterogeneity and arbitrary missingness in multimodal time series via symbolic tokenization, adaptive attention budgeting, sparse cross-modal attention, and dynamic MoE routing, substantially outperforming baselines under both complete and missing modality scenarios.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Time Series"
  - "Multimodal time series"
  - "sparse attention"
  - "mixture of experts"
  - "missing modality"
  - "sensor fusion"
date: 2026-05-08
content_hash: 30ab0450eabbd53c
---

# MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.25278](https://arxiv.org/abs/2509.25278)  
**Code**: [GitHub](https://github.com/payalmohapatra/MAESTRO)  
**Area**: Time Series
**Keywords**: Multimodal time series, sparse attention, mixture of experts, missing modality, sensor fusion

## TL;DR

This paper proposes MAESTRO, a framework that addresses modality heterogeneity and arbitrary missingness in multimodal time series via symbolic tokenization, adaptive attention budgeting, sparse cross-modal attention, and dynamic MoE routing, substantially outperforming baselines under both complete and missing modality scenarios.

## Background & Motivation

Multimodal sensor monitoring (e.g., ECG, electrodermal activity, accelerometers) is widely used in clinical healthcare and daily-life settings. However, existing multimodal learning methods suffer from three key limitations: (1) reliance on a single dominant modality as an alignment anchor, leading to over-dependence on a pre-specified primary modality; (2) pairwise modality interaction modeling, which becomes computationally infeasible as the number of modalities grows; and (3) the assumption that all modalities are fully available, precluding robust handling of arbitrary missingness caused by sensor failures. The authors argue that naively treating multi-source sensor data as multivariate time series is suboptimal, since modalities are semantically disjoint and require explicit modeling of both intra- and inter-modal interactions.

## Method

### Overall Architecture

MAESTRO comprises four core modules: (1) missing-aware symbolic tokenization that converts raw time series into discrete symbolic representations; (2) adaptive attention budget gating that allocates sparse attention capacity according to modality availability and task relevance; (3) sparse cross-modal attention for processing long concatenated multimodal sequences; and (4) sparse MoE routing for input-conditioned dynamic expert assignment.

### Key Designs

1. **Missing-Aware Symbolic Tokenization (SAX Tokenization)**: Each modality is independently compressed and symbolized via Symbolic Aggregate approXimation (SAX). The key innovation is the reservation of an extra symbol $s_0$ to represent missing modalities. Exploiting the Gaussian distribution of normalized time series, continuous values are mapped to $\alpha$ equiprobable symbols. It is theoretically shown (Corollary 3.2) that the SAX transformation not only preserves distance lower bounds within each modality but also maintains the relative contrastive structure of sample similarity across modalities, ensuring that meaningful cross-modal structure is retained.

2. **Adaptive Sparse Attention and Attention Budgeting**: Each modality has an independent encoder employing sparse self-attention (selecting top-$\upsilon$ queries based on a max-mean sparsity measure). The central innovation is that the attention budget $\mathbf{u}_i$ is adaptively controlled by a learnable gating function $\mathcal{G}(\mathbf{m}_i; \theta_a)$, where $\mathbf{m}_i$ is the modality availability vector. This gate is trained end-to-end with the task objective, automatically allocating greater attention resources to more task-relevant modalities.

3. **Sparse Cross-Modal Attention**: The outputs of all modality encoders are concatenated along the temporal dimension, augmented with modality and positional embeddings, to form a unified long multimodal sequence $\mathbf{c} \in \mathbb{R}^{\hat{D} \times \hat{L}}$. Sparse attention is applied to this sequence, reducing complexity from $\mathcal{O}(\hat{L}^2)$ to $\mathcal{O}(\hat{L} \log \hat{L})$. This design naturally accommodates modalities with different sampling rates, enables time-varying cross-modal attention, and avoids the combinatorial explosion of pairwise interaction modeling.

4. **Sparse MoE Dynamic Routing**: A standard sparse MoE layer processes the cross-modal representations, with a trainable router $\mathcal{R}$ selecting the top-$k$ experts. Implicit expert specialization is achieved through progressive modality dropping (curriculum learning)—training initially on complete modality samples and gradually increasing the missing probability to $p_{max}$—without requiring auxiliary load-balancing losses.

### Loss & Training

The standard cross-entropy loss $\mathcal{L}_{CE}$ is used. Progressive modality dropping is applied: the modality dropout probability $p(\tau)$ increases linearly to $p_{max}$ after a warmup period $\tau_{warmup}$, serving as a regularizer that encourages the model to adapt to diverse modality combinations. The MoE employs top-$k$ (typically $k=1$) routing; for $k>1$, expert outputs are aggregated by averaging logits. An 80/10/10 train/validation/test split is used, with three different splits per dataset and three trials per split; means are reported.

## Key Experimental Results

### Main Results

| Dataset | Metric | MAESTRO | Best Baseline | Gain |
|---------|--------|---------|---------------|------|
| WESAD (10 modalities) | Acc | 0.77 | 0.71 (FlexMoE) | +8% relative |
| DaliaHAR (5 modalities) | F1 | 0.84 | 0.79 (FuseMoE) | +5% relative |
| DSADS (5 positions, 9 axes) | Acc | 0.88 | 0.85 (FuseMoE) | +4% relative |
| MIMIC-III (17 modalities) | F1 | 0.30 | 0.27 (FlexMoE) | +11% relative |

### Ablation Study

| Configuration | Full Modality Acc Drop | 40% Missing Acc Drop | Notes |
|---------------|------------------------|----------------------|-------|
| w/o SAX | −8% | −7% | Symbolic tokenization contributes substantially |
| w/o MoE | −5% | −22% | MoE is critical under missing modality scenarios |
| w/o modality dropping | −2% | −9% | Training strategy key for missing robustness |
| Replace all with full attention | +3% Acc | — | GFLOPs increase by 20% |

### Key Findings

- Under 40% modality missingness, MAESTRO achieves an average F1 improvement of 59% over the strongest missing-robust baseline.
- Sparse attention sacrifices at most 3% accuracy while reducing GFLOPs by approximately 20% (only 6.13 GFLOPs, 1.39M parameters).
- Multimodal processing outperforms multivariate processing by an average of 4%; MAESTRO further exceeds the best multimodal baseline by 8%.
- MAESTRO exhibits inherent robustness to noisy inputs (Gaussian noise and electrical interference spikes).
- Symbolic representations contribute an average relative performance gain of 6%.
- Sensitivity analysis on the compression ratio shows that both excessively large and small ratios degrade performance.

## Highlights & Insights

- Reframing sensor time series from a "multivariate" to a "multimodal" perspective represents a valuable paradigm shift.
- Symbolic representation simultaneously achieves sequence compression, missingness representation, and noise robustness—an elegantly multifunctional design.
- Implicit MoE specialization without auxiliary losses is realized through progressive dropout, yielding a clean and effective solution.
- Sparse cross-modal attention avoids pairwise modeling and improves computational efficiency by over 200% compared to MULT.

## Limitations & Future Work

- Validation is limited to classification tasks; regression and time series forecasting settings remain unexplored.
- SAX hyperparameters (compression ratio and alphabet size) require per-dataset tuning.
- Absolute performance on MIMIC-III remains modest (Acc 0.78), with limited gains on complex tasks.
- Extension to online learning scenarios with asynchronous sampling warrants future investigation.

## Related Work & Insights

- FlexMoE and FuseMoE handle missing modalities via MoE but rely on pairwise modeling or missing-modality imputation banks.
- MULT's pairwise Transformer incurs high computational cost (~27 GFLOPs vs. MAESTRO's 6 GFLOPs).
- The SAX symbolization approach is transferable to other time series tasks requiring robust handling of missing sensors.
- ShaSpec addresses missingness through shared modality representations but underperforms at high missing rates.
- Multivariate methods such as InceptionTime and ResNet1D overlook modality heterogeneity, resulting in a lower performance ceiling.
- iTransformer's inverted attention performs poorly on certain datasets (only 0.62 on DSADS).
- The adaptive attention budgeting design shares conceptual similarities with resource allocation in Neural Architecture Search.
- Symbolic representation is related to the discretization philosophy of VQ-VAE, but SAX requires no learned codebook.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of multiple design innovations (SAX-based missing representation, adaptive budgeting, lossless MoE) is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 4 datasets, 10 baselines, and ablation, complexity, sensitivity, and noise experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with good integration of theory and experiments.
- **Value**: ⭐⭐⭐⭐ — Meaningful practical contribution to the multimodal sensor fusion field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Soft Sparse Shapes for Efficient Time-Series Classification](../../ICML2025/time_series/learning_soft_sparse_shapes_for_efficient_time-series_classification.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICML 2026\] DistMatch: Adaptive Binning via Distribution Matching for Robust Sequential Conformal](../../ICML2026/time_series/distmatch_adaptive_binning_via_distribution_matching_for_robust_sequential_confo.md)
- [\[ICLR 2026\] Enhancing Sparse Event Detection in Healthcare Time-Series via Adaptive Gate of Context–Detail Interaction](../../ICLR2026/time_series/enhancing_sparse_event_detection_in_healthcare_time-series_via_adaptive_gate_of_.md)
- [\[ICLR 2026\] GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care](../../ICLR2026/time_series/garlic_graph_attention-based_relational_learning_of_multivariate_time_series_in_.md)

</div>

<!-- RELATED:END -->
