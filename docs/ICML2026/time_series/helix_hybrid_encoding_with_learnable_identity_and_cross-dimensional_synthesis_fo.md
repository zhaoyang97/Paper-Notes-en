---
title: >-
  [Paper Note] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation
description: >-
  [ICML 2026][Time Series][Feature Identity Embedding] A learnable "identity embedding" is assigned to each feature as a persistent semantic anchor…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Feature Identity Embedding"
  - "Time Series Imputation"
  - "Spatiotemporal Transformer"
  - "Double Helix Encoding"
date: 2026-05-08
content_hash: 103a5f548c5b4f75
---

# HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation

**Conference**: ICML 2026  
**arXiv**: [2605.02278](https://arxiv.org/abs/2605.02278)  
**Code**: https://github.com/milaogou/HELIX (integrated into PyPOTS)  
**Area**: Time Series / Missing Value Imputation / Transformer  
**Keywords**: Feature Identity Embedding, Time Series Imputation, Spatiotemporal Transformer, Double Helix Encoding

## TL;DR
A learnable "identity embedding" is assigned to each feature as a persistent semantic anchor, combined with a time-feature double helix attention mechanism. HELIX achieves first place across all 21 missing data scenarios on 5 public multivariate time series datasets, outperforming the next-best ImputeFormer by over 25% MAE reduction on datasets like ETT-h1.

## Background & Motivation

**Background**: Multivariate time series imputation is a crucial preprocessing step for downstream tasks in healthcare, meteorology, transportation, etc. Mainstream methods fall into three categories: RNN-based (BRITS, GRU-D), Transformer-based (SAITS, ImputeFormer), and diffusion models (CSDI, PriSTI). Recently, GNN-based methods (GRIN, SPIN) have also attempted to explicitly model inter-feature dependencies.

**Limitations of Prior Work**: (1) Existing attention-based methods "rediscover" feature relationships at each layer, lacking cross-layer consistent anchors—leading to collapse of feature relations under heavy missingness; (2) GNN methods rely on predefined graph topology, assuming feature homogeneity (e.g., all are spatial sensors of the same type), and cannot handle mixed feature types; (3) Learning adjacency matrices incurs $O(F^2)$ cost and is still affected by missing data; (4) Bidirectional attention methods like Crossformer use only value patches for embedding, so cross-feature attention degenerates when values are missing.

**Key Challenge**: For "cross-feature reasoning," the model needs each token to possess both temporal and feature identities; however, existing solutions only provide a persistent anchor on one axis (either temporal PE or graph topology), while the other axis must be dynamically inferred from values—which fails when values are missing.

**Goal**: (1) Provide each feature with a cross-layer stable semantic identity, (2) Design an encoding structure that enables full bidirectional interaction between time and feature axes, (3) Maintain stable cross-feature reasoning even under severe missingness.

**Key Insight**: The authors treat token embedding as a soft prompt in NLP—each feature learns a $d_f$-dimensional vector $f_i$ as a "feature-specific prompt," ensuring feature identity information is always present regardless of value missingness. A "parallel-then-cross" double helix attention mechanism is designed to alternately process time and feature dimensions.

**Core Idea**: The embedding at each $(t, i)$ position is $e_{t,i} = [\tilde x_{t,i}; \text{PE}(t); f_i; m_{t,i}]$, where $f_i$ is a learnable identity embedding. $L$ layers of "double helix" encoding (each layer first parallelizes time and feature attention, then crosses them) allow full information flow between the two axes.

## Method

### Overall Architecture
Input $\tilde X \in \mathbb{R}^{T \times F}$ and missing mask $M$; each position forms $e_{t,i} \in \mathbb{R}^{d_e}$ (value + sinusoidal PE + identity + mask), projected via a linear layer to hidden dimension $d$ to obtain $H^{(0)}$. This is followed by $L$ Hybrid Encoding Layers, each outputting four branches $H_T^{(l)}, H_F^{(l)}, H_{TF}^{(l)}, H_{FT}^{(l)}$, which are averaged to get $H^{(l)}$. Finally, multi-level fusion is performed: $\tilde H = \frac{1}{1+4L}(H^{(0)} + \sum_l \text{sum of branches})$, followed by LayerNorm and a linear layer to produce $\hat X$.

### Key Designs

1. **Feature Identity Embedding (FeatID) as a Soft Adjacency Prior for Cross-feature Attention**:

    - **Function**: Provides each feature with a cross-layer, cross-time stable semantic vector, enabling attention even when values are missing.
    - **Mechanism**: Concatenate $e_{t,i} = [\tilde x_{t,i}; \text{PE}(t); f_i; m_{t,i}]$, where $f_i \in \mathbb{R}^{d_f}$ is the $i$-th row of a learnable identity matrix. The attention score $s_{ij}^{(t)} = e_{t,i}^\top A e_{t,j}$ decomposes into an identity prior $f_i^\top A_{ff} f_j$, identity-context cross terms, and dynamic context $r_{t,i}^\top A_{rr} r_{t,j}$. When $x_{t,i}$ and $x_{t,j}$ are both missing, the dynamic term vanishes, but the identity prior still provides cross-feature compatibility.
    - **Design Motivation**: In ImputeFormer, the "static feature embedding" is only a soft spatial index and does not interact with missingness; SPIN's graph embedding is tied to predefined topology. FeatID requires no graph prior and remains anchored under heavy missingness. Ablation on BeijingAir shows removing FeatID increases Subseq-50% MAE from 0.166 to 0.398, highlighting its critical role.

2. **Double Helix Hybrid Encoding Layer (Parallel-then-Cross)**:

    - **Function**: Allows both time and feature dimensions to be refined independently and to exchange information across dimensions.
    - **Mechanism**: Each layer has two stages. Stage 1: Parallel $H_T = \text{TimeMHA}(H^{(l-1)})$ and $H_F = \text{FeatMHA}(H^{(l-1)})$, optimized independently. Stage 2: Serial cross $H_{TF} = \text{FeatMHA}(H_T)$ and $H_{FT} = \text{TimeMHA}(H_F)$. The four branches are averaged: $H^{(l)} = \frac{1}{4}(H_T + H_F + H_{TF} + H_{FT})$. The structure resembles a DNA double helix.
    - **Design Motivation**: Purely serial Time→Feature→Time encoding (w/o Hybrid) increases Subseq-50% MAE from 0.166 to 0.294, indicating that parallel+cross bidirectional flow is key for handling long missing gaps. Serial encoding compresses information from the other dimension before propagation, causing bottlenecks.

3. **Multi-level Fusion**:

    - **Function**: Aggregates outputs from all layers (including the 0-th embedding layer) by weighted averaging, avoiding loss of shallow details if only the last layer is used.
    - **Mechanism**: $\tilde H = \frac{1}{1+4L}(H^{(0)} + \sum_{l=1}^L (H_T^{(l)} + H_F^{(l)} + H_{TF}^{(l)} + H_{FT}^{(l)}))$, deliberately omitting $H^{(l)}$ to avoid double counting since it is already an average of the four branches. Simple averaging outperforms learnable gating (see ablation in Appendix D).
    - **Design Motivation**: Imputation requires pixel-level reconstruction at $(t, i)$; deep abstraction often loses local details, while shallow layers retain "raw signals" that aid filling. This aligns with ResNet's finding that direct connections are beneficial.

### Loss & Training
Follows SAITS's two-part loss: observed reconstruction $\mathcal{L}_{ORT}$ + artificial mask imputation $\mathcal{L}_{MIT}$, equally weighted $\mathcal{L} = \mathcal{L}_{ORT} + \mathcal{L}_{MIT}$. $d_{pe} \in [6, 24], d_f \in [6, 32], d \in [32, 576], L \in [2, 3]$.

## Key Experimental Results

### Main Results (Benchmark Ranking)

| Model | Avg. Rank ↓ | Notes |
|-------|-------------|-------|
| **HELIX (Ours)** | **1.00** | 1st in all 21/21 |
| ImputeFormer | 3.29 | KDD'24 SOTA |
| SAITS | 3.76 | 88M params |
| StemGNN | 5.71 | GNN |
| Linear Interpolation | 6.67 | Surprisingly ranked 5th |
| PatchTST | 7.24 | — |

ETT-h1 MAE for each missing pattern (mean of 5 runs ± std):

| Pattern | HELIX | ImputeFormer | SAITS | Linear Interpolation |
|---------|-------|--------------|-------|----------------------|
| Point-10% | **0.128 ± 0.005** | 0.202 ± 0.044 | 0.150 ± 0.007 | 0.197 |
| Point-50% | **0.189 ± 0.012** | 0.296 ± 0.036 | 0.208 ± 0.009 | 0.267 |
| Block-50% | **0.372 ± 0.015** | 0.404 ± 0.021 | 0.422 ± 0.019 | 0.527 |
| Subseq-50% | **0.489 ± 0.014** | 0.520 ± 0.017 | 0.620 ± 0.016 | 0.722 |

Parameter count is 803K, 100x smaller than SAITS (88M). Wilcoxon significance $p < 0.001$.

### Ablation Study (BeijingAir)

| Config | Point-50% | Block-50% | Subseq-50% |
|--------|-----------|-----------|------------|
| Full HELIX | **0.102 ± 0.005** | **0.131 ± 0.005** | **0.166 ± 0.009** |
| w/o Fusion | 0.104 | 0.147 | 0.173 |
| w/o Sinusoidal | 0.108 | 0.142 | 0.173 |
| w/o Hybrid | 0.104 | 0.137 | 0.294 (collapse) |
| w/o FeatEmb | 0.144 | 0.223 | 0.398 (major collapse) |

### Key Findings
- **FeatID is vital**: Removing it causes significant degradation in all missing patterns, especially Subseq-50% (MAE jumps to 0.398), proving the irreplaceable role of persistent identity anchors for long gaps.
- **Double helix excels under long gaps**: Removing Hybrid drops performance by only 2% on Point-50%, but by 77% on Subseq-50%, showing bidirectional crossing is crucial for "severely missing context."
- **Sublinear scaling of identity embedding dimension**: PeMS with 862 features only needs $d_f = 32$ (27:1 compression), but ETT-h1 with 7 features requires $d_f = 12$ (0.6:1 expansion)—with fewer features, FeatID compensates for "intrinsic structure."
- **Feature attention aligns with physical topology layer by layer**: On BeijingAir, the correlation between feature attention and the geographic proximity of 12 Beijing weather stations increases from 0.589 at Layer 0 to 0.712 at Layer 2, indicating fully unsupervised spatial structure discovery.
- **Structural utilization increases with correlation**: HELIX's improvement over ImputeFormer rises from 16.5% in low-correlation groups to 22.1% in high-correlation groups, proving FeatID truly "utilizes structure" rather than just fitting superficially.

## Highlights & Insights
- **"Persistent token identity" concept**: Adapts the NLP soft prompt idea to time series, giving each feature a never-missing ID card, enabling cross-feature attention to reason via identity even when all data is missing. This idea is generalizable to any "column-sparse" tabular or multimodal scenario.
- **Three independent lines of evidence**: Degradation in ablation, unsupervised spatial structure discovery, and progressive cross-layer attention alignment all independently support the necessity of FeatID, making the argument robust.
- **Small model outperforms large models**: 803K parameters outperforming SAITS (88M) and MOMENT (109M) demonstrates that "embedding design" is more important than "parameter stacking" in time series.
- **Physical metaphor of the double helix**: The parallel-then-cross structure immediately evokes DNA replication, tightly linking architecture and motivation, which aids in paper dissemination.

## Limitations & Future Work
- Feature identity embeddings are learned per dataset, making cross-dataset transfer difficult—for example, FeatID learned on BeijingAir is meaningless for PeMS; further work is needed for foundation models.
- For feature count $F > 10^3$, cross-feature attention $O(TF^2)$ remains a bottleneck; the authors acknowledge this scalability issue.
- Visualization of initial alignment under heavy missingness is only done on BeijingAir; generalization to other spatiotemporal data needs more empirical evidence.
- No direct comparison with diffusion imputation (CSDI); the authors provide a single-point comparison on BeijingAir (HELIX 0.073 vs CSDI 0.102, a 28.4% improvement), but not a systematic one.

## Related Work & Insights
- **vs ImputeFormer** (KDD 2024): ImputeFormer learns static feature embeddings but does not interact with mask state; HELIX incorporates the mask into the embedding, linking identity information with missingness.
- **vs SPIN** (NeurIPS 2022): SPIN uses predefined graphs, while HELIX learns soft adjacency end-to-end, requiring no spatial prior.
- **vs SAITS** (ESWA 2023): SAITS was the attention-based imputation SOTA; HELIX outperforms it in all 21 settings with 100x fewer parameters.
- **vs Crossformer** (ICLR 2023): Both use two-stage time-feature attention, but Crossformer’s tokens are derived from value patches, while HELIX’s explicit FeatID is the key difference.

## Rating
- Novelty: ⭐⭐⭐⭐ "Persistent feature identity embedding" is a clear new component; double helix encoding is a novel combination, though not individually innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets × 5 missing patterns = 21 settings, all ranked first; 16 baselines; mean and variance over 5 seeds fully reported; visualization is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative, covering ablation, interpretability, and cross-domain visualization; DNA analogy enhances readability.
- Value: ⭐⭐⭐⭐⭐ Integrated into the PyPOTS open-source toolkit and ready to use; FeatID concept is broadly applicable to wide multivariate time series tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](../../ICLR2026/time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](../../NeurIPS2025/time_series/diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](../../NeurIPS2025/time_series/statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)
- [\[AAAI 2026\] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](../../AAAI2026/time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)
- [\[ICML 2025\] Risk and Cross Validation in Ridge Regression with Correlated Samples](../../ICML2025/time_series/risk_and_cross_validation_in_ridge_regression_with_correlated_samples.md)

</div>

<!-- RELATED:END -->
