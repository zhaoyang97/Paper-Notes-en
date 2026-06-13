---
title: >-
  [Paper Note] HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation
description: >-
  [ICML 2026][Time Series][Feature Identity Embedding] Learn a "Feature Identity Embedding" for each feature as a persistent semantic anchor…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Feature Identity Embedding"
  - "Time Series Imputation"
  - "Spatio-temporal Transformer"
  - "Double-helix Encoding"
date: 2026-05-08
content_hash: a99d4d1e6d089dce
---

# HELIX: Hybrid Encoding with Learnable Identity and Cross-dimensional Synthesis for Time Series Imputation

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.02278](https://arxiv.org/abs/2605.02278)  
**Code**: https://github.com/milaogou/HELIX (integrated into PyPOTS)  
**Area**: Time Series / Time Series Imputation / Transformer  
**Keywords**: Feature Identity Embedding, Time Series Imputation, Spatio-temporal Transformer, Double-helix Encoding

## TL;DR
Learn a "Feature Identity Embedding" for each feature as a persistent semantic anchor, combined with time-feature double-helix attention. It achieves the top rank across all 21 missing scenarios on 5 public multivariate time series datasets, with more than 25% MAE reduction on datasets like ETT-h1 compared to the sub-optimal ImputeFormer.

## Background & Motivation

**Background**: Multivariate time series imputation is a critical preprocessing step for downstream tasks in healthcare, meteorology, and transportation. Mainstream methods are divided into three categories: RNN-based (BRITS, GRU-D), Transformer-based (SAITS, ImputeFormer), and Diffusion-based (CSDI, PriSTI). Recently, GNN methods (GRIN, SPIN) also attempt to explicitly model dependencies between features.

**Limitations of Prior Work**: (1) Existing attention methods "re-discover" relationships between features at every layer, lacking consistent anchors across layers—leading to the collapse of feature relationships under heavy missingness; (2) GNN methods rely on predefined graph topologies, assuming feature homogeneity (e.g., uniform spatial sensors), which cannot handle scenarios with mixed feature types; (3) Learning adjacency matrices incurs $O(F^2)$ cost and remains susceptible to data missingness; (4) Cross-feature attention methods like Crossformer rely solely on numerical patch embeddings, causing attention to degrade when values are entirely missing.

**Key Challenge**: To perform "cross-feature reasoning," a model needs each token to possess dual identities: time and feature. However, existing schemes only have persistent anchors on one axis (either time PE or graph topology), while the other axis must be dynamically inferred from values—this inference fails when values are missing.

**Goal**: (1) Provide each feature with a stable semantic identity across layers, (2) design an encoding structure with sufficient bidirectional interaction between time and features, and (3) maintain stable cross-feature reasoning even under heavy missingness.

**Key Insight**: The authors treat token embedding as a soft prompt in NLP—each feature learns a $d_f$-dimensional vector $f_i$ as a "feature-specific prompt." Regardless of whether values at that position are missing, feature identity information persists. Subsequently, a "parallel-then-cross" double-helix attention is designed to alternately process temporal and feature dimensions.

**Core Idea**: The embedding for each $(t, i)$ position is formulated as $e_{t,i} = [\tilde x_{t,i}; \text{PE}(t); f_i; m_{t,i}]$, where $f_i$ is a learnable identity embedding. $L$ layers of "double-helix" encoding (where each layer first performs time and feature attention in parallel, then crosses feature and time attention) allow information to flow sufficiently between the two dimensions.

## Method

### Overall Architecture
Input $\tilde X \in \mathbb{R}^{T \times F}$ and missing mask $M$. Each position constructs $e_{t,i} \in \mathbb{R}^{d_e}$ (value + sinusoidal PE + identity + mask), which is projected to a hidden dimension $d$ as $H^{(0)}$ via a linear layer. This is followed by $L$ Hybrid Encoding Layers, each outputting four branches $H_T^{(l)}, H_F^{(l)}, H_{TF}^{(l)}, H_{FT}^{(l)}$ which are averaged to obtain $H^{(l)}$. Finally, multi-level fusion is applied: $\tilde H = \frac{1}{1+4L}(H^{(0)} + \sum_l \text{sum of branches})$, followed by LayerNorm and a linear layer to produce $\hat X$.

### Key Designs

1. **Feature Identity Embedding (FeatID) as Soft-Adjacency Prior for Cross-Feature Attention**:

    - **Function**: Assigns a semantic vector to each feature that is stable across layers and time, enabling identity-based attention even if numerical values are missing.
    - **Mechanism**: Concatenates $e_{t,i} = [\tilde x_{t,i}; \text{PE}(t); f_i; m_{t,i}]$, where $f_i \in \mathbb{R}^{d_f}$ is the $i$-th row of a learnable identity matrix. The attention score $s_{ij}^{(t)} = e_{t,i}^\top A e_{t,j}$ can be decomposed into an identity prior $f_i^\top A_{ff} f_j$, identity-context cross terms, and dynamic context $r_{t,i}^\top A_{rr} r_{t,j}$. When both $x_{t,i}$ and $x_{t,j}$ are missing, dynamic terms degrade, but the identity prior remains intact to provide cross-feature compatibility.
    - **Design Motivation**: In ImputeFormer, "static feature embeddings" serve only as soft spatial indices and do not interact with missing status; SPIN’s graph embeddings are tied to predefined topologies. FeatID requires no graph prior and maintains anchoring under extreme missingness. Ablations on BeijingAir show that removing FeatID causes Subseq-50% MAE to surge from 0.166 to 0.398, identifying it as the most critical component.

2. **Double-Helix Hybrid Encoding Layer (Parallel-then-Cross)**:

    - **Function**: Allows temporal and feature dimensions to be refined independently while facilitating cross-dimensional information exchange.
    - **Mechanism**: Each layer consists of two stages. Stage 1 executes $H_T = \text{TimeMHA}(H^{(l-1)})$ and $H_F = \text{FeatMHA}(H^{(l-1)})$ in parallel for independent optimization. Stage 2 executes serial crossing: $H_{TF} = \text{FeatMHA}(H_T)$ and $H_{FT} = \text{TimeMHA}(H_F)$. The final output is the average of the four branches: $H^{(l)} = \frac{1}{4}(H_T + H_F + H_{TF} + H_{FT})$. It is named for its resemblance to a DNA double helix.
    - **Design Motivation**: Ablating the structure to pure serial Time→Feature→Time ("w/o Hybrid") caused MAE on Subseq-50% to jump from 0.166 to 0.294, indicating that bidirectional flow through parallel and cross structures is vital for handling long gaps. Serial encoding compresses one dimension's information before propagation, creating an information bottleneck.

3. **Multi-level Fusion**:

    - **Function**: Aggregates multi-branch outputs from every layer (including $H^{(0)}$) via weighted averaging to prevent loss of shallow details typically caused by using only the final layer.
    - **Mechanism**: $\tilde H = \frac{1}{1+4L}(H^{(0)} + \sum_{l=1}^L (H_T^{(l)} + H_F^{(l)} + H_{TF}^{(l)} + H_{FT}^{(l)}))$. The intermediate $H^{(l)}$ is intentionally omitted as it is already the average of its four branches. Simple averaging outperformed learnable gating (see Appendix D).
    - **Design Motivation**: Imputation requires pixel-level $(t, i)$ reconstruction. Deep abstractions often lose local details, whereas "original signals" preserved in shallow layers assist in filling. This aligns with findings that direct connections in ResNet perform better.

### Loss & Training
Ours follows the dual loss of SAITS: Observed Reconstruction $\mathcal{L}_{ORT}$ and Masked Imputation $\mathcal{L}_{MIT}$, with equal weighting: $\mathcal{L} = \mathcal{L}_{ORT} + \mathcal{L}_{MIT}$. Hyperparameters: $d_{pe} \in [6, 24], d_f \in [6, 32], d \in [32, 576], L \in [2, 3]$.

## Key Experimental Results

### Main Results

| Model | Avg Rank ↓ | Remarks |
|------|-------------|------|
| **HELIX (Ours)** | **1.00** | 1st place in all 21/21 scenarios |
| ImputeFormer | 3.29 | KDD'24 SOTA |
| SAITS | 3.76 | 88M parameters |
| StemGNN | 5.71 | GNN |
| Linear Interpolation | 6.67 | Naive baseline ranks 5th |
| PatchTST | 7.24 | — |

MAE on ETT-h1 across missing patterns (mean of 5 runs ± std):

| Pattern | HELIX | ImputeFormer | SAITS | Linear Interp. |
|------|-------|--------------|-------|--------------|
| Point-10% | **0.128 ± 0.005** | 0.202 ± 0.044 | 0.150 ± 0.007 | 0.197 |
| Point-50% | **0.189 ± 0.012** | 0.296 ± 0.036 | 0.208 ± 0.009 | 0.267 |
| Block-50% | **0.372 ± 0.015** | 0.404 ± 0.021 | 0.422 ± 0.019 | 0.527 |
| Subseq-50% | **0.489 ± 0.014** | 0.520 ± 0.017 | 0.620 ± 0.016 | 0.722 |

Parameters: 803K, 100x smaller than SAITS (88M). Wilcoxon significance $p < 0.001$.

### Ablation Study (BeijingAir)

| Configuration | Point-50% | Block-50% | Subseq-50% |
|------|-----------|-----------|------------|
| Full HELIX | **0.102 ± 0.005** | **0.131 ± 0.005** | **0.166 ± 0.009** |
| w/o Fusion | 0.104 | 0.147 | 0.173 |
| w/o Sinusoidal | 0.108 | 0.142 | 0.173 |
| w/o Hybrid | 0.104 | 0.137 | 0.294 (collapse) |
| w/o FeatEmb | 0.144 | 0.223 | 0.398 (extreme collapse) |

### Key Findings
- **FeatID is the Lifeline**: Removing it leads to significant degradation across all patterns, especially on Subseq-50% (collapsing to 0.398), proving that persistent identity anchors are irreplaceable for long gaps.
- **Double-Helix excels in long gaps**: Removing Hybrid on Point-50% only dropped performance by 2%, but on Subseq-50%, performance dropped by 77%; indicating bidirectional crossing is critical for "missing context" scenarios.
- **Sub-linear Scaling of Identity Dimensions**: PeMS with 862 features only requires $d_f = 32$ (27:1 compression), while ETT-h1 with 7 features needs $d_f = 12$ (0.6:1 expansion). Fewer features require more FeatID capacity to define "intrinsic structure."
- **Feature Attention aligns with Physical Topology**: On BeijingAir, the correlation between feature attention and geographic proximity of 12 stations increased from 0.589 at Layer 0 to 0.712 at Layer 2, demonstrating unsupervised discovery of spatial structure.
- **Structural Utilization increases with Correlation**: HELIX's gain over ImputeFormer rose from 16.5% for low-correlation groups to 22.1% for high-correlation groups, proving FeatID genuinely "utilizes structure" rather than just surface fitting.

## Highlights & Insights
- **"Persistent Token Identity" Philosophy**: Porting NLP soft prompts to time series, granting each feature a permanent "ID card." This allows cross-feature attention to perform compatibility reasoning based on identity even when data is entirely missing. This concept is generalizable to any "column-sparse" tabular or multi-modal scenario.
- **Triple Independent Evidence**: The necessity of FeatID is supported by three independent lines of evidence: extreme ablation degradation, unsupervised spatial discovery, and progressive layer-wise attention alignment.
- **Small Model Beats Large Models**: 803K parameters outperformed the 88M SAITS and 109M MOMENT, proving that "embedding design" is more important than "parameter stacking" in time series tasks.
- **Physical Metaphor of the Double Helix**: The parallel-then-cross structure evokes DNA replication, linking architecture to motivation effectively for paper dissemination.

## Limitations & Future Work
- Feature Identity Embeddings are learned per-dataset, making cross-dataset transfer difficult; e.g., transferring FeatID from BeijingAir to PeMS is meaningless. Foundation models would require alternative approaches.
- For features $F > 10^3$, cross-feature attention cost $O(TF^2)$ remains a bottleneck; the authors acknowledge this scalability issue.
- Visualization of alignment under heavy missingness was only performed on BeijingAir; more empirical evidence is needed for other spatio-temporal datasets.
- No direct systematic comparison was made with diffusion-based imputation (CSDI), though a single-point comparison on BeijingAir was provided (HELIX 0.073 vs CSDI 0.102, a 28.4% improvement).

## Related Work & Insights
- **vs ImputeFormer** (KDD 2024): ImputeFormer learns static embeddings without mask interaction. HELIX concatenates the mask into the embedding, linking identity information with missing status.
- **vs SPIN** (NeurIPS 2022): SPIN relies on predefined graphs. HELIX learns soft adjacency end-to-end without spatial priors.
- **vs SAITS** (ESWA 2023): SAITS was the previous attention-based SOTA. HELIX outperforms it in all 21 settings with 100x fewer parameters.
- **vs Crossformer** (ICLR 2023): Both use two-stage time-feature attention, but Crossformer tokens are derived from patch values. The addition of explicit FeatID is the key differentiator for HELIX.

## Rating
- Novelty: ⭐⭐⭐⭐ "Persistent Feature Identity Embedding" is a clear novel component, while double-helix encoding is a new but perhaps incremental combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 datasets × 5 patterns = 21 settings all at 1st place, 16 baselines, results reported with mean/variance over 5 seeds, excellent visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative covering ablation, interpretability, and cross-domain visualization; DNA analogy improves readability.
- Value: ⭐⭐⭐⭐⭐ Integrated into the PyPOTS open-source toolkit for immediate use. The FeatID concept has broad applicability for multi-variable time series tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](../../ICLR2026/time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[NeurIPS 2025\] Statistical Guarantees for High-Dimensional Stochastic Gradient Descent](../../NeurIPS2025/time_series/statistical_guarantees_for_high-dimensional_stochastic_gradient_descent.md)
- [\[AAAI 2026\] HydroDCM: Hydrological Domain-Conditioned Modulation for Cross-Reservoir Inflow Prediction](../../AAAI2026/time_series/hydrodcm_hydrological_domain-conditioned_modulation_for_cross-reservoir_inflow_p.md)
- [\[ICML 2026\] TimeOmni-VL: Unified Models for Time Series Understanding and Generation](timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)
- [\[CVPR 2026\] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](../../CVPR2026/time_series/sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)

</div>

<!-- RELATED:END -->
