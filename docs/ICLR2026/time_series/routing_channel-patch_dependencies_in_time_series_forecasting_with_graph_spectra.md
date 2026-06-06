---
title: >-
  [Paper Note] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition
description: >-
  [ICLR 2026][Time Series][Channel dependency] This paper proposes xCPD, a plug-and-play plugin that refines the modeling unit of multivariate time series from "channels" to "channel-patches." It constructs spectral embedd…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Channel dependency"
  - "graph spectral decomposition"
  - "frequency-aware"
  - "MoE routing"
  - "plug-and-play"
date: 2026-05-08
content_hash: 13ea7f9e60bb6379
---

# Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition

**Conference**: ICLR 2026
**arXiv**: [2603.13702](https://arxiv.org/abs/2603.13702)  
**Code**: [GitHub](https://github.com/Clearloveyuan/xCPD)  
**Area**: Time Series Forecasting
**Keywords**: Channel dependency, graph spectral decomposition, frequency-aware, MoE routing, plug-and-play

## TL;DR

This paper proposes xCPD, a plug-and-play plugin that refines the modeling unit of multivariate time series from "channels" to "channel-patches." It constructs spectral embeddings via a shared graph Fourier basis, groups nodes into low/mid/high frequency bands based on spectral energy responses, and applies dynamic MoE routing to adaptively select frequency-specific filter experts. xCPD can be seamlessly integrated into any existing CI/CD model to consistently improve both long- and short-term forecasting performance, and supports zero-shot transfer.

## Background & Motivation

**Background**: Multivariate time series forecasting (MTSF) is a core AI task with broad applications in traffic, finance, energy, and meteorology. Recent advances have proceeded along two main axes: model architecture (Linear/CNN/Transformer/MLP/GNN) and channel strategy (CI/CD/CP), with the latter emerging as a key performance bottleneck.

**Limitations of the Three Channel Paradigms**: (1) CI (Channel-Independent), e.g., DLinear/PatchTST, models each channel independently—robust but ignores inter-channel relations; (2) CD (Channel-Dependent), e.g., TSMixer/TimesNet, aggregates all channels and may introduce irrelevant information leading to over-smoothing; (3) CP (Channel-Partiality), e.g., DUET/CCM/TimeFilter, attempts to balance the two but has fundamental shortcomings.

**Coarse-Granularity Bottleneck in CP Methods**: Existing CP methods operate at the channel level, treating entire channels as relational units and failing to model local patch-level interactions. For instance, channel A may exhibit a smooth seasonal trend in segment $T_1$ and sharp anomalies in segment $T_2$, yet channel-level models produce only a single averaged weight and cannot differentiate the distinct interaction patterns across segments.

**Frequency Coupling Problem**: CD/CD models compute attention weights in the time domain, where low-frequency trends, mid-frequency fluctuations, and high-frequency noise are entangled in the same embedding. A high attention score between two channels may simultaneously reflect meaningful low-frequency seasonal dependencies and irrelevant high-frequency noise correlations—the model cannot disentangle them, leading to spurious correlations.

**Key Insight**: The modeling unit is refined from "channels" to "channel-patches" (with patches as graph nodes). Dependency modeling is conducted in the graph spectral domain (rather than the time domain), followed by frequency-energy-based grouping and MoE routing of frequency-specific filter experts, achieving frequency-decoupled fine-grained channel-patch dependency modeling.

**Practical Value**: xCPD is designed as a post-processing plugin that requires no retraining of the backbone model. With linear computational complexity, it can be directly embedded into existing forecasting pipelines, making it suitable for large-scale real-time scenarios.

## Method

### Overall Architecture

xCPD consists of three core modules: (A) Spectral Channel-Patch Embedding → (B) Channel-Patch Grouping → (C) Channel-Patch Routing with MoE. The input is the backbone model's prediction output $\hat{X}^{\text{model}} \in \mathbb{R}^{C \times T'}$; after processing through xCPD, the refined prediction $\hat{X}^{\text{predict}}$ is produced. The overall pipeline is: patching → linear embedding → channel-patch graph construction → shared graph Fourier transform → spectral energy-based grouping → MoE routing for frequency expert selection → adaptive graph learning → gated residual correction for output.

### Key Design 1: Spectral Channel-Patch Embedding

- **Function**: Divides the backbone output into patches, applies linear embedding, constructs a channel-patch graph, and projects node embeddings into the spectral domain via a shared graph Fourier basis.
- **Mechanism**: The prediction output $\hat{X}^{\text{model}}$ is divided into $N = \lceil T'/P \rceil$ non-overlapping patches and linearly mapped to $d$-dimensional embeddings $X^{\text{emb}} \in \mathbb{R}^{n \times d}$ (where $n = C \times N$). An adjacency matrix is constructed using cosine similarity: $A_{ij}^t = \cos(X_i^{\text{emb},t}, X_j^{\text{emb},t})$. The normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$ is computed, and eigendecomposition yields the shared Fourier basis $U$. The spectral embedding is $X^{\text{spc}} = U^\top X^{\text{emb}}$.
- **Design Motivation**: (1) Cosine similarity-based graph construction is scale-invariant with respect to variable magnitudes, suitable for multivariate settings. (2) Per-batch eigendecomposition produces inconsistent Fourier bases across batches, making cross-batch comparison infeasible. A **shared graph Fourier basis** (Theorem 4.1) is therefore learned from the averaged Laplacian, ensuring all time steps are mapped to a consistent spectral domain. The theoretical guarantee $\|U^t - UR^t\|_F \leq C\|L^t - L_{\text{avg}}\|_F$ establishes that the shared basis provides a linear approximation to each per-batch basis.

### Key Design 2: Spectral Channel-Patch Grouping

- **Function**: Assigns channel-patch nodes to three frequency groups—low, mid, and high—based on each node's spectral energy response, and constructs ego-graph subgraphs for frequency-aware message passing.
- **Mechanism**: Learnable boundaries $\tau_1, \tau_2$ define three frequency bands; sigmoid-based soft partitioning computes per-frequency membership weights $\alpha_j^{\text{low/mid/high}}$. The spectral energy response is defined as $S_{i,j} = \|U_{i,j} \cdot X_{j,:}^{\text{spc}}\|_2^2$ (Theorem 4.2 guarantees energy conservation: $\sum_j S_{i,j} = \|X_{i,:}^{\text{emb}}\|_2^2$). Nodes are then assigned to the frequency band with the highest energy via softmax. For each node, an ego-graph is constructed using $k$-NN neighbor selection, and intra-group subgraphs are built based on frequency labels.
- **Design Motivation**: (1) Learnable frequency boundaries adapt to the frequency structure of different datasets. (2) Spectral energy responses directly quantify each node's sensitivity to different frequency components, enabling precise grouping. (3) Ego-graphs reduce noise by retaining only dependencies relevant to the center node. (4) Frequency-based subgraphs ensure message passing occurs between nodes in the same frequency band, preventing mixing of trend nodes and noise nodes.

### Key Design 3: Dynamic MoE Routing (Spectral Channel-Patch Routing with DyMoE)

- **Function**: Dynamically selects a variable number of frequency-specific filter experts (low/mid/high-frequency filters) for each ego-graph, constructs a sparse adjacency matrix, and performs graph learning.
- **Mechanism**: Three frequency filters construct adjacency matrices from low/mid/high spectral components, respectively. The routing network computes $\psi(x_i) = \text{Linear}_c(x_i) + \epsilon \cdot \text{Softplus}(\text{Linear}_n(x_i))$ (comprising deterministic and stochastic noise components). Experts are selected by choosing the minimum number required such that their cumulative probability $\geq \tau$ (Eq. 7), differing from fixed Top-K selection. The selected experts' edge sets are merged per Eq. (8) to construct a sparse adjacency matrix, followed by $L$-layer graph learning (Eq. 9–10) for neighborhood aggregation. The final output is produced via **gated dual-path residual correction**: $\hat{X}^{\text{predict}} = \hat{X}^{\text{model}} + \sigma(g_{\text{GNN}}) \odot \delta_{\text{GNN}} + \sigma(g_{\text{Lin}}) \odot \delta_{\text{Lin}}$.
- **Design Motivation**: (1) Three frequency-specific experts separately capture smooth trends (low), local fluctuations (mid), and abrupt changes/anomalies (high), enabling frequency-decoupled modeling. (2) DyMoE dynamically allocates experts—different inputs receive different combinations—offering greater flexibility than fixed Top-K. (3) The gated residual design degrades gracefully to the original backbone prediction when gate values approach zero, ensuring safe integration. (4) Entropy loss $\mathcal{L}_{\text{Entropy}}$ and balance loss $\mathcal{L}_{\text{Balance}}$ are added to the training objective to prevent expert collapse.

### Key Design 4: Gated Dual-Path Residual Correction and Optimization

- **Function**: Combines a GNN path (cross-channel spectral dependencies) and a Linear path (CI-style refinement), with learnable gates determining each path's contribution.
- **Mechanism**: $\delta_{\text{GNN}} = W_{\text{proj}} H^{(L)}$ captures cross-variable spectral dependencies; $\delta_{\text{Lin}} = f_{\text{lin}}(\hat{X}^{\text{model}})$ preserves channel-independent refinement. Gate parameters $g_{\text{GNN}}, g_{\text{Lin}} \in \mathbb{R}^C$ operate per channel. Total loss: $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mu\mathcal{L}_{\text{Entropy}} + \beta\mathcal{L}_{\text{Balance}}$.
- **Design Motivation**: The dual-path design simultaneously leverages the strengths of CD (GNN path) and CI (Linear path), enabling adaptive balancing. Per-channel gating allows different variables to select different degrees of dependency modeling.

## Key Experimental Results

### Table 1: Long-Term Forecasting Main Results (9 datasets, 4 backbones, MSE↓)

| Setting | TSMixer → +xCPD | DLinear → +xCPD | PatchTST → +xCPD | TimesNet → +xCPD |
|------|:---:|:---:|:---:|:---:|
| ETTh1 avg | 0.412 → **0.401** | 0.456 → **0.445** | 0.469 → **0.455** | 0.458 → **0.447** |
| Weather avg | 0.234 → **0.221** | 0.265 → **0.253** | 0.259 → **0.248** | 0.259 → **0.249** |
| Electricity avg | 0.167 → **0.158** | 0.212 → **0.197** | 0.205 → **0.194** | 0.192 → **0.175** |
| Traffic avg | 0.408 → **0.394** | 0.625 → **0.606** | 0.482 → **0.467** | 0.620 → **0.558** |

xCPD achieves consistent improvements across nearly all 144 experimental configurations, with the most significant gains on high-dimensional datasets (Electricity: 321 variables; Traffic: 862 variables).

### Table 2: Comparison with LIFT and CCM Baselines (TSMixer/DLinear backbone)

| Dataset | TSMixer+LIFT | TSMixer+CCM | TSMixer+**xCPD** | DLinear+LIFT | DLinear+CCM | DLinear+**xCPD** |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ETTh2 | 0.351 | 0.351 | **0.345** | 0.553 | 0.552 | **0.507** |
| Weather | 0.231 | 0.225 | **0.221** | 0.262 | 0.262 | **0.253** |
| Traffic | 0.405 | 0.396 | **0.394** | 0.620 | 0.614 | **0.606** |

xCPD outperforms both LIFT and CCM on all 9 datasets.

### Table 3: Comparison with 5 CP Baselines under General Settings

| Dataset + Architecture | +PRReg | +LIFT | +PCD | +CCM | +**xCPD** |
|------------|:---:|:---:|:---:|:---:|:---:|
| ETTm1 Transformer | 0.349 | 0.356 | 0.404 | 0.300 | **0.289** |
| Exchange Linear | 0.048 | 0.050 | — | 0.045 | **0.042** |
| Weather Transformer | 0.180 | 0.178 | 0.198 | 0.164 | **0.161** |

xCPD achieves the best performance across all 10 configurations.

## Key Findings

1. **High-dimensional datasets benefit most**: The performance gain from xCPD increases with the number of channels (Electricity: 321; Traffic: 862), as spectral frequency decoupling more effectively suppresses irrelevant channel noise in high-dimensional settings.
2. **CI models benefit more from xCPD**: In zero-shot experiments, CI models (DLinear: 12.0%, PatchTST: 15.2%) exhibit significantly larger improvements than CD models (TSMixer: 6.7%, TimesNet: 11.1%), demonstrating that xCPD injects the cross-channel interaction capability that CI models lack.
3. **Larger gains at longer forecast horizons**: In zero-shot settings, performance improvements increase with forecast horizon length, suggesting that frequency knowledge transfer is more effective for long-range dependencies.
4. **Linear computational complexity**: Time complexity $\mathcal{O}(nkd + Lnkd)$ and space complexity $\mathcal{O}(nd + nk)$ introduce only 9%–11% training time overhead, far below CCM's quadratic complexity.
5. **Ablation study**: Removing any individual component—shared Fourier basis, frequency partitioning, node grouping, or filters—degrades performance. Replacing DyMoE with Top-K, Random-K, RegionTop-K, or TimeFilter also underperforms the full xCPD, validating the necessity of each component.

## Highlights & Insights

- **Dependency modeling in the graph spectral domain**: xCPD is the first method to model channel interactions entirely in the graph spectral domain (rather than the time domain). In the spectral domain, low/mid/high-frequency components are naturally decoupled, eliminating the spurious correlations caused by frequency coupling in time-domain attention—this is the core innovation distinguishing xCPD from all prior CP methods including LIFT, CCM, PCD, and TimeFilter.
- **Granularity improvement from channels to channel-patches**: Different temporal segments of the same channel may interact differently with other channels; patch-level modeling is the first to capture this segment-level heterogeneity.
- **Dynamic expert allocation via DyMoE**: Unlike fixed Top-K, DyMoE adaptively selects 1–3 experts based on a cumulative probability threshold, routing smooth segments to low-frequency experts and abrupt segments to high-frequency experts—enabling input-aware fine-grained modeling.
- **Visualization validates theory**: Figure 3 demonstrates the correspondence between spectral energy and time-domain patterns—nodes with high low-frequency energy indeed correspond to smooth trends, while nodes with high high-frequency energy correspond to rapid fluctuations—confirming the energy conservation guarantee of Theorem 4.2.
- **Plug-and-play and zero-shot transfer**: As a post-processing plugin, xCPD requires no backbone retraining, and the learned frequency filtering knowledge transfers across datasets (zero-shot improvements across 48 configurations).

## Limitations & Future Work

1. **Zero-shot transfer evaluated only on ETT series**: Cross-domain transfer (e.g., Weather → Traffic) is not validated; effectiveness between domains with substantially different frequency structures remains uncertain.
2. **Quadratic cost in graph construction**: Although overall complexity is linear, computing the cosine similarity adjacency matrix remains $O(n^2d)$, which may become a bottleneck when both the number of channels and the number of patches are large.
3. **Prior assumption of three frequency groups**: The fixed partition into low/mid/high frequency bands may be suboptimal for certain datasets that require finer or coarser divisions; a mechanism for adaptively determining the number of groups is absent.
4. **Validation limited to forecasting tasks**: Generalizability to other time series tasks (classification, anomaly detection, imputation) is not assessed.
5. **Approximation error of the shared graph Fourier basis**: The approximation bound in Theorem 4.1 depends on the spectral gap of $L_{\text{avg}}$; when data distribution shifts dramatically, the gap may be small, degrading approximation quality.

## Related Work & Insights

| Dimension | xCPD (Ours) | CCM (Chen et al., NeurIPS 2024) | TimeFilter (Hu et al., 2025) |
|------|------------|------|------|
| Modeling granularity | **Channel-patch level** | Channel level (channel clustering) | Patch level but in time domain |
| Modeling domain | **Graph spectral domain** | Time domain | Time domain |
| Adaptivity | **Frequency-specific MoE routing** | Similarity-based clustering | Spatiotemporal attention filtering |
| Frequency decoupling | ✓ Spectral energy grouping | ✗ Frequency coupling | ✗ Frequency coupling |
| Plugin compatibility | ✓ Post-processing, no retraining | ✓ But quadratic complexity | Requires integration into specific architecture |
| Zero-shot transfer | ✓ Frequency knowledge transferable | Not validated | Not validated |

## Rating

- **Novelty**: ⭐⭐⭐⭐ Spectral-domain channel-patch dependency modeling combined with DyMoE represents a new perspective, with simultaneous innovation along three axes (granularity / domain / adaptivity)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 9 datasets × 4 backbones × 144 configurations, plus short-term / zero-shot / efficiency / ablation / visualization analyses—extremely comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ Clear method description, rigorous theoretical derivations (two theorems), rich figures and tables
- **Value**: ⭐⭐⭐⭐ As a general plug-and-play plugin, xCPD offers direct practical value to the time series forecasting community; linear complexity makes it deployment-friendly

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] TimeSliver: Symbolic-Linear Decomposition for Explainable Time Series Classification](timesliver_symbolic-linear_decomposition_for_explainable_time_series_classificat.md)
- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)

</div>

<!-- RELATED:END -->
