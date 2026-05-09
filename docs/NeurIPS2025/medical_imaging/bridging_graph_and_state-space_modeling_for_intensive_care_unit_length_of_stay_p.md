---
title: >-
  [Paper Note] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction
description: >-
  [NeurIPS 2025 (GenAI for Health Workshop)][Medical Imaging][ICU length-of-stay prediction] This paper proposes S2G-Net, a dual-branch architecture that integrates Mamba state-space temporal encoding with a multi-view graph neural network (GraphGPS) for ICU length-of-stay (LOS) prediction, achieving comprehensive improvements over sequential, graph-based, and hybrid baselines on MIMIC-IV.
tags:
  - NeurIPS 2025 (GenAI for Health Workshop)
  - Medical Imaging
  - ICU length-of-stay prediction
  - graph neural networks
  - state-space models
  - Mamba
  - multi-view graphs
date: 2026-05-08
content_hash: d7791d9823a6758d
---

# Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction

**Conference**: NeurIPS 2025 (GenAI for Health Workshop)
**arXiv**: [2508.17554](https://arxiv.org/abs/2508.17554)
**Code**: [GitHub](https://github.com/ShuqiZi1/S2G-Net)
**Area**: Medical Informatics
**Keywords**: ICU length-of-stay prediction, graph neural networks, state-space models, Mamba, multi-view graphs

## TL;DR
This paper proposes S2G-Net, a dual-branch architecture that integrates Mamba state-space temporal encoding with a multi-view graph neural network (GraphGPS) for ICU length-of-stay (LOS) prediction, achieving comprehensive improvements over sequential, graph-based, and hybrid baselines on MIMIC-IV.

## Background & Motivation

**State of the Field**: ICU LOS prediction is critical for hospital resource allocation. Existing approaches fall into two categories: temporal models (LSTM, Transformer) that capture individual patient trajectories, and graph models (GCN, GAT) that capture inter-patient relationships.

**Limitations of Prior Work**: Pure temporal models neglect clinical similarity between patients; pure graph models rely on single-view static graphs and cannot model multimodal heterogeneous clinical features; Transformer-based graph backbones (e.g., GraphGPS) suffer from quadratic complexity that hinders scaling to large clinical datasets.

**Root Cause**: ICU data simultaneously exhibits long-range irregular temporal dependencies and multimodal inter-patient relationships, yet no unified modeling framework exists to address both.

**Paper Goals**: To simultaneously capture temporal dynamics and population-level relational structure within a single end-to-end framework while preserving computational efficiency and interpretability.

**Starting Point**: Mamba SSM processes long sequences in linear time and is complementary to multi-view GraphGPS — the former models patient state evolution along the time axis, while the latter models patient similarity across diagnostic, semantic, and administrative dimensions.

**Core Idea**: Dual-branch architecture + multi-view graph construction + replacing the Transformer global layer in GraphGPS with an SSM.

## Method

### Overall Architecture
S2G-Net consists of three branches: (1) a temporal encoder (Mamba SSM) processing 48-hour ICU time-series data $\mathbf{X}_i^{TS} \in \mathbb{R}^{T \times d}$; (2) a graph encoder (optimized GraphGPS) learning population-level relationships from multi-view patient similarity graphs; and (3) a static feature encoder handling demographic and admission metadata $\mathbf{x}_i^{Flat}$. The three feature streams are fused via weighted concatenation and fed into a regression head for LOS prediction.

### Key Designs

1. **Multi-View Patient Similarity Graph Construction**:

    - Function: Constructs patient similarity graphs from two views — ICD-9/10 diagnostic codes and BERT semantic embeddings.
    - Mechanism: The diagnostic-code view computes similarity via TF-IDF cosine similarity / FAISS approximate KNN / penalized co-occurrence, selecting top-$k$ neighbors to build $\mathcal{G}_{diag}$; the semantic view encodes diagnostic descriptions with DistilBERT and constructs $\mathcal{G}_{bert}$ using a Gaussian kernel distance $w_{ij}^{bert} = \exp(-\|\mathbf{b}_i - \mathbf{b}_j\|_2^2 / 2\sigma^2)$.
    - Design Motivation: Single-view graphs cannot capture semantic proximity between diagnostic codes (e.g., distinct codes with similar clinical meanings); multi-view fusion covers both structured and unstructured clinical relationships.

2. **Temporal Encoder (Mamba SSM)**:

    - Function: Extracts patient state representations from 48-hour multivariate time-series data.
    - Mechanism: A linear projection followed by RMSNorm and GELU yields $\mathbf{H}_i^{(0)}$, which is then processed by $L$ stacked Mamba blocks; mask-aware pooling $\mathbf{z}_i^{TS} = \text{MaskPool}(\mathbf{H}_i^{(L)}, \mathbf{m}_i)$ handles missing values.
    - Design Motivation: Mamba's input-dependent recurrence captures long-range dependencies in linear time, making it better suited than LSTM or Transformer for irregular, long-sequence ICU data.

3. **Graph Encoder (Local GENConv + Global Mamba)**:

    - Function: Combines local neighborhood aggregation with global context modeling over multi-view graphs.
    - Mechanism: Stacks $L_g$ GraphGPS blocks, each comprising GENConv (typed weighted edge attributes for local message passing) and a Mamba global encoder (modeling global dependencies over degree-sorted node sequences), followed by BN and residual connection: $\mathbf{x}_i^{(\ell+1)} = \mathbf{x}_i^{(\ell)} + \tilde{\mathbf{u}}_i^{(\ell)}$.
    - Design Motivation: Replacing the Transformer layer in GraphGPS with Mamba reduces global attention complexity from $O(N^2)$ to $O(N)$ while preserving global context modeling capacity.

4. **Weighted Fusion and Auxiliary Supervision**:

    - Three feature streams are concatenated with softmax-normalized weights $\boldsymbol{\lambda}$: $\mathbf{z}_i^{fused} = \text{Concat}(\lambda_{Graph}\mathbf{z}_i^{Graph}, \lambda_{TS}\mathbf{z}_i^{TS}, \lambda_{Flat}\mathbf{z}_i^{Flat})$.
    - Huber loss is applied in the log domain $\tilde{y}_i = \log(1+y_i)$, supplemented by an auxiliary temporal branch loss to facilitate gradient flow.
    - Sample reweighting $w(y_i) = 1 + \gamma \mathbb{I}(y_i > \tau)$ increases the contribution of extreme LOS values.

### Loss & Training
$$\mathcal{L}_i = (1-\alpha)\mathcal{L}_{Huber}(\tilde{\hat{y}}_i^{Main}, \tilde{y}_i) + \alpha \mathcal{L}_{Huber}(\tilde{\hat{y}}_i^{TS}, \tilde{y}_i)$$

Training uses AdamW with gradient clipping and early stopping based on validation $R^2$. Hyperparameters are searched with Optuna over 75 trials.

## Key Experimental Results

### Main Results
Dataset: MIMIC-IV v3.1, 65,347 adult patients, 216 features (174 temporal + 42 static).

| Model | R² ↑ | Kappa ↑ | MSE ↓ | MSLE ↓ | MAD ↓ | log-MAPE ↓ |
|------|------|---------|-------|--------|-------|------------|
| **S2G-Net** | **0.43±0.01** | **0.42±0.00** | **14.25±0.18** | **0.25±0.01** | **1.88±0.02** | **35.74±1.24** |
| GraphGPS | 0.40±0.01 | 0.40±0.00 | 15.08±0.24 | 0.27±0.02 | 1.93±0.05 | 37.27±1.26 |
| Mamba | 0.33±0.00 | 0.36±0.00 | 16.89±0.04 | 0.28±0.02 | 2.03±0.01 | 40.46±1.03 |
| BiLSTM | 0.31±0.01 | 0.35±0.01 | 17.23±0.16 | 0.28±0.00 | 2.01±0.01 | 42.09±2.38 |
| LSTM-GAT | 0.31±0.01 | 0.35±0.01 | 17.36±0.21 | 0.28±0.00 | 2.01±0.00 | 41.07±1.58 |
| XGBoost | 0.32±0.00 | 0.39±0.00 | 17.32±0.03 | 0.27±0.00 | 1.96±0.01 | 47.14±0.17 |

S2G-Net improves $R^2$ by approximately 7.5% over the best baseline (GraphGPS) and reduces log-MAPE by 4.1%, with statistically significant differences ($p < 0.05$).

### Ablation Study

| Configuration | R² ↑ | MSE ↓ | Note |
|------|------|-------|------|
| Baseline (48h) | 0.43 | 14.25 | Full model |
| Last 6h | 0.28 | 18.02 | 35% drop in R² from shortened window |
| Last 24h | 0.40 | 15.09 | Marginal gains plateau after 24h |
| No Physiology | 0.39 | 15.40 | Largest drop when physiological features removed |
| No Vitals | 0.40 | 15.16 | Vital signs second most important |
| No Ethnicity | 0.42 | 14.21 | Ethnicity contributes least |
| Static Only | — | 34.69 | Static features alone perform poorly |
| Drop 30% edges | 0.43 | 14.35 | Graph structure is relatively robust |
| Drop 70% edges | 0.38 | 15.63 | Significant degradation beyond 70% edge removal |

### Key Findings
- Performance improves substantially as the observation window extends from 6h to 24h ($R^2$: 0.28→0.40), with diminishing returns from 24h to 48h.
- Physiological indicators and vital signs contribute most to prediction, while ethnicity carries the least information (validated via SHAP).
- Graph structure remains robust after removing 30%–50% of edges but degrades sharply beyond 70%.
- The model contains fewer than 2.5M parameters, achieving the best computational efficiency among models with comparable performance.

## Highlights & Insights
- **Multi-view graph construction**: Combining diagnostic codes, BERT semantic embeddings, and MST/GDC augmentation covers three relational dimensions — code co-occurrence, semantic proximity, and long-range connectivity — a strategy transferable to other clinical prediction tasks.
- **SSM as global layer replacing Transformer**: Substituting Mamba for the Transformer global attention in GraphGPS reduces complexity from $O(N^2)$ to $O(N)$ with no performance degradation — suggesting that full pairwise attention across all nodes may be unnecessary.
- **Mask-aware pooling**: Designed for the high missingness characteristic of ICU time-series, this approach yields more accurate aggregation than naive mean pooling.

## Limitations & Future Work
- Validation is conducted on a single dataset (MIMIC-IV); cross-hospital and cross-regional generalization experiments are absent.
- Graph construction relies on static thresholds (top-$k$, bottom 30% pruning) without considering dynamic graph evolution.
- A final performance of $R^2 = 0.43$ may still be insufficient for direct clinical deployment.
- Online prediction scenarios with dynamically expanding time windows are not explored.
- Incorporating Mamba2 or multi-scale SSM architectures into the temporal encoder warrants future investigation.

## Related Work & Insights
- **vs. LSTM-GNN**: LSTM-GNN encodes temporal features with LSTM and then aggregates on a static graph as two decoupled stages. S2G-Net unifies sequential modeling across both branches through a shared Mamba paradigm.
- **vs. GraphGPS**: The original GraphGPS employs a Transformer as the global layer, incurring high computational cost; replacing it with Mamba in S2G-Net preserves global modeling while substantially reducing complexity.
- **vs. XGBoost**: XGBoost as a non-neural baseline ($R^2 = 0.32$) remains competitive, highlighting the continued relevance of feature engineering in clinical prediction.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multi-view graph construction and dual-branch SSM+GNN fusion is novel, though each individual component is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes 16 baseline comparisons, detailed ablations, and interpretability analysis, but is limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations and rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides an efficient and interpretable unified framework for ICU clinical prediction with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[NeurIPS 2025\] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)
- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)

<!-- RELATED:END -->
