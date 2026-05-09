---
title: >-
  [Paper Note] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs
description: >-
  [NeurIPS 2025][Medical Imaging][State Space Models] DyG-Mamba introduces continuous state space models (SSMs) into dynamic graph learning. It proposes a temporal span-aware continuous SSM that models irregular time intervals via an exponential decay function inspired by the Ebbinghaus forgetting curve, combined with input-dependent parameters constrained by spectral norm for Lipschitz robustness. The method achieves an average rank of 2.42 across 12 dynamic graph benchmarks (vs. DyGFormer's 2.92) while maintaining $O(bdL)$ linear complexity.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - State Space Models
  - Dynamic Graphs
  - Temporal Span Awareness
  - Mamba
  - Link Prediction
date: 2026-05-08
content_hash: 6875858122d6cc9d
---

# DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs

**Conference**: NeurIPS 2025
**arXiv**: [2408.06966](https://arxiv.org/abs/2408.06966)
**Code**: [https://github.com/Clearloveyuan/DyG-Mamba](https://github.com/Clearloveyuan/DyG-Mamba)
**Area**: Medical Imaging
**Keywords**: State Space Models, Dynamic Graphs, Temporal Span Awareness, Mamba, Link Prediction

## TL;DR
DyG-Mamba introduces continuous state space models (SSMs) into dynamic graph learning. It proposes a temporal span-aware continuous SSM that models irregular time intervals via an exponential decay function inspired by the Ebbinghaus forgetting curve, combined with input-dependent parameters constrained by spectral norm for Lipschitz robustness. The method achieves an average rank of 2.42 across 12 dynamic graph benchmarks (vs. DyGFormer's 2.92) while maintaining $O(bdL)$ linear complexity.

## Background & Motivation

**Background**: Dynamic graph methods are dominated by RNN-based approaches (GRU/LSTM, capable of capturing long-term dependencies but prone to gradient issues) and Transformer-based approaches (DyGFormer, with $O(N^2)$ attention and a poor efficiency–accuracy trade-off). Mamba/SSM has demonstrated advantages in sequence modeling, offering linear complexity with long-range dependency capture.

**Limitations of Prior Work**: (a) Real-world dynamic graph interaction intervals are highly irregular — fixed-step discretization loses temporal information; (b) graph data is heavily noisy, requiring robust parameterization; (c) MLP/GNN methods are fast but discard long-term temporal patterns.

**Key Challenge**: Standard Mamba assumes fixed step sizes, whereas interaction time intervals in dynamic graphs span from seconds to days. The SSM time step must adapt to actual temporal intervals.

**Goal**: Design a temporal span-aware continuous SSM that handles dynamic graphs with irregular time intervals under linear complexity.

**Key Insight**: The Ebbinghaus memory forgetting curve $R = \exp(-t/S)$ describes exponential decay of memory over time — it serves as the temporal step adaptation function for the SSM.

**Core Idea**: Forgetting-curve-inspired time step $\Delta t$ + diagonal negative-eigenvalue matrix $A$ for stability + spectral-norm-constrained input-dependent $B, C$ for robustness = temporal-aware continuous SSM for dynamic graphs.

## Method

### Overall Architecture
Dynamic graph interaction sequence → node/edge/time/co-occurrence frequency encoding → temporal span-aware parameterization ($\Delta t_k, A, B, C$) → continuous SSM state transition + parallel scan $O(bdL)$ → link prediction / node classification.

A key property is that all parameters carry theoretical guarantees — negative eigenvalues of $A$ ensure stability (Theorem 4.1), and spectral norm constraints on $B, C$ ensure robustness (Theorem 4.3).

### Key Designs

1. **Temporal Span-Aware $\Delta t$**:

    - **Function**: Adaptively adjusts the SSM step size based on actual interaction time intervals.
    - **Mechanism**: $\Delta t_k = w_1 \odot (1 - \exp(-w_2 \odot \frac{t_{k+1} - t_k}{\tau - t_1}))$, where $w_1, w_2$ are learnable. Larger intervals yield larger $\Delta t$ (more "forgetting"); smaller intervals yield smaller $\Delta t$ (more retention).
    - **Design Motivation**: The Ebbinghaus forgetting curve establishes an exponential relationship between memory retention and time interval — naturally encoding the prior that recent interactions are more informative than distant ones.

2. **Diagonal Negative-Eigenvalue Matrix $A$**:

    - **Function**: Guarantees SSM stability (states do not diverge).
    - **Mechanism**: $A = \text{diag}(\lambda_1, ..., \lambda_d)$ with $\lambda_i < 0$. The discretized form yields $\bar{A}_k = \text{diag}(e^{\lambda_i \Delta t_k})$, all eigenvalues of which have absolute value less than 1 (Theorem 4.1).
    - **Design Motivation**: Theorem 4.1 proves that diagonal negative eigenvalues are a necessary and sufficient condition for stability — avoiding gradient explosion/vanishing in RNNs.

3. **Spectral-Norm-Constrained Input-Dependent $B, C$**:

    - **Function**: Dynamically adjusts SSM parameters based on input content while ensuring Lipschitz continuity.
    - **Mechanism**: $B = W_B x + b_B$, $C = W_C x + b_C$, with constraints $\|W_B\|_2 \leq 1$, $\|W_C\|_2 \leq 1$ (spectral norm). Theorem 4.3 proves that the sensitivity of the output to input perturbations is bounded.
    - **Design Motivation**: Dynamic graph data is heavily noisy — input dependence improves expressiveness while spectral norm constraints ensure robustness.

### Loss & Training
- Link prediction: contrastive loss (positive samples vs. randomly sampled negatives).
- Parallel scan optimization: $O(bdL)$ vs. Transformer $O(bdL^2)$.
- Encoding: node + edge + absolute time + co-occurrence frequency + encoding alignment.

## Key Experimental Results

### Main Results (Link Prediction AP on 12 Datasets)

| Dataset | DyG-Mamba | DyGFormer | CAWN | Gain |
|--------|-----------|-----------|------|------|
| Wikipedia | 99.06±0.01 | 99.03±0.02 | 98.50 | +0.03 |
| Reddit | 99.25±0.00 | 99.22±0.01 | 98.82 | +0.03 |
| MOOC | **90.17±0.19** | 87.52±0.49 | 85.67 | **+2.65** |
| LastFM | **94.22±0.04** | 93.00±0.12 | 88.21 | **+1.22** |
| **Average Rank** | **2.42** | 2.92 | 4.33 | — |

### Ablation Study

| Configuration | Effect |
|------|------|
| Temporal span function vs. fixed step | Temporal span function consistently superior (MOOC +3%+) |
| Spectral norm constraint vs. unconstrained | Constraint improves robustness (Theorem 4.3) |
| Input-dependent $B, C$ vs. fixed $B, C$ | Input-dependent variant performs better (noise filtering) |
| Negative eigenvalue init. vs. random init. | Negative eigenvalues ensure stable convergence |

### Key Findings
- Improvements are most pronounced on datasets with high interaction interval variance such as MOOC and LastFM (+2.65/+1.22 AP), validating the value of temporal span awareness.
- Gains are marginal on dense interaction datasets such as Wikipedia and Reddit (+0.03), where time intervals are relatively uniform.
- Linear complexity enables scaling to longer historical sequences (vs. DyGFormer, which is constrained by its attention window).
- Ranking advantages are maintained under the inductive setting (unseen nodes).

## Highlights & Insights
- The analogy **forgetting curve → SSM step size** is both intuitive and theoretically grounded: time intervals directly reflect the degree of information decay.
- **Dual theoretical guarantees of stability and robustness** (Theorem 4.1 + 4.3) make the method reliable on noisy graph data.
- **Linear complexity** enables processing of extremely long interaction histories — critical for dynamic graphs that accumulate continuously, such as social networks.

## Limitations & Future Work
- Improvements are marginal on dense interaction datasets (Wikipedia +0.03) — when interactions are frequent, temporal interval differences are less pronounced.
- The method assumes interactions are ordered chronologically — out-of-order or concurrent interactions require additional handling.
- Multi-modal dynamic graphs (e.g., social networks with text/images) are not evaluated — extension to multi-modal settings may require modifications to the encoding scheme.
- The temporal span function's parameterization (exponential decay) may not generalize to all distributions, such as periodic interaction patterns.
- Evaluation is limited to link prediction — node classification, graph classification, and other tasks are not validated.
- The diagonal structure of $A$ limits expressiveness — non-diagonal formulations may be more powerful at the cost of parallelism.

## Related Work & Insights
- **vs. DyGFormer**: Transformer attention is $O(L^2)$; DyG-Mamba achieves $O(L)$ with comparable or superior accuracy.
- **vs. CAWN/TGN etc.**: Earlier methods do not model long-term dependencies; DyG-Mamba achieves this via SSM state memory.
- **vs. standard Mamba**: Standard Mamba assumes fixed step sizes; DyG-Mamba adapts these via the forgetting curve.
- **Transferability**: The temporal span-aware SSM can be extended to other irregular time-series tasks such as medical EHR and financial transactions.
- **Theoretical value**: Theorems 4.1 and 4.3 provide dual stability and robustness guarantees for applying SSMs to noisy graph data.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of continuous SSM with forgetting-curve-inspired step sizes to dynamic graphs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 datasets + inductive/transductive settings + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations with clearly stated theorem guarantees.
- Value: ⭐⭐⭐⭐ Provides a strong linear-complexity baseline for dynamic graph learning, particularly well-suited for irregular time interval scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bridging Graph and State-Space Modeling for Intensive Care Unit Length of Stay Prediction](bridging_graph_and_state-space_modeling_for_intensive_care_unit_length_of_stay_p.md)
- [\[NeurIPS 2025\] BarcodeMamba+: Advancing State-Space Models for Fungal Biodiversity Research](barcodemamba_advancing_state-space_models_for_fungal_biodiversity_research.md)
- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)

</div>

<!-- RELATED:END -->
