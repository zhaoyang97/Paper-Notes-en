---
title: >-
  [Paper Note] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper proposes PEPSY, a framework that learns client-side embedding controls to encode data-missing patterns, reconfiguring globally aggregated representations into complete-data features adapted to each client's local context, addressing both modality-missing and feature-missing scenarios in multimodal federated learning.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "Multimodal Learning"
  - "Missing Data"
  - "Reconfigurable Representations"
  - "Embedding Controls"
date: 2026-05-08
content_hash: 418eb9aa19625327
---

# Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data

**Conference**: NeurIPS 2025
**arXiv**: [2510.22880](https://arxiv.org/abs/2510.22880)  
**Code**: [GitHub](https://github.com/nmduonggg/PEPSY)  
**Area**: Optimization
**Keywords**: Federated Learning, Multimodal Learning, Missing Data, Reconfigurable Representations, Embedding Controls

## TL;DR

This paper proposes PEPSY, a framework that learns client-side embedding controls to encode data-missing patterns, reconfiguring globally aggregated representations into complete-data features adapted to each client's local context, addressing both modality-missing and feature-missing scenarios in multimodal federated learning.

## Background & Motivation

**Background**: In multimodal federated learning (MMFL), multiple clients observe different subsets of modalities and collaboratively train a shared model. Recent methods include FedMSplit, MIFL, FedInMM, and FedMAC.

**Limitations of Prior Work**: In practice, two types of missing data events arise: (1) clients possess only a subset of modalities (e.g., one device captures audio, another captures physiological signals); and (2) features within each modality are partially missing (e.g., due to sensor failure). Existing methods address only one type of missingness and cannot handle both simultaneously.

**Key Challenge**: When local models are optimized on different feature subsets, incompatible representation spaces emerge. Aggregating without alignment leads to information collapse or degeneration. The server cannot observe training data, and clients cannot fully interpret the globally aggregated representations.

**Goal**: Design a mechanism to capture and convey the missing-pattern characteristics of each client's local data, enabling the shared model to adapt to each client's specific missingness configuration.

**Key Insight**: Encode missing-pattern characteristics as a set of learnable embedding controls, serving as reconfiguration signals to align global representations with local context.

**Core Idea**: Learn a data-missing profile comprising multiple embedding controls to reconfigure biased representations into complete-data features—clients with similar missing patterns can share aggregated embedding controls.

## Method

### Overall Architecture

PEPSY operates through multi-round client-server communication. **Client side**: extracts modality-specific and data-specific features, queries the data-missing profile to select relevant embedding controls, and constructs complete-data representations. **Server side**: aggregates neural network parameters via FedAvg and synchronizes data-missing profiles via non-parametric clustering.

### Key Designs

1. **Data-Missing Representations**: Decomposes multimodal instance information into three components:

    - **Modality-specific features** $\mathbf{w}_{di}^{\text{mod}}$: learnable embeddings $W^{\text{mod}} = \{\mathbf{w}_i^{\text{mod}}\}_{i=1}^{|\mathcal{M}|}$, invariant across data samples, encoding modality identity.
    - **Data-specific features** $\mathbf{w}_{di}^{\text{ins}}$: maps each observed modality to a representation $\mathbf{h}_{di}$; missing modalities are substituted by the mean of available modality features: $\mathbf{w}_{di}^{\text{ins}} = \mathbf{I}(i \notin \mathcal{S}_d)\mathbf{h}_{di} + \mathbf{I}(i \in \mathcal{S}_d) \frac{1}{|\mathcal{M}|-|\mathcal{S}_d|}\sum_{j \notin \mathcal{S}_d}\mathbf{h}_{dj}$
    - **Data-specific contrastive loss** $\mathcal{L}_{ds}$: pulls features from different modalities of the same instance closer together while pushing those from different instances apart.

2. **Embedding Controls Selection**: A query-key matching mechanism enables interaction between data-missing features and embedding controls. Relevance is defined as:
   $\gamma(\mathbf{x}_{di}, \boldsymbol{\psi}_p) = e(\mathbf{q}(\mathbf{x}_{di}), \mathbf{k}(\boldsymbol{\psi}_p))$
   Only $\kappa$ most relevant embedding controls are selected per instance ($\kappa \ll |\Psi|$), with a regularization term $\mathcal{R}$ encouraging sparse selection. The final missing-pattern representation $\mathbf{w}_{di}^{\text{mis}}$ is the mean of selected embeddings.

3. **Reconfiguration Regularization**: A contrastive loss $\mathcal{L}_{rc}$ ensures that the final representation $\mathbf{w}_{di} = [\mathbf{w}_{di}^{\text{mod}} \circ \mathbf{w}_{di}^{\text{ins}} \circ \mathbf{w}_{di}^{\text{mis}}]$, which concatenates missing-pattern information, faithfully reflects complete-modality information.

4. **Modality Fusion**: Cross-modal information is fused using pairwise similarities between high-level representations $\hat{\mathbf{w}}_{di}$ as attention weights. An adaptive gate $\boldsymbol{\alpha}_{di}$ combines cross-modal and original representations to yield the final representation $\mathbf{c}_{di}$.

5. **Server Aggregation**: Data-missing profiles cannot be directly merged due to differing client learning orders. A non-parametric clustering method, PFPT, dynamically groups similar embeddings and adaptively adjusts the number of clusters to reflect the system-wide missing complexity.

### Loss & Training

The overall training objective is:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda(\mathcal{L}_{ds} + \mathcal{L}_{rc}) - \eta\mathcal{R}$$

where $\mathcal{L}_{task}$ is the task-specific loss, $\mathcal{L}_{ds}$ and $\mathcal{L}_{rc}$ are the data-specific and reconfiguration contrastive losses respectively, and $\mathcal{R}$ is the embedding relevance regularization term.

## Key Experimental Results

### Main Results

**PTBXL Dataset (12 modalities, IID, $p_m=0.2$) Accuracy (%):**

| Method | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|--------|-----------|-----------|-----------|-----------|-----------|
| FedProx | 73.43 | 73.64 | 71.42 | 71.37 | 69.93 |
| FedMAC | 78.56 | 77.30 | 76.25 | 75.49 | 74.70 |
| FedMSplit | 54.84 | 53.63 | 52.12 | 52.50 | 55.84 |
| **PEPSY** | **78.81** | **77.43** | **76.75** | **76.13** | **75.41** |

**PTBXL Non-IID Setting ($p_m=0.2$):**

| Method | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|--------|-----------|-----------|-----------|-----------|-----------|
| FedProx | 54.01 | 51.15 | 50.06 | 54.89 | 44.17 |
| FedMAC | 58.26 | 58.55 | 54.98 | 50.94 | 48.38 |
| **PEPSY** | **71.45** | **69.70** | **66.92** | **68.26** | **66.75** |

**EDF Dataset (5 modalities, Non-IID, $p_m=0.8$):**

| Method | $p_s$=0.2 | $p_s$=0.4 | $p_s$=0.6 | $p_s$=0.8 | $p_s$=1.0 |
|--------|-----------|-----------|-----------|-----------|-----------|
| FedMAC | 46.01 | 45.73 | 45.66 | 46.22 | 34.21 |
| **PEPSY** | **48.95** | **51.52** | **50.97** | **50.96** | **46.07** |

### Ablation Study

PEPSY achieves the greatest performance gains under severe data incompleteness:

| Scenario | Max Gain |
|----------|----------|
| PTBXL Non-IID ($p_m=0.2, p_s=1.0$) | +18.37% (vs FedMAC) |
| PTBXL Non-IID ($p_m=0.8, p_s=0.6$) | +32.24% (vs FedMAC) |
| EDF Non-IID ($p_m=0.8, p_s=0.4$) | +5.79% (vs FedMAC) |

### Key Findings

- PEPSY demonstrates the largest advantage under Non-IID settings, achieving up to 36.45% performance improvement in high-missingness Non-IID scenarios ($p_m=0.8$).
- Under IID conditions, PEPSY performs comparably to FedMAC, indicating that data-missing profiles contribute less when data distributions are consistent.
- Existing methods (FedMSplit, FedInMM) address only a single missingness type and degrade significantly when both types co-occur.
- Theoretical analysis shows that prediction bias induced by missing modalities is directly controlled by $\mathcal{L}_{ds}$, validating the loss design.

## Highlights & Insights

- **Effective problem formulation**: Decomposing missing patterns in multimodal FL into three orthogonal components—modality-specific, data-specific, and missing-pattern—provides a principled and clean design.
- **Novel embedding control mechanism**: Encoding missing patterns as learnable embeddings selected via query-key matching offers an elegant solution.
- **Non-parametric clustering aggregation**: Elegantly resolves the alignment problem of local missing profiles across heterogeneous clients.
- Theorem 3.1 directly links training loss to prediction stability under missing modalities, yielding strong consistency between theory and experiment.

## Limitations & Future Work

- Non-parametric clustering (PFPT) in server aggregation introduces additional communication and computational overhead; scalability requires further validation.
- Experiments are conducted only on medical and sleep datasets; other multimodal domains (e.g., autonomous driving, environmental monitoring) remain untested.
- The impact of hyperparameters such as the number of embedding controls $\tau$ and selection count $\kappa$ on performance is not thoroughly analyzed.
- Substituting missing modalities with feature means in data-specific representations is relatively simple and may limit feature quality.

## Related Work & Insights

- FedMAC is the closest baseline, handling feature missingness but not modality missingness.
- FedMSplit addresses modality missingness but not feature missingness.
- PEPSY unifies both types of missingness through embedding controls, offering a more comprehensive solution.
- The data-missing profile concept is potentially generalizable to other FL scenarios requiring adaptation to heterogeneous clients.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core mechanism of embedding controls combined with reconfiguration is novel, and the unified treatment of two missingness types is a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic comparisons across two datasets and multiple missingness configurations, though validation on additional domains is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear, theoretical analysis is rigorous, and notation is consistent throughout.
- **Value**: ⭐⭐⭐⭐ Addresses an important and practical problem in MMFL; the 36.45% improvement demonstrates clear practical effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Multiplayer Federated Learning: Reaching Equilibrium with Less Communication](multiplayer_federated_learning_reaching_equilibrium_with_less_communication.md)
- [\[NeurIPS 2025\] Personalized Subgraph Federated Learning with Differentiable Auxiliary Projections](personalized_subgraph_federated_learning_with_differentiable_auxiliary_projectio.md)

</div>

<!-- RELATED:END -->
