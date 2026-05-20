---
title: >-
  [Paper Note] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data
description: >-
  [ICCV 2025][Optimization][Federated Learning] This paper proposes FED-PRIME, a federated prompt-tuning framework for multimodal settings with missing modalities. It maintains two sets of learnable prompts — inter-client…
tags:
  - "ICCV 2025"
  - "Optimization"
  - "Federated Learning"
  - "Prompt-Tuning"
  - "Multimodal"
  - "Missing Modality"
  - "Heterogeneous Data"
date: 2026-05-08
content_hash: b6f9ddc3bdfd13d8
---

# Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data

**Conference**: ICCV 2025
**arXiv**: [2602.07081](https://arxiv.org/abs/2602.07081)  
**Code**: [github.com/hangpt01/FedPrime](https://github.com/hangpt01/FedPrime)  
**Area**: Optimization
**Keywords**: Federated Learning, Prompt-Tuning, Multimodal, Missing Modality, Heterogeneous Data

## TL;DR

This paper proposes FED-PRIME, a federated prompt-tuning framework for multimodal settings with missing modalities. It maintains two sets of learnable prompts — inter-client and intra-client — to capture cross-client alignable missing patterns and client-specific missing patterns, respectively, and employs a clustering-alignment mechanism for server-side aggregation. FED-PRIME substantially outperforms existing baselines across diverse missing-data configurations.

## Background & Motivation

### State of the Field

Fine-tuning large pretrained models has become the dominant paradigm. Prompt-tuning, as a parameter-efficient fine-tuning approach, adapts models to downstream tasks by prepending learnable prompt tokens to the input. Federated learning (FL) enables collaborative model training across devices without sharing raw data.

### Limitations of Prior Work

**Federated prompt-tuning supports only unimodal settings**: Existing methods assume identical data modalities across clients and cannot handle multimodal scenarios.

**Multimodal federated learning does not leverage pretrained models**: Existing multimodal FL methods (FedMSplit, FedMAC, etc.) rely on customized architectures and cannot benefit from fine-tuning pretrained multimodal foundation models (e.g., CLIP, ViLT).

**Dual heterogeneity of missing modalities**:
   - **Intra-heterogeneity**: Different samples within a single client dataset exhibit different missing modality patterns.
   - **Inter-heterogeneity**: Different clients exhibit different distributions of missing modality patterns.

**Failure of naive aggregation**: Prompts from different clients may encode different missing patterns and cannot be directly averaged; simple FedAvg collapses prompts into low-information representations.

### Root Cause

In multimodal federated learning, clients' missing modality patterns differ, causing learned prompts to encode heterogeneous information patterns. Directly aggregating these prompts leads to information conflicts and performance degradation. A mechanism is needed to identify, align, and aggregate prompts that encode similar missing patterns across clients.

## Method

### Overall Architecture

FED-PRIME builds on a pretrained ViLT model. Each client maintains two sets of learnable prompt pools (inter-client and intra-client) and selects the most relevant prompt subsets via an input-adaptive retrieval mechanism. The server aggregates intra-client prompts via standard FedAvg, and inter-client prompts via a clustering-based alignment-aggregation mechanism.

### Key Designs

#### 1. **Dual Prompt Pool Design**

- **Function**: Decomposes fine-tuning knowledge into two prompt sets that respectively encode different types of missing-pattern information.
- **Mechanism**:

**Inter-client prompts** $\mathbf{w}_p^{inter} = \{\mathbf{p}_1^{inter}, \ldots, \mathbf{p}_\tau^{inter}\}$: Encode input-level missing data distribution patterns and can be aligned and aggregated across clients.

**Intra-client prompts** $\mathbf{w}_p^{intra} = \{\mathbf{p}_1^{intra}, \ldots, \mathbf{p}_\tau^{intra}\}$: Encode input-agnostic missing modality patterns (e.g., image-only missing vs. text-only missing) and can be aggregated directly via FedAvg.

- **Design Motivation**: The aggregation mechanism implicitly constrains how knowledge is encoded. If input-level pattern knowledge is incorrectly encoded into intra-prompts, it will be averaged away by FedAvg; if general knowledge is incorrectly encoded into inter-prompts, it wastes their representational capacity. This separation design enables automatic, correct knowledge allocation through implicit gradient signals.

#### 2. **Input-Adaptive Prompt Retrieval**

- **Function**: Selects the $\kappa$ most relevant prompts from each pool for each input sample as adaptation instructions.
- **Mechanism**: Learns a key function $k(\mathbf{p})$ and a query function $q(\mathbf{x}(M))$, measuring relevance via cosine distance $d(\mathbf{x}(M), \mathbf{p}) = \cos(q(\mathbf{x}(M)), k(\mathbf{p}))$. A regularization term is added to the local loss:

$$L'_t(\mathbf{w}) = \sum_{s=1}^m \ell(F(\mathbf{x}(M_{t,s}); \mathbf{w}'), z_{t,s}) + \sum_{s=1}^m r(\mathbf{x}(M_{t,s}), \mathbf{w}'_p)$$

where $r(\mathbf{x}(M), \mathbf{w}'_p) = \sum_{\mathbf{p} \in \mathbf{w}'_p} d(\mathbf{x}(M), \mathbf{p})$ penalizes the distance between selected prompts and the input.

- **Design Motivation**: Different samples have different missing patterns and thus require different prompt instructions. The regularization term prevents prompt overloading — each prompt specializes in samples whose patterns are in its "neighborhood," enabling knowledge distillation and separation.

#### 3. **Server-Side Clustering-Alignment Aggregation**

- **Function**: Identifies inter-client prompts encoding similar missing patterns across clients, clusters them, and merges them into more comprehensive prompts.
- **Mechanism**: The alignment problem is formalized as a constrained clustering optimization:

$$\min_{\boldsymbol{\alpha}, \boldsymbol{\theta}, \gamma} G(\boldsymbol{\alpha}, \boldsymbol{\theta}, \gamma) + R(\boldsymbol{\alpha}, \zeta)$$

where $\alpha_t^{p,q} \in \{0,1\}$ indicates whether the $p$-th prompt of client $t$ is matched to the $q$-th cluster, and $\boldsymbol{\theta}_q$ denotes the cluster center (i.e., the aggregated prompt). Constraints ensure prompts from the same client are not assigned to the same cluster. $R(\boldsymbol{\alpha}, \zeta)$ prioritizes updating more generalizable prompts via a learnable popularity function $U(\boldsymbol{\theta}_q; \zeta)$. The discrete optimization subproblem is solved using the Hungarian algorithm.

- **Design Motivation**: At the same position, inter-client prompts from different clients may encode entirely different missing patterns (since certain patterns may not exist at some clients). Naive position-based alignment causes incompatible prompts to be mixed. The clustering mechanism aligns prompts by semantic similarity rather than by position.

### Loss & Training

- Model: Frozen ViLT + learnable prompt pools + classification head.
- Client update: Minimize $L'_t(\mathbf{w})$ (local loss with regularization term).
- Server aggregation: Inter-prompts are aggregated via the clustering-alignment algorithm; intra-prompts are aggregated via FedAvg.
- Alternating optimization: (1) Fix $\boldsymbol{\alpha}$, optimize $(\boldsymbol{\theta}, \zeta, \gamma)$; (2) Fix $(\boldsymbol{\theta}, \zeta, \gamma)$, solve for $\boldsymbol{\alpha}$ via the Hungarian algorithm.

## Key Experimental Results

### Main Results

UPMC Food-101 dataset (classification accuracy %):

| Training Setting | Method | Test(~Train) | Test(Miss Both) | Test(Full Modal) | Test(Text only) | Test(Image only) |
|---------|------|-------------|----------------|-----------------|----------------|-----------------|
| Miss Text | FEDAVG-P | 15.71 | 14.90 | 21.56 | 16.91 | 15.36 |
| Miss Text | FED-INTER | 54.82 | 48.87 | 59.17 | 35.13 | 56.59 |
| Miss Text | **FED-PRIME** | **78.88** | **80.38** | **92.12** | **73.01** | **76.83** |
| Miss Image | FEDAVG-P | 17.35 | 15.12 | 16.84 | 18.12 | 14.81 |
| Miss Image | FED-INTER | 77.96 | 64.62 | 82.08 | 77.69 | 37.56 |
| Miss Image | **FED-PRIME** | **90.55** | **79.12** | **92.89** | **90.18** | **54.14** |
| Miss Both | FEDAVG-P | 14.57 | - | 17.17 | 16.40 | 13.24 |
| Miss Both | FED-INTER | 56.32 | - | 69.57 | 45.15 | 59.30 |
| Miss Both | **FED-PRIME** | **84.44** | - | **93.64** | **87.95** | **72.41** |

FED-PRIME's improvement over the second-best method ranges from 1.73% to 107.83% on Food-101 and from 4.41% to 69.65% on MM-IMDB.

### Ablation Study

| Method | Components | Food-101 Miss Text (Full Modal) | MM-IMDB Miss Text (Full Modal) |
|------|------|-------------------------------|-------------------------------|
| FEDAVG-P | No prompt separation | 21.56 | 30.78 |
| FED-INTRA | Intra-prompt only | 62.06 | 12.55 |
| FED-INTER | Inter-prompt only | 59.17 | 18.67 |
| **FED-PRIME** | **Both combined** | **92.12** | **37.67** |

Robustness (Miss Both, Food-101, varying missing rate η):

| Missing Rate η | FED-PRIME | FEDAVG-P | Centralized-P |
|---------|-----------|---------|--------------|
| 0.00 | ~93% | ~90% | ~93% |
| 0.25 | ~88% | ~60% | ~85% |
| 0.50 | ~85% | ~45% | ~80% |
| 0.75 | ~82% | ~30% | ~75% |
| 1.00 | ~80% | ~15% | ~70% |

### Key Findings

- **Both prompt types are indispensable**: Using FED-INTER or FED-INTRA alone falls far short of the full FED-PRIME, validating that inter- and intra-heterogeneity must be addressed separately.
- **Alignment mechanism is critical**: Prompt-tuning with FedAvg and no alignment degrades sharply at high missing rates (from ~90% to ~15%), while FED-PRIME remains above 80%.
- **FED-PRIME approaches the centralized upper bound**: At high missing rates, FED-PRIME even surpasses Centralized-P (both using prompt-tuning).
- **Faster and more stable convergence**: FED-PRIME's training/test loss converges significantly faster and more stably than FED-INTER and FED-INTRA.
- **Noteworthy Miss Text experiment**: After training with 70% text missing, the model still performs well on Text Only evaluation, suggesting that prompt alignment effectively recovers information about missing modalities.

## Highlights & Insights

1. **Systematic problem formulation**: The paper clearly distinguishes intra-heterogeneity from inter-heterogeneity and designs corresponding prompt sets and aggregation strategies for each.
2. **Implicit knowledge separation mechanism**: The aggregation mechanism inversely constrains how knowledge is encoded — an elegant design philosophy that enables the model to automatically learn how to allocate different types of knowledge to different prompts.
3. **Formal clustering-alignment framework**: The prompt alignment problem is cast as a constrained clustering optimization; the popularity function $U(\boldsymbol{\theta}_q; \zeta)$ further differentiates general from specialized prompts.
4. **Comprehensive experimental design**: 3 training missing scenarios × 5 test scenarios = 15 experimental configurations, providing broad coverage.

## Limitations & Future Work

1. **Validated only on bimodal (image + text) settings**: Scalability to three or more modalities remains unknown.
2. **Inherent text bias in ViLT**: Experiments show that Image Only test performance is consistently lower, likely stemming from ViLT's text-centric pretraining.
3. **Only 8 categories selected**: The most frequent 8 classes are sampled from the original datasets, potentially underestimating challenges at larger class scales.
4. **Scalability of the Hungarian algorithm**: At $n \times \tau$ clusters, the $O(n^3\tau^3)$ complexity may limit large-scale deployment.
5. **Missing patterns are randomly simulated**: Real-world missing modalities may exhibit more complex structure (e.g., correlated with geographic location).
6. **Not combined with stronger foundation models**: ViLT is no longer state-of-the-art among multimodal models; integration with CLIP and similar models is unexplored.

## Related Work & Insights

- Missing Prompt-Tuning (Lee et al.) learns dedicated prompts for each missing modality subset in a centralized setting; FED-PRIME extends this to the federated scenario.
- FedMSplit and FedMAC address multimodal federated learning but do not leverage pretrained foundation models.
- The clustering-alignment idea can be generalized to heterogeneity alignment problems in other federated learning settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First work to connect federated learning with multimodal prompt-tuning; dual prompt design and clustering-alignment are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 15 training-test scenario combinations provide comprehensive coverage, though only 2 datasets is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formalization is clear, but the dense notation and formulas leave room for improved readability.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in federated prompt-tuning for multimodal missing-data settings with broad practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data](../../NeurIPS2025/optimization/learning_reconfigurable_representations_for_multimodal_federated_learning_with_m.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](federated_continual_instruction_tuning.md)
- [\[NeurIPS 2025\] Exact and Linear Convergence for Federated Learning under Arbitrary Client Participation is Attainable](../../NeurIPS2025/optimization/exact_and_linear_convergence_for_federated_learning_under_arbitrary_client_parti.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)

</div>

<!-- RELATED:END -->
