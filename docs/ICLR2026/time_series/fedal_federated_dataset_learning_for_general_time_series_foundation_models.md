---
title: >-
  [Paper Note] FeDaL: Federated Dataset Learning for General Time Series Foundation Models
description: >-
  [ICLR 2026][Time Series][Time Series Foundation Model] This paper proposes FeDaL, a federated framework that trains a general time series foundation model from scratch via client-side Domain Bias Elimination (DBE) and se…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Time Series Foundation Model"
  - "Federated Learning"
  - "Dataset Heterogeneity"
  - "Domain Bias Elimination"
  - "Cross-domain Generalization"
date: 2026-05-08
content_hash: a0d689c5c4766e21
---

# FeDaL: Federated Dataset Learning for General Time Series Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2508.04045](https://arxiv.org/abs/2508.04045)  
**Code**: [GitHub](https://github.com/shengchaochen82/FeDaL)  
**Area**: Time Series / Federated Learning
**Keywords**: Time Series Foundation Model, Federated Learning, Dataset Heterogeneity, Domain Bias Elimination, Cross-domain Generalization

## TL;DR

This paper proposes FeDaL, a federated framework that trains a general time series foundation model from scratch via client-side Domain Bias Elimination (DBE) and server-side Global Bias Elimination (GBE), achieving competitive or superior performance across 8 downstream task types with significantly fewer parameters than centralized TSFMs.

## Background & Motivation

**Background**: Time series foundation models (TSFMs) such as Moirai, Chronos, and Time-MoE acquire transferable representations through large-scale multi-domain pretraining, yet still rely on centralized data access and are typically tailored to specific tasks (e.g., forecasting). Time Series Pattern Machines (TSPMs) pursue architecture-level generality but are trained per dataset, limiting zero-shot generalization.

**Limitations of Prior Work**: FFTS, a pioneer in the federated foundation model (FFM) direction, addresses only coarse-grained domain-level heterogeneity (e.g., climate vs. healthcare), ignores intra-dataset structural biases, and does not support zero-shot inference—and thus cannot be considered a true foundation model.

**Key Challenge**: Time series data are inherently siloed and heterogeneous, yet federated aggregation assumes client updates are unbiased estimates of the global gradient. This assumption breaks down under severe dataset heterogeneity, causing the aggregated global model to be dominated by bias. The paper systematically identifies three dataset-level biases: *temporal resolution bias* (different sampling rates under identical window lengths lead to unequal information density), *physical constraint bias* (differing physical laws reduce cross-domain transferability), and *pattern shift bias* (exogenous events cause initially similar trends to diverge, which is amplified during aggregation).

**Goal**: To train a general TSFM from scratch under federated constraints that supports zero-shot inference across multiple downstream tasks while handling dataset-level heterogeneity.

**Key Insight**: The distributed architecture of federated learning is itself a natural mechanism for decomposing heterogeneity—local biases are eliminated via DBE at the client side, and global representations are aligned via GBE at the server side.

**Core Idea**: Reframe federated learning from a "privacy-preservation tool" to a "heterogeneity decomposition paradigm," producing domain-invariant time series representations through a dual DBE+GBE mechanism.

## Method

### Overall Architecture

FeDaL follows the standard federated "client-training / server-aggregation" paradigm. Each client holds one time series dataset and performs unsupervised pretraining via patch-wise masked reconstruction. Input sequences are split into patches and randomly masked (75%), encoded by a backbone, passed through a DBE module to separate dataset-specific biases, and then reconstructed via a reconstruction head. Upon receiving client model updates, the server executes two GBE steps—gradient-level dynamic correction and core-set fine-tuning—to produce the corrected global model. After each communication round, the server broadcasts both the updated global model $\theta^g$ and the global bias reference $\mathbf{b}^g$.

### Key Designs

1. **Domain Bias Elimination (DBE)**

    - **Function**: Separates dataset-specific non-transferable biases from client-side latent representations.
    - **Mechanism**: Applies trend–seasonality decomposition to the latent representation of the masked input: $\mathbf{h}_t, \mathbf{h}_s = \text{TimeDecomp}(f_{\theta^b}(\tilde{X}), \tau)$. Each component is averaged and scaled by learnable factors to obtain a bias vector $\mathbf{b} = \text{Mean}(\mathbf{h}_t) \odot \gamma_t + \text{Mean}(\mathbf{h}_s) \odot \gamma_s$. During reconstruction, the bias is injected into the latent features: $\mathcal{L} = \mathbb{E}[\|f_{\theta_h}(f_{\theta_b}(\tilde{X}) + \mathbf{b}) - X\|^2] + \lambda\|\mathbf{b} - \mathbf{b}^g\|^2$, where $\mathbf{b}^g$ is the server-aggregated global bias reference. EMA is used to stabilize bias estimation across mini-batches.
    - **Design Motivation**: Compared to simple averaging, trend–seasonality decomposition introduces inductive bias—$\mathbf{b}_t$ captures low-frequency drift and $\mathbf{b}_s$ captures high-frequency periodicity. Once the bias vector absorbs dataset-specific shifts, the backbone is forced to focus on transferable temporal structure. The regularization term prevents client-side bias from drifting excessively.

2. **Global Bias Elimination (GBE)**

    - **Function**: Eliminates residual cross-client biases during server-side aggregation.
    - **Mechanism**: Comprises two sub-components. (a) **Gradient-level dynamic correction**: maintains a server state vector $\mathbf{s}^r = \mathbf{s}^{r-1} - \beta\sum_i(\theta_i^r - \theta_g^{r-1})$ that records accumulated client–server drift, and corrects the FedAvg result as $\hat{\theta}_g^r = \tilde{\theta}_g^r - (1/\beta)\cdot\mathbf{s}^r$. (b) **Core-set fine-tuning**: each client samples a small batch from local data and optimizes learnable core-set vectors via gradient matching $\mathcal{L}_{\text{match}} = \sum_{x}\|\nabla_\theta f_\theta(\mathcal{C}) - \nabla_\theta f_\theta(x)\|_2^2$; privacy is protected by adding noise to the amplitude in the Fourier domain (only amplitude is perturbed; phase is preserved since it encodes semantic information such as periodicity). The server fine-tunes the corrected model with the aggregated core-sets, followed by convex combination: $\theta^{g,r} = \alpha\hat{\theta}^{g,r} + (1-\alpha)\theta^{gt,r}$.
    - **Design Motivation**: Because DBE debiases clients inconsistently, residual bias persists after aggregation. Gradient correction compensates for client drift, while core-set fine-tuning further aligns global representations using privacy-protected knowledge summaries.

### Loss & Training

The client loss consists of masked patch reconstruction and bias regularization. The server performs three sequential steps: weighted-average aggregation, gradient correction, and core-set fine-tuning followed by convex combination. Pretraining is conducted on the LOTSA dataset (231B time points, 174 datasets treated as 174 clients). For non-zero-shot downstream tasks, a single epoch of fine-tuning suffices for adaptation.

## Key Experimental Results

### Main Results

**Federated Representation Learning** (Table 1; average Reconstruction MSE across 5 masking ratios; lower is better):

| Method | UTSD-H1 | UTSD-H2 | CTSD | Comm. Params |
|--------|---------|---------|------|--------------|
| FedAvg | 0.586 | 0.592 | 0.455 | 108.41 MB |
| FedProx | 0.583 | 0.586 | 0.444 | 108.41 MB |
| FFTS | 0.562 | 0.531 | 0.416 | 118.94 MB |
| Standalone | 0.571 | 0.567 | 0.447 | — |
| **FeDaL** | **0.551** | **0.511** | **0.387** | 110.41 MB |

Compared to FFTS, FeDaL reduces MSE by 4.16% on UTSD and 8.86% on CTSD.

**Zero-shot Forecasting** (Table 4; average over ETT series + Weather):

| Method | Type | Avg MSE | Avg MAE | # 1st Place |
|--------|------|---------|---------|-------------|
| **FeDaL** | FL | **0.335** | **0.365** | 3 |
| FFTS | FL | 0.348 | 0.379 | 1 |
| Moirai-base | Centralized | 0.357 | 0.361 | 4 |
| Chronos-large | Centralized | 0.434 | 0.400 | 1 |
| Time-MoE-ultra | Centralized | 0.337 | 0.370 | 2 |

### Ablation Study

| Configuration | UTSD MSE | CTSD MSE | Avg. Change |
|---------------|----------|----------|-------------|
| FeDaL (full) | 0.573 | 0.405 | — |
| w/o bias alignment | 0.602 | 0.434 | ↓6.11% |
| w/o DBE | 0.637 | 0.452 | ↓9.17% |
| w/o core-set tuning | 0.590 | 0.430 | ↓4.57% |
| w/o gradient correction | 0.600 | 0.431 | ↓5.57% |
| w/o GBE | 0.610 | 0.444 | ↓8.05% |

### Key Findings

- DBE contributes most (removing it causes an average drop of 9.17%), indicating that local bias is the primary challenge.
- Both gradient correction and core-set fine-tuning within GBE contribute independently; removing the entire GBE causes an 8.05% drop.
- In full-shot long-term forecasting (Table 3), FeDaL achieves the best MSE on ETTh1/ETTm1/Weather/ILI, ranking first in 9 out of 12 metrics.
- The first systematic analysis of federated scaling behavior for TSFMs shows that more clients with moderate participation rates yields the best results, and performance improves steadily with increasing data volume.

## Highlights & Insights

- Federated learning is not merely a privacy-preservation mechanism but a natural computational paradigm for handling heterogeneity—turning the disadvantage of "data silos" into the advantage of "bias decomposition."
- DBE is a plug-and-play module compatible with any Transformer-based time series model without modifying the main architecture.
- The Fourier-domain amplitude perturbation strategy for core-sets is elegant—only the amplitude is perturbed while phase is preserved, since phase encodes semantic information such as periodicity.
- This work presents the first systematic study of TSFM scaling behavior in a federated setting, providing empirical guidance for decentralized large model training.

## Limitations & Future Work

- Core-set fine-tuning adds approximately 2 MB of communication overhead per round, which may accumulate in large-scale deployments.
- Validation is limited to Transformer-based architectures; newer time series architectures such as SSMs (e.g., Mamba) have not been tested.
- Hyperparameters ($\lambda$, $\alpha$, $\beta$, core-set size $K$) require careful tuning; sensitivity analysis shows that extreme values significantly degrade performance.
- In-depth comparison with personalized federated methods (e.g., Per-FedAvg) is lacking.

## Related Work & Insights

- **vs. FFTS**: FFTS handles only coarse-grained domain-level heterogeneity and does not support zero-shot inference. FeDaL addresses dataset-level biases and supports 8 task types via zero-shot inference or single-epoch adaptation.
- **vs. Moirai/Chronos/Time-MoE**: These centralized TSFMs require full data aggregation and have larger parameter counts. FeDaL achieves competitive performance under privacy constraints with fewer parameters (zero-shot Avg MSE: 0.335 vs. Time-MoE-ultra's 0.337).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The intersection of federated learning and time series foundation models is novel; the DBE/GBE designs are original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 8 task types, 54 baselines, and federated scaling behavior analysis comprehensively.
- **Writing Quality**: ⭐⭐⭐⭐ — The illustrations of the three bias types are clear; the overall structure is well-organized.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for general time series modeling under privacy-preserving constraints.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICML 2026\] FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](../../ICML2026/time_series/factorynet_a_large-scale_dataset_toward_industrial_time-series_foundation_models.md)
- [\[AAAI 2026\] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](../../AAAI2026/time_series/optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)
- [\[ICLR 2026\] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)

</div>

<!-- RELATED:END -->
