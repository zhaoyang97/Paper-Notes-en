---
title: >-
  [Paper Note] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting
description: >-
  [ICLR 2026][Time Series][Multivariate time series forecasting] This paper proposes the CPiRi framework, which achieves channel permutation invariance (CPI) without sacrificing cross-channel modeling capability by combining a frozen pretrained temporal encoder, a trainable permutation-equivariant spatial module, and a channel shuffling training strategy. CPiRi achieves state-of-the-art performance on multiple traffic benchmarks.
tags:
  - ICLR 2026
  - Time Series
  - Multivariate time series forecasting
  - channel permutation invariance
  - spatiotemporal decoupling
  - foundation models
  - channel interaction
date: 2026-05-08
content_hash: 1bddc6df4319f5db
---

# CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2601.20318](https://arxiv.org/abs/2601.20318)
**Code**: [https://github.com/JasonStraka/CPiRi](https://github.com/JasonStraka/CPiRi)
**Area**: Time Series
**Keywords**: Multivariate time series forecasting, channel permutation invariance, spatiotemporal decoupling, foundation models, channel interaction

## TL;DR
This paper proposes the CPiRi framework, which achieves channel permutation invariance (CPI) without sacrificing cross-channel modeling capability by combining a frozen pretrained temporal encoder, a trainable permutation-equivariant spatial module, and a channel shuffling training strategy. CPiRi achieves state-of-the-art performance on multiple traffic benchmarks.

## Background & Motivation

1. **Background**: Multivariate time series forecasting (MTSF) comprises two major paradigms—channel-dependent (CD) models that learn cross-channel features, and channel-independent (CI) models that process each channel independently.
2. **Limitations of Prior Work**: CD models (e.g., Informer, Crossformer) effectively memorize the fixed positional order of channels rather than learning semantic relationships. When channels are reordered or new channels are introduced at inference time, performance collapses catastrophically (Informer's error increases by >400% on PEMS-08). CI models are naturally immune to channel ordering but completely ignore cross-channel dependencies, limiting forecasting performance.
3. **Key Challenge**: CD models capture interactions but lack robustness, while CI models guarantee robustness but forgo relational reasoning—the two properties appear mutually exclusive.
4. **Goal**: How can a model simultaneously capture cross-channel relationships and maintain channel permutation invariance (CPI), enabling deployment in real-world scenarios where channels change dynamically?
5. **Key Insight**: The authors observe that the strengths of CI and CD models are complementary. By thoroughly decoupling temporal feature extraction from spatial relational modeling, the advantages of both paradigms can be inherited independently. A channel shuffling strategy during training then forces the spatial module to learn content-based rather than position-based relationships.
6. **Core Idea**: Use a frozen foundation model for temporal encoding (CI advantage), employ a permutation-equivariant Transformer spatial module to learn cross-channel relationships (CD advantage), and enforce content-driven relational reasoning via a channel shuffling training strategy.

## Method

### Overall Architecture
CPiRi is a three-stage pipeline. The input is $\mathcal{X} \in \mathbb{R}^{L \times C}$ ($L$ time steps, $C$ channels), and the output is a forecast $\mathcal{Y} \in \mathbb{R}^{T \times C}$ over the next $T$ steps. The three stages are: (1) a frozen temporal encoder that independently extracts temporal features for each channel; (2) a trainable spatial module that learns cross-channel relationships; and (3) a frozen decoder that independently generates predictions for each channel.

### Key Designs

1. **Frozen Temporal Encoder (Stage 1)**:
    - **Function**: Uses the encoder of the pretrained Sundial foundation model to independently extract a temporal feature vector $\mathbf{h}_i \in \mathbb{R}^D$ for each channel.
    - **Mechanism**: Directly reuses large-scale pretrained temporal priors; encoder parameters are completely frozen. Processing each channel independently provides natural permutation invariance.
    - **Design Motivation**: (a) Transfers robust temporal priors learned from large-scale datasets, alleviating the data scarcity problem in MTSF; (b) Freezing prevents overfitting to specific datasets; (c) Independent processing preserves the noise-immunity advantage of CI models.

2. **Permutation-Equivariant Spatial Module (Stage 2)**:
    - **Function**: Takes the temporal features $\{\mathbf{h}_1, \ldots, \mathbf{h}_C\}$ of all channels as an **unordered set** and learns cross-channel relationships via the self-attention mechanism of a Transformer encoder block.
    - **Mechanism**: Self-attention is inherently permutation-equivariant—$f(\mathbf{h}_{\pi(1)}, \ldots, \mathbf{h}_{\pi(C)}) = (f(\mathcal{H})_{\pi(1)}, \ldots, f(\mathcal{H})_{\pi(C)})$—meaning a permutation of the input induces the corresponding permutation of the output.
    - **Design Motivation**: No positional encodings are added, forcing the spatial module to determine inter-channel relationships solely from the content of feature vectors, thereby eliminating positional bias. The complexity is $O(C^2)$, substantially lower than iTransformer's $O((T \times C)^2)$.

3. **Channel Shuffling Training Strategy (Permutation-Invariant Regularization)**:
    - **Function**: At each training batch, a random channel permutation $\pi \leftarrow \Pi_C$ is applied to both the input and the target.
    - **Mechanism**: The optimization objective becomes $\min_\theta \mathbb{E}_{(\mathcal{X},\mathcal{Y})\sim\mathcal{D},\pi\sim\Pi_C}[\mathcal{L}(f_\theta(\mathcal{X}_\pi), \mathcal{Y}_\pi)]$. Any non-equivariant component that relies on a specific ordering incurs high loss under most permutations, so optimization naturally drives parameters toward equivariant solutions.
    - **Design Motivation**: Although the self-attention architecture is structurally equivariant, random initialization and gradient noise during training may introduce subtle positional dependencies. Channel shuffling, as a form of data augmentation, eliminates all positional shortcuts and forces the model to learn content-driven relational reasoning as a generalizable meta-skill.

### Loss & Training
- Standard MSE/MAE loss; $L = T = 336$.
- Spatial module dropout set to 0.3 to encourage sparse spatial relationship learning.
- Only the spatial module parameters are trained; the encoder and decoder are completely frozen.
- A new channel permutation is randomly generated for each batch, analogous to task distribution sampling in meta-learning.

## Key Experimental Results

### Main Results
CPiRi is compared against CI and CD models on five traffic datasets, achieving state-of-the-art performance on 4 out of 5:

| Dataset | Metric | CPiRi | iTransformer | STID | PatchTST (CI) | Gain |
|--------|------|-------|-------------|------|--------------|------|
| PEMS-BAY | WAPE | **3.90%** | 4.21% | 3.91% | 4.87% | vs iT: −7.4% |
| PEMS-04 | WAPE | **11.67%** | 12.99% | 12.43% | 15.54% | vs STID: −6.1% |
| PEMS-08 | WAPE | **9.43%** | 10.70% | 10.90% | 12.37% | vs iT: −11.9% |
| SD | WAPE | **12.25%** | 12.45% | 12.51% | 13.41% | vs iT: −1.6% |
| Electricity | WAPE | **9.90%** | 10.67% | 10.65% | 10.68% | vs STID: −7.0% |

### Ablation Study

| Configuration | PEMS-08 WAPE | Note |
|------|-------------|------|
| CPiRi (full) | 9.43% | Full model |
| w/o spatiotemporal decoupling (encoder unfrozen) | 10.80% | −1.37%, overfitting |
| w/o shuffling strategy | 10.08% | −0.65%, CPI lost |
| w/o pretrained weights | 52.29% | Catastrophic collapse |
| 3-layer encoder from scratch | 11.17% | Clearly inferior to frozen pretrained |
| Frozen Chronos-2 encoder | 13.16% | Chronos designed for short-term forecasting; incompatible |
| w/o spatial module | 22.69% | Degrades to CI; large drop |
| Mean pooling vs. last token | 12.42% | Last token outperforms mean aggregation |

### Key Findings
- **Channel shuffling robustness**: Under 100% channel shuffling, CPiRi's WAPE changes by less than 0.25%, whereas Informer's error increases by >400% and STID's by >235%.
- **Inductive generalization**: Training on only 25% of channels and testing on all channels results in only ~2% accuracy degradation, while reducing training time by 70%.
- **Large-scale scalability**: On the CA dataset (8,600 channels), CPiRi requires only 0.41s/sample inference time and 8 GB of GPU memory, compared to 75.68 GB for Timer-XL.

## Highlights & Insights
- The design philosophy of **thorough spatiotemporal decoupling** is particularly elegant: freezing the encoder simultaneously transfers pretrained priors and inherently guarantees CI properties, while the spatial module focuses exclusively on relational learning. This modular design allows the two sub-problems (temporal modeling and channel interaction) to be optimized independently.
- **Channel shuffling as regularization** is essentially a meta-learning idea—by exposing the model to all possible permutations during training, the learned relational reasoning capability becomes permutation-agnostic. This technique is transferable to any scenario requiring set-valued inputs (e.g., point cloud processing, graph node classification).
- The **CPI diagnostic test** itself constitutes a valuable contribution, as it can rapidly expose the positional memorization deficiencies of existing CD models.

## Limitations & Future Work
- CPiRi does not achieve state-of-the-art on METR-LA, where STID and Crossformer leverage exogenous holiday features. CPiRi currently processes only raw sequence data and lacks an interface for exogenous variables.
- The framework is highly dependent on the quality of the Sundial pretrained foundation model—substituting the Chronos-2 encoder leads to a significant performance drop, indicating sensitivity to encoder selection.
- The spatial module currently consists of only a single-layer Transformer block, which may be insufficient for modeling complex relationships in ultra-large-scale settings (>8,000 channels).
- Dynamic graph structure learning is not explored. The current self-attention implicitly learns fully connected relationships, whereas channel relationships in many real-world scenarios are sparse.

## Related Work & Insights
- **vs. iTransformer**: iTransformer performs joint spatiotemporal attention at each layer with complexity $O((T \times C)^2)$; while also CPI, the cost is high. CPiRi reduces spatial attention to $O(C^2)$ through decoupling.
- **vs. PatchTST**: PatchTST is a representative CI model—naturally CPI but completely disregarding cross-channel relationships. CPiRi inherits its robustness while adding relational modeling.
- **vs. STID**: STID relies on fixed spatial ID embeddings, which is essentially positional memorization. CPiRi employs dynamic, content-driven relationships.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of spatiotemporal decoupling and channel shuffling is novel, though individual components (frozen encoders, self-attention equivariance) are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five benchmarks, large-scale extension, progressive shuffling, partial-channel training, and detailed ablations; very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical analysis is clear (equivariance proofs), experimental design is systematic, and the CPI diagnostic test is a notable highlight.
- **Value**: ⭐⭐⭐⭐ — Addresses an important practical deployment problem (dynamically changing sensors) with a simple and efficient solution.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

<!-- RELATED:END -->
