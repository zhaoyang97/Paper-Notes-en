---
title: >-
  [Paper Note] QuITE: Query-based Irregular Time Series Embedding
description: >-
  [ICML 2026][Time Series][Irregular time series] QuITE is a **plug-and-play embedding module**—utilizing learnable query tokens to directly aggregate irregular observations via self-attention…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Irregular time series"
  - "embedding"
  - "multivariate forecasting"
  - "query tokens"
date: 2026-05-08
content_hash: 8fb1f2a4a016960c
---

# QuITE: Query-based Irregular Time Series Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.28166](https://arxiv.org/abs/2605.28166)  
**Code**: To be confirmed  
**Area**: Time Series / Irregular Sampling  
**Keywords**: Irregular time series, embedding, multivariate forecasting, query tokens  

## TL;DR
QuITE is a **plug-and-play embedding module**—utilizing learnable query tokens to directly aggregate irregular observations via self-attention, thereby adapting any MTS model to Irregular Multivariate Time Series (IMTS) without architectural modifications or artificial value generation; it achieves an average relative improvement of 54.7% in forecasting when applied to iTransformer + QuITE.

## Background & Motivation

**Background**: Irregular Multivariate Time Series (IMTS) are prevalent in fields such as healthcare, climate, and industrial monitoring. Existing methods follow two paradigms: architecture design (e.g., GRU-D, Latent ODE, GNN) which creates specialized structures for irregularity, and data adaptation (e.g., mTAND, IP-Nets) which maps IMTS to regular grids via interpolation.

**Limitations of Prior Work**: Architectural methods fail to reuse powerful, well-validated MTS models (e.g., PatchTST, iTransformer); interpolation methods allow model reuse but damage true temporal dynamics by generating artificial values, leading to performance degradation. Both paradigms involve significant trade-offs.

**Key Challenge**: The true bottleneck **lies not in the backbone architecture, but in the input embedding layer**—existing embedding schemes assume uniform sampling and are naturally incompatible with irregular inputs. Simple fusion strategies (adding or concatenating time and value embeddings) remain constrained by the uniform sampling paradigm. Although attention mechanisms can capture interactions between irregular observations, their observation-level outputs require additional pooling to match the variable-level or patch-level representations expected by modern MTS models, which dilutes fine-grained temporal information.

**Goal**: To design a simple and efficient adaptation mechanism at the input embedding layer, allowing existing MTS models to handle IMTS directly without architectural changes.

**Key Insight**: Address irregularity directly at the embedding layer rather than at the architecture or preprocessing stages. The key observation is that learnable query tokens can serve as **structured aggregation anchors**, transforming irregular observations into fixed-dimensional representations compatible with backbone networks through a single layer of self-attention.

**Core Idea**: Utilize learnable **query tokens** to extract structured embedding representations directly from irregular observations via a self-attention mechanism, bypassing lossy pooling and artificial value generation, ensuring the output is directly usable as input for backbone models.

## Method

### Overall Architecture
QuITE is a plug-and-play embedding module consisting of: (1) Observation Tokenization: encoding each observation $(x_{n, i}, t_{n, i}, m_{n, i})$ as a token; (2) Query Token Aggregation: using learnable query tokens to aggregate variable-level or patch-level observations via masked self-attention; (3) Outputting backbone-compatible representations. It can be flexibly configured for **variable-level aggregation** (one query token per variable) or **patch-level aggregation** (one query token per time-variable patch).

### Key Designs

1.  **Observation Tokenization**:
    - **Function**: Transforms irregular observation triplets $(x_{n, i}, t_{n, i}, m_{n, i})$ into a unified set of tokens.
    - **Mechanism**: Employs **harmonic time embedding** $\phi(t)[k] = \omega_0 t + \alpha_0$ (for $k = 0$) or $\sin(\omega_k t + \alpha_k)$ (for $k > 0$) to encode continuous timestamps, with frequency and phase parameters learned. Values are mapped to the latent space via linear projection $f_{\text{val}}$, resulting in the final token $z_{n, i} = f_{\text{val}}(x_{n, i}) + \phi(t_{n, i})$.
    - **Design Motivation**: Harmonic embeddings periodically encode any time span; the masking mechanism $m_{n, i}$ identifies missing or padded observations to be excluded from subsequent attention.

2.  **Query Token Aggregation**:
    - **Function**: Aggregates the set of irregular observations into a structured representation via self-attention.
    - **Mechanism**: Assigns a learnable query token $q_n$ to each variable, interacting with all observations $Z_n$ of that variable via masked self-attention: $H_n = \text{SelfAttn}([q_n; Z_n], A_n = [1 | m_n])$. The updated output of the query token $e_n = H_n[0]$ serves as the variable-level embedding. Patch-level aggregation follows a similar logic, assigning a query token to each (patch, variable) pair to produce a patch-variable matrix $E_{\text{patch}} \in \mathbb{R}^{M \times N \times D}$.
    - **Design Motivation**: Query tokens act as anchors to avoid information loss from simple pooling; masked self-attention skips missing observations without data imputation; it functions similarly to the BERT [CLS] token but for irregular sequence embedding; compared to post-attention pooling, query tokens directly produce the representation shape expected by the backbone.

3.  **QuITE++ Hierarchical Encoder**:
    - **Function**: Extends QuITE into a full forecasting architecture that explicitly models temporal and intra-variable dependencies.
    - **Mechanism**: Stacks $L$ hierarchical encoder layers, each containing two attention blocks: (a) Patch-level self-attention: variable tokens are prepended to patch sequences to model temporal dependencies; (b) Variable-level self-attention: models interactions between all variable tokens. The decoder uses cross-attention to extract global and local context.
    - **Design Motivation**: The hierarchical structure captures local temporal patterns within patches while modeling global and cross-variable dependencies via variable tokens; the cross-attention decoder avoids constraints like patch flattening.

## Key Experimental Results

### Main Results: Performance Gains across Different Backbones

| Backbone Type | Dataset | Model | Without QuITE (MSE) | With QuITE (MSE) | Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Patch-level | Human Activity | PatchTST | 3.10 | 2.76 | +10.97% |
| Patch-level | USHCN | PatchMixer | 5.31 | 5.02 | +5.46% |
| Variable-level | PhysioNet | iTransformer | 16.48 | 4.99 | +69.72% |
| Variable-level | MIMIC-III | iTransformer | 6.05 | 1.56 | +74.19% |
| Hybrid | Human Activity | TimeXer | 2.99 | 2.53 | +15.52% |
| **Average** | **All Datasets** | **iTransformer+QuITE** | 8.37 | 3.79 | **+54.70%** |

### Classification Performance

| Dataset | Metric | Without QuITE | PatchMixer+QuITE | iTransformer+QuITE |
| :--- | :--- | :--- | :--- | :--- |
| P12 | AUROC | 78.2 | 83.9 | 85.3 |
| P19 | AUPRC | 26.4 | 55.8 | 51.7 |
| PAM | F1 | 75.7 | 83.7 | 91.5 |

### Ablation Study (Comparison of Embedding Strategies)

| Embedding Strategy | PatchTST | iTransformer | QuITE++ | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Add (Time + Value) | 4.00 | 4.98 | 3.44 | Direct summation |
| Concat | 3.90 | 5.77 | 3.35 | Concatenation |
| mTAND (Latent Interpolation) | 3.74 | 3.50 | 3.34 | Data-level interpolation |
| Mean Pooling | 3.75 | 3.59 | 3.31 | Post-attention mean |
| **QuITE (Learnable Query)** | **3.69** | **3.31** | **3.18** | **Best** |

### Key Findings
- **Backbone Agnostic**: QuITE provides consistent improvements across 6 diverse MTS backbones, ranging from 5.1% to 54.7%.
- **Differentiated Benefits**: Variable-level models (e.g., iTransformer, S-Mamba) benefit most (25%-74%) as they are more sensitive to irregular sampling; patch-level models benefit less (5%-11%) due to independent variable modeling.
- **Dataset Differences**: Improvements are most significant in medical data (MIMIC-III, PhysioNet, 30%-74%) and more conservative in climate data (USHCN, 1%-33%).
- **Robustness**: QuITE++ performance remains stable even when 50% of observations are randomly removed; performance drops sharply after 75%, suggesting a sparsity limit near 50%.

## Highlights & Insights
- **Accurate Problem Diagnosis**: The study identifies that the bottleneck is the embedding layer rather than the architecture, avoiding large-scale modification of existing powerful models—significant gains are achieved by simply replacing the input module.
- **Generality of Query Tokens**: While drawing inspiration from the BERT [CLS] token, the innovation lies in using learnable queries as **structured anchors** rather than de-semanticized general tokens, directly aggregating irregular observations without extra pooling.
- **Plug-and-Play**: QuITE is entirely decoupled from backbone architectures and can be seamlessly inserted into the front end of any MTS model, significantly lowering application barriers.
- **Thorough Ablation**: Sequential validation against Add/Concat/Mean Pooling/mTAND ensures each design choice is supported by empirical data.
- **Transfer Potential**: The hierarchical encoding structure could generalize to other sequence models (e.g., language models processing varying lengths); the learnable token aggregation paradigm could extend to other irregular data types (e.g., point clouds, dynamic graphs).

## Limitations & Future Work
- Patch-level MTS models exhibit weaker performance on medical data (e.g., PhysioNet, MIMIC-III) that requires intensive variable interaction, which is a limitation of the patch-level paradigm itself.
- Actual runtime depends on backbone implementation details and may not always be faster.
- Robustness tests indicate rapid performance decline when observation missing rates exceed 75%, posing a challenge for ultra-sparse scenarios.
- Improvements: Extension to multi-head hierarchical architectures to refine variable-patch dependencies; introduction of sparse attention for lower complexity; cross-dataset pre-training for zero-shot transfer.

## Related Work & Insights
- **vs GRU-D / P-LSTM**: RNN methods handle missing values via decay or gating but are constrained by the RNN architecture; QuITE allows for the reuse of modern backbones like Transformers.
- **vs Continuous-Time ODE (Latent-ODE, ContiFormer)**: ODE methods learn continuous dynamics between observations with high expressivity but high computational cost; QuITE provides more efficient aggregation via attention.
- **vs GNN Methods (GraFITi, tPatchGNN)**: GNNs model variable-time bipartite graphs but introduce complexity in graph construction; QuITE's self-attention is more streamlined.
- **vs Interpolation Methods (mTAND, IP-Nets)**: mTAND interpolates in latent space to avoid explicit artificial values but still alters the true sampling pattern; QuITE processes raw observations directly to maintain integrity.

## Rating
- Novelty: ⭐⭐⭐⭐ The query token aggregation approach is simple yet effective; it provides a new perspective on irregular sequence adaptation by shifting focus from architecture/data to the embedding layer.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive design with 7 datasets, 6 backbone categories, 17 baselines, 4 ablation comparisons, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and in-depth motivation; some experimental details (e.g., patch partitioning strategy) could be discussed further.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play characteristics significantly lower the barrier for practical application; the massive gains (50%-75%) in high-sparsity scenarios like healthcare are highly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Latent Laplace Diffusion for Irregular Multivariate Time Series](latent_laplace_diffusion_for_irregular_multivariate_time_series.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[AAAI 2026\] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting](../../AAAI2026/time_series/revitalizing_canonical_pre-alignment_for_irregular_multivariate_time_series_fore.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)

</div>

<!-- RELATED:END -->
