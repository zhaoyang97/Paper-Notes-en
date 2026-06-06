---
title: >-
  [Paper Note] Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting
description: >-
  [ICML 2026][Time Series][Time Series Forecasting] This paper proposes the KUP-BI framework, which constructs a "post-target continuation" knowledge base from the training set. By retrieving continuation patterns of simil…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Time Series Forecasting"
  - "Retrieval-Augmented"
  - "Post-target Continuation"
  - "Bidirectional Inspiration"
  - "Gated Fusion"
date: 2026-05-08
content_hash: fc17bad54a836e6f
---

# Beyond Extrapolation: Knowledge Utilization Paradigm with Bidirectional Inspiration for Time Series Forecasting

**Conference**: ICML 2026  
**arXiv**: [2605.19249](https://arxiv.org/abs/2605.19249)  
**Code**: To be confirmed  
**Area**: Time Series Forecasting  
**Keywords**: Time Series Forecasting, Retrieval-Augmented, Post-target Continuation, Bidirectional Inspiration, Gated Fusion  

## TL;DR

This paper proposes the KUP-BI framework, which constructs a "post-target continuation" knowledge base from the training set. By retrieving continuation patterns of similar historical trajectories through ratio-based transformations, it generates a continuation-style auxiliary stream and fuses it with backbone features via gating. The method consistently improves long-term forecasting performance across 6 datasets and 4 backbone architectures.

## Background & Motivation

**Background**: Time series forecasting is widely applied in energy, transportation, and finance. Prevailing deep learning methods (Transformer, MLP, CNN, etc.) follow a **unidirectional inference paradigm**, mapping historical sequences to future target sequences.

**Limitations of Prior Work**: Unidirectional extrapolation is prone to error accumulation and trend drift in long-term forecasting. Some recent works (e.g., RAFT) attempt to retrieve **target segments** from the training set as auxiliary information, but since target segments align closely with supervision signals, they often act as **excessively strong shortcuts** during training, damaging generalization.

**Key Challenge**: A natural three-part chain of "History $\to$ Target $\to$ Post-target Continuation" exists in training data, but existing methods only utilize the first two segments. The post-target continuation segment shares the same dynamical system with the target segment but is temporally decoupled, providing **weaker but more transferable** evolutionary cues.

**Goal**: To distill continuation-style structural priors from the training set and inject them into standard forecasting backbones, without observing the post-target continuation during inference.

**Key Insight**: By representing the change of the post-target continuation relative to the history as a ratio and retrieving ratio patterns from similar historical trajectories, an approximate post-target continuation proxy for the current input can be generated.

**Core Idea**: Construct a continuation-style auxiliary stream using retrieval and ratio transformations, and fuse it with the original input stream via gating to achieve "bidirectional inspiration" forecasting.

## Method

### Overall Architecture

The KUP-BI pipeline consists of three stages: (1) Constructing a retrieval library from the training set (used only during training); (2) Channel-wise retrieval of similar histories for a new input and aggregating their ratio transformations to generate an auxiliary signal $\mathbf{Z}$; (3) Extracting features from both $\mathbf{Z}$ and the original input $\mathbf{X}$ and fusing them via a lightweight gating module before feeding them into an unmodified forecasting backbone. This process introduces no extra information beyond the training set and only provides structured inductive bias.

### Key Designs

1.  **Ratio-based Retrieval Library Construction**:

    For each trajectory in the training set, the chain $(\mathbf{H}, \mathbf{Y}, \mathbf{F})$ (history, target, post-target continuation) is extracted. The ratio representation of the post-target continuation relative to the history is calculated as:

    $$\mathbf{R} = (\mathbf{F} - \mathbf{H}) \oslash (\mathbf{H} + \epsilon \, \text{sign}(\mathbf{H}))$$

    where $\oslash$ denotes element-wise division and $\epsilon$ is a numerical stability term. This ratio representation describes the **relative change** (magnitude scaling, seasonal fluctuations) of the continuation segment, offering better scale invariance than residual representations. The historical segments, after last-step offsetting to remove local level differences, serve as retrieval keys paired with the ratio matrix $\mathbf{R}$ in the library $\mathcal{D} = \{(\tilde{\mathbf{H}}_j, \mathbf{R}_j)\}_{j=1}^N$. In the retrieval phase, Top-$k$ candidates are selected via **channel-wise** Pearson correlation and aggregated using temperature-controlled softmax. After applying quantile-$\tanh$ clipping to suppress extremes, the result is applied to the current input to generate the auxiliary sequence $\mathbf{Z}$, followed by channel mean/standard deviation alignment.

2.  **Lightweight Gated Fusion**:

    The main stream feature $\mathbf{X}_\text{main} = \text{Fea}(\mathbf{X})$ and the auxiliary stream feature $\mathbf{X}_\text{aux} = \text{Fea}(\mathbf{Z})$ are fused via gating weights $\boldsymbol{\gamma}$:

    $$\widetilde{\mathbf{X}} = \boldsymbol{\gamma} \odot \mathbf{X}_\text{main} + (1 - \boldsymbol{\gamma}) \odot \mathbf{X}_\text{aux}$$

    A convex combination with residual weight $\alpha$ is then applied: $\mathbf{X}' = \alpha \mathbf{X}_\text{main} + (1 - \alpha) \widetilde{\mathbf{X}}$, ensuring the main stream remains dominant. The gating supports both static (learnable scalar $g$) and dynamic (lightweight MLP $\phi$) modes. Ablation studies show that $\alpha$ is the most critical hyperparameter; removing it causes the MSE on the ILI dataset to surge from 1.366 to 1.929.

3.  **Backbone-agnostic Plug-and-play Design**:

    The retrieval library construction and ratio transformation in KUP-BI are non-parametric and fully decoupled from the backbone. The same library can be reused across different architectures (Transformer / MLP / CNN / Hybrids). Two integration modes are supported: Plugin-only (tuning only KUP-BI hyperparameters) and Joint-tune (fine-tuning with the backbone), with the former already yielding stable gains.

## Key Experimental Results

| Dataset | Backbone | Original MSE | +KUP-BI (Plugin) MSE | +KUP-BI (Joint) MSE | Best Relative Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ETTh2 | DLinear | 0.469 | 0.453 | **0.394** | -16.0% |
| ILI | TimesNet | 2.438 | 2.328 | **2.114** | -13.3% |
| Exchange | DLinear | 0.369 | 0.362 | **0.313** | -15.2% |
| ETTh1 | xPatch | 0.444 | 0.431 | **0.409** | -7.9% |
| ETTm2 | PatchTST | 0.258 | 0.257 | **0.255** | -1.2% |
| ILI | xPatch | 1.383 | 1.366 | **1.365** | -1.3% |

| Ablation Study (xPatch, Avg. over all lengths) | ETTh1 MSE | ETTm1 MSE | ILI MSE |
| :--- | :--- | :--- | :--- |
| KUP-BI (Full) | **0.431** | **0.352** | **1.366** |
| w/o $\alpha$ | 0.457 | 0.412 | 1.929 |
| Random Retrieval | 0.443 | 0.352 | 1.378 |
| Direct Target Usage | 0.466 | 0.352 | 1.382 |
| Concat. vs Gating | 0.411 | 0.388 | 1.713 |

## Highlights & Insights

-   **Post-target Continuation vs. Target Segment**: Utilizing post-target continuation instead of target segments as auxiliary information avoids over-reliance on label neighbors during training and providing more transferable structural priors.
-   **Ratio vs. Residual**: The ratio-based representation is scale-invariant. On ETTh1, it achieves an MSE of 0.431 compared to 0.488 for the residual-based approach, showing a significant advantage.
-   **Weak Backbones Benefit More**: DLinear, with weaker modeling capacity, benefits the most from continuation auxiliary signals (16% reduction on ETTh2), while stronger backbones like xPatch show more modest but consistent improvements.
-   Recommended default hyperparameters: $\alpha = 0.75$, Top-$k = 1$, $\tau = 0.01$.

## Limitations & Future Work

-   Current retrieval strategies do not explicitly handle **phase shifts**, potentially leading to imprecise retrieval matches.
-   To reach full potential, KUP-BI may require backbone-specific hyperparameter tuning rather than being completely plug-and-play, which increases training costs.
-   The ratio transformation is a heuristic design; future work could explore learnable encoders to replace non-parametric ratios.
-   Accurately capturing extreme fluctuations, such as sudden spikes, remains difficult.

## Related Work & Insights

-   **RAFT (Han et al., 2025)**: Retrieves target segments for auxiliary forecasting, but target segments align too strongly with supervision; KUP-BI uses post-target continuation segments to avoid this.
-   **RAF (Tire et al., 2024)**: Provides retrieval-augmented prompts for foundational time series models, used only during inference.
-   **xPatch (Stitsyuk & Choi, 2025)**: A dual-stream MLP+CNN hybrid backbone, used as the strongest baseline in experiments.

## Rating

-   Novelty: ⭐⭐⭐⭐ — The "post-target continuation" perspective is unique, and incorporating the third segment of the training chain is a novel entry point for time series forecasting.
-   Experimental Thoroughness: ⭐⭐⭐⭐ — 6 datasets × 4 backbones, including comprehensive analyses on ablations, hyperparameter sensitivity, ratio vs. residual, and retrieval vs. prediction.
-   Writing Quality: ⭐⭐⭐⭐ — Clear logic, natural derivation of motivation, and consistent mathematical notation.
-   Value: ⭐⭐⭐⭐ — Provides a general, pluggable enhancement paradigm, though absolute gains on strong backbones are somewhat limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](../../ACL2026/time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)
- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](../../ICLR2026/time_series/from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)
- [\[ICML 2026\] From Observations to States: Latent Time Series Forecasting](from_observations_to_states_latent_time_series_forecasting.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
