---
title: >-
  [Paper Note] Predicting the Future by Retrieving the Past
description: >-
  [AAAI 2026][Time Series][Retrieval-Augmented Forecasting] This paper proposes PFRP (Predicting the Future by Retrieving the Past), which constructs a Global Memory Bank (GMB) to store historical patterns, trains an encoder via Predictive Contrastive Learning (PCL) for efficient retrieval, and dynamically integrates retrieved global predictions with any local forecasting model. PFRP achieves an average improvement of 8.4% in forecasting performance across 7 datasets.
tags:
  - AAAI 2026
  - Time Series
  - Retrieval-Augmented Forecasting
  - Global Memory Bank
  - Contrastive Learning
  - Univariate Time Series
  - Plug-and-Play
date: 2026-05-08
content_hash: ac25d776a2e70202
---

# Predicting the Future by Retrieving the Past

**Conference**: AAAI 2026
**arXiv**: [2511.05859](https://arxiv.org/abs/2511.05859)
**Code**: [github.com/ddz16/PFRP](https://github.com/ddz16/PFRP)
**Area**: Time Series
**Keywords**: Retrieval-Augmented Forecasting, Global Memory Bank, Contrastive Learning, Univariate Time Series, Plug-and-Play

## TL;DR

This paper proposes PFRP (Predicting the Future by Retrieving the Past), which constructs a Global Memory Bank (GMB) to store historical patterns, trains an encoder via Predictive Contrastive Learning (PCL) for efficient retrieval, and dynamically integrates retrieved global predictions with any local forecasting model. PFRP achieves an average improvement of 8.4% in forecasting performance across 7 datasets.

## Background & Motivation

Existing deep learning time series forecasting models (MLP, Transformer, TCN, etc.) are trained on sliding windows and implicitly compress historical information into model parameters. Once training is complete, the original training data is discarded; at inference time, the model can only access local context within the current lookback window and cannot explicitly leverage global historical knowledge.

**Key Observation**: Time series frequently contain highly similar subsequences across different cycles. For example, the electricity consumption pattern of a household in a given week of 2019 may closely resemble that of a corresponding week in 2018—both exhibiting daily periodicity, with peak values decreasing over the first three days and remaining elevated over the following four days. This implies that if the current lookback window is similar to a historical segment, the future trajectory is likely to be similar as well.

**Core Motivation**:
1. Existing models are "local forecasting models"—they only observe the current window, thereby wasting rich global historical patterns.
2. RAG (Retrieval-Augmented Generation) has proven effective in NLP, but existing time series RAG methods (RATD, TimeRAF, TimeRAG) either rely on diffusion models or LLMs, resulting in low efficiency, or require traversing the entire training set, leading to slow retrieval.
3. An efficient, model-agnostic retrieval-augmented forecasting framework is needed.

## Method

### Overall Architecture

PFRP operates in two stages:
- **Stage 1 — Building the Global Memory Bank (GMB)**: Train the contrastive learning encoder → encode the lookback windows of all historical samples → apply K-medoids clustering to reduce redundancy → store $K$ representative (feature, prediction sequence) pairs.
- **Stage 2 — Retrieval-Augmented Forecasting**: Encode the current lookback window → retrieve top-k most similar historical entries from the GMB → apply confidence gate and output gate modulation → dynamically fuse with the local forecasting model.

### Key Designs

1. **Predictive Contrastive Learning (PCL)**: The selection of positive samples is the key innovation when training the lookback window encoder. Unlike conventional contrastive learning, which selects positive samples based on MSE over lookback sequences, PCL **selects positive samples based on MSE over prediction horizon sequences**:

    $i^+ = \arg\min_{1 \leq j \leq B, j \neq i} \|y_i - y_j\|_2^2$

   The InfoNCE loss then pulls the positive sample pairs closer in the feature space:

    $\mathcal{L}_{pcl} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\epsilon^{(i)} \cdot \epsilon^{(i^+)}/\tau)}{\sum_{j=1}^{B} \exp(\epsilon^{(i)} \cdot \epsilon^{(j)}/\tau)}$

   **Design Motivation**: The goal is not to find historical segments that *look* similar, but those that have *similar futures*. Selecting positive samples based on future trajectories enables the encoder to learn features that more directly serve the retrieval objective—identifying historical segments whose future evolution is most likely to resemble the current one. Furthermore, to prevent temporally overlapping samples from acting as false positives, samples overlapping with the anchor by more than 48 steps are excluded.

2. **K-medoids Clustering for GMB Construction**: After encoding all training samples, K-medoids clustering is applied in the feature space, retaining only $K$ cluster medoids as memory entries $\{(\epsilon^{(i)}, y^{(i)})\}_{i=1}^{K}$.

   **Design Motivation**:
    - Redundancy reduction: avoids storing highly similar historical samples and improves retrieval efficiency.
    - K-medoids vs. K-means: K-medoids uses actual historical samples rather than synthetic averages as cluster centers, ensuring that patterns in the memory bank correspond to real, coherent historical sequences.
    - The GMB is constructed with the maximum prediction length of 720; shorter predictions simply use the first few steps, eliminating the need to rebuild the GMB for different prediction lengths.

3. **Confidence Gate + Output Gate + Dynamic Fusion**: After retrieving the top-k most similar historical entries, three modulation steps are applied to generate the final prediction:

   **Confidence Gate**: Determines whether concatenating the retrieved historical entry with the current window forms a plausible sequence:
    $p_i = \text{Sigmoid}(\text{MLP}([x; y^{(a_i)}]))$
   This probability modulates the original similarity weights, downweighting historical entries that appear similar but whose futures do not match.

   **Output Gate**: Applies an affine transformation to the weighted global prediction $\bar{y}_1$ to adapt to the current scale and shift:
    $y_1 = \alpha \cdot \bar{y}_1 + \beta$
   where $\alpha, \beta \in \mathbb{R}^H$ are generated by an MLP from the current lookback window; $\alpha$ is initialized to all ones and $\beta$ to all zeros.

   **Dynamic Fusion**: Dynamically determines the weights of the global prediction $y_1$ and the local prediction $y_2$ based on retrieval similarity:
    $y = w_1 \cdot y_1 + w_2 \cdot y_2, \quad w_1, w_2 = \text{Softmax}(\text{MLP}(\bar{w}^{(a_1)}, \ldots, \bar{w}^{(a_k)}))$

   **Design Motivation**: When no highly similar historical sequence exists, the modulation weight remains small and fusion automatically favors the local model. For strongly periodic data, the modulation weight is larger and the global prediction dominates.

### Loss & Training

- **Stage 1** (PCL training): batch size 256, temperature $\tau=0.05$, learning rate 0.001; samples with temporal overlap > 48 steps excluded.
- **Stage 2** (PFRP training): Adam optimizer, L2 loss, initial learning rate 0.0001, following the official hyperparameters of each baseline model.
- **Fair comparison**: Baseline model hyperparameters and training configurations remain unchanged regardless of whether PFRP is applied.
- **Lookback window** $L=96$, prediction lengths $H \in \{96, 192, 336, 720\}$.

## Key Experimental Results

### Main Results

Univariate forecasting on 7 datasets (last variable), averaged over all prediction lengths:

| Baseline | Original MSE | +PFRP MSE | Gain |
|----------|----------|-----------|------|
| SparseTSF | 0.2404 (Traffic) | 0.1919 | **20.2%** |
| DLinear | 0.2778 (Traffic) | 0.1793 | **35.5%** |
| PatchTST | 0.1797 (Traffic) | 0.1712 | **4.7%** |
| TimesNet | 0.2165 (Traffic) | 0.1799 | **16.9%** |
| SparseTSF | 0.0841 (ETTh1) | 0.0766 | **8.9%** |
| DLinear | 0.3951 (Electricity) | 0.3666 | **7.2%** |

Average gains across 7 datasets: SparseTSF +8.4%, DLinear +7.1%, with smaller but consistent gains for PatchTST and TimesNet. Datasets with stronger periodicity yield the largest improvements (Traffic +17.4%, Electricity +10.1%).

### Ablation Study

| Configuration | Traffic MSE | Electricity MSE | Note |
|------|-------------|-----------------|------|
| SparseTSF (baseline) | 0.2404 | 0.4968 | No PFRP |
| +PFRP (full) | **0.1919** | **0.3561** | All modules |
| w/o confidence gate | 0.2385 | 0.3960 | Contribution of confidence gate |
| w/o output gate | 0.2130 | 0.5140 | Contribution of output gate |
| w/o both gates | 0.2128 | 0.5763 | Both gates removed |
| w/o prediction model | **0.1686** | 0.3952 | Global prediction only |

GMB-related ablations:

| Dimension | Optimal Choice | Alternatives | Note |
|------|----------|------|------|
| Retrieval criterion | Feature cosine (Ours) | MSE/DTW/PCC | Feature-level similarity outperforms raw sequence-level |
| Encoder type | MLP (default) | PatchTST/TimesNet | Trade-offs exist; MLP is most efficient |
| Training strategy | PCL (Ours) | CL/PL | Selecting positives by future similarity yields best results |

### Key Findings

1. **Stronger periodicity yields larger gains**: Traffic/Electricity datasets have periodicity scores of 0.32/0.19 respectively; the global prediction weight $w_1$ is higher for these datasets, resulting in the most significant improvements.
2. **Global prediction alone can surpass the baseline**: On Traffic, the MSE of the w/o prediction model variant (0.1686) even outperforms the full PFRP (0.1919), indicating that historical retrieval alone is sufficient for strongly periodic data.
3. **Simpler models benefit more**: MLP-based models such as DLinear and SparseTSF gain more than more complex models such as PatchTST and TimesNet.
4. **Compatible with large pretrained models**: Freezing the parameters of TimeCMA/Moirai/Sundial and fine-tuning only the PFRP parameters also improves performance.
5. **Negligible computational overhead**: GMB construction is a one-time cost of 186 seconds (PCL: 134s + clustering: 52s), with a model size increase of only 1.57 MB.

## Highlights & Insights

- **Model-agnostic plug-and-play architecture**: PFRP integrates seamlessly into any univariate forecasting model without modifying baseline hyperparameters.
- **Novel positive sample selection in PCL**: Selecting positive samples based on future trajectories rather than historical shapes enables the encoder to learn a feature space better aligned with the retrieval objective.
- **Dynamic fusion adapts to periodicity**: The global prediction weight is proportional to data periodicity, automatically switching between retrieval-driven and model-driven regimes.
- **Interpretability**: Retrieved historical sequences provide an intuitive explanation of why a particular prediction is made, offering local interpretability.

## Limitations & Future Work

1. **Univariate only**: PFRP currently handles only univariate time series; extending it to the multivariate setting requires redesigning the retrieval and fusion mechanisms.
2. **Hyperparameter tuning for $K$ and $k$**: The optimal values of GMB size $K$ and retrieval count $k$ vary substantially across datasets ($K$ ranging from 1000 to 4000, $k$ from 10 to 200).
3. **Limited gains on weakly periodic data**: Average improvements on ETT datasets (periodicity < 0.13) are only 2–3%.
4. **Static GMB**: Once constructed from the training set, the GMB remains fixed and cannot adapt to distribution shift.

## Related Work & Insights

- **RAG for time series**: RATD uses diffusion models, TimeRAF uses foundation models, and TimeRAG uses LLMs—all requiring heavyweight inference. PFRP achieves more efficient retrieval-augmented forecasting with a simple MLP encoder and GMB retrieval.
- **Memory-augmented networks**: The GMB can be viewed as a form of external memory, analogous to the Neural Turing Machine but simpler and more efficient.
- **Effective use of K-medoids**: Retaining actual historical samples (rather than synthetic averages) as memory entries ensures the interpretability of retrieved results.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of PCL + GMB + dynamic fusion offers a distinctive contribution to time series RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 datasets × 4 baselines, comprehensive GMB/PFRP ablations, hyperparameter sensitivity analysis, and large model enhancement experiments.
- Writing Quality: ⭐⭐⭐⭐ — Figures are clear, method descriptions are complete, and comparative analysis is well-articulated.
- Value: ⭐⭐⭐⭐ — Plug-and-play, minimal overhead, and consistent gains; however, the univariate limitation constrains practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching](detecting_the_future_all-at-once_event_sequence_forecasting_with_horizon_matchin.md)
- [\[AAAI 2026\] Finding Time Series Anomalies using Granular-ball Vector Data Description](finding_time_series_anomalies_using_granular-ball_vector_data_description.md)
- [\[AAAI 2026\] HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)
- [\[AAAI 2026\] C3RL: Rethinking the Combination of Channel-independence and Channel-mixing from Representation Learning](c3rl_rethinking_the_combination_of_channel-independence_and_channel-mixing_from_.md)
- [\[AAAI 2026\] Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)

</div>

<!-- RELATED:END -->
