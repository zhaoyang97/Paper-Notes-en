---
title: >-
  [Paper Note] Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching
description: >-
  [AAAI 2026][Time Series][Event sequence forecasting] This paper proposes DEF (Detection-based Event Forecasting), which draws on the set-matching idea from DETR in object detection and employs the Hungarian algorithm to align predicted and ground-truth event sequences, achieving high-accuracy and high-diversity long-horizon event forecasting with state-of-the-art results on five datasets.
tags:
  - AAAI 2026
  - Time Series
  - Event sequence forecasting
  - temporal point processes
  - matching loss
  - long-horizon forecasting
  - Hungarian algorithm
date: 2026-05-08
content_hash: 090213b995514a73
---

# Detecting the Future: All-at-Once Event Sequence Forecasting with Horizon Matching

**Conference**: AAAI 2026  
**arXiv**: [2408.13131](https://arxiv.org/abs/2408.13131)  
**Code**: [github.com/ivan-chai/hotpp-benchmark](https://github.com/ivan-chai/hotpp-benchmark)  
**Area**: Time Series / Event Sequence Forecasting  
**Keywords**: Event sequence forecasting, temporal point processes, matching loss, long-horizon forecasting, Hungarian algorithm

## TL;DR

This paper proposes DEF (Detection-based Event Forecasting), which draws on the set-matching idea from DETR in object detection and employs the Hungarian algorithm to align predicted and ground-truth event sequences, achieving high-accuracy and high-diversity long-horizon event forecasting with state-of-the-art results on five datasets.

## Background & Motivation

Event sequence forecasting is a core task in retail, finance, healthcare, and social networks. Traditional marked temporal point process (MTPP) models primarily address next-event prediction; however, practical applications often require forecasting **multiple events** within a future time window (e.g., purchasing behavior over the next month, long-term medical prognosis).

Existing methods face three critical challenges:

**Degeneration in autoregressive methods**: Traditional autoregressive models predict the next event step by step, but as the prediction horizon grows, outputs rapidly converge to constant values or repetitive patterns, due to compounding errors when the model uses its own predictions as subsequent inputs.

**Deficiencies of position-wise paired losses**: Existing horizon forecasting methods (including GAN- and diffusion-based approaches) apply position-wise pairwise losses, matching predicted and ground-truth events by position. Since the number and positions of predicted events may not align with the ground truth, this leads to erroneous alignment (as illustrated in Figure 1a).

**Insufficient prediction diversity**: Autoregressive and diffusion-based methods tend to produce repetitive event types in long-horizon forecasting, failing to reflect the true diversity of event distributions.

**Core insight**: Long-horizon event forecasting is fundamentally more akin to **object detection** (detecting all events within a future time window) than to sequence generation.

## Method

### Overall Architecture

The overall architecture of DEF (Figure 2) consists of three core components:
1. **Backbone model** (GRU): Extracts contextual embeddings from the historical event sequence.
2. **K prediction heads**: Outputs K candidate events in parallel, each comprising an occurrence probability $\hat{o}$, a timestamp $\hat{t}$, and a label distribution $\hat{p}(l)$.
3. **Horizon Matching loss**: Dynamically aligns predictions with ground-truth events via the Hungarian algorithm and computes the matching loss.

At inference, candidate events are filtered by occurrence probability and sorted chronologically to produce the final output.

### Key Designs

#### 1. **Probabilistic Prediction Heads**

Each prediction head outputs three components within a unified probabilistic framework:

- **Occurrence probability** $\hat{o}$: Obtained via sigmoid activation, indicating whether the slot corresponds to a real event.
- **Label distribution** $\hat{p}(l)$: Outputs probabilities over $L$ event types via softmax.
- **Time distribution**: Modeled as a Laplace distribution with density $P(t) = \frac{1}{2}e^{-|t-\hat{t}|}$.

The complete log-likelihood of an event is:

$$\log P(y) = \log\hat{o} + \log\hat{p}(l) - |t-\hat{t}| - \log R(t)$$

The probability of a non-occurring event is: $\log P(\emptyset) = \log(1-\hat{o}) + C_{\emptyset}$

#### 2. **Horizon Matching Loss**

This is the paper's central contribution. Inspired by DETR, the Hungarian algorithm is used to find the optimal matching between predicted and ground-truth events:

$$\mathcal{L}_{\text{matching}} = \min_{\sigma \in \mathcal{A}} \left[\sum_{i=1}^{T} \mathcal{L}_{\text{pair}}(y_i, \hat{y}_{\sigma(i)}) + \mathcal{L}_{\text{BCE}}(\sigma, \hat{y})\right]$$

where:
- $\mathcal{A}$ is the set of all possible alignments.
- $\sigma$ is a specific alignment solved by the Hungarian algorithm.
- **Pairwise loss**: $\mathcal{L}_{\text{pair}}(y_i, \hat{y}_{\sigma(i)}) = |t_i - \hat{t}_{\sigma(i)}| - \log\hat{p}_{\sigma(i)}(l_i)$
- **Binary cross-entropy loss**: $\mathcal{L}_{\text{BCE}}$ trains the model to predict whether each slot is matched to a ground-truth event.

Unlike DETR, DEF uses the **same loss function** for both matching and model training, incorporating the alignment loss $\mathcal{L}_{\text{BCE}}$ into the matching cost.

#### 3. **Conditional Head Architecture**

To avoid parameter explosion and overfitting caused by $K$ independent feed-forward networks, a conditional prediction head design is adopted (Figure 3):
- A **single shared feed-forward network** is used.
- Each output head has a trainable **query vector**.
- The query vector is concatenated with the context vector and fed into the shared network.
- Each query vector encodes information specific to its corresponding output head.

This design substantially reduces parameter count, accelerates convergence, and improves prediction quality.

### Loss & Training

The final training objective combines the matching loss with a next-event prediction auxiliary loss:

$$\mathcal{L}_{\text{DEF}} = \mathcal{L}_{\text{matching}} + \lambda[|t_1 - \hat{t}_1| - \log\hat{p}_1(l_1)]$$

- $\lambda=4$ is fixed across all experiments.
- The weight of $\mathcal{L}_{\text{BCE}}$ is typically 8× that of the label and time losses.
- **Inference-time calibration**: Occurrence probability thresholds are calibrated by tracking each head's matching frequency during training, aligning the prediction rate with matching probability.
- Hyperparameter $K$ is set to 4× the average number of horizon events ($K=32$–$64$ in experiments).

## Key Experimental Results

### Main Results (Long-horizon Forecasting, OTD↓ / T-mAP↑)

| Dataset | IFTPP | Diffusion | DEF | Relative Gain |
|--------|-------|-----------|-----|---------|
| StackOverflow | 13.64/8.31% | 13.01/15.07% | **12.14/22.72%** | +6.7%/+50.8% |
| Amazon | 6.52/22.56% | 6.52/30.29% | **5.98/37.20%** | +8.3%/+22.8% |
| Retweet | 172.7/31.75% | 158.0/52.24% | **132.9/57.93%** | +15.9%/+10.9% |
| MIMIC-IV | 11.53/21.67% | 13.28/22.82% | **-/30.35%** | -/+28.2% |
| Transactions | 6.90/5.88% | 6.88/6.04% | **6.70/9.26%** | +2.2%/+31.3% |

DEF achieves state-of-the-art results in 9 out of 10 comparisons.

### Ablation Study

| Configuration | Description |
|------|------|
| Including $\mathcal{L}_{BCE}$ in matching | Significantly improves performance on most datasets |
| Conditional heads vs. independent heads | Conditional heads reduce parameters and improve quality |
| Next-event auxiliary loss ($\lambda$) | Encourages the first output head to focus on near-term predictions |
| Choice of $K$ | Setting $K$ to 4× the average horizon event count is optimal |

### Key Findings

1. **Prediction diversity**: DEF achieves the best diversity–accuracy trade-off on 4/5 datasets (Figure 6), producing naturally diverse predictions without temperature tuning.
2. **Next-event prediction**: DEF not only excels at long-horizon forecasting but also achieves state-of-the-art performance on next-event prediction (Figure 5), with particularly pronounced gains on the Transactions dataset.
3. **Inference efficiency**: DEF is among the fastest methods at inference (Figure 7), as it predicts $K$ events in parallel without autoregressive generation.
4. **Extension to longer horizons**: A hybrid autoregressive strategy—appending DEF's outputs to the input sequence and predicting recursively—enables forecasting over even longer horizons.

## Highlights & Insights

1. **Cross-domain idea transfer**: Migrating the set-prediction idea of DETR from object detection to event sequence forecasting is an elegant analogy—future events are treated like objects in an image, to be "detected" rather than "generated."
2. **Matching loss addresses the root problem**: The fundamental limitation of position-wise paired losses is resolved by the matching loss, which allows predicted events to align with the nearest ground-truth events rather than enforcing positional correspondence.
3. **Unified probabilistic framework**: Occurrence probability, label distribution, and time distribution are unified within a single probabilistic framework used consistently for both matching and backpropagation.
4. **Engineering design of conditional heads**: Replacing $K$ independent networks with query vectors and a shared network balances efficiency and expressiveness.

## Limitations & Future Work

1. **Conditional independence assumption over event attributes**: The model assumes that, given the history, event time, label, and occurrence are conditionally independent, whereas dependencies may exist in practice.
2. **Inter-event dependencies not modeled**: Similar to diffusion-based methods, DEF does not model dependencies among the $K$ predicted events.
3. **Computational complexity of the Hungarian algorithm**: The $O(K^3)$ complexity may become a bottleneck for large $K$; custom CUDA kernels could offer speedups.
4. Integrating beam search or rescoring strategies could further improve performance.
5. Combining intensity-based modeling approaches (NHP/RMTPP) with DEF may improve temporal predictions.

## Related Work & Insights

- **DETR** (Carion et al., 2020): Set-prediction method for object detection; the primary source of inspiration for this work.
- **HYPRO** (Xue et al., 2022): Another long-horizon forecasting method that generates multiple candidate sequences and selects the best, but is computationally inefficient.
- **Diffusion-based MTPP** (Zhou et al., 2025): Applies diffusion models to event sequences but still relies on position-wise losses.
- **Insight**: When the conventional sequence generation paradigm reaches its limits, reframing the problem—from "generation" to "detection"—can open entirely new perspectives.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Cross-domain transfer from object detection to event forecasting is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five datasets, diverse baselines, and comprehensive diversity/efficiency/ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, though the dense notation requires careful reading in places.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a fundamental problem in long-horizon event forecasting with a broadly applicable method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Predicting the Future by Retrieving the Past](predicting_the_future_by_retrieving_the_past.md)
- [\[AAAI 2026\] Optimal Look-back Horizon for Time Series Forecasting in Federated Learning](optimal_look-back_horizon_for_time_series_forecasting_in_federated_learning.md)
- [\[AAAI 2026\] A Theoretical Analysis of Detecting Large Model-Generated Time Series](a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](../../ICLR2026/time_series/unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)
- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](../../ACL2026/time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)

</div>

<!-- RELATED:END -->
