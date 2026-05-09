---
title: >-
  [Paper Note] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming
description: >-
  [ICLR 2026][Time Series][video saliency] This paper proposes HiVid, the first framework to leverage LLMs as human proxies for generating content importance weights for video chunks. Through a Perception module (sliding-window scoring), a Ranking module (LLM-guided merge sort to eliminate scoring bias), and a Prediction module (multimodal time series forecasting with adaptive latency), HiVid enables content-aware streaming, achieving an 11.5% improvement in VOD PLCC, a 26% gain in live streaming prediction, and a 14.7% improvement in human MOS correlation.
tags:
  - ICLR 2026
  - Time Series
  - video saliency
  - LLM-as-judge
  - content-aware streaming
  - time series forecasting
  - adaptive bitrate
date: 2026-05-08
content_hash: d41e252074064fde
---

# HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming

**Conference**: ICLR 2026
**arXiv**: [2602.14214](https://arxiv.org/abs/2602.14214)
**Code**: To be confirmed
**Area**: Time Series
**Keywords**: video saliency, LLM-as-judge, content-aware streaming, time series forecasting, adaptive bitrate

## TL;DR
This paper proposes HiVid, the first framework to leverage LLMs as human proxies for generating content importance weights for video chunks. Through a Perception module (sliding-window scoring), a Ranking module (LLM-guided merge sort to eliminate scoring bias), and a Prediction module (multimodal time series forecasting with adaptive latency), HiVid enables content-aware streaming, achieving an 11.5% improvement in VOD PLCC, a 26% gain in live streaming prediction, and a 14.7% improvement in human MOS correlation.

## Background & Motivation
**Background**: Content-aware video streaming assigns higher bitrates to more important chunks via $QoE = \sum_i w_i \cdot q_i$. Existing approaches include CV-based highlight detection models (DETR, VASNet, etc.) and human crowdsourced annotation (SENSEI).

**Limitations of Prior Work**: CV models suffer from insufficient semantic understanding and poor generalization; large video understanding models (VideoLLaMA3, VILA) exhibit severe hallucination on subjective scoring tasks; human annotation is prohibitively expensive (78 min / $100 per video) and infeasible for live streaming scenarios.

**Key Challenge**: A weight generation approach that simultaneously achieves accuracy (semantic understanding) and efficiency (real-time + low cost) is lacking.

**Goal**: Three challenges: (1) LLMs cannot directly process video and are subject to token limits; (2) local scores within sliding windows are inconsistent across windows; (3) live streaming requires real-time inference but LLM latency is nondeterministic.

**Key Insight**: Using LLMs as "human proxies" for zero-shot subjective reasoning, bypassing token limits through windowing and context summarization.

**Core Idea**: LLM perception + merge-sort debiasing + multimodal predictive adaptive latency = end-to-end content-aware streaming.

## Method

### Overall Architecture
HiVid consists of three modules: Perception (base) → Ranking (VOD) / Prediction (live streaming), ultimately outputting chunk weights $w_i$ integrated into the QoE model.

### Key Designs
1. **Perception Module**: Every $m$ frames are fed to an LLM (default GPT-4o) via a sliding window; the prompt requests scores and an updated summary:
    $R_{(k-1)m+1}^{km}, S_{km} = LLM(F_{(k-1)m+1}^{km}, S_{(k-1)m})$
   Only $\lceil D/m \rceil$ LLM calls are needed for arbitrarily long videos; the summary $S$ serves as a compressed historical context.

2. **Ranking Module (VOD)**: LLM-guided merge sort is employed to eliminate inter-window scoring bias. At each merge step, $m/2$ frames are sampled from each sorted group to form a new list for LLM ranking; overall complexity is $O(k \log k)$ where $k = \lceil D/m \rceil$. Scores are then normalized to $[0,1]$ and smoothed with Gaussian smoothing $w_i = GS(s, \sigma, w_i)$.

3. **Prediction Module (Live Streaming)**: A multimodal time series prediction model comprising:

    - **CLIP Alignment**: Frozen CLIP encodes historical frames and text summaries.
    - **Content-Aware Attention**: Temporal features serve as Q; concatenated image and text features serve as K/V:
    $Attn(F(x_w), F(x_{cat}), F(x_{cat})) = softmax\left(\frac{Q_w K_{cat}^T}{\sqrt{d}}\right) \cdot V_{cat}$
    - **Adaptive Prediction Dimension**: Output length is dynamically adjusted based on LLM latency $\Delta t$ and prediction latency $\delta$:
    $L_{out} = \lceil(\Delta t + \delta)/d\rceil + m + N$
    - **Correlation Loss**: $loss = MSE(x, x_{gt}) + \lambda(1 - \text{Pearson}(x, x_{gt}))$

### Loss & Training
- The Perception and Ranking modules require no training (zero-shot LLM inference).
- For the Prediction module, multiple models with different $L_{out}$ values are trained; at inference time, the smallest model satisfying the latency requirement is selected.

## Key Experimental Results

### Main Results
Saliency scoring (PLCC/mAP50) across three datasets:

| Method | Youtube-8M PLCC | TVSum PLCC | SumMe PLCC |
|--------|----------------|-----------|------------|
| DETR | 0.57 | 0.42 | 0.38 |
| SL-module | 0.59 | 0.43 | 0.39 |
| VideoLLaMA3 | 0.54 | 0.41 | 0.35 |
| **HiVid** | **0.66** | **0.50** | **0.47** |

### Ablation Study
Effect of window parameter $m$ on overhead and accuracy (201s video):

| m | Total API Calls | Total Cost | Total Time/h |
|---|----------------|------------|--------------|
| 2 | 1458 | $8.12 | 1.26 |
| 6 | 384 | $2.41 | 0.67 |
| 10 | 202 | $1.35 | 0.54 |

$m=10$ achieves the optimal accuracy–cost trade-off.

### Key Findings
- HiVid outperforms the second-best method SL-module by 11.5% in average PLCC and 6% in mAP50.
- In live streaming scenarios, HiVid's multimodal prediction outperforms the strongest time series baseline iTransformer by 26%.
- Human MOS correlation improves by 14.7%, validating real-world streaming QoE gains.
- Video understanding models (VILA, Flamingo) underperform CV baselines on subjective scoring tasks.

## Highlights & Insights
- **First framework to systematically leverage LLMs for video-level content-aware streaming**: Extends the LLM-as-judge paradigm from text to video streaming.
- **LLM-guided merge sort**: An elegant algorithm design using LLMs as a comparison function, with manageable $O(k \log k)$ overhead.
- **Adaptive prediction dimension**: Dynamic adjustment for asynchronous LLM inference latency, a critical design for practical deployment.
- **End-to-end validation**: A complete validation chain from scoring accuracy to real-world streaming QoE.
- **Content-Aware Attention for multimodal fusion**: A novel attention design that aligns CLIP image and text features and combines them with temporal sequences.

## Limitations & Future Work
- Relies on the closed-source GPT-4o API; cost remains relatively high ($1.35/video), limiting large-scale deployment.
- The Perception module uses only the first frame as an anchor, potentially missing intra-window dynamic changes (e.g., rapid motion).
- In live streaming, the initial $\lceil(\Delta t + \delta)/d\rceil + m$ chunks lack LLM scores and are filled with a default weight of 1.
- The Ranking module incurs significant overhead for very long videos; the practical API cost of $O(k \log k)$ LLM calls is non-trivial.
- Only GPT-4o is evaluated; open-source LLM alternatives (e.g., Llama, Qwen) are not explored.
- Scoring quality is heavily dependent on the subjective judgment capability of the LLM, and different LLMs may introduce different biases.
- Dynamic video characteristics (e.g., scene transitions, camera motion) are not considered as temporal features.
- Category-specific strategies for different video genres (sports, news, education, etc.) are not explored.

## Related Work & Insights
- SENSEI obtains precise weights via human crowdsourcing at prohibitive cost; HiVid achieves an accuracy–efficiency balance through LLMs.
- Compared to attention-based highlight detection methods (DETR, SL-module), LLMs demonstrate clear advantages in semantic content understanding.
- Compared to VideoLLaMA3/VILA: large video understanding models suffer severe hallucination on subjective scoring tasks and underperform the LLM vision-plus-text strategy.
- Offers methodological insights for the intersection of networked systems and AI: asynchronous design patterns for integrating LLM inference into online systems.
- Suggests that CLIP-aligned image and text features can serve as effective contextual signals in multimodal time series prediction.

## Rating
- Novelty: ⭐⭐⭐⭐ LLM-as-judge applied to video streaming is a novel combination, though individual modules are more engineering integration than technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, 17 baselines, ablations, and a human user study — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem-driven structure with three challenges mapped to three modules; rigorous exposition.
- Value: ⭐⭐⭐⭐ Practically significant for content-aware streaming, though generalizability to the broader academic community is somewhat limited.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths](../../AAAI2026/time_series/interpreting_fedspeak_with_confidence_a_llm-based_uncertainty-aware_framework_gu.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[ICLR 2026\] TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](tsrating_time_series_quality_llm.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)

<!-- RELATED:END -->
