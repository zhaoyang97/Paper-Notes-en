---
title: >-
  [Paper Note] Aha: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead
description: >-
  [NeurIPS 2025][Autonomous Driving][Online Highlight Detection] Aha proposes the first autoregressive framework for Online Highlight Detection (OHD), featuring a decoupled multi-objective prediction head (relevance / informativeness / uncertainty) and a novel Dynamic SinkCache memory mechanism. Under strict causal constraints with no access to future frames, Aha surpasses prior offline methods on TVSum and Mr.Hisum benchmarks by +5.9% and +8.3% mAP, respectively.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Online Highlight Detection
  - Streaming Video
  - Autoregressive
  - Video-Language Model
  - Uncertainty Modeling
date: 2026-05-08
content_hash: 8d23a954b532e33a
---

# Aha: Predicting What Matters Next — Online Highlight Detection Without Looking Ahead

**Conference**: NeurIPS 2025
**arXiv**: [2509.16421](https://arxiv.org/abs/2509.16421)
**Code**: [GitHub](https://github.com/aiden200/Aha-)
**Area**: Autonomous Driving / Video Understanding
**Keywords**: Online Highlight Detection, Streaming Video, Autoregressive, Video-Language Model, Uncertainty Modeling

## TL;DR
Aha proposes the first autoregressive framework for Online Highlight Detection (OHD), featuring a decoupled multi-objective prediction head (relevance / informativeness / uncertainty) and a novel Dynamic SinkCache memory mechanism. Under strict causal constraints with no access to future frames, Aha surpasses prior offline methods on TVSum and Mr.Hisum benchmarks by +5.9% and +8.3% mAP, respectively.

## Background & Motivation

**State of the Field** Highlight Detection (HD) aims to identify key segments from videos. Nearly all modern Transformer-based HD methods rely on offline, full-sequence access. Streaming Video-LLMs can process streaming video, but HD as an auxiliary function yields limited performance.

**Limitations of Prior Work** (1) Offline methods require complete videos and are unsuitable for real-time decision-making (autonomous driving / surveillance / search-and-rescue); (2) Existing Video-LLMs performing HD either require benchmark modification or post-processing smoothing that violates online constraints; (3) No robust method has been specifically designed for OHD.

**Root Cause** Effective HD requires understanding of temporal context, yet online constraints mandate the exclusive use of past and current information — the central challenge is achieving high-accuracy frame-level scoring under strict causal constraints.

**Paper Goals** Design a real-time, task-conditioned online highlight detection framework that uses neither future frames nor modifications to standard benchmarks.

**Starting Point** Establish an autoregressive scoring framework that captures three complementary dimensions — "Is it relevant? Is it novel? Is it certain?" — via a decoupled multi-objective head, coupled with a task-aware memory mechanism for unbounded streaming inference.

**Core Idea** Formalize online HD as an autoregressive multi-objective scoring problem, and employ Dynamic SinkCache to support unbounded-length inference with constant memory overhead.

## Method

### Overall Architecture
Aha consists of four components: (1) a frozen SigLIP visual encoder for frame feature extraction; (2) a single-layer linear projection mapping features to the LLM space; (3) a Qwen2-based autoregressive decoder processing interleaved text and visual token sequences; and (4) three multi-objective prediction heads plus an auxiliary LM head.

### Key Designs

1. **Decoupled Multi-Objective Prediction Heads**:
    - Function: Predict three complementary signals from decoder hidden state $h_t$ — relevance, informativeness, and uncertainty.
    - Mechanism: The relevance head $\hat{r}_t = W_r h_t$ is supervised with Smooth L1 loss plus total variation regularization ($\mathcal{L}_{rel-total} = \mathcal{L}_{rel} + \lambda_{TV}\mathcal{L}_{TV}$); the informativeness head predicts whether a frame introduces new information (supervised with BCE); the uncertainty head predicts Gaussian variance (supervised with NLL plus a variance diversity penalty to prevent mode collapse).
    - Design Motivation: Effective HD requires not only task relevance but also informational novelty and prediction reliability. The decoupled design allows each head to independently optimize its complementary objective.

2. **Dynamic SinkCache**:
    - Function: Enable unbounded streaming inference with constant memory overhead.
    - Mechanism: An extension of SinkCache that dedicates the sink region exclusively to task-description text tokens (~45 tokens), while a sliding window (2,048 tokens) retains recent visual context. Formalized as $\mathcal{K}_t = \{\mathcal{Q}, k_{t-n:t}\}$, requiring only 17% of the memory of a standard KV cache.
    - Design Motivation: Standard KV caches grow linearly with sequence length and lead to OOM; static SinkCache uses generic initial tokens as sinks without task specificity; Dynamic SinkCache preserves task objectives to achieve long-range semantic alignment.

3. **Uncertainty-Aware Fusion Scoring**:
    - Function: Fuse the outputs of the three heads into a final highlight score.
    - Mechanism: A piecewise linear function $\hat{y}_t = \alpha\hat{i}_t + \beta\hat{r}_t - \epsilon(\hat{u}_t - \tau_u)\mathbf{1}[\hat{u}_t > \tau_u]$ performs weighted fusion under low uncertainty and suppresses scores when uncertainty exceeds a threshold.
    - Design Motivation: High uncertainty indicates unreliable model judgment for the current frame, whose influence should therefore be reduced.

### Loss & Training
The total loss is $\mathcal{L}_{total} = \lambda_r\mathcal{L}_{rel-total} + \lambda_i\mathcal{L}_{info} + \lambda_u\mathcal{L}_{unc} + \lambda_{LM}\mathcal{L}_{LM}$, with fixed weights to ensure training stability. Training data includes the HIHD dataset (22,463 videos with YouTube engagement signals derived from Mr.Hisum) and Shot2Story/COIN for informativeness head supervision.

## Key Experimental Results

### Main Results — TVSum Highlight Detection

| Model | Fine-tuned | mAP | Kendall τ | Spearman ρ |
|------|---------|-----|-----------|------------|
| TR-DETR (Offline, SOTA) | Y | 87.1 | - | - |
| LLMVS | N | - | 0.211 | 0.275 |
| **Aha (Zero-shot)** | N | **91.6** | **0.304** | **0.433** |
| **Aha (Domain-adapted)** | N | **93.0** | 0.285 | 0.406 |

### Main Results — Mr.Hisum

| Model | mAP@50 | mAP@15 |
|------|--------|--------|
| PGL-SUM | 55.89 | 27.45 |
| **Aha** | **64.19** | **32.66** |

### Ablation Study

| Configuration | mAP | Note |
|------|-----|------|
| Full Aha | 93.0 | Baseline |
| w/o Relevance Head | 77.3 | −15.7, most critical component |
| w/o Informativeness Head | 83.2 | −9.8, substantial contribution |
| w/o Language Conditioning | 81.2 | −11.8, task text is essential |
| Dynamic SinkCache | 93.0 | Outperforms unbounded cache and standard SinkCache |

### Key Findings
- The fully online model surpasses all fine-tuned offline methods in zero-shot setting (91.6 vs. 87.1 mAP).
- Dynamic SinkCache supports long-video inference over 127K+ tokens using only 17% of standard cache memory.
- Language conditioning is critical for task-oriented HD (removal causes −11.8 mAP).

## Highlights & Insights
- This work is the first to demonstrate that strict online causal constraints need not preclude surpassing offline full-context methods, challenging the intuition that "a complete video is necessary for high-quality HD."
- Anchoring task semantics as long-term memory via Dynamic SinkCache is an elegant and principled design choice.
- The three-head decoupled architecture provides interpretability — relevance, novelty, and prediction reliability can each be analyzed independently.

## Limitations & Future Work
- Engagement signals as a proxy for highlights may introduce bias (e.g., clickbait content).
- The current framework supports only frame-level scoring, lacking segment-level output and structured summarization.
- HIHD is derived from YouTube data, which may limit generalization to safety-critical domains (e.g., medical, military).

## Related Work & Insights
- **vs. TR-DETR**: An offline bidirectional attention method; Aha surpasses it by 6 mAP points using purely causal inference.
- **vs. MMDuet**: A streaming Video-LLM where HD is an auxiliary function; Aha is specifically optimized for HD.
- **vs. StreamingLLM**: Aha's Dynamic SinkCache is a task-aware extension of the StreamingLLM SinkCache.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First strictly online HD framework; results surpassing offline methods are compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ TVSum + Mr.Hisum + ablations + robot video, multi-dimensional validation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; method description is thorough.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to real-time video understanding systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[NeurIPS 2025\] DINO-Foresight: Looking into the Future with DINO](dino-foresight_looking_into_the_future_with_dino.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](../../ICCV2025/autonomous_driving/where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

<!-- RELATED:END -->
