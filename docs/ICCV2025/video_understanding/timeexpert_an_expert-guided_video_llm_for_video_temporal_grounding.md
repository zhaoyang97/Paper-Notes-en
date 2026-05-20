---
title: >-
  [Paper Note] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding
description: >-
  [ICCV 2025][Video Understanding][Video Temporal Grounding] This paper proposes TimeExpert — the first MoE-based Video-LLM framework that routes timestamps, saliency scores…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "MoE"
  - "Dynamic Routing"
  - "Video-LLM"
  - "Dense Video Captioning"
  - "Moment Retrieval"
date: 2026-05-08
content_hash: 6d01f7e528a488a5
---

# TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding

**Conference**: ICCV 2025
**arXiv**: [2508.01699](https://arxiv.org/abs/2508.01699)  
**Code**: [Project Page](https://mwxely.github.io/projects/yang2025time/index)  
**Area**: Video Understanding / Temporal Grounding
**Keywords**: Video Temporal Grounding, MoE, Dynamic Routing, Video-LLM, Dense Video Captioning, Moment Retrieval

## TL;DR

This paper proposes TimeExpert — the first MoE-based Video-LLM framework that routes timestamps, saliency scores, and text descriptions to specialized experts via **task-aware dynamic gating** and **token-adaptive routing**, complemented by task-dependent auxiliary losses. TimeExpert achieves state-of-the-art performance across three VTG task categories: Dense Video Captioning, Moment Retrieval, and Video Highlight Detection.

## Background & Motivation

Video Temporal Grounding (VTG) aims to precisely localize event intervals in video based on text queries. VTG outputs contain three heterogeneous components: **timestamps**, **saliency scores**, and **text descriptions**. Existing methods face fundamental limitations:

### Three-Level Deficiencies of Prior Work

**Temporal coarse-grainedness of general Video-LLMs**: Success on coarse-grained tasks such as VideoQA does not transfer well to VTG, which requires precise temporal localization — explicit temporal modeling mechanisms are absent.

**Shared-parameter bottleneck in VTG-specific methods**: Methods such as TimeChat and TRACE introduce temporal tokens but process timestamps, scores, and text tokens **indiscriminately** through the same LLM. Shared parameters lead to **task interference**:
   - Timestamp prediction requires precise numerical regression
   - Saliency scoring requires global importance judgment
   - Text generation requires semantic understanding and linguistic organization
   These three capabilities demand fundamentally different feature representations.

**Static computation allocation**: All tokens receive identical computational resources, ignoring the varying importance of different task tokens.

### Discovery of Implicit Task Preferences

The motivation for TimeExpert stems from a key observation: even in a vanilla MoE **without explicit expert specialization training**, certain experts already exhibit implicit preferences for specific task tokens (Figure 4). For example, a particular expert is consistently activated by score tokens. This suggests that **explicitly reinforcing** such preferences can substantially improve performance.

## Method

### Overall Architecture

The core modification in TimeExpert lies in replacing the LLM backbone with an MoE decoder, while introducing independent temporal encoders, score encoders, and corresponding decoding heads.

1. **Visual Encoder**: Lightweight ViT (438M parameters), compressing each frame into 8 visual tokens via slot-based token compression
2. **Temporal/Score Encoders**: Independent tokenizers (11 numeric tokens + separator token + switch token)
3. **MoE Decoder**: Replaces the monolithic LLM to enable dynamic expert routing

### Task-Aware Dynamic Gating

**Problem with Vanilla MoE**: Fixed top-k selection lacks flexibility and treats all tokens uniformly.

TimeExpert's gating function introduces two innovations:

**1. Cosine similarity replacing linear projection**:

$$s(\mathbf{x}) = \cos(\mathbf{x}, \mathbf{W}_g)$$

**2. Task activation rate weighting**:

$$g(\mathbf{x}) = \text{sign}\left(\sigma\left(\frac{s(\mathbf{x}) + \alpha A_t}{1 + \alpha}\right) - \sigma(\mathbf{G})\right)$$

where:
- $A_t$: the **historical activation rate** for the current task token type — experts frequently activated by a token type are more likely to receive similar tokens
- $\alpha$: task importance scaling factor
- $\mathbf{G} \in \mathbb{R}^K$: **learnable threshold** — a token is routed to an expert only if its similarity exceeds the threshold
- A straight-through estimator is applied to make the sign function differentiable

**Key effect**: Different tokens can activate different numbers of experts (adaptive-k); timestamp tokens may activate more experts (requiring precise processing), while text tokens may activate fewer.

### Token-Adaptive Routing

The routing mechanism comprises three dynamic components:

**1. Task-level routing records**:
- Records the activation timestamps of each expert $\mathbf{R}_E \in \mathbb{R}^K$
- Aggregates embeddings of unmatched tokens $\mathbf{R}_S \in \mathbb{R}^d$
- Maintains the activation rate $A_t$ for each task token type

**2. Adaptive expert addition**: When a large number of task tokens fail to activate any expert, a new expert is added:

$$\mathbf{W}_{g,K+1} = \frac{\mathbf{R}_S}{\|\mathbf{R}_S\|}, \quad \mathbf{G}_{K+1} = 0$$

The representation vector of the new expert is initialized as the mean embedding of unmatched tokens.

**3. Redundant expert pruning**: Experts whose activation rate falls below threshold $\tau_{\min}$ are removed:

$$\mathcal{E}_{\text{remove}} = \{e \mid A_e < \tau_{\min}\}$$

### Task-Dependent Auxiliary Loss

$$\mathcal{L}_{\text{aux}} = \lambda_1 \sum_{e=1}^{K}\left(\frac{A_e}{\sum_j A_j} - \frac{N_e}{\sum_j N_j}\right)^2 + \lambda_2 \sum_{e=1}^{K}\|\mathbf{w}_{g,e}\|_2^2$$

- **Left term (task-aware concentration)**: Encourages experts with higher activation rates to handle more tokens of the same task type
- **Right term (activation regularization)**: Prevents any single expert from being over-activated
- Unlike conventional load-balancing losses, this loss does **not** pursue uniform distribution but instead **reinforces specialization**

### Three-Stage Training

| Stage | Objective | Data Scale |
|-------|-----------|------------|
| Stage 1: Task Module Pre-training | Visual compression layer + task encoders + task heads | 1.9M |
| Stage 2: MoE Decoder Pre-training | Expert routing alignment with VTG task tokens | 0.9M |
| Stage 3: Supervised Fine-tuning | Joint optimization of the full model | 2.3M |

## Key Experimental Results

### Zero-Shot VTG Performance (Table 2)

| Method | Active Params | DVC-SODAc | DVC-F1 | MR-R@1₀.₅ | MR-R@1₀.₇ | VHD-mAP | VHD-HIT@1 |
|--------|--------------|-----------|--------|-----------|-----------|---------|-----------|
| TimeChat | 7B | 1.2 | 12.6 | 32.2 | 13.4 | 14.5 | 23.9 |
| TRACE | 7B | 2.2 | 22.4 | 40.3 | 19.4 | 26.8 | 42.7 |
| **TimeExpert** | **~4–6B** | **2.5** | **23.6** | **42.8** | **20.3** | **29.6** | **46.9** |

TimeExpert comprehensively outperforms TRACE with **fewer active parameters**: MR R@1₀.₅ +2.5%, VHD HIT@1 +4.2%.

### Fine-Tuned VTG Performance (Table 3)

| Method | DVC-CIDEr | DVC-F1 | MR-R@1₀.₅ | MR-R@1₀.₇ |
|--------|-----------|--------|-----------|-----------|
| TRACE | 35.5 | 31.8 | 61.7 | 41.4 |
| **TimeExpert** | **39.0** | **33.5** | **64.1** | **43.3** |

CIDEr +3.5, F1 +1.7, R@1₀.₅ +2.4.

### Ablation Study (Table 5)

| Configuration | DVC-SODAc | MR-R@1₀.₅ | VHD-HIT@1 |
|---------------|-----------|-----------|-----------|
| w/o token-adaptive routing | 2.1 | 40.5 | 42.6 |
| w/o task-dependent loss | 2.4 | 41.3 | 45.2 |
| Vanilla MoE (k=2) | 2.3 | 42.1 | 45.8 |
| Vanilla MoE (k=6) | 2.5 | 42.8 | 46.9 |
| **TimeExpert (adaptive k)** | **2.5** | **42.8** | **46.9** |

- Removing token-adaptive routing causes VHD HIT@1 to drop by 4.3%, representing the largest individual impact
- Adaptive-k achieves performance on par with k=6 while being more computationally efficient (fewer experts activated on average)
- Increasing the number of frames from 8 to 128 yields substantial gains, validating VTG's sensitivity to temporal resolution

## Highlights & Insights

1. **First work to discover and exploit "implicit task preferences" in VTG**: MoE experts naturally tend to handle specific token types; explicitly reinforcing this tendency substantially improves performance.
2. **Dynamic k is more elegant than fixed k**: Different tokens activate different numbers of experts according to their own needs, achieving both efficiency and precision.
3. **The adaptive expert addition/pruning mechanism** offers a new perspective for MoE research: rather than fixing the number of experts before training, the expert count adapts dynamically during training.
4. Independent encoders for temporal/score/text components completely decouple heterogeneous tasks — processing temporal tokens directly through a text tokenizer collapses instruction-following capability.

## Limitations & Future Work

1. The three-stage training pipeline and 5.1M data requirement impose high costs, with data undergoing extensive manual filtering and re-annotation.
2. The MoE architecture of the backbone model ARIA limits the generalizability of the approach — applicability to non-MoE backbones remains unexplored.
3. Thresholds in the expert addition/pruning strategy (e.g., $\tau_{\min}$) require careful tuning.
4. Evaluation is limited to minute-level videos; applicability to hour-level long videos (e.g., films) is not verified.

## Related Work & Insights

- **VTG**: TimeChat, VTimeLLM, HawkEye, TRACE, VTG-LLM
- **Video-LLM**: ARIA, LLaVA-Video, Share-GPT4Video
- **MoE**: DeepSeekMoE, Switch Transformer, Llama-MoE

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A complete MoE innovation chain comprising task-aware dynamic gating, adaptive expert addition/pruning, and task-dependent auxiliary losses
- **Value**: ⭐⭐⭐⭐ — Unified handling of three VTG sub-tasks, though training cost is relatively high
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Zero-shot and fine-tuned evaluations, four datasets, three task categories, detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐ — Clear figures and tables, rigorous formulations

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](moment_quantization_for_video_temporal_grounding.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](../../CVPR2026/video_understanding/cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
