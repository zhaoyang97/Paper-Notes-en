---
title: >-
  [Paper Note] Less is More: Empowering GUI Agent with Context-Aware Simplification
description: >-
  [ICCV 2025][LLM Agent][GUI Agent] This paper proposes SimpAgent — a context-aware simplification framework that achieves SOTA on multiple GUI navigation benchmarks while reducing FLOPs by 27%, via masking-based element pruning (randomly masking irrelevant element regions during training) and consistency-guided history compression (directly dropping historical visual tokens at intermediate LLM layers with a KL divergence consistency constraint).
tags:
  - "ICCV 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Context Simplification"
  - "Element Pruning"
  - "History Compression"
  - "Computational Efficiency"
date: 2026-05-08
content_hash: 3aa1f43773d9eb7e
---

# Less is More: Empowering GUI Agent with Context-Aware Simplification

**Conference**: ICCV 2025
**arXiv**: [2507.03730](https://arxiv.org/abs/2507.03730)  
**Code**: [github.com/JiuTian-VL/SimpAgent](https://github.com/JiuTian-VL/SimpAgent)  
**Area**: LLM Agent
**Keywords**: GUI Agent, Context Simplification, Element Pruning, History Compression, Computational Efficiency

## TL;DR

This paper proposes SimpAgent — a context-aware simplification framework that achieves SOTA on multiple GUI navigation benchmarks while reducing FLOPs by 27%, via masking-based element pruning (randomly masking irrelevant element regions during training) and consistency-guided history compression (directly dropping historical visual tokens at intermediate LLM layers with a KL divergence consistency constraint).

## Background & Motivation

### Problem Definition

GUI Agents must generate actions (clicks, swipes, etc.) in graphical interfaces to complete complex tasks, given a task goal, the current screenshot, and historical context. Pure vision-based approaches represent a promising direction but face severe challenges in context modeling.

### Limitations of Prior Work

Existing pure-vision GUI Agents primarily pursue large-scale pretraining data to improve GUI understanding, overlooking two critical challenges:

**High density and loose correlation of element context**: Each screenshot contains on average 56–180 UI elements, yet these elements are loosely correlated. Experiments show that masking irrelevant elements in half the screenshot actually improves performance (66.0% → 68.8%).

**High redundancy in historical context**: Introducing 4 historical frames increases computational overhead by 3.4×, yet yields only a 3.0% performance gain. Historical visual information is severely redundant.

### Root Cause

The cost-effectiveness tradeoff between pretraining on more data versus optimizing context modeling. For example, OS-Atlas uses 1.9M pretraining samples for only a 0.8% improvement, whereas SimpAgent achieves comparable or superior gains without any pretraining data.

## Method

### Overall Architecture

SimpAgent consists of two core components:
1. **Training phase**: Irrelevant element regions in screenshots are randomly pruned via masking operations.
2. **Training and inference phase**: Historical visual tokens are dropped at an intermediate LLM layer, with a consistency loss guiding the compression process.

### Key Designs

#### 1. **Masking-based Element Pruning**
- **Function**: During training, a rectangular region of the current screenshot is randomly masked.
- **Mechanism**:
    - Rectangle dimensions $h, w \sim U(a, b)$; center point $p_c$ sampled from a uniform distribution.
    - The masking operation is applied with probability $p$; no masking is applied at inference.
    - Pixels in the masked region are set to a fixed value $v$.

$$o_t^m = \mathcal{M}(o_t) = \begin{cases} v, & (x,y) \in \mathcal{R} \\ o_t(x,y), & \text{otherwise} \end{cases}$$

- **Design Motivation**:
    - The target action region occupies on average only 2% of the screenshot area; irrelevant elements dominate.
    - UI design follows modular principles, resulting in loose inter-element coupling, so masking irrelevant elements does not impair comprehension.
    - Eliminates the need for complex element relationship modeling — no explicit identification of irrelevant elements is required.

#### 2. **LLM-based History Dropping**
- **Function**: All historical visual tokens are dropped directly at layer $k$ of the LLM.
- **Mechanism**: Shallow LLM layers compress visual information into adjacent action tokens via causal self-attention (verified through attention visualization), so historical visual tokens can be discarded while preserving information within action tokens.
- **Design Motivation**:
    - Requires no additional parameters.
    - Unlike the attention mask adjustment approach of VoCo-LLaMA, this method is compatible with efficient attention implementations such as FlashAttention.
    - Achieves a 27% reduction in FLOPs.

#### 3. **Consistency Guidance**
- **Function**: During training, both a truncated branch and a full branch are maintained simultaneously; the KL divergence between their output distributions is minimized.
- **Mechanism**:

$$\mathcal{L} = -\mathbb{D}_{KL}[\pi_\theta(\tilde{a}_t|o_t^m, H_t, G) \| \pi_\theta(a_t|o_t^m, H_t^c, G)] - \sum_t \log \pi_\theta(\tilde{a}_t|o_t^m, H_t, G) - \sum_t \log \pi_\theta(a_t|o_t^m, H_t^c, G)$$

- **Design Motivation**: Naive dropping is implicit compression and incurs information loss (1.7% drop on AITW). Consistency guidance provides explicit supervision, reducing compression loss to near zero.

### Loss & Training

The total training objective comprises three components:
1. Cross-entropy loss on the full branch.
2. Cross-entropy loss on the truncated branch.
3. KL divergence consistency loss between the two branches.

At inference, only the truncated branch is used, achieving a 27% reduction in FLOPs.

## Key Experimental Results

### Main Results

| Method | Pretraining Data | Params | AITW | Mind2Web-CT | GUI-Odyssey |
|--------|-----------------|--------|------|-------------|-------------|
| SeeClick | 850K | 9.6B | 59.3 | 25.5 | - |
| ShowUI | 256K | 2B | 70.0 | 37.2 | - |
| OdysseyAgent | - | 9.6B | - | - | 74.3 |
| Qwen2VL (baseline) | - | 2B | 69.0 | 46.7 | 74.9 |
| **SimpAgent** | **-** | **2B** | **71.3** | **47.1** | **76.0** |
| **SimpAgent-M** | **-** | **2B** | **71.5** | **48.7** | **77.4** |

AndroidControl results:

| Method | Pretraining Data | Params | SR |
|--------|-----------------|--------|-----|
| OS-Atlas | 1.9M | 4B | 67.5 |
| Qwen2VL (baseline) | - | 2B | 68.4 |
| **SimpAgent** | **-** | **2B** | **69.1** |

### Ablation Study

| Component | FLOPs (T) | AITW | GUI-Odyssey | Notes |
|-----------|-----------|------|-------------|-------|
| Baseline (Qwen2VL) | 11.90 | 69.0 | 74.9 | Full history |
| + History compression | 8.71 (↓27%) | 67.3 (↓1.7) | 71.8 (↓3.1) | Implicit compression, information loss |
| + Consistency guidance | 8.71 | 68.9 (↑1.6) | 73.7 (↑1.9) | Explicit guidance recovers performance |
| + Element pruning | 8.71 | **71.3 (↑2.4)** | **76.0 (↑2.3)** | All three components exceed baseline |

### Comparison with Other Compression Methods

| Method | Step SR | FLOPs (T) |
|--------|---------|-----------|
| No compression | 69.0 | 11.90 |
| TokenMerger | 68.9 | 9.28 |
| Victor | 67.6 | 9.28 |
| FastV-50 | 66.0 | 10.15 |
| FastV-0 | 63.8 | 8.71 |
| **Ours** | **68.9** | **8.71** |

### Key Findings

1. **Element pruning yields substantial gains**: Performance improves even when 50% of the screenshot is masked, confirming that a large proportion of UI elements constitute "noise."
2. **Uniform distribution outperforms Gaussian distribution**: The spatial distribution of UI elements is more complex than expected; a simple uniform masking strategy is more robust.
3. **Consistency guidance is critical for successful compression**: Recovers the 1.7% performance drop caused by compression to near-lossless levels.
4. **Outperforms methods requiring large pretraining data without any pretraining**: SimpAgent-2B's 0.7% gain is comparable to OS-Atlas-4B's 0.8% gain achieved with 1.9M pretraining samples.

## Highlights & Insights

1. **Rigorous analysis of GUI screenshot characteristics**: Systematically quantifies element density (56–180 elements per screenshot) and historical redundancy (3.4× computational overhead yields only 3.0% performance gain).
2. **Elegant masking-based pruning strategy**: No identification of irrelevant elements is needed — random masking eliminates them with high probability since the target action region occupies only 2% of the screenshot area.
3. **Asymmetric train-inference design**: Masking during training improves robustness, while full screenshots are retained at inference to preserve complete information.
4. **Significant computational efficiency**: A 27% FLOPs reduction is achieved simultaneously with performance improvement, substantiating the "less is more" claim.

## Limitations & Future Work

1. The masking strategy is data-agnostic and does not exploit UI structural information such as component hierarchies or layout patterns.
2. Validation is limited to Qwen2-VL-2B as the backbone; generalizability to larger models remains unknown.
3. History compression discards all historical visual tokens, potentially losing critical visual change information.
4. The choice of dropping layer $k$ is empirical ($k=3$), lacking an adaptive selection mechanism.
5. Evaluation is restricted to mobile and web scenarios; applicability to desktop applications has not been verified.

## Related Work & Insights

- FastV demonstrates that shallow LLM layers can aggregate visual features into anchor tokens; SimpAgent builds on this insight to develop a direct dropping strategy.
- ShowUI and Iris attempt to eliminate backgrounds using low-level visual cues, but show limited effectiveness in high-density element scenarios.
- OdysseyAgent proposes a history resampling module to compress history, but introduces additional parameters and neglects multimodal interaction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The masking-based pruning idea is simple and intuitive yet counterintuitively effective; consistency guidance is a solid technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, 10+ baseline comparisons, comprehensive ablation and sensitivity analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem motivation is deeply analyzed; pilot experiments are convincing.
- **Value**: ⭐⭐⭐⭐ — Provides a computationally efficient training paradigm for GUI Agents; the "no pretraining data required" aspect offers substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration](../../CVPR2025/llm_agent/gui-xplore_empowering_generalizable_gui_agents_with_one_exploration.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[ICLR 2026\] Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management](../../ICLR2026/llm_agent/sculptor_empowering_llms_with_cognitive_agency_via_active_context_management.md)
- [\[CVPR 2026\] ReFAct: Empowering Multimodal Web Agents with Visual and Context Focusing](../../CVPR2026/llm_agent/refact_empowering_multimodal_web_agents_with_visual_and_context_focusing.md)

</div>

<!-- RELATED:END -->
