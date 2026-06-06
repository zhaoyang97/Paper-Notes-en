---
title: >-
  [Paper Note] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding
description: >-
  [ICML 2026][Video Understanding][RL] Video-MTR is a reinforcement learning-based **multi-turn reasoning** framework. It guides MLLMs to iteratively select key video segments via a **Gated Dual-Level Reward** mechanism…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "RL"
  - "Multi-turn Reasoning"
  - "Long Video Understanding"
  - "Keyframe Retrieval"
date: 2026-05-08
content_hash: d0e1785a0b1f3007
---

# Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding

**Conference**: ICML 2026  
**arXiv**: [2508.20478](https://arxiv.org/abs/2508.20478)  
**Code**: To be confirmed  
**Area**: Video Understanding / Reinforcement Learning / Multimodal Reasoning  
**Keywords**: RL, Multi-turn Reasoning, Long Video Understanding, Keyframe Retrieval

## TL;DR
Video-MTR is a reinforcement learning-based **multi-turn reasoning** framework. It guides MLLMs to iteratively select key video segments via a **Gated Dual-Level Reward** mechanism, achieving SOTA performance in long video understanding using only **8K data**, outperforming baseline methods by two orders of magnitude in data efficiency (which require 257K to 4.4M samples).

## Background & Motivation

**Background**: Long video understanding has become a critical application for MLLMs. Current approaches generally follow two paradigms: (1) Instruction fine-tuning, which relies on uniform sampling and large-scale data; (2) Agent-based paradigms, which integrate external VLM tools, introducing complex heterogeneous components.

**Limitations of Prior Work**:
- Uniform sampling strategies fail to handle information loss in long videos as they cannot adaptively locate key segments.
- Reliance on external VLMs leads to high system complexity, suboptimal tool utilization strategies, and a lack of end-to-end training.
- Existing RL methods mostly employ single-turn reasoning or sparse rewards based only on the final answer, making it difficult to guide multi-turn intermediate behaviors.

**Key Challenge**: Long videos contain multiple events and long-term temporal dependencies. Existing methods either suffer from information loss due to fixed sampling or sacrifice efficiency due to external tool dependency. How can adaptive, multi-turn key segment retrieval be achieved within a limited computational budget?

**Goal**:
1. Propose a pure RL post-training paradigm without the need for large-scale supervised fine-tuning.
2. Design a fine-grained multi-turn reward mechanism to guide intermediate frame retrieval behaviors.
3. Achieve SOTA performance with minimal data (8K vs. 257K–4.4M).

**Key Insight**: Reframe long video understanding as a **multi-turn interactive decision-making process**, where the MLLM acts as an agent, the video acts as the environment, and the model iteratively retrieves key segments to update the context. This simulates the natural human process of watching long videos: starting with a global understanding, followed by targeted reviews of details, and finally synthesizing evidence.

**Core Idea**: Use a **Gated Dual-Level Reward** (coupling goal rewards with intermediate frame rewards) combined with **Exploration Bootstrapping** (no cold-start SFT) to enable MLLMs to learn multi-turn evidence seeking via pure RL while significantly reducing data requirements.

## Method

### Overall Architecture
Long video understanding is reframed as an MDP:
- **State** $s_k = (\mathcal{F}_{k-w}, x_{k-w}, \ldots, \mathcal{F}_k, x_k)$: Past $w$ turns of interaction + current set of observed frames.
- **Action** $a_k$: Retrieve a new segment (specifying a timestamp range) or output the final answer.
- **Environment Response**: Returns a new sampled frame set $\mathcal{F}_{k+1}$ based on the retrieval action, or a reward based on answer correctness.
- **Trajectory** $\tau = \{(\mathcal{F}_k, x_k, y_k)\}_{k=0}^K$.

Initially, $n_0$ frames are uniformly sampled. Each subsequent turn allows up to $K_{\max} = 3$ retrievals. The model generates reasoning text + executable actions; the system determines whether to continue retrieval or conclude the answer.

### Key Designs

1. **Multi-turn Reasoning Paradigm**:
    - **Function**: Overcomes the information loss bottleneck of uniform sampling by iteratively retrieving and adaptively locating key segments.
    - **Mechanism**: Differing from single-turn processing of fixed frame sets, the MLLM actively retrieves new segments based on evolving context (processed frames + reasoning progress). The first turn uses 50% of the sampling budget, and subsequent turns use 25% each, ensuring the total count stays below the limit—enabling dense sampling for complex regions and coarse sampling for simpler ones.
    - **Design Motivation**: To solve the failure of uniform sampling in locating critical details; multi-turn gains grow with task complexity and video length (+3.8% overall, +8.1% for multi-detail tasks; +6.3% for long videos on VideoMME vs. +1.7% for short videos).

2. **Gated Dual-Level Reward Mechanism (Core Innovation)**:
    - **Function**: Guides multi-turn optimization using fine-grained, multi-level reward signals to prevent reward hacking.
    - **Mechanism**: Reward is layered: **Trajectory-level** $R_{\text{acc}}$ is 1 for a correct final answer and 0 otherwise. **Turn-level** $R_{\text{fms}}^k$ is an IoU improvement reward (max 0.5, half of $R_{\text{acc}}$) between retrieved frames and ground truth, encouraging marginal improvement and penalizing redundancy. **Format-level** $R_{\text{format}}^k = 0.1$ rewards adherence to formatting. **Gating Mechanism**: Intermediate rewards $\sum_{k=0}^{K-1}(R_{\text{fms}}^k + R_{\text{format}}^k)$ are activated only if $R_{\text{acc}} > 0$, ensuring frame retrieval behavior is reinforced only when it leads to a correct answer. Final combination: $R(\tau) = \mathbb{1}_{\{R_{\text{acc}} > 0\}} \cdot \sum_{k=0}^{K-1}(R_{\text{fms}}^k + R_{\text{format}}^k) + R_{\text{acc}} + R_{\text{format}}^K$.
    - **Design Motivation**: Absolute terminal rewards fail to guide intermediate segment selection (-4.4% on LVBench); unconstrained intermediate rewards lead to "gaming" for cumulative turn rewards rather than accuracy. Gating forces the coupling of frame retrieval and answer accuracy.

3. **Data-efficient Training Strategy**:
    - **Function**: Enables RL post-training using only 8K samples.
    - **Mechanism**: **Data Curation**: Reuses existing temporal localization datasets (NExT-GQA with native QA + timestamps, and QVHighlights converted via GPT-4o). Selection of samples with short key segments yields an 8K compact dataset. **Exploration Bootstrapping**: Since pre-trained MLLMs lack original active retrieval capabilities (cold start) without SFT warm-up, a small reward is automatically given per mini-batch if the retrieval rate falls below a threshold. Once retrieval becomes regular, this is disabled, and learning is driven by dual-level signals.
    - **Design Motivation**: Large-scale SFT data is expensive; RL converges rapidly to multi-turn strategies via precise data curation (quality over quantity) and adaptive exploration.

### Training Strategy
PPO algorithm is used, treating multi-turn trajectories as a single token sequence. Two-tier discount factors are applied: inter-turn $\gamma_{\text{turn}} = 0.95$ (propagating final signals back to early decisions) and intra-turn $\gamma_{\text{token}} = 1.0$. Batch size is 32, actor $lr$ $1 \times 10^{-6}$, critic $lr$ $1 \times 10^{-5}$, using 8 A800-80GB GPUs.

## Key Experimental Results

### Main Results

| Model | Params | Frames | VideoMME | MLVU | LongVideoBench | LVBench | EgoSchema |
|------|------|----|--------|------|----------------|---------|-----------|
| GPT-4o | — | 384 | 71.9 | 54.9 | 66.7 | 48.9 | 72.2 |
| Gemini-1.5-Pro | — | 0.5fps | 75.0 | — | 64.0 | 33.1 | 71.1 |
| LongVA-7B | 7B | 256 | 52.6 | 41.1 | 47.8 | 37.9 | — |
| Video-R1-7B | 7B | 32 | 59.3 | 45.4 | — | 35.9 | 48.8 |
| Video-R1-7B | 7B | 64 | 61.4 | 47.6 | — | 38.0 | 51.8 |
| **Video-MTR** | 7B | 32 | 59.0 | 48.4 | 52.3 | 38.2 | 62.4 |
| **Video-MTR** | 7B | 64 | 62.2 | 49.8 | 54.8 | 41.8 | 63.4 |
| **Video-MTR** | 7B | 80 | **62.7** | **50.4** | **57.1** | **42.3** | **68.8** |

Under the same frame budget, Video-MTR significantly outperforms Qwen2.5-VL-7B (+5.4 to +6.3% at 32 frames). Video-MTR with 80 frames nearly matches Qwen2.5-VL-7B using 768 frames. Data efficiency is exceptional: 8K data vs. 2M for VideoChat2 or 260K for Video-R1.

### Ablation Study

| Configuration | VideoMME Short | Mid | Long | Total | LVBench |
|------|-----------|-----|-----|-----|---------|
| **Full Model** | 74.8 | 60.6 | 52.7 | 62.7 | 42.3 |
| w/o Dual-Level Reward | 69.4 | 56.2 | 49.4 | 58.3 | 37.7 |
| **Gain (Loss)** | -5.4 | -4.4 | -3.3 | -4.4 | **-4.6** |
| Single-turn Baseline | 68.8 | 54.8 | 47.9 | 57.2 | 35.3 |
| **Gain (Loss)** | -6.0 | -5.8 | -4.8 | -5.5 | **-7.0** |

### Key Findings
- **Differential Effects of Multi-turn Reasoning**: On MLVU, multi-turn gains show a linear relationship with task difficulty: +3.8% overall, +7.5% for single details, and +8.1% for multiple details.
- **Scalability with Video Length**: On VideoMME (32-frame budget), gains are +4.6% for short, +5.3% for medium, and **+6.3% for long videos**, highlighting that multi-turn reasoning is most effective in long-form scenarios.
- **Prevention of Reward Hacking**: Without gating, the model learns a false strategy of increasing turns just to accumulate rewards without improving accuracy. With gating, it learns to retrieve only as needed.
- **Exploration Bootstrapping Effectiveness**: RL directly enables multi-turn capabilities without SFT warm-up. This was also successful on Qwen2.5-VL-3B, demonstrating paradigm scalability.

## Highlights & Insights
- **Paradigm Innovation**: Reframes long video understanding as a multi-turn MDP for the first time, breaking the single-turn sampling bottleneck. The coefficient of reward for multi-turn processes is significantly larger on complex tasks compared to single-turn RL.
- **Ingenious Reward Design**: The combination of dual-level rewards and gating is elegant—preventing reward hacking while maintaining fine-grained supervision. Ablation proves both are essential (each contributing >4% drop if removed).
- **Data Efficiency Breakthrough**: Utilizing 8K data to compete with million-scale methods proves that "high-quality data + refined rewards" can far exceed "large-scale low-quality" data. This is highly portable for RL post-training.
- **Alignment with Human Cognition**: Iterative multi-turn reasoning aligns with natural human video consumption, enhancing interpretability (case studies show clear 3-turn reasoning trajectories).

## Limitations & Future Work
- Due to compute constraints, training was limited to 80 frames; future work aims to expand to hundreds of frames.
- Focus is currently on Video QA; expansion to video captioning or event detection remains unexplored.
- RL Training Instability: Despite dual-level rewards and gating, inherent RL variance may still affect stability across different seeds.
- Dependence on Temporal Labels: Dual-level rewards require frame-level ground truth; transferability might decrease in new domains with low-quality labels.
- Future directions: Explore weak supervision for dual-level rewards; research multi-task RL (QA + Caption + Detection); analyze marginal returns of the turn limit.

## Related Work & Insights
- **vs. Uniform Sampling Methods** (VideoChat2 / Video-LLaVA): Fixed sampling lacks adaptivity. Video-MTR achieves dynamic localization via multi-turn retrieval + fine-grained rewards, improving long video accuracy by 23pp (39.3% → 57.1%).
- **vs. External Tool Methods** (VideoAgent / VideoMemAgent): These require multiple external VLMs (captioning, tracking), leading to system complexity and poor end-to-end training. Video-MTR unifies reasoning within one model, eliminating tool coupling while remaining efficient.
- **vs. RL Methods** (Video-R1): While both use RL, Video-R1 requires 260K SFT data to achieve multi-turn capabilities. Video-MTR's pure RL approach with 8K samples proves that multi-level rewards + gating act as a multiplier for data efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First to combine multi-turn reasoning with dual-level gated rewards for long videos; innovative paradigm with clever anti-hacking mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers five major benchmarks (VideoMME / MLVU / LongVideoBench / LVBench / EgoSchema) with detailed ablation and stratified analysis.
- Writing Quality: ⭐⭐⭐⭐  Clear logic, precise methodology, and intuitive visualizations; related work section is slightly concise.
- Value: ⭐⭐⭐⭐⭐  8K data makes it ready for industrial application; the dual-level gated reward paradigm is transferable to other long-sequence RL tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Multi-Agent Perception-Action Alliance for Efficient Long Video Reasoning](../../CVPR2026/video_understanding/a_multi-agent_perception-action_alliance_for_efficient_long_video_reasoning.md)
- [\[AAAI 2026\] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](../../AAAI2026/video_understanding/reason_reinforced_causal_search_with_information_bottleneck_for_video_understand.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/video_understanding/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](../../CVPR2026/video_understanding/vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)

</div>

<!-- RELATED:END -->
