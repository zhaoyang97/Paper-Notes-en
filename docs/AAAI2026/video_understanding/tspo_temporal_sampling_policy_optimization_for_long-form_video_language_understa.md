---
title: >-
  [Paper Note] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding
description: >-
  [AAAI 2026][Video Understanding][Long-form video understanding] This paper formulates keyframe selection and language generation as a joint decision-making process…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Long-form video understanding"
  - "keyframe sampling"
  - "reinforcement learning"
  - "temporal policy optimization"
  - "video multimodal large language models"
date: 2026-05-08
content_hash: cc84218d53bfdf0c
---

# TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding

**Conference**: AAAI 2026
**arXiv**: [2508.04369](https://arxiv.org/abs/2508.04369)
**Code**: [https://github.com/Hui-design/TSPO](https://github.com/Hui-design/TSPO)
**Area**: Video Understanding / Multimodal Large Language Models
**Keywords**: Long-form video understanding, keyframe sampling, reinforcement learning, temporal policy optimization, video multimodal large language models

## TL;DR

This paper formulates keyframe selection and language generation as a joint decision-making process, and optimizes a lightweight temporal agent's sampling policy end-to-end via GRPO-based reinforcement learning. It achieves state-of-the-art results on four long-form video understanding benchmarks (LongVideoBench +5.0%, MLVU +6.0% on LLaVA-Video-7B) and transfers zero-shot to other Video-MLLMs.

## Background & Motivation

**Background**: Video-MLLMs (e.g., LLaVA-Video, Qwen2.5-VL) have achieved notable progress in video understanding, but processing long videos is constrained by LLM context length and training cost, necessitating sparse frame sampling. The dominant approaches are uniform sampling or training-free keyframe search using pretrained models such as DINOv2 or LLaVA-1.5.

**Limitations of Prior Work**: (1) Uniform sampling frequently misses query-relevant information. (2) Training-free keyframe search methods (e.g., LongVU with DINOv2-1B, CoS with LLaVA-1.5-13B) are limited by the cross-modal understanding capacity of pretrained feature extractors and incur high inference overhead (CoS requires a 13B-scale model). (3) None of these methods can be further optimized through training.

**Key Challenge**: Building a trainable frame sampling method faces two fundamental challenges: (1) *Lack of supervision*: General video understanding training data lacks frame-level annotations, providing no precise localization supervision. (2) *Non-differentiability*: Frame sampling is a discrete subset selection problem whose outputs are frame indices rather than continuous variables, precluding optimization via backpropagation in standard SFT.

**Goal**: How to design a trainable sparse frame sampling method that can be optimized end-to-end, requires no frame-level annotations, and significantly improves long-form video understanding performance?

**Key Insight**: Inspired by DeepSeek-R1's use of GRPO to enhance reasoning, the authors model keyframe selection and language generation as a joint decision-making process—the discrete selection of frames is analogous to actions in RL, and the accuracy of the MLLM's answers serves directly as a reward signal to supervise the sampling policy.

**Core Idea**: A lightweight event-aware temporal agent performs probabilistic keyframe selection and is jointly optimized with the Video-MLLM's language generation via GRPO, using answer accuracy as the reward signal for end-to-end training of the sampling policy.

## Method

### Overall Architecture

The TSPO framework consists of three components: (1) an **Event-aware Temporal Agent** that uses CLIP features and local window attention to capture event–query relevance, outputting per-frame selection probabilities; (2) the **TSPO reinforcement learning optimization**, which models frame selection and language generation as a joint policy and optimizes only the temporal agent while freezing the Video-MLLM; and (3) a **dual-style training data pipeline with dual reward functions**. Given a long video and a text query, the temporal agent selects keyframes from candidate frames, which are then fed into the frozen Video-MLLM to generate the answer.

### Key Designs

1. **Event-aware Temporal Agent**

    - **Function**: Probabilistically selects the most query-relevant keyframes from candidate frames.
    - **Mechanism**: Takes CLIP frame-level visual features $\mathbf{F}_f \in \mathbb{R}^{T_c \times D}$ and text features $\mathbf{F}_t \in \mathbb{R}^{1 \times D}$ as input. Local window attention (window size $w=12$) combined with sinusoidal positional encoding captures intra-event temporal dependencies to produce event representations $\mathbf{F}_e$. The final similarity score fuses event-level and frame-level similarities: $S = \text{Sim}_{event}(\mathbf{F}_e, \mathbf{F}_t) + \text{Sim}_{frame}(\mathbf{F}_f, \mathbf{F}_t)$. Gumbel-Softmax sampling is used for exploration: $\mathcal{P}, \mathcal{I} = \text{TopK}(\text{Softmax}(S/\tau + \gamma))$, where $\gamma \sim \text{Gumbel}(0,1)$.
    - **Design Motivation**: CLIP provides only frame-level features and lacks cross-frame event-level understanding. Local window attention enables neighboring frames to interact, forming event-level temporal representations. Gumbel-Softmax ensures differentiability while providing the exploration required by RL. Temperature annealing ($\tau$ decays from 0.025 to 0.01) gradually shifts training from exploration to exploitation. The entire agent has only 3.5M learnable parameters, making it extremely lightweight.

2. **Temporal Sampling Policy Optimization (TSPO Algorithm)**

    - **Function**: End-to-end optimization of the temporal agent's sampling policy without frame-level annotations.
    - **Mechanism**: The joint policy is factored as $\pi(\mathbf{o}, \mathbf{V_s} | \mathbf{q}, \mathbf{V}_c) = \pi_l(\mathbf{o} | \mathbf{q}, \mathbf{V}_s, \mathbf{V}_c) \cdot \pi_{ts}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)$. Since the Video-MLLM is frozen, the language generation policy ratio equals 1, and the TSPO objective reduces to optimizing only the temporal agent: $\mathcal{J}^*_{tspo} = \frac{1}{G}\sum_{i=1}^{G}\frac{\pi_{ts}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)}{\pi_{ts_{old}}(\mathbf{V}_s | \mathbf{q}, \mathbf{V}_c)} A_i$. For each question, $G$ different keyframe combinations are sampled; each generates an answer and receives a reward. The within-group relative advantage $A_i$ is computed to update the policy.
    - **Design Motivation**: Directly training the frame selector with SFT (the SFT* baseline in experiments) is less effective than TSPO, because SFT gradients originate from language generation cross-entropy loss, providing only indirect supervision for frame selection. The GRPO framework allows the sampling policy to receive direct reward signals, and estimates the advantage function through multiple sampled rollouts, enabling more effective policy optimization.

3. **Dual-style Training Data Construction Pipeline**

    - **Function**: Provides high-quality long-form video training data for TSPO.
    - **Mechanism**: (1) *Comprehensive temporal data*: Multi-choice QA samples longer than one minute are filtered from LLaVA-Video-178K, excluding those answerable with 4 frames (too easy) or unanswerable with 64 frames (too hard). (2) *Video needle-in-a-haystack data*: Target videos are concatenated with irrelevant videos and shuffled to form 10–60 minute long videos; Qwen2.5-VL generates event descriptions for target segments, which are converted into multi-choice questions to simulate long-range temporal localization. The combined dataset constitutes **TSPO-10K**.
    - **Design Motivation**: Comprehensive temporal data builds general video understanding capability, while needle-in-a-haystack data trains long-range temporal localization. The two are complementary—ablations show that needle-in-a-haystack data alone improves LongVideoBench but hurts VideoMME, while their combination achieves the best overall results.

### Loss & Training

A dual reward mechanism is used: answer accuracy reward $R_A = \mathbf{1}(y = \bar{y})$ (whether the selected option is correct) and temporal localization reward $R_T = T_t / T_a$ (fraction of sampled frames falling within the target video). The total reward for comprehensive temporal data is $R_A + 1$, and for needle-in-a-haystack data is $R_A + R_T$. Training uses 8×A800 GPUs, learning rate 5e-4, batch size 1, for 1 epoch. During training, 16 frames are selected; at inference, 64 frames are selected.

## Key Experimental Results

### Main Results

| Model | LongVideoBench | MLVU | VideoMME (Long) | LVBench |
|-------|---------------|------|-----------------|---------|
| LLaVA-Video-7B (uniform) | 58.9 | 70.3 | 53.6 | 40.2 |
| LLaVA-Video-7B + TSPO | **63.9** (+5.0) | **76.3** (+6.0) | **54.7** (+1.1) | **45.3** (+5.1) |
| Qwen2.5VL (uniform) | 59.0 | 65.1 | 53.3 | 38.3 |
| Qwen2.5VL + TSPO | **64.2** (+5.2) | **74.3** (+9.2) | **56.4** (+3.1) | **46.4** (+8.1) |
| LLaVA-Video + CoS (13B selector) | 58.9 | 71.4 | 53.8 | — |
| LLaVA-Video + AKS (0.5B selector) | 62.7 | — | 54.0 | — |

### Ablation Study

| Configuration | LongVideoBench | VideoMME | Notes |
|---------------|---------------|---------|-------|
| Baseline (uniform) | 58.9 | 64.4 | No training |
| Comprehensive temporal only + $R_A$ | 62.8 | 65.5 | Substantial gain on general understanding |
| Needle-in-a-haystack only + $R_T$ | 63.4 | 64.6 | Localization improves; understanding slightly drops |
| Dual data + dual reward | **63.9** | **65.5** | Optimal combination |
| SFT* (30K data) | 62.8 | 64.8 | TSPO (10K) still outperforms |

### Key Findings

- **Zero-shot cross-architecture transfer**: The temporal agent trained on LLaVA-Video transfers directly to Qwen2VL/Qwen2.5VL, yielding average improvements of 4.5%/6.3%, and even transfers to LLaVA-Video-72B with a 3.6% gain.
- **Inference efficiency advantage**: TSPO keyframe extraction takes only 1.2 seconds (vs. 28.4 seconds for CoS), saving 90% of inference time. Using 32 frames approaches the performance of 64-frame uniform sampling, halving both token count and LLM inference time.
- **TSPO outperforms SFT**: Even with 3× more training data (30K vs. 10K), SFT* underperforms TSPO, validating that RL exploration with reward feedback is better suited than supervised learning for discrete sampling optimization.
- **Modest gains on VideoMME**: VideoMME emphasizes holistic video understanding rather than keyframe localization, which precisely highlights that TSPO's advantage lies in long-range temporal localization.

## Highlights & Insights

- **Recasting frame sampling as an RL problem** is the paper's most fundamental contribution. Frame sampling is inherently discrete and unsupervised, and the GRPO framework perfectly matches this setting—no frame-level labels are required; answer accuracy serves as the reward signal, and the advantage function is estimated via multiple sampled rollouts. This modeling paradigm generalizes to any scenario requiring discrete selection from a large candidate set.
- **Freezing the MLLM and optimizing only the sampler** is a highly pragmatic design choice: (1) it avoids the training cost of large models; (2) an already SFT-trained MLLM can answer correctly when given the right frames, so only the "frame selection" step needs optimization; (3) the trained sampler can be plug-and-play across different MLLMs.
- **The 3.5M-parameter ultra-lightweight temporal agent** outperforms CoS's 13B-parameter MLLM-based selector while using 3,700× fewer parameters and achieving 23× faster inference.

## Limitations & Future Work

- The candidate frame count is fixed at 128 (1 FPS), which may be insufficient for multi-hour ultra-long videos and motivates exploring more efficient multi-stage sampling strategies.
- The Video-MLLM remains frozen throughout; joint optimization of the sampler and the MLLM has not been explored, likely due to prohibitive computational cost.
- Training data is limited to 10K samples with videos up to 60 minutes; generalization to longer videos (e.g., feature-length films) remains unverified.
- The reward design is relatively simple (accuracy + temporal localization); finer-grained reward signals such as visual information density or frame diversity could be incorporated.

## Related Work & Insights

- **vs. LongVU (shen2024longvu)**: LongVU uses DINOv2-1B for cross-frame deduplication, a training-free approach with 285× more parameters than TSPO's temporal agent; TSPO surpasses it by 10.9% on MLVU.
- **vs. CoS (hu2025cos)**: CoS uses LLaVA-1.5-13B for per-frame scoring, incurring extremely high inference cost (28.4s vs. TSPO's 1.2s) and cannot be trained further. TSPO outperforms it by 5.0% on LongVideoBench.
- **vs. TPO**: TPO uses DPO to improve the LLM's temporal reasoning capability but does not optimize frame selection. TSPO directly optimizes at the frame selection level and is complementary to TPO.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to model frame sampling as an RL decision-making process and optimize it end-to-end with GRPO, resolving the fundamental challenges of unsupervised and non-differentiable discrete selection.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, cross-MLLM transfer validation across three architectures, and comprehensive ablations covering data, rewards, SFT comparison, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear, though the GRPO derivation could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical: the lightweight plug-and-play sampler has direct engineering value for long-form video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](../../ICCV2025/video_understanding/hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[CVPR 2026\] VideoARM: Agentic Reasoning over Hierarchical Memory for Long-Form Video Understanding](../../CVPR2026/video_understanding/videoarm_agentic_reasoning_over_hierarchical_memory_for_long-form_video_understa.md)
- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](../../ICCV2025/video_understanding/videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[CVPR 2026\] DIvide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding](../../CVPR2026/video_understanding/divide_then_ground_adapting_frame_selection_to_query_types_for_long-form_video_u.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](causality_matters_how_temporal_information_emerges_in_video_language_models.md)

</div>

<!-- RELATED:END -->
