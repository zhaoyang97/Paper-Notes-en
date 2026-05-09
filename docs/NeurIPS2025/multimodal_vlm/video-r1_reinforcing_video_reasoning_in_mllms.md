---
title: >-
  [Paper Note] Video-R1: Reinforcing Video Reasoning in MLLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][video reasoning] Inspired by DeepSeek-R1, this paper presents the first systematic exploration of applying the R1 paradigm (rule-based RL) to video reasoning. It proposes the T-GRPO algorithm to explicitly encourage temporal reasoning, constructs a mixed image-video training dataset, and achieves 37.1% accuracy on VSI-Bench, surpassing GPT-4o.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - video reasoning
  - reinforcement learning
  - GRPO
  - temporal modeling
  - chain-of-thought
date: 2026-05-08
content_hash: d531e4f39d8769ae
---

# Video-R1: Reinforcing Video Reasoning in MLLMs

**Conference**: NeurIPS 2025
**arXiv**: [2503.21776](https://arxiv.org/abs/2503.21776)
**Code**: [GitHub](https://github.com/tulerfeng/Video-R1)
**Area**: Multimodal VLM
**Keywords**: video reasoning, reinforcement learning, GRPO, temporal modeling, chain-of-thought

## TL;DR

Inspired by DeepSeek-R1, this paper presents the first systematic exploration of applying the R1 paradigm (rule-based RL) to video reasoning. It proposes the T-GRPO algorithm to explicitly encourage temporal reasoning, constructs a mixed image-video training dataset, and achieves 37.1% accuracy on VSI-Bench, surpassing GPT-4o.

## Background & Motivation

DeepSeek-R1 demonstrated that rule-based RL can elicit strong reasoning capabilities and long chain-of-thought in the text domain. Subsequent works such as Kimi k1.5 and Skywork R1V began extending RL to image reasoning. However, **RL-based exploration in video reasoning remains nearly absent**.

Directly applying GRPO to video reasoning faces two fundamental challenges:

**Lack of explicit reward signals for temporal modeling**: Standard GRPO provides no mechanism to encourage temporal reasoning. Models may take shortcuts—focusing on surface-level visual patterns in a single frame rather than temporal changes across frames. For instance, when asked about the direction of an object's motion, a model may guess from a single frame rather than reasoning over multi-frame changes. Both this work and Video-UTR observe such shortcut behavior.

**Scarcity of high-quality video reasoning data**: Existing video datasets primarily target simple recognition tasks rather than reasoning. Samples requiring strong reasoning capabilities or long reasoning chains are extremely rare, limiting the effectiveness of RL.

## Method

### Overall Architecture

Video-R1 is built on Qwen2.5-VL-7B-Instruct and adopts a two-stage training pipeline: (1) SFT cold start (Video-R1-CoT-165k) → (2) T-GRPO reinforcement learning (Video-R1-260k). Training data consists of a mixture of images and videos.

### Key Designs

1. **T-GRPO (Temporal Group Relative Policy Optimization)**: Introduces a contrastive reward mechanism on top of GRPO to explicitly encourage temporal reasoning. The core idea is: for the same video question, two groups of responses are generated using **temporally ordered frames** and **randomly shuffled frames**, respectively. Their accuracy rates are compared to assess whether the model genuinely exploits temporal information.

The temporal reward is defined as:

$$r_t = \begin{cases} \alpha, & \text{if } p \geq \tilde{p} \\ 0, & \text{otherwise} \end{cases}$$

where $p$ and $\tilde{p}$ are the proportions of correct answers in the ordered and shuffled groups, respectively, and $\alpha=0.3$. A positive temporal reward is granted only when the ordered group performs at least as well as the shuffled group. **Crucially**, $r_t$ is applied only to correct responses to avoid diluting the reward signal.

The final reward is:

$$R_i = \begin{cases} r_i + r_t, & \text{if } o_i \text{ is correct} \\ r_i, & \text{otherwise} \end{cases}$$

2. **Mixed Image-Video Training Data Strategy**: To compensate for the scarcity of video reasoning data, high-quality image reasoning data is incorporated into training. The composition of Video-R1-260k is:

    - Video general data: 116K (temporal understanding and reasoning)
    - Image general: 15K + charts: 21K + OCR: 16K + math: 37K + knowledge: 37K + spatial: 20K

   Image data provides a broad foundation of reasoning skills (mathematics, spatial logic, domain knowledge), while video data provides temporal reasoning complexity. The model can transfer reasoning capabilities learned from images to dynamic video scenarios.

3. **Multi-type Rule-based Reward Design**:

    - Multiple-choice: exact answer match
    - Numerical QA: exact match of predicted numbers
    - OCR: Word Error Rate (WER)
    - Free-form QA: average of ROUGE-1/2/L
    - Regression: $1 - \text{relative error}$

4. **Length Reward Control**: A length reward $r_l = \omega$ ($\omega = 0.2$) is introduced, granting an additional reward when correct responses fall within the length range $[l_{\min}=320, l_{\max}=512]$, balancing "deep thinking" against "overthinking."

### Loss & Training

- SFT stage: trained for 1 epoch on Video-R1-CoT-165k; CoT annotations generated by Qwen2.5-VL-72B
- RL stage: T-GRPO training for only 1K steps (≈15 hours); Adam optimizer, learning rate 1e-6
- Ordered group size $G=8$, shuffled group size $\tilde{G}=4$ (for efficiency)
- Up to 16 frames during training; up to 64 frames during inference
- KL divergence coefficient $\beta=0.04$; maximum response length 768 tokens

## Key Experimental Results

### Main Results (64-frame inference)

| Model | VSI-Bench | VideoMMMU | MMVU(mc) | MVBench | TempCompass | VideoMME |
|-------|-----------|-----------|----------|---------|-------------|---------|
| GPT-4o | 34.0 | 61.2 | 75.4 | - | - | 71.9 |
| Qwen2.5-VL-7B (CoT) | 31.4 | 50.4 | 60.0 | 59.2 | 72.9 | 59.6 |
| Qwen2.5-VL-7B-SFT | 34.8 | 49.4 | 61.6 | 60.6 | 70.0 | 58.8 |
| **Video-R1-7B** | **37.1** | **52.4** | **63.8** | **64.8** | **73.2** | **61.4** |

Video-R1-7B achieves 37.1% on VSI-Bench, **surpassing GPT-4o** (34.0%).

### Ablation Study

| Variant | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|---------|-----------|-----------|------|---------|-------------|---------|
| wo-image (video data only) | 32.3 | 45.8 | 60.6 | 60.9 | 69.8 | 53.8 |
| wo-temporal (GRPO instead of T-GRPO) | 32.7 | 48.3 | 62.1 | 61.1 | 71.3 | 54.5 |
| zero (skip SFT cold start) | 31.8 | 49.5 | 63.8 | 60.4 | 70.9 | 53.8 |
| **Video-R1-7B (full)** | **34.6** | **49.8** | **64.2** | **62.7** | **72.6** | **57.4** |

Removing image data, T-GRPO, or SFT cold start all lead to consistent performance degradation.

### Temporal Reasoning Ratio Analysis

| Model | Proportion of Responses with Temporal Reasoning |
|-------|------------------------------------------------|
| Video-R1 (T-GRPO) | **75.0%** |
| Video-R1-wo-temporal (GRPO) | 60.2% |

T-GRPO increases the proportion of temporal reasoning behavior by 14.8 percentage points.

### Key Findings

- **RL outperforms SFT**: SFT even slightly degrades performance on benchmarks such as VideoMME (possibly due to overfitting), whereas just 1K steps of RL yields substantial improvements, corroborating the "SFT memorizes, RL generalizes" perspective.
- **More frames = better reasoning**: Performance consistently improves across nearly all benchmarks as the number of frames increases from 16 → 32 → 64.
- **Emergent behavior (Aha Moment)**: When confronted with ambiguous temporal cues, the model self-reflects and re-examines video evidence, indicating active learning rather than simple pattern memorization.
- Training curves show that response length initially decreases at the onset of RL (discarding suboptimal SFT-style reasoning), then rises and stabilizes (forming a new reasoning strategy).
- The hyperparameter $\alpha$ is insensitive within the range 0.2–0.3; values of 0.1 and 0.4 yield slightly worse results.
- Extending training to 10K steps yields further gains (TempCompass: 73.2→74.2; MVBench: 64.8→65.5).

## Highlights & Insights

1. **First systematic exploration of the R1 paradigm applied to video reasoning**, establishing a foundational framework for this direction.
2. **Elegant T-GRPO design**: Contrastive learning between ordered and shuffled frames achieves temporal reasoning capability at minimal additional cost, with broadly applicable methodology.
3. **Practical value of mixed image-video training**: Using image reasoning data to compensate for insufficient video reasoning data is a pragmatic solution under resource constraints.
4. **Significant improvements with only 1K steps of RL**, demonstrating the efficiency of the data and algorithmic design.
5. **Emergence of self-reflective behavior**: The model exhibits reasoning loops resembling "aha moments" when processing ambiguous temporal information.

## Limitations & Future Work

- Training uses only 16 frames, limiting the capture of long-range temporal dependencies.
- T-GRPO introduces additional computational overhead (requiring an equivalent number of shuffled-order inferences), which can be mitigated via inference acceleration frameworks such as vLLM.
- The length reward uses a fixed interval rather than adaptively adjusting to question complexity.
- The image-to-video knowledge transfer approach is coarse (simple mixing), lacking a more principled design.
- Rule-based rewards are manually designed for specific task types; a general video reward model is absent.

## Related Work & Insights

- DeepSeek-R1 demonstrated that rule-based RL can elicit reasoning capabilities; this work extends that paradigm to the video modality.
- Video-UTR also identifies the shortcut problem of GRPO on video, but addresses it with a different approach.
- The "SFT memorizes, RL generalizes" perspective (Chu et al. 2025) is validated in the video domain.
- All code, models, and data are open-sourced, providing a foundation for subsequent community research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The contrastive temporal reward design in T-GRPO is elegant; first systematic exploration of R1 + video reasoning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six benchmarks, comprehensive ablations, training curve analysis, and temporal reasoning ratio analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with explicit problem-solution correspondence.
- **Value**: ⭐⭐⭐⭐⭐ Fully open-sourced; surpasses GPT-4o on VSI-Bench; high community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[ICLR 2026\] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](../../ICLR2026/multimodal_vlm/vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)

</div>

<!-- RELATED:END -->
