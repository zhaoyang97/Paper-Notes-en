---
title: >-
  [Paper Note] Seeing the Arrow of Time in Large Multimodal Models
description: >-
  [NeurIPS 2025][Video Understanding][Arrow of Time] This paper reveals that current large multimodal models (LMMs) are surprisingly insensitive to the temporal directionality of video (i.e., the Arrow of Time)—producing nearly identical answers for forward and reversed playback. The authors propose ArrowRL, a GRPO-based training strategy that introduces a reverse video reward to elicit temporal direction awareness, and construct AoTBench for evaluation. The approach achieves significant gains across multiple VQA benchmarks, including a 65.9% relative improvement on Vinoground.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Arrow of Time
  - Temporal Awareness in LMMs
  - Reinforcement Learning Fine-tuning
  - Video Understanding Benchmark
  - GRPO
date: 2026-05-08
content_hash: 59e3911071bd3104
---

# Seeing the Arrow of Time in Large Multimodal Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.03340](https://arxiv.org/abs/2506.03340)
**Code**: [Project Page](https://vision.cs.utexas.edu/projects/SeeAoT)
**Area**: Video Understanding / Multimodal Temporal Perception
**Keywords**: Arrow of Time, Temporal Awareness in LMMs, Reinforcement Learning Fine-tuning, Video Understanding Benchmark, GRPO

## TL;DR

This paper reveals that current large multimodal models (LMMs) are surprisingly insensitive to the temporal directionality of video (i.e., the Arrow of Time)—producing nearly identical answers for forward and reversed playback. The authors propose ArrowRL, a GRPO-based training strategy that introduces a reverse video reward to elicit temporal direction awareness, and construct AoTBench for evaluation. The approach achieves significant gains across multiple VQA benchmarks, including a 65.9% relative improvement on Vinoground.

## Background & Motivation

The Arrow of Time (AoT) is a fundamental property of the physical world—events such as cream dissolving into coffee or glass shattering appear immediately unnatural when reversed. While humans possess an innate sense of temporal directionality, current large multimodal models lack this capability.

The central finding is striking: when video frames are shuffled or reversed, state-of-the-art LMMs (e.g., LLaVA-OV-7B) exhibit almost no performance degradation on standard VQA benchmarks—indicating that these models do not genuinely exploit temporal ordering information. More concretely, when presented with forward and reversed videos, models frequently produce identical descriptions (e.g., predicting "ignite" for both), exposing a fundamental temporal insensitivity.

The paper adopts a two-pronged approach: (1) on the model side, reinforcement learning is used to encourage divergent responses for forward and reversed videos; (2) on the evaluation side, a benchmark is constructed that genuinely tests temporal direction awareness. The core idea is to use reversed video as a natural contrastive signal, employing a reward mechanism to force the model to distinguish forward from reversed temporal ordering.

## Method

### Overall Architecture

ArrowRL is a post-training strategy based on GRPO (Group Relative Policy Optimization). Given a pretrained LMM $\pi_\theta$, for each question $q=(v, l)$, a group of candidate responses $\{o_i\}_{i=1}^G$ is generated alongside reverse responses $\tilde{o}$ produced from the reversed video $\tilde{v}$. A dual reward signal—comprising a fidelity reward and a reverse reward—guides optimization, and the policy is updated via the GRPO objective.

### Key Designs

1. **Temporal Divergence Score (TDS)**:

    - To quantify the temporal sensitivity of individual samples and benchmarks, a KL divergence-based scoring method is proposed.
    - For each sample, the first-token probability distributions of the model under forward and reversed video are compared: $\text{TDS}_i = D_{KL}[p_i \| \tilde{p}_i]$
    - This provides finer granularity than simple accuracy differences, capturing changes in model confidence.
    - Applied to systematically analyze temporal sensitivity across 8 mainstream VQA benchmarks; TVBench, Vinoground, and TempCompass are found to be highly temporally sensitive, while VITATECS and TemporalBench exhibit low sensitivity.

2. **Target Fidelity Reward**:

    - Measures the similarity between a candidate response $o_i$ and the target response $o^*$.
    - For MCQ tasks: exact match (1.0 or 0.0).
    - For open-ended QA and video captioning: LLM-based scoring.
    - Ensures model outputs remain aligned with correct answers.

3. **Reverse Reward**:

    - Maximizes the divergence between a forward candidate response $o_i$ and the reverse response $\tilde{o}$: $r_i^{rev} = 1 - \text{Similarity}(o_i, \tilde{o})$
    - Motivation: a model sensitive to AoT should produce different responses for forward and reversed video.
    - Dynamic weighting mechanism: when $\text{Similarity}(\tilde{o}, o^*) > \gamma$ (indicating the sample is temporally insensitive), $\alpha_i = 0$ disables the reverse reward.
    - Final reward: $r_i = r_i^{fid} + \alpha_i \cdot r_i^{rev}$

4. **AoTBench**:

    - Three tasks: (a) sequence direction classification (613 videos, forward vs. reversed); (b) directional description matching (2,000 videos, V2T and T2V tasks); (c) AoT-sensitive VQA (1,800 samples selected from 8 benchmarks based on high TDS).
    - Purpose-built to evaluate temporal direction perception.

### Loss & Training

The standard GRPO objective is adopted with group size $G=8$, reverse reward weight $\alpha=0.25$, and dynamic threshold $\gamma=0.75$. Training data includes: 1.1K MCQ samples (UCF101 forward/reverse classification), 11.8K high-temporality open-ended QA samples (filtered from LLaVA-Video-178K by perplexity difference), and 11.7K video captions (RTime dataset). Only 2,000 RL training steps are required on 6 GH200 GPUs. No SFT stage is needed; ArrowRL is applied directly to pretrained models.

## Key Experimental Results

### Main Results

**AoTBench Results**:

| Model | Dir. Cls. RFilm | Dir. Cls. UCF | Desc. Match T2V | Desc. Match V2T | AoT-VQA |
|-------|----------------|--------------|----------------|----------------|---------|
| GPT-4o | 52.8 | 54.0 | 56.5 | 69.5 | 67.8 |
| Qwen2.5-VL-7B | 50.0 | 51.6 | 53.4 | 66.6 | 49.6 |
| + ArrowRL | **51.4** | **54.8** | **55.6** | **69.6** | **58.8** |
| Qwen2-VL-7B | 50.0 | 51.6 | 56.3 | 62.3 | 44.3 |
| + ArrowRL | **69.1** | **72.6** | **57.1** | **68.8** | **51.1** |

**Existing Temporally Sensitive Benchmarks**:

| Model | TempCompass | TVBench | Vinoground Group |
|-------|------------|---------|-----------------|
| Qwen2.5-VL-7B | 73.8 | 54.7 | 16.4 |
| + ArrowRL | **75.5** | **56.2** | **27.2** (+65.9% relative gain) |

### Ablation Study

| Configuration | AoTBench Avg. Acc. | Notes |
|--------------|-------------------|-------|
| Qwen2.5-VL-7B baseline | 56.2% | No training |
| + SFT (same data) | 57.4% | Limited effect |
| + ArrowRL (LLaVA captions) | 57.7% | Non-high-temporality data |
| + ArrowRL (RTime captions) | 60.4% | High-temporality data |
| + ArrowRL (full data) | **61.4%** | Best multi-task combination |

### Key Findings
- Nearly all open-source LMMs perform at chance level (50%) on direction classification, producing identical responses for forward and reversed video.
- ArrowRL substantially outperforms SFT (+4.0 vs. +1.2), demonstrating that RL is markedly more effective at eliciting temporal direction awareness.
- The reverse reward is the core component: removing it ($\alpha=0$) degrades performance below the baseline.
- ArrowRL does not harm general video understanding: performance on temporally insensitive benchmarks such as VideoMME and NExT-QA is maintained or slightly improved.

## Highlights & Insights

- **Exposes a fundamental deficiency**: State-of-the-art LMMs are nearly blind to temporal directionality in video—a critical flaw that had been largely overlooked by the community.
- **Elegant reverse reward design**: Reversed video serves as a natural contrastive signal without requiring additional annotation; temporal awareness is induced by driving the model to produce divergent forward/reverse responses.
- **Methodological contribution of TDS**: Provides a systematic tool for assessing the temporal sensitivity of video benchmarks, revealing that many benchmarks ostensibly targeting temporal understanding are in fact insensitive to temporal ordering.
- **Highly efficient training**: Significant temporal awareness gains are achieved with only 2,000 RL steps, applicable directly to pretrained models without an SFT stage.

## Limitations & Future Work

- Training is limited to a maximum of 16 video frames, restricting temporal awareness gains for long videos.
- The reverse reward assumes semantic differences exist between forward and reversed video, making it inapplicable to static or looping content (though dynamic weighting partially mitigates this).
- Performance on direction classification tasks remains near chance for some baselines, indicating that ArrowRL alone is insufficient to fully resolve the problem.
- Integration with chain-of-thought reasoning has not been explored.

## Related Work & Insights

- **vs. Video-R1**: Video-R1 focuses on video reasoning but similarly performs at chance level on AoT tasks (50% on direction classification), whereas ArrowRL yields effective improvements.
- **vs. Early AoT Self-Supervision** (Pickup 2014, Wei 2018): Prior work employs AoT solely as a pretext task for visual feature learning; this paper is the first to address temporal direction perception at the language generation level in LMMs.
- **Implications for RL fine-tuning**: The reverse reward concept is broadly generalizable—specific input transformations (e.g., reversal, cropping, speed variation) can serve as contrastive signals to reinforce model sensitivity to targeted attributes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Exposes a fundamental deficiency in LMM temporal perception; the reverse reward design is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic analysis across 8 benchmarks, validation on 3 base models, and comprehensive ablation and hyperparameter studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous argumentation with a clear narrative arc from problem discovery through analysis, solution, and validation.
- Value: ⭐⭐⭐⭐⭐ Identifies a fundamental deficiency in the field and provides an effective remedy, with broad implications for the video LMM community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)
- [\[ACL 2026\] ArrowGEV: Grounding Events in Video via Learning the Arrow of Time](../../ACL2026/video_understanding/arrowgev_grounding_events_in_video_via_learning_the_arrow_of_time.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)

<!-- RELATED:END -->
