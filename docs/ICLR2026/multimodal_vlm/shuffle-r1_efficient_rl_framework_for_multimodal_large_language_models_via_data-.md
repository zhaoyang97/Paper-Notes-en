---
title: >-
  [Paper Note] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle
description: >-
  [ICLR 2026][Multimodal VLM][Reinforcement Learning] Shuffle-R1 is proposed as an RL training framework that addresses two key efficiency bottlenecks—Advantage Collapsing and Rollout Silencing—through Pairwise Trajectory Sampling (selecting high-contrast trajectory pairs) and Advantage-based Batch Shuffle (redistributing training batches by advantage values). The framework achieves a 22% improvement over the baseline on Geo3K and surpasses GPT-4o on MathVerse.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Reinforcement Learning
  - Multimodal Reasoning
  - Data-centric Optimization
  - Trajectory Sampling
  - GRPO
date: 2026-05-08
content_hash: 5958eebd5b6dfc42
---

# Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle

**Conference**: ICLR 2026
**arXiv**: [2508.05612](https://arxiv.org/abs/2508.05612)
**Code**: [https://xenozlh.github.io/Shuffle-R1](https://xenozlh.github.io/Shuffle-R1)
**Area**: Multimodal VLM
**Keywords**: Reinforcement Learning, Multimodal Reasoning, Data-centric Optimization, Trajectory Sampling, GRPO

## TL;DR
Shuffle-R1 is proposed as an RL training framework that addresses two key efficiency bottlenecks—Advantage Collapsing and Rollout Silencing—through Pairwise Trajectory Sampling (selecting high-contrast trajectory pairs) and Advantage-based Batch Shuffle (redistributing training batches by advantage values). The framework achieves a 22% improvement over the baseline on Geo3K and surpasses GPT-4o on MathVerse.

## Background & Motivation

**Background**: Reinforcement learning (RL) has become the dominant post-training paradigm for enhancing reasoning capabilities in LLMs and MLLMs. Works such as DeepSeek-R1 leverage verifiable outcome reward signals and have achieved notable progress in mathematical reasoning and code generation. RL has also been extended to multimodal settings, encompassing visual reasoning, object detection, and video understanding.

**Limitations of Prior Work**: Two overlooked efficiency issues exist in current RL training pipelines:
   - **Advantage Collapsing**: The advantage values of most rollouts within a batch concentrate near zero, resulting in extremely weak gradient signals; informative trajectories are drowned out by a large number of uninformative ones.
   - **Rollout Silencing**: As training progresses, the proportion of rollouts contributing non-zero gradients continuously declines (easy questions have converged while hard questions remain unsolved), leading to wasted computation.

**Key Challenge**: The static sampling paradigm treats all trajectories equally and cannot distinguish which data warrants parameter updates. Increasing the number of rollouts partially alleviates the issue but incurs linearly growing computational cost without addressing the root cause.

**Goal**: To dynamically filter informative trajectories and optimize batch composition without significantly increasing computational overhead, thereby improving gradient signal quality and computational efficiency during RL training.

**Key Insight**: Approaching the problem from the data side, the framework shifts the focus of RL training from "how to update" to "what data to update with," designing adaptive trajectory selection and batch reorganization mechanisms.

**Core Idea**: High-contrast trajectory pairs are selected via pairwise sampling, and training batches are reshuffled according to advantage values to amplify critical signals, realizing dynamic data priority scheduling.

## Method

### Overall Architecture
Shuffle-R1 extends standard GRPO by inserting two modules after the advantage computation step: (1) Pairwise Trajectory Sampling (PTS), which selects high-value trajectory pairs from an expanded rollout pool; and (2) Advantage-based Batch Shuffle (ABS), which reorganizes the selected trajectories into training batches by importance. The two modules jointly implement dynamic data priority scheduling.

### Key Designs

1. **Pairwise Trajectory Sampling (PTS)**:

    - **Function**: Constructs $N$ positive–negative contrast trajectory pairs from $2N$ rollouts and retains only the top-$k$ high-contrast pairs.
    - **Mechanism**: Rollouts are sorted in descending order of advantage, and a max-min pairing strategy is applied—the highest-advantage rollout is paired with the lowest, the second-highest with the second-lowest, and so on. The advantage gap within each pair represents contrast intensity. A sampling ratio $\alpha$ is used to retain the top $M = \alpha N$ high-contrast pairs.
    - **Design Motivation**: High-contrast positive–negative pairs provide stronger gradient signals. Although the total rollout generation doubles to $2N$, gradient computation is performed only on the selected top-$k$ pairs, keeping computational cost unchanged. This alleviates Advantage Collapsing.

2. **Advantage-based Batch Shuffle (ABS)**:

    - **Function**: Assigns sampling probabilities to the effective trajectory pairs output by PTS based on their advantage values, and performs $S$ rounds of sub-sampling to reorganize the training batch.
    - **Mechanism**: An importance weight $W(p) = |\hat{A}_1| + |\hat{A}_2|$ is computed for each trajectory pair and normalized into a sampling distribution $\Phi$. Each round draws $T$ pairs without replacement; $S$ rounds are combined to form the reshuffled batch $B'$, maintaining $|B'| = |B|$.
    - **Design Motivation**: Trajectories with high advantage receive more update opportunities, while low-value trajectories are naturally down-weighted, alleviating Rollout Silencing. This is essentially a soft priority ranking mechanism.

### Loss & Training
- The base objective follows PPO-clip-style policy gradient (consistent with GRPO), with advantages normalized within groups.
- PTS sampling ratio $\alpha = 0.5$ (retaining half the pairs); ABS sub-sampling capacity $T = 256$, number of rounds $S = 8$.
- Each query generates $2N = 16$ rollouts, forming 8 pairs of which 4 are retained.
- Learning rate $1\text{e-}6$, rollout temperature $1.0$, visual encoder frozen.

## Key Experimental Results

### Main Results

| Model | Method | Geo3K | Math Avg. | HallBench | ChartQA |
|------|------|-------|-----------|-----------|---------|
| Qwen2.5-VL-3B | Baseline | 25.79 | 41.71 | 59.83 | 73.08 |
| Qwen2.5-VL-3B | +GRPO | 42.64 | 46.74 | 63.09 | 76.20 |
| Qwen2.5-VL-3B | +DAPO | 45.09 | 48.08 | 63.24 | 76.70 |
| Qwen2.5-VL-3B | **+Ours** | **47.88 (+22.09)** | **48.70 (+6.99)** | 63.19 | **77.04** |
| Qwen2.5-VL-7B | Baseline | 38.12 | 49.82 | 65.19 | 79.84 |
| Qwen2.5-VL-7B | +GRPO | 52.60 | 53.13 | 68.56 | 80.84 |
| Qwen2.5-VL-7B | **+Ours** | **55.89 (+17.77)** | **54.63 (+4.81)** | **69.51** | **81.64** |

On cross-domain benchmarks (30k training data), the 7B model achieves 52.2% on MathVerse, surpassing GPT-4o (50.8%).

### Ablation Study

| Component | Geo3K (3B) | Math Avg. (3B) |
|------|------------|----------------|
| GRPO baseline | 42.64 | 46.74 |
| +PTS only | 46.52 | 47.89 |
| +ABS only | 44.18 | 47.35 |
| +PTS+ABS (Full) | **47.88** | **48.70** |

### Key Findings
- PTS contributes the most, yielding ~4% improvement on Geo3K alone; ABS provides an additional ~1.5% gain.
- Shuffle-R1 matches the full-training performance of GRPO using only 50% of training steps, achieving a 2× improvement in training efficiency.
- Consistent results across 3B and 7B scales demonstrate cross-scale generalizability.
- The advantage is particularly pronounced on Geo3K (2.1k samples), indicating greater utility in low-data regimes.

## Highlights & Insights
- **Precise problem formulation**: Advantage Collapsing and Rollout Silencing are systematically identified and quantified as RL training efficiency bottlenecks for the first time.
- **Simple yet effective design**: Both PTS and ABS are lightweight, plug-and-play modules implemented via straightforward operations (sorting, pairing, and weighted sampling).
- **Comprehensive evaluation**: Validated across in-domain/out-of-domain settings, small/large data regimes, and 3B/7B model scales.
- **Controlled computational overhead**: Doubling rollout generation does not increase gradient computation cost, as gradients are computed only over the selected trajectories.

## Limitations & Future Work
- The max-min pairing strategy in PTS is heuristic; alternative pairing schemes (e.g., based on semantic similarity) warrant further exploration.
- The resampling in ABS risks repeated use of the same trajectory, potentially leading to overfitting; a more systematic analysis is needed.
- Validation is currently limited to mathematical reasoning; generalization to broader tasks such as general VQA and visual dialogue remains to be demonstrated.
- The fixed sampling ratio $\alpha = 0.5$ could potentially be replaced by an adaptive schedule for further gains.

## Related Work & Insights
- Complementary to NoisyRollout (increasing rollout diversity) and VL-Rethinker (reflection tokens): the former focuses on data diversity while the present work focuses on data quality filtering.
- Conceptually aligned with curriculum learning: both approaches direct the model's attention toward more informative training samples.
- Provides insights for data-centric optimization in other RL training frameworks such as DPO and RLHF.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Novel problem formulation (Advantage Collapsing / Rollout Silencing) with a concise and effective method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-scale, multi-data-regime, multi-benchmark validation with complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem analysis, intuitive figures, and coherent argumentation.
- **Value**: ⭐⭐⭐⭐ The data-centric RL optimization perspective offers significant inspiration to the community, and the plug-and-play nature ensures strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL](vidguard-r1_ai-generated_video_detection_and_explanation_via_reasoning_mllms_and.md)
- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](../../ICCV2025/multimodal_vlm/jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
