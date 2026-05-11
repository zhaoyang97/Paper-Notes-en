---
title: >-
  [Paper Note] Scaling RL to Long Videos
description: >-
  [NeurIPS 2025][Video Generation][Long video reasoning] This paper proposes LongVILA-R1, a full-stack framework that extends VLM reasoning to long videos (up to 8192 frames) via a 104K long-video reasoning dataset…
tags:
  - "NeurIPS 2025"
  - "Video Generation"
  - "Long video reasoning"
  - "reinforcement learning"
  - "vision-language models"
  - "sequence parallelism"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 5a56363445987bfa
---

# Scaling RL to Long Videos

**Conference**: NeurIPS 2025
**arXiv**: [2507.07966](https://arxiv.org/abs/2507.07966)
**Code**: [GitHub](https://github.com/NVlabs/Long-RL)
**Area**: Video Generation
**Keywords**: Long video reasoning, reinforcement learning, vision-language models, sequence parallelism, chain-of-thought

## TL;DR

This paper proposes LongVILA-R1, a full-stack framework that extends VLM reasoning to long videos (up to 8192 frames) via a 104K long-video reasoning dataset, a two-stage CoT-SFT + RL training pipeline, and the MR-SP multimodal reinforcement sequence parallelism system, achieving 65.1%/71.1% on VideoMME.

## Background & Motivation

1. **Background**: Long video understanding demands temporal, spatial, goal-oriented, and narrative reasoning capabilities. Closed-source models such as GPT-4o and Gemini-1.5-Pro have demonstrated strong performance, while open-source VLMs have also made progress on short videos.

2. **Limitations of Prior Work**: (1) High-quality long-video reasoning datasets are lacking — unlike math or code reasoning with structured annotations, long-video reasoning requires labeling complex temporal dynamics and narrative elements. (2) RL training frameworks for long videos are difficult to construct — processing hundreds to thousands of frames incurs enormous memory overhead and extremely long rollout times.

3. **Key Challenge**: Existing RL frameworks (e.g., R1-V, EasyR1) are not designed for long videos; GRPO's group sampling is computationally prohibitive under long contexts, and visual encoding requires redundant computation.

4. **Goal**: To holistically address the three core challenges of long-video reasoning: data, training methodology, and training systems.

5. **Key Insight**: On the data side, NVILA-8B and DeepSeek-R1-671B are used to automatically generate long-video CoT annotations; on the system side, sequence parallelism and video embedding caching are employed to accelerate RL training.

6. **Core Idea**: Caching video embeddings combined with sequence parallelism makes long-video RL training feasible, while high-quality CoT data and difficulty filtering are critical to the emergence of reasoning capabilities.

## Method

### Overall Architecture

Three major components:
1. **LongVideo-Reason Dataset**: 104K QA pairs derived from 18K long videos
2. **Two-stage training**: Stage-1 CoT-SFT (36K) → Stage-2 GRPO RL (68K + 102K)
3. **MR-SP training system**: Sequence parallelism + embedding caching

### Key Designs

**1. LongVideo-Reason Data Construction**

- **Function**: Provides large-scale, high-quality reasoning annotations for long videos
- **Mechanism**: Videos are segmented into ~10s clips → NVILA-8B generates per-clip descriptions → all clip descriptions are aggregated → DeepSeek-R1-671B generates Question-Reasoning-Answer triplets based on the full-video description. Four reasoning types are covered: temporal, goal/purpose, spatial, and narrative reasoning. Data filtering: each question is rolled out 10 times; samples that are too easy (all correct) or too hard (all wrong) are discarded, retaining only moderate-difficulty instances (GRPO requires diversity across rollouts).
- **Design Motivation**: GRPO is sensitive to batch sampling — if all rollouts are uniformly correct or incorrect, gradients vanish; data of appropriate difficulty is therefore required.

**2. MR-SP Multimodal Reinforcement Sequence Parallelism**

- **Function**: Makes long-video RL training feasible and efficient
- **Mechanism**:
    - **Stage 1 — Rollout parallel encoding**: Video frames are evenly distributed across multiple GPUs, each encoding independently, with embeddings aggregated via all-gather. Key optimization: video embeddings are cached and reused across 8–16 rollouts to avoid redundant encoding.
    - **Stage 2 — Prefilling sequence parallelism**: The prefilling step for both the policy and reference models is parallelized — global embeddings are padded to a uniform length and sharded across GPUs, with each GPU computing logits for only a subset of tokens.
- **Design Motivation**: Long videos produce massive numbers of visual tokens ($10^4$–$10^5$), which cannot fit on a single GPU. Embedding caching eliminates $G$-fold redundant encoding (where $G$ is the number of rollouts).

**3. Two-Stage Training Strategy**

- **Function**: First establishes a reasoning foundation, then scales it via RL
- **Mechanism**: Stage-1 applies SFT on 36K high-quality CoT data (formatted as `<think></think><answer></answer>`) to equip the model with basic reasoning and instruction-following capabilities. Stage-2 applies GRPO on 68K curated data plus 102K open-source data (rule-based rewards for accuracy and format), further scaling reasoning ability.
- **Design Motivation**: Skipping SFT and directly applying RL degrades performance; SFT provides necessary reasoning warm-up.

### Loss & Training

- Stage-1: Standard SFT cross-entropy loss
- Stage-2: GRPO objective $\mathcal{J}(\theta)$ with clipping and KL regularization. Group size $G=8$; advantage $A_i$ is computed via within-group normalization
- Reward: Rule-based (format correctness + answer accuracy)

## Key Experimental Results

### Main Results

**VideoMME Benchmark**

| Model | w/o sub | Short | Medium | Long | w/ sub |
|-------|---------|-------|--------|------|--------|
| LongVILA-7B | 60.1 | 69.0 | 58.3 | 53.0 | 65.1 |
| **LongVILA-R1-7B** | **65.1** | **76.8** | **63.2** | **55.2** | **71.1** |
| Video-R1-7B | 61.4 | - | - | - | - |
| Gemini-1.5-Pro | 75.0 | - | - | - | 81.3 |
| LLaVA-Video-7B | 63.3 | - | - | - | 69.7 |

**LongVideo-Reason-eval**

| Model | Temporal | Goal | Plot | Spatial | Overall |
|-------|----------|------|------|---------|---------|
| Video-R1-7B | 61.4 | 85.0 | 62.0 | 58.5 | 68.1 |
| Gemini-1.5-Pro | 65.4 | 81.9 | 67.8 | 53.3 | 69.3 |
| LongVILA-7B | 58.0 | 80.2 | 57.1 | 46.7 | 62.7 |
| **LongVILA-R1-7B** | **68.1** | **85.7** | **70.6** | **53.3** | **72.0** |

### Ablation Study

| Setting | CoT-SFT | RL Data | LongVideo-Reason-eval |
|---------|---------|---------|----------------------|
| Base only | ✗ | ✗ | 62.7 |
| SFT only | ✓ Ours | ✗ | Significant gain |
| Direct RL (no SFT) | ✗ | ✓ | Performance drop |
| **Full pipeline** | ✓ | ✓ | **72.0** |

**MR-SP Training Efficiency (8×A100, LongVILA-7B)**

| Frames | w/o MR-SP | MR-SP Stage 1 | Full MR-SP |
|--------|-----------|---------------|-----------|
| 256 | Normal | Accelerated | Accelerated |
| 512 | Slow | Accelerated but OOM | **2.1× speedup** |
| 1024 | OOM | OOM | **Runnable** |

### Key Findings

- Reasoning gains from RL continue to scale with increasing frame count (LongVILA-R1 improves consistently from 16→512 frames, whereas LongVILA saturates or degrades beyond 256 frames)
- MR-SP achieves a 2.1× speedup at 512 frames and is the only viable solution at 1024 frames
- CoT-SFT is a necessary prerequisite for RL — skipping it leads to performance degradation
- RL training on hour-long videos (3600 frames) is feasible on a single A100 node

## Highlights & Insights

- Full-stack solution: data, training, and system components are all internally consistent
- MR-SP transforms long-video RL from infeasible to practically viable; embedding caching to eliminate $G$-fold redundant encoding is the key optimization
- The data filtering strategy (removing too-easy and too-hard samples) is critical for GRPO — this is a practical answer to the theoretical gradient-vanishing condition
- The sustained improvement in reasoning ability with increasing frame count validates the value of long-video RL

## Limitations & Future Work

- Data generation requires approximately 80,000 H100 GPU hours, resulting in substantial cost
- The reasoning pipeline relies on segment-level captioning followed by LLM-based reasoning generation, which may introduce caption noise
- Experiments are currently limited to the 7B model scale; performance at larger scales remains unexplored
- Spatial reasoning scores show limited improvement (53.3%), which is identified as a known weakness

## Related Work & Insights

- Built upon LongVILA's MM-SP infrastructure; data generation leverages NVILA and DeepSeek-R1
- Video-R1 handles only 16-frame short videos; this work extends RL for VLMs to long-video settings
- Insight: The essence of RL training lies in aligning data difficulty with model capability and in robust systems engineering

## Rating

- **Novelty**: ⭐⭐⭐⭐ Full-stack integration rather than a single-point contribution; MR-SP represents a significant systems contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-benchmark comparisons, comprehensive ablations, and quantified training efficiency
- **Writing Quality**: ⭐⭐⭐⭐ Content-dense but well-organized
- **Value**: ⭐⭐⭐⭐⭐ Provides a reproducible, complete solution for long-video VLM reasoning with high open-source utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](../../ICCV2025/video_generation/omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[ICCV 2025\] Long Context Tuning for Video Generation](../../ICCV2025/video_generation/long_context_tuning_for_video_generation.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](../../CVPR2026/video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)

</div>

<!-- RELATED:END -->
