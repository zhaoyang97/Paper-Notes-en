---
title: >-
  [Paper Note] Video-T1: Test-Time Scaling for Video Generation
description: >-
  [ICCV 2025][LLM Reasoning][Test-time scaling] This paper transfers the test-time scaling (TTS) paradigm from LLMs to video generation by reformulating TTS as a search problem over trajectories from Gaussian noise space t…
tags:
  - "ICCV 2025"
  - "LLM Reasoning"
  - "Test-time scaling"
  - "video generation"
  - "Tree-of-Frames"
  - "diffusion models"
  - "autoregressive video"
date: 2026-05-08
content_hash: 57e03e5f59173378
---

# Video-T1: Test-Time Scaling for Video Generation

**Conference**: ICCV 2025
**arXiv**: [2503.18942](https://arxiv.org/abs/2503.18942)
**Code**: [liuff19.github.io/Video-T1](https://liuff19.github.io/Video-T1)
**Area**: LLM Reasoning
**Keywords**: Test-time scaling, video generation, Tree-of-Frames, diffusion models, autoregressive video

## TL;DR

This paper transfers the test-time scaling (TTS) paradigm from LLMs to video generation by reformulating TTS as a search problem over trajectories from Gaussian noise space to the target video distribution. It proposes the Tree-of-Frames (ToF) search algorithm for efficient inference-time compute scaling, achieving consistent quality improvements across diverse video generation models on VBench.

## Background & Motivation

### State of the Field

Video generation has seen remarkable progress through training-time scaling (more data, larger models, more compute), yet the cost of further scaling has become prohibitive. Meanwhile, the LLM community (e.g., DeepSeek-R1, OpenAI o1) has demonstrated that TTS can substantially improve model performance by increasing inference-time computation.

### Limitations of Prior Work

**Prohibitive training-time scaling costs**: Training video generation models demands enormous data and computational resources, with scaling costs far exceeding those of LLMs.

**Unexplored inference-time potential**: Virtually no systematic study has investigated how increasing inference-time computation can improve video generation quality.

**Domain-specific challenges**: Unlike LLM token sequences, video requires simultaneous spatial quality and temporal coherence; the multi-step denoising process of diffusion models further complicates compute scaling.

### Core Problem

To what extent can video generation quality be improved when models are granted additional inference-time computation?

### Starting Point

The paper reformulates video generation TTS as a search problem: finding better trajectories through the space from Gaussian noise to the target video distribution, guided by test-time verifiers and heuristic search algorithms.

## Method

### Overall Architecture

The Video-T1 framework comprises three core components:
- **Video Generator** $\mathcal{G}$: a pretrained model that generates video conditioned on text
- **Test-Time Verifier** $\mathcal{V}$: a multimodal evaluation model that assesses generated video quality
- **Heuristic Search Algorithm** $f$: an optimization method that uses verifier feedback to guide the search

Two search strategies are proposed: Random Linear Search and Tree-of-Frames (ToF) Search.

### Key Designs

#### 1. **Random Linear Search**

- **Function**: Samples $N$ Gaussian noise vectors, performs complete denoising for each to generate $N$ videos, and selects the one with the highest verifier score (Best-of-N strategy).
- **Mechanism**: Can be viewed as a forest of $N$ degenerate trees (each with a single path), from which the optimal path is selected.
- **Time Complexity**: $O(TN)$, where $T$ is the number of frames and $N$ is the number of noise samples.
- **Limitations**: The linear structure is simplistic, lacks efficient optimization, and the independent trees provide no cross-candidate feedback, leading to redundant computation.

#### 2. **Tree-of-Frames (ToF) Search**

- **Function**: Exploits the frame-by-frame generation property of autoregressive models to introduce inference-time reasoning along the temporal dimension, adaptively expanding and pruning video branches via a tree structure.
- **Mechanism**: Generation proceeds in three stages:
    - **(a) Image-level alignment**: The first frame is aligned with the text prompt on core semantics (color, object count, spatial layout), influencing all subsequent frames.
    - **(b) Hierarchical prompting**: Intermediate frames focus on motion stability and physical plausibility, with the verifier's evaluation emphasis adjusted dynamically.
    - **(c) Holistic evaluation**: The final stage assesses overall video quality and text alignment.
- **Three Core Techniques**:
    - **Image-level alignment**: Incrementally evaluates frame quality during denoising, enabling early rejection of low-quality candidates.
    - **Hierarchical prompting**: Different verifier prompts are used at different stages — the first frame emphasizes semantic consistency, intermediate frames emphasize motion continuity, and the final frames emphasize overall quality.
    - **Heuristic pruning**: Retains the top-$k_t$ nodes at each timestep with a dynamic branching factor $b_t$, balancing exploration and convergence.
- **Complexity**: In practice, $b_t = 1$ for most timesteps and branching occurs only at stage transitions, reducing complexity to $O(N+T)$.

#### 3. **Multi-Verifier Ensemble**

- **Function**: Aggregates scores from multiple verifiers to select the final video.
- **Core Formula**: $\hat{i} = \arg\max_{0<i<n} \frac{1}{|\mathcal{M}|} \sum_{v \in \mathcal{M}} c_v \text{Rank}_v(f^{(i)})$
- **Design Motivation**: Different verifiers capture different evaluation dimensions; ensemble aggregation reduces individual bias and selects the globally optimal video.

### Verifier Selection

Three multimodal reward models are employed: VisionReward (human preference across 29 dimensions), VideoScore (multi-dimensional LMM-based scoring), and VideoLLaMA3 (a state-of-the-art multimodal understanding model). VBench serves as the ground-truth verifier for measuring the performance upper bound.

## Key Experimental Results

### Main Results (VBench Total Score Improvement)

| Model | w/o TTS | +TTS | Gain% | Semantic Gain% |
|------|-------|------|-------|------------|
| CogVideoX-5B | 81.61 | **84.42** | +3.44% | +10.1% |
| CogVideoX-2B | 80.91 | 83.89 | +3.68% | +3.38% |
| OpenSora-v1.2 | 79.76 | 81.65 | +2.37% | +9.87% |
| Pyramid-Flow (FLUX) | 81.61 | **86.51** | +5.86% | **+18.6%** |
| Pyramid-Flow (SD3) | 81.72 | 85.31 | +4.39% | +13.8% |
| NOVA | 78.56 | 79.80 | +1.58% | +2.43% |

**Dimension-level highlights** (Pyramid-Flow FLUX):
- Multiple Objects: 61.08 → 88.93 (+45.6%)
- Scene: 47.65 → 56.07 (+17.7%)
- Object Class: 93.49 → 99.69 (+6.6%)

### Ablation Study

| Configuration | Description | Outcome |
|------|------|------|
| ToF vs. Linear Search | Comparison under equal compute budget | ToF achieves comparable performance with ~70% fewer GFLOPs |
| Single vs. Multi-Verifier | VisionReward / VideoScore / VideoLLaMA3 | Ensemble further raises the TTS curve |
| Small model (NOVA 0.6B) vs. Large model (CogVideoX-5B) | Effect of model scale on TTS gains | Larger models benefit substantially more from TTS |
| Different generation dimensions | Scene / Object / Motion etc. | Common semantic dimensions show large gains; Motion Smoothness shows limited improvement |

**GFLOPs comparison** ($N=7$):

| Model | Linear Search | ToF Search | Ratio |
|------|---------|--------|------|
| Pyramid-Flow (FLUX) | 5.22×10⁷ | 1.62×10⁷ | 31% |
| Pyramid-Flow (SD3) | 3.66×10⁷ | 1.13×10⁷ | 31% |
| NOVA | 4.02×10⁶ | 1.41×10⁶ | 35% |

### Key Findings

- **TTS consistently improves quality across all video generation models**, eventually converging to an upper bound.
- **Stronger base models benefit more from TTS**: CogVideoX-5B achieves larger gains and higher efficiency than NOVA.
- **ToF search is far more efficient than linear search**: it achieves comparable or superior performance at approximately one-third of the computation.
- **Semantic alignment dimensions show the most significant gains** (Object Class +19.5%, Scene +18.6%), while implicit attributes such as motion smoothness show limited improvement.
- **Multi-verifier ensemble further raises the TTS performance ceiling.**

## Highlights & Insights

1. **Paradigm transfer**: This is the first systematic adaptation of TTS from LLMs to video generation, establishing a general search framework for the domain.
2. **Elegant ToF design**: Leveraging the frame-by-frame nature of autoregressive models, tree search is constructed along the temporal dimension, reducing complexity from $O(TN)$ to $O(N+T)$.
3. **Hierarchical prompting strategy**: Attending to different quality dimensions at different stages mirrors human multi-level judgment of video quality.
4. **Verifiers as the critical bottleneck**: The framework's performance ceiling is determined by verifier quality, underscoring the importance of advancing video evaluation models.
5. **Implicit conclusion**: Inference-time computation may be more cost-effective than additional training — offering a new optimization pathway beyond training-time scaling.

## Limitations & Future Work

1. **ToF is limited to autoregressive models**: Full-frame diffusion models (e.g., CogVideoX) can only use linear search and cannot benefit from ToF's efficiency advantages.
2. **Verifier capability constrains the upper bound**: Current verifiers are insufficient for evaluating motion smoothness, temporal flickering, and related dimensions, limiting gains on these metrics.
3. **Inference overhead remains substantial**: Even with ToF's efficiency improvements, inference costs in large-scale deployment far exceed those of single-pass generation.
4. **More sophisticated search strategies unexplored**: Techniques common in LLM TTS such as MCTS and beam search have not been investigated.
5. **Evaluation limited to VBench**: Human preference studies are absent from the evaluation.

## Related Work & Insights

- **OpenAI o1 / DeepSeek-R1** established the immense potential of TTS in LLMs; this paper transfers that insight to visual generation.
- **TTS for image generation** provides direct inspiration, though the temporal dimension of video introduces new challenges.
- **NOVA, Pyramid-Flow**, and other autoregressive video models are naturally compatible with ToF's tree-search structure.
- Verifier quality is the decisive factor in TTS effectiveness — improvements to video evaluation models (e.g., VideoLLaMA3) directly translate into stronger TTS performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic introduction of TTS to video generation; the ToF search design is both novel and efficient.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Six generation models, three verifiers, and multi-dimensional analysis; human evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Concepts are clearly presented, the framework is well-structured, and algorithm pseudocode is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a fundamentally new optimization pathway for video generation beyond training-time scaling, with broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation](../../NeurIPS2025/llm_reasoning/tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)
- [\[ICLR 2026\] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation](../../ICLR2026/llm_reasoning/prismaudio_decomposed_chain-of-thoughts_and_multi-dimensional_rewards_for_video-.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](../../NeurIPS2025/llm_reasoning/limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)

</div>

<!-- RELATED:END -->
