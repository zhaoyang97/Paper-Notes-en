---
title: >-
  [Paper Note] PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation
description: >-
  [ICLR 2026][LLM Reasoning][Video-to-Audio] This work is the first to integrate decomposed Chain-of-Thought reasoning with multi-dimensional reinforcement learning (RL) for video-to-audio (V2A) generation. It addresses the objective entanglement problem via four specialized CoT modules (semantic/temporal/aesthetic/spatial) paired with corresponding reward functions, and proposes the Fast-GRPO algorithm to substantially reduce RL training cost.
tags:
  - ICLR 2026
  - LLM Reasoning
  - Video-to-Audio
  - Chain-of-Thought
  - Reinforcement Learning
  - Multi-dimensional Rewards
  - Diffusion Models
date: 2026-05-08
content_hash: 057789224eea1f80
---

# PrismAudio: Decomposed Chain-of-Thoughts and Multi-dimensional Rewards for Video-to-Audio Generation

**Conference**: ICLR 2026
**arXiv**: [2511.18833](https://arxiv.org/abs/2511.18833)
**Code**: [https://PrismAudio.github.io](https://PrismAudio.github.io)
**Area**: LLM Reasoning
**Keywords**: Video-to-Audio, Chain-of-Thought, Reinforcement Learning, Multi-dimensional Rewards, Diffusion Models

## TL;DR

This work is the first to integrate decomposed Chain-of-Thought reasoning with multi-dimensional reinforcement learning (RL) for video-to-audio (V2A) generation. It addresses the objective entanglement problem via four specialized CoT modules (semantic/temporal/aesthetic/spatial) paired with corresponding reward functions, and proposes the Fast-GRPO algorithm to substantially reduce RL training cost.

## Background & Motivation

- **Background**: V2A generation must simultaneously satisfy four human perceptual dimensions—semantic consistency, temporal synchronization, aesthetic quality, and spatial accuracy—which inherently compete and trade off against one another.
- **Limitations of Prior Work**: Existing methods employ a single loss function that jointly optimizes multiple competing objectives, preventing the model from achieving a good balance across dimensions. For instance, emphasizing semantic consistency can degrade aesthetic quality. Pioneer works such as ThinkSound introduce CoT reasoning but use a monolithic reasoning path for all audio analysis tasks, leading to conflated analytical frameworks and frequent multimodal hallucinations. Furthermore, existing V2A methods lack mechanisms for learning from human perceptual preferences, producing audio that is technically correct but perceptually unsatisfying. On the training side, existing GRPO implementations (e.g., Flow-GRPO) require SDE sampling at every denoising step, incurring prohibitive computational cost.
- **Key Challenge**: Monolithic objective optimization and monolithic reasoning pipelines conflate fundamentally different analytical tasks, making it impossible to achieve well-balanced multi-dimensional performance.
- **Goal**: Introduce decomposed CoT reasoning and multi-dimensional RL into V2A generation to achieve comprehensive, balanced improvements across all perceptual dimensions with high training efficiency.

## Method

### Overall Architecture

PrismAudio comprises three main stages: (1) a CoT-aware audio foundation model built on a diffusion Transformer with flow matching; (2) a four-dimensional decomposed CoT reasoning module, whose training data is constructed using Gemini 2.5 Pro and used to fine-tune VideoLLaMA2; and (3) a Fast-GRPO multi-dimensional CoT-RL framework for efficient multi-objective optimization.

### Key Designs

**1. CoT-Aware Audio Foundation Model Enhancement**

- **Function**: Upgrades the video encoder and text encoder to support multi-dimensional CoT reasoning.
- **Mechanism**: Replaces CLIP with VideoPrism as the video encoder to capture richer video semantics, and replaces T5 with T5-Gemma as the text encoder to better handle structured reasoning text.
- **Design Motivation**: CLIP processes video frame-by-frame and lacks holistic understanding; standard T5 struggles with the complex reasoning text produced by the CoT modules.

**2. Four-Dimensional Decomposed CoT Reasoning**

- **Function**: Decomposes monolithic reasoning into four specialized modules—Semantic CoT, Temporal CoT, Aesthetic CoT, and Spatial CoT.
- **Mechanism**: Gemini 2.5 Pro generates training data; VideoLLaMA2 is fine-tuned to produce four types of specialized reasoning text, which are concatenated to form enhanced text conditions.
- **Design Motivation**: Different analytical tasks require fundamentally different analytical frameworks—semantic analysis focuses on content identification, spatial reasoning requires directional localization logic, and aesthetic evaluation requires subjective quality assessment. Mixing them in a single path leads to insufficient treatment of each dimension.

**3. Multi-dimensional Reward Functions**

- **Function**: Designs four specialized reward functions aligned with the CoT dimensions.
- **Mechanism**: The semantic reward uses MS-CLAP to evaluate audio-text alignment; the temporal reward uses Synchformer to assess audio-visual synchronization; the aesthetic reward uses Audiobox Aesthetics to predict MOS scores; the spatial reward uses StereoCRW to verify directional accuracy.
- **Design Motivation**: A single reward function leads to suboptimal trade-offs across dimensions.

**4. Fast-GRPO Algorithm**

- **Function**: Proposes a hybrid ODE-SDE sampling strategy that substantially reduces GRPO training overhead.
- **Mechanism**: Deterministic ODE steps are used for the majority of the denoising trajectory, with SDE steps applied only within a randomly selected small window for exploration. This reduces the policy model NFE from $T$ to $w$ (window width $w \ll T$).
- **Design Motivation**: Pure SDE methods (e.g., Flow-GRPO) create an efficiency bottleneck by evaluating the policy at every step; the hybrid strategy achieves a balance between exploration and efficiency.

### Loss & Training

- A Windowed GRPO objective is used, computing the clipped surrogate objective only over SDE window steps.
- Multi-reward weighted aggregation: $R_{total}^i = \sum_{k=1}^K \lambda_k R_k(\mathbf{x}_T^i, c)$; advantage scores are computed via within-group mean and standard deviation normalization.
- Training proceeds in three stages: pre-training → CoT fine-tuning → RL post-training.

## Key Experimental Results

### Main Results

| Method | Params | CLAP↑ | DeSync↓ | PQ↑ | CE↑ | CRW↓ | FD↓ | MOS-Q↑ | MOS-C↑ |
|--------|--------|-------|---------|-----|-----|------|-----|--------|--------|
| ThinkSound | 1.3B | 0.43 | 0.55 | 6.15 | 3.95 | 13.47 | 1.17 | 4.05 | 4.18 |
| MMAudio | 1.03B | 0.40 | 0.46 | 5.94 | 3.88 | - | 2.17 | 3.95 | 4.03 |
| **PrismAudio** | **518M** | **0.47** | **0.41** | **6.38** | **4.29** | **7.72** | **1.08** | **4.21** | **4.22** |

On the VGGSound test set, PrismAudio achieves state-of-the-art performance across all perceptual dimensions with only 40% of ThinkSound's parameter count.

### Ablation Study

| Reward Strategy | CLAP↑ | DeSync↓ | CE↑ | CRW↓ | FD↓ |
|-----------------|-------|---------|-----|------|-----|
| Baseline (No RL) | 0.47 | 0.42 | 3.81 | 15.30 | 1.90 |
| Semantic Only | 0.54 | 0.58 | 3.93 | 11.89 | 1.84 |
| Temporal Only | 0.46 | 0.35 | 3.63 | 13.08 | 1.88 |
| Aesthetic Only | 0.46 | 0.42 | 3.92 | 13.51 | 4.50 |
| **Multi-dimensional** | **0.52** | **0.36** | **4.26** | **12.87** | **1.53** |

### Key Findings

1. **Single-dimension rewards cause severe objective entanglement**: Semantic Only achieves the highest CLAP of 0.54 but degrades DeSync to 0.58; Aesthetic Only raises PQ to 7.06 but doubles FD to 4.50.
2. **Multi-dimensional rewards are the only approach that achieves comprehensive balanced improvements**, simultaneously benefiting semantic, temporal, aesthetic, and spatial dimensions.
3. **Fast-GRPO converges approximately 3× faster than Flow-GRPO**, surpassing the latter's 600-step final performance in only 200 steps and achieving a higher final reward score (0.51 vs. 0.47).
4. **Decomposed CoT substantially outperforms monolithic CoT**: MultiCoT leads by large margins on semantic (CLAP 0.52 vs. 0.46) and aesthetic (CE 4.26 vs. 3.79) dimensions.

## Highlights & Insights

- The **CoT-reward correspondence design** is the central contribution: each CoT module is paired with a specialized reward function, enabling RL optimization to precisely guide improvements in each reasoning dimension.
- This work is the **first to introduce RL into V2A generation**, establishing a new paradigm for human preference alignment in audio generation.
- The AudioCanvas benchmark (3,177 videos, 300 single-event categories, 501 multi-event samples) fills a gap in V2A evaluation.
- On the out-of-domain AudioCanvas benchmark, PrismAudio's semantic and synchronization metrics even surpass ground-truth audio (GT), indicating that the RL framework can generate audio that better satisfies proxy evaluation metrics than natural audio.

## Limitations & Future Work

1. The phenomenon of surpassing GT on out-of-domain data reflects a gap between proxy metrics and human perception, motivating the need for better evaluation metrics.
2. Whether the four-dimensional CoT categorization is optimal remains an open question; finer-grained or alternative decompositions may exist.
3. CoT annotations in AudioCanvas rely on Gemini 2.5 Pro, potentially introducing model bias.
4. The current system supports only 9-second audio generation; scalability to longer videos remains to be validated.

## Related Work & Insights

- **ThinkSound**: The first V2A method to introduce CoT reasoning, but employs monolithic reasoning without RL alignment—this paper directly addresses its three major limitations.
- **Flow-GRPO / DanceGRPO**: Extend GRPO to flow matching models but support only single-objective optimization with low efficiency—Fast-GRPO's hybrid ODE-SDE strategy provides an efficient alternative.
- **RLHF in LLMs**: RL-based preference alignment is mature in the text domain; this paper extends it to multi-dimensional audio generation with diffusion models, with implications transferable to image and video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to bring decomposed CoT and multi-dimensional RL to V2A; the CoT-reward correspondence design is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ In-domain and out-of-domain evaluation, objective and subjective metrics, and detailed ablations covering every design decision.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-motivated arguments, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for V2A generation; Fast-GRPO and AudioCanvas offer broad community value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[NeurIPS 2025\] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing](../../NeurIPS2025/llm_reasoning/thinksound_chain-of-thought_reasoning_in_multimodal_large_language_models_for_au.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](../../CVPR2026/llm_reasoning/reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[ACL 2026\] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards](../../ACL2026/llm_reasoning/trojail_trajectory-level_optimization_for_multi-turn_large_language_model_jailbr.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)

<!-- RELATED:END -->
