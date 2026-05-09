---
title: >-
  [Paper Note] TrajTok: Learning Trajectory Tokens Enhances Video Understanding
description: >-
  [CVPR 2026][Video Understanding][Video Tokenization] This paper proposes TrajTok—the first end-to-end differentiable trajectory-based video tokenizer—which encodes video into object trajectory tokens via implicit spatiotemporal clustering, requiring no external segmentation or tracking pipeline. TrajTok achieves +4.8% on K400, +4.1% on SSv2, and +8.8% on long-video QA benchmarks, with inference efficiency on par with the most efficient baselines.
tags:
  - CVPR 2026
  - Video Understanding
  - Video Tokenization
  - Trajectory Tokens
  - End-to-End Differentiable
  - Token Compression
  - Video LLM
date: 2026-05-08
content_hash: 02b9ca4198d4517e
---

# TrajTok: Learning Trajectory Tokens Enhances Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2602.22779](https://arxiv.org/abs/2602.22779)
**Code**: None
**Area**: Video Understanding
**Keywords**: Video Tokenization, Trajectory Tokens, End-to-End Differentiable, Token Compression, Video LLM

## TL;DR

This paper proposes TrajTok—the first end-to-end differentiable trajectory-based video tokenizer—which encodes video into object trajectory tokens via implicit spatiotemporal clustering, requiring no external segmentation or tracking pipeline. TrajTok achieves +4.8% on K400, +4.1% on SSv2, and +8.8% on long-video QA benchmarks, with inference efficiency on par with the most efficient baselines.

## Background & Motivation

**Background**: The dominant approach in video Transformers is to partition videos into spatiotemporal patches to generate tokens; however, token count grows linearly or quadratically with video length, leading to severe redundancy. TrajViT first demonstrated that grouping tokens by object trajectories outperforms patch-based tokens.

**Limitations of Prior Work**: TrajViT relies on an external SAM+SAM2 segmentation-tracking pipeline, which introduces three fundamental limitations: (1) the pipeline is slow and non-differentiable, operating as an independent preprocessing step; (2) segmentation granularity is fixed by a general-purpose segmentation model and cannot adapt to downstream task requirements (e.g., body-part-level segmentation for dance understanding vs. person-level segmentation for formation recognition); (3) performance gains diminish as data scale increases—poor scalability.

**Key Challenge**: While the superiority of the trajectory token paradigm has been established, the method for generating trajectories (i.e., the external pipeline) becomes the bottleneck for both performance and efficiency.

**Goal**: To design an end-to-end differentiable, lightweight, and efficient trajectory tokenizer that decouples token count from video duration and allows segmentation granularity to be driven by downstream task objectives.

**Key Insight**: Trajectory generation is reformulated as an implicit spatiotemporal clustering problem—rather than pursuing pixel-level segmentation accuracy, the focus shifts to optimizing semantic-level grouping capability.

**Core Idea**: Learnable queries perform implicit spatiotemporal clustering to generate trajectory masks, jointly trained end-to-end with downstream objectives, allowing task targets to inversely shape segmentation granularity.

## Method

### Overall Architecture

TrajTok consists of a Universal Segmenter and a Trajectory Encoder, trained jointly. Given an input video $\mathbf{V}\in\mathbb{R}^{T\times H\times W\times 3}$, the segmenter generates soft/hard segmentation masks; the encoder aggregates masked regions into $N$ trajectory tokens $\mathbf{Z}\in\mathbb{R}^{N\times d}$ (where $N$ varies dynamically with scene complexity), which are then passed to a downstream Transformer or LLM.

### Key Designs

1. **Universal Segmenter**:
    - **Function**: Partitions the video into semantically coherent trajectory regions in a single forward pass.
    - **Mechanism**: ConvNeXt-tiny extracts multi-scale features at 1/4 resolution per frame, $\mathbf{F}\in\mathbb{R}^{T\times h\times w\times d}$; 128 learnable queries $\mathbf{Q}$ perform cross-attention over features via a Perceiver layer (with 1D RoPE encoding applied to $\mathbf{F}$ for spatiotemporal position); output softmax soft segmentation maps $\mathbf{M}^{\text{soft}}_{k,t,i,j}=\text{softmax}_k(\hat{\mathbf{q}}_k\cdot\mathbf{F}_{t,i,j})$; empty-mask queries are automatically discarded, and long videos are processed in parallel chunks. A key trick is that patch feature gradients are detached before entering the Perceiver to prevent unstable co-adaptation.
    - **Design Motivation**: "Pixel-perfect segmentation is not required"—downstream understanding tasks only need semantic grouping capability. Dice + Focal loss (rather than cross-entropy) is used to emphasize discovering all object regions over pixel-level precision.

2. **Trajectory Encoder**:
    - **Function**: Aggregates segmented regions into compact trajectory token representations.
    - **Mechanism**: Initial embeddings are computed via soft-mask-weighted aggregation to maintain differentiability: $\mathbf{z}_k^{\text{init}}=\sum_{t,i,j}\mathbf{M}^{\text{soft}}_{k,t,i,j}\cdot\mathbf{F}_{t,i,j}$. A refinement stage applies a second Perceiver with masked cross-attention (hard masks), where each query attends only to features within its corresponding region to ensure decoupling. An adaptive Matryoshka mechanism allows each trajectory to output $n\in\{1,2,4\}$ sub-tokens (initialized with Fourier positional embeddings to ensure diversity); $n$ is sampled randomly during training and adjusted according to the computational budget at inference.
    - **Design Motivation**: Soft aggregation ensures gradient flow back to the segmenter; hard masks ensure decoupling between trajectories; adaptive token counts balance efficiency and expressiveness (4 tokens for motion-complex trajectories, 1 for simple ones).

3. **Three Application Scenarios**:
    - **Function**: Validates the cross-scenario applicability of TrajTok as a general-purpose module.
    - **Mechanism**: **TrajViT2** (training a CLIP video encoder from scratch); **TrajAdapter** (freezing a pretrained ViT and inserting TrajTok as a feature adapter); **TrajVLM** (replacing patch pooling with TrajTok as the visual-language connector in a LLaVA architecture, processing 128 frames).
    - **Design Motivation**: Demonstrates that trajectory tokens serve not only as a tokenizer but also as a general feature reorganization module.

### Loss & Training

The segmenter uses Dice + Focal loss with pseudo-labels generated from the TrajViT pipeline on 8M annotated videos and 15M images. Downstream objectives include CLIP contrastive loss (TrajViT2), classification loss (TrajAdapter), and language modeling loss (TrajVLM). The segmenter can be jointly trained with downstream tasks (TrajViT2) or pretrained and frozen for reuse (TrajAdapter/TrajVLM). Training uses a global batch of 1024 images + 128 videos on 8×A100 for 20 epochs.

## Key Experimental Results

### Main Results

| Model | K400 Top-1↑ | SSv2 Top-1↑ | ActivityNet vid2txt R@5↑ | VATEX vid2txt R@5↑ |
|------|------------|------------|-------------------------|-------------------|
| ViT3D | 54.2 | 46.3 | 35.6 | 60.2 |
| TokenLearner | 52.9 | 42.4 | 36.2 | 58.8 |
| TrajViT | 55.3 | 45.7 | 38.1 | 61.1 |
| **TrajViT2** | **59.1 (+4.8)** | **48.7 (+4.1)** | **42.2 (+4.1)** | **65.0 (+3.9)** |

| VLM Connector | LongVideoBench | LVBench |
|-----------|---------------|---------|
| PatchVLM (pool=3, 32 frames) | Baseline | Baseline |
| **TrajVLM (128 frames)** | **+8.8%** | **+5.4%** |

| Probing Method | K400 (VideoMAE-v2) | SSv2 (V-JEPA2) |
|------------|-------------------|----------------|
| Linear probing | 79.4 | 73.7 |
| Attentive probing | 80.2 | 74.2 |
| **TrajAdapter (4 tok/traj)** | **82.5** | **75.1** |

### Ablation Study

| Module | Variation | VEQ(%) | STQ(%) | R@5 |
|------|------|--------|--------|-----|
| Default architecture | — | 42.3 | 70.1 | 22.1 |
| Perceiver | No gradient detach | 34.1 (↓8.2) | 59.3 (↓10.8) | 18.3 (↓3.8) |
| Segmentation loss | Remove Dice loss | 39.0 (↓3.3) | 68.9 (↓1.2) | 16.7 (↓5.4) |
| Backbone | No hierarchical features | 39.3 (↓3.0) | 66.2 (↓3.9) | 19.2 (↓2.9) |

### Key Findings

- Gradient detaching is the most critical design choice (removing it causes VEQ to drop by 8.2%)—it prevents unstable co-adaptation between patch features and queries.
- End-to-end training enables segmentation granularity to adapt to downstream tasks: the CLIP objective drives finer foreground segmentation and coarser background merging (verified by Figure 3 visualizations).
- TrajViT2 scales with data far better than TrajViT—from 1M to 8M training samples, it consistently maintains a large margin over ViT3D.
- The tokenizer has only 46M parameters, an order of magnitude smaller than the ViT-Large backbone (304M).
- TrajViT2 slightly underperforms ViT3D on ImageNet, as the segmenter produces too few tokens for simple single-object scenes.

## Highlights & Insights

- "Pixel-perfect segmentation is not required" is the core insight—for understanding tasks, semantic grouping capability matters far more than boundary precision.
- Applying the Matryoshka idea to trajectory tokens is elegant: motion-complex trajectories use multiple tokens while simple ones use a single token, with flexible adjustment at inference time.
- End-to-end training allows segmentation granularity to be inversely shaped by downstream tasks, offering far greater flexibility than fixed pipelines.
- The three application scenarios of TrajTok (encoder/adapter/connector) validate its versatility as a general-purpose module.

## Limitations & Future Work

- Slight underperformance vs. ViT3D on ImageNet—the segmenter generates too few tokens for simple single-object scenes; adaptive strategies are needed.
- TrajVLM is currently validated at small scale (Qwen3-4B); scaling to larger LLMs with more data is a future direction.
- Segmenter pretraining relies on TrajViT pipeline-generated pseudo-labels; fully self-supervised trajectory discovery is worth exploring.
- Temporal chunking for long videos may lose cross-chunk trajectory continuity information.

## Related Work & Insights

- **vs. TrajViT**: End-to-end differentiability replaces the external pipeline, improving efficiency by an order of magnitude with better data scalability, at the cost of slightly lower segmentation precision (which does not affect understanding tasks).
- **vs. TokenLearner/ToMe/RLT and other token compression methods**: Achieves comparable inference FLOPs but significantly higher accuracy, demonstrating that trajectory-level grouping is more effective than simple token merging.
- **vs. patch pooling VLM connectors (Molmo/LLaVA)**: Large advantage on long video (+8.8% on LongVideoBench), as trajectory token count is decoupled from frame count.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First end-to-end differentiable trajectory tokenizer; a paradigm-level contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three-scenario validation with comprehensive ablations and data scaling experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ The trajectory token framework directly inspires visual token compression and video understanding research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_video.md)
- [\[CVPR 2026\] AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing](autogaze_attend_before_attention_efficient_video.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)

</div>

<!-- RELATED:END -->
