---
title: >-
  [Paper Note] TrajTok: Learning Trajectory Tokens Enhances Video Understanding
description: >-
  [CVPR2026][Video Understanding][Video tokenization] This paper proposes TrajTok — an end-to-end differentiable trajectory tokenizer that implicitly clusters video pixels into object trajectory tokens, replacing external segmentation-and-tracking pipelines. It achieves significant improvements across three settings: training from scratch (TrajViT2), feature adaptation (TrajAdapter), and vision-language model connectors (TrajVLM), with particularly large gains on long-video QA over patch pooling.
tags:
  - CVPR2026
  - Video Understanding
  - Video tokenization
  - trajectory token
  - end-to-end segmentation
  - video CLIP
  - VLM connector
  - token compression
  - object trajectory
date: 2026-05-08
content_hash: 177dfbddef4623a0
---

# TrajTok: Learning Trajectory Tokens Enhances Video Understanding

**Conference**: CVPR2026
**arXiv**: [2602.22779](https://arxiv.org/abs/2602.22779)
**Code**: To be confirmed
**Area**: Video Segmentation / Video Understanding
**Keywords**: Video tokenization, trajectory token, end-to-end segmentation, video CLIP, VLM connector, token compression, object trajectory

## TL;DR

This paper proposes TrajTok — an end-to-end differentiable trajectory tokenizer that implicitly clusters video pixels into object trajectory tokens, replacing external segmentation-and-tracking pipelines. It achieves significant improvements across three settings: training from scratch (TrajViT2), feature adaptation (TrajAdapter), and vision-language model connectors (TrajVLM), with particularly large gains on long-video QA over patch pooling.

## Background & Motivation

1. **Explosion of video token counts**: Current video Transformers tokenize via spatiotemporal patches, causing token counts to grow linearly or even quadratically with resolution and frame count, resulting in severe memory bottlenecks.
2. **Insufficiency of existing token reduction methods**: Token pruning/merging methods (e.g., TokenLearner, RLT) either require a pre-specified token count and cannot adapt to input complexity, or are sensitive to scene motion and lack robustness.
3. **Limitations of TrajViT**: The prior work TrajViT introduced a trajectory-based tokenization paradigm and first demonstrated that grouped tokens outperform raw patch tokens across all tasks, but it relies on an external SAM+SAM2 segmentation-tracking pipeline — which is slow, non-trainable, and produces semantically fixed granularity.
4. **Task-agnostic semantic granularity**: The trajectory granularity produced by general-purpose segmentation models may not be optimal for downstream tasks (e.g., dance analysis requires fine-grained body parts, whereas formation recognition benefits from holistic tokens), and cannot be adapted per task.
5. **Pixel-perfect segmentation is unnecessary**: Conventional segmentation models expend substantial computation on pixel-precise masks, whereas high-level understanding tasks rely more on correct semantic grouping than on boundary precision.
6. **Scalability bottleneck**: TrajViT shows sharply diminishing performance gains when scaling data from 1M to 8M, indicating that the fixed segmentation pipeline limits the model's scalability.

## Method

### Overall Architecture

TrajTok consists of two differentiable modules trained jointly:

- **Universal Segmenter**: Implicitly clusters the input video in a single forward pass to produce object trajectory masks.
- **Trajectory Encoder**: Aggregates pixels/features according to the masks to generate compact trajectory tokens.

Given input $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$, the output is $\mathbf{Z} \in \mathbb{R}^{N \times d}$, where $N$ varies dynamically with the semantic complexity of the scene.

### Key Designs

**1. Universal Segmenter**

- **Per-frame feature extraction**: A lightweight ConvNeXt-Tiny extracts multi-scale feature maps, which are upsampled to 1/4 resolution and summed to obtain dense features $\mathbf{F} \in \mathbb{R}^{T \times h \times w \times d}$.
- **Learnable query clustering**: $N_q=128$ learnable queries $\mathbf{Q}$ interact with the features via cross-attention in Perceiver layers, with 1D RoPE encoding applied to features for spatiotemporal positional awareness.
- **Soft segmentation**: Dot products between queries and feature points are passed through softmax to yield soft masks $\mathbf{M}^{\text{soft}} \in [0,1]^{N_q \times T \times h \times w}$; queries corresponding to empty masks are discarded, enabling a dynamic token count.
- **Gradient detachment**: Gradients through the features $\mathbf{F}$ are detached before entering the Perceiver, preventing unstable co-adaptation between patch features and queries.

**2. Trajectory Encoder**

- **Soft aggregation initialization**: The soft mask is used to compute a weighted sum of features, yielding an initial trajectory embedding $\mathbf{z}_k^{\text{init}}$ that supports gradient backpropagation.
- **Hard mask refinement**: Argmax is applied to $\mathbf{M}^{\text{soft}}$ to obtain hard masks $\mathbf{M}^{\text{hard}}$, which are used in masked cross-attention to refine token representations and ensure disentanglement.
- **Adaptive token count**: Inspired by Matryoshka representations, each trajectory can emit $n \in \{1,2,4\}$ tokens; multi-token trajectories are initialized with Fourier positional encodings to encourage diversity; $n$ is sampled randomly during training and adjusted according to the compute budget at inference.

### Loss & Training

- **Segmentation loss**: Dice loss + Focal loss (no cross-entropy); Dice loss ensures all target regions are discovered, and Focal loss handles class imbalance.
- **Downstream loss**: Contrastive learning loss for CLIP (TrajViT2), classification loss (TrajAdapter), or autoregressive VLM loss (TrajVLM).
- In the TrajViT2 setting, segmentation and downstream losses are optimized jointly; in the TrajAdapter/TrajVLM settings, the segmenter is pre-trained and then frozen.

## Key Experimental Results

### Setting 1: TrajViT2 (Training Video Encoder from Scratch with CLIP Objective)

A ViT-Large-scale encoder is trained on 4M videos + 15M images:

| Model | K400 (Top-1) | SSv2 (Top-1) | ActivityNet txt2vid R@5 | VATEX vid2txt R@5 |
|-------|-------------|-------------|------------------------|-------------------|
| ViT3D | 54.2 | 46.3 | 37.1 | 60.2 |
| TokenLearner | 52.9 | 42.4 | 36.4 | 58.8 |
| TrajViT | 55.3 | 45.7 | 38.4 | 61.1 |
| **TrajViT2** | **59.1** | **48.7** | **40.1** | **65.0** |

- K400 outperforms ViT3D by **+4.9%** and TrajViT by **+3.8%**.
- ActivityNet vid2txt R@5 exceeds TrajViT by **+4.1%**.
- Inference FLOPs are comparable to the most efficient ViViT variants, far below the overhead of TrajViT's external pipeline.

### Setting 2: TrajAdapter (Feature Adapter)

TrajTok is inserted on top of frozen features from VideoMAE-v2-Huge and V-JEPA2-Huge:

| Method | V-JEPA2 K400 | V-JEPA2 SSv2 |
|--------|-------------|-------------|
| Linear probing | 84.5 | 73.7 |
| Attentive probing | 85.1 | 74.2 |
| **TrajAdapter (4 tok/traj)** | **88.0** | **75.1** |

TrajAdapter improves K400 accuracy on V-JEPA2 from 85.1% to **88.0%** (+2.9%).

### Ablation Study

**Segmenter Design Ablation** (Table 4):

| Variant | VEQ (%) | STQ (%) | Retrieval R@5 |
|---------|---------|---------|---------------|
| Default | 42.3 | 70.1 | 22.1 |
| w/o Dice loss | 39.0 (↓3.3) | 68.9 (↓1.2) | 16.7 (↓5.4) |
| w/o gradient detach | 34.1 (↓8.2) | 59.3 (↓10.8) | 18.3 (↓3.8) |
| w/o hierarchical features | 39.3 (↓3.0) | 66.2 (↓3.9) | 19.2 (↓2.9) |

**Encoder Design Ablation** (Table 5): Removing the hard attention mask causes R@5 to drop by 4.7–5.1%, confirming that trajectory disentanglement is critical.

## Highlights & Insights

- **End-to-end differentiability**: This is the first work to unify trajectory segmentation and video tokenization into a fully end-to-end trainable module, enabling downstream tasks to backpropagate gradients to adjust segmentation granularity.
- **Generality across three settings**: The same module serves as a tokenizer (TrajViT2), feature adapter (TrajAdapter), or VLM connector (TrajVLM), demonstrating strong universality.
- **Adaptive semantic granularity**: After training with a CLIP objective, segmentation granularity adjusts automatically — foreground objects are segmented more finely while background regions are merged more aggressively (as shown in Figure 3).
- **Strong advantage on long videos**: TrajVLM outperforms PatchVLM by **+8.8%** on LongVideoBench and **+5.4%** on LVBench, demonstrating that trajectory tokens are naturally suited for long-range reasoning.
- **Parameter and efficiency advantages**: The entire tokenizer contains only 46M parameters (1/7 of ViT-Large) and achieves inference FLOPs comparable to the best token merging methods.

## Limitations & Future Work

1. **Limited pixel-level segmentation precision**: The lightweight design and low-resolution output lead to missed small objects, over-merged backgrounds, and imprecise boundaries, making the method unsuitable for tasks requiring accurate masks (e.g., instance segmentation benchmarks).
2. **Slightly lower ImageNet performance**: In simple single-object scenes, the segmenter produces too few tokens, limiting fine-grained discriminative capacity.
3. **Inconsistent short-video performance in TrajVLM**: On certain short-video QA benchmarks, performance is lower than patch pooling, suggesting that trajectory tokens may be less effective than patches for simple short videos.
4. **Dependency on pseudo-labels**: Segmenter pre-training still relies on pseudo-labels generated by the TrajViT external pipeline, and the method has not fully freed itself from dependence on SAM/SAM2.
5. **Limited scale of TrajVLM**: Validation is performed only on Qwen3-4B; the method has not been tested on models at the 70B+ scale, leaving large-scale effectiveness to be confirmed.

## Related Work & Insights

| Dimension | TrajViT (Prior Work) | TrajTok (Ours) |
|-----------|---------------------|----------------|
| Trajectory generation | External SAM+SAM2 pipeline | End-to-end lightweight segmenter |
| Segmentation precision | Pixel-level accurate | Coarse-grained semantic grouping |
| Trainability | Non-differentiable, frozen | Fully differentiable, jointly trained |
| Task adaptation | Fixed granularity | Adaptive to downstream objective |
| Scalability | Diminishing returns with more data | Continues to scale |
| Parameter overhead | SAM2 itself 304M+ | Tokenizer only 46M |
| Efficiency | High pipeline latency | Single forward pass |

Compared with token merging methods such as TokenLearner and RLT, TrajTok achieves substantially higher performance on both retrieval and classification at comparable inference efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ — Advances trajectory tokenization from an external pipeline to a fully end-to-end differentiable framework with clear motivation and broad impact
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across three settings (pre-training / adaptation / VLM), thorough ablations, and complete scalability analysis
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with progressively built motivation and information-rich figures and tables
- Value: ⭐⭐⭐⭐⭐ — The trajectory tokenizer is highly generalizable and makes important contributions to video understanding efficiency and long-video reasoning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochat-m1_collaborative_policy_planning_for_video_understanding_via_multi-age.md)
- [\[CVPR 2026\] Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing](attend_before_attention_efficient_and_scalable_video_understanding_via_autoregre.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)

</div>

<!-- RELATED:END -->
