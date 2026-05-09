---
title: >-
  [Paper Note] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards
description: >-
  [NeurIPS 2025][Multimodal VLM][Spatial Reasoning] This paper proposes SpatialThinker, which trains MLLMs to construct scene graphs and perform structured spatial reasoning via online RL with multi-objective dense spatial rewards (lexicographic gating over format → count → accuracy → spatial localization). Using only 7K samples, it surpasses GPT-4o on 3DSRBench by 12.1%.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Spatial Reasoning
  - Scene Graph
  - Reinforcement Learning
  - Dense Reward
  - 3D Understanding
date: 2026-05-08
content_hash: 263df177e6aed2be
---

# SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2511.07403](https://arxiv.org/abs/2511.07403)
**Code**: [https://github.com/SpatialThinker](https://github.com/SpatialThinker)
**Area**: Multimodal VLM
**Keywords**: Spatial Reasoning, Scene Graph, Reinforcement Learning, Dense Reward, 3D Understanding

## TL;DR
This paper proposes SpatialThinker, which trains MLLMs to construct scene graphs and perform structured spatial reasoning via online RL with multi-objective dense spatial rewards (lexicographic gating over format → count → accuracy → spatial localization). Using only 7K samples, it surpasses GPT-4o on 3DSRBench by 12.1%.

## Background & Motivation

### State of the Field

Spatial understanding remains a core weakness of MLLMs. Existing approaches either require massive data (SpatialVLM uses 2B samples), rely on explicit 3D inputs (depth maps/point clouds), or apply RL with sparse rewards.

### Limitations of Prior Work

Sparse rewards (based solely on whether the final answer is correct) provide insufficient guidance for visually grounded spatial reasoning; SFT learns static patterns rather than reasoning strategies.

### Root Cause

**Scene-graph-guided reasoning**: The model first constructs a question-relevant sub-scene graph (objects + bounding boxes + relations), then performs reasoning over this structured representation.

**Dense spatial rewards**: A four-component reward (format + count + accuracy + CIoU spatial localization) with lexicographic gating to prevent reward hacking.

## Method

### Key Designs
1. **Reasoning Template**: `<observe>` scene description → `<scene>` JSON scene graph (object IDs + bboxes + relation triples) → `<think>` reasoning → `<answer>` answer
2. **Dense Spatial Rewards + Lexicographic Gating**:

    - Format reward $R_f$ ($w=0.1$): JSON parsability + field completeness
    - Count reward $R_c$ ($w=0.2$): matching degree of object and relation counts
    - Accuracy reward $R_a$ ($w=0.5$): correctness of the final answer
    - Spatial reward $R_s$ ($w=0.2$): activated only when the answer is correct; evaluates localization precision via Hungarian matching + CIoU
    - Gating: $R_{total} = \mathbb{I}[R_f=1] \cdot (w_f R_f + w_c R_c + w_a R_a + \mathbb{I}[R_a=1] \cdot w_s R_s)$
3. **STVQA-7K Dataset**: 7.5K high-quality spatial VQA samples generated from Visual Genome scene graphs, covering 9 categories of spatial reasoning.

### Loss & Training
Built on Qwen2.5-VL-7B, trained with GRPO (no critic network) on 4×H100 GPUs for 15 hours.

## Key Experimental Results

### Main Results

| Model | 3DSRBench | CV-Bench | BLINK Avg. |
|------|-----------|---------|-----------|
| GPT-4o | 44.3 | 79.4 | 80.4 |
| Qwen2.5-VL-7B (base) | 48.4 | 68.6 | 68.2 |
| + Sparse RL | 52.4 (+4.0) | 72.1 | 72.8 |
| + **SpatialThinker (Dense)** | **55.6 (+7.2)** | **75.3** | **76.8** |

### Ablation Study

| Reward Configuration | 3DSRBench | Δ vs base |
|----------|-----------|-----------|
| No RL | 48.4 | - |
| Sparse reward | 52.4 | +4.0 |
| + Format + Count | 53.8 | +5.4 |
| + Spatial reward (no gating) | 54.1 | +5.7 |
| + Spatial reward (with gating) | **55.6** | **+7.2** |

### Key Findings
- The gain from dense rewards is 1.8× that of sparse rewards (+7.2 vs. +4.0).
- Lexicographic gating is critical — without it, the model generates excessive objects to exploit the spatial reward.
- Only 7K samples are sufficient to surpass methods trained on million-scale SFT datasets.

## Highlights & Insights
- The **"observe→localize→think→answer" reasoning template** emulates human spatial reasoning: perceive the scene, localize key objects, reason, then respond.
- **Dense spatial rewards on only 7K samples outperform million-scale SFT** — demonstrating that *what guidance is provided* matters more than *how much data is used*.
- Lexicographic gating is the key design for preventing reward hacking and is directly transferable to other multi-objective RL settings.

## Limitations & Future Work
- Scene graph construction quality depends on annotation accuracy in the training data; sub-graph extraction may be incomplete for highly complex scenes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to combine scene graphs with dense RL for spatial reasoning
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 12 benchmarks with detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Reward design and derivation are clearly presented
- Value: ⭐⭐⭐⭐⭐ Addresses spatial reasoning challenges with remarkably few training samples

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)

<!-- RELATED:END -->
