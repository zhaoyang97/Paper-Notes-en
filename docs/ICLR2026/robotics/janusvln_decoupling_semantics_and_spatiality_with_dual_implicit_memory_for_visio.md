---
title: >-
  [Paper Note] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation
description: >-
  [ICLR2026][Robotics][Vision-Language Navigation] Inspired by the left-brain/right-brain division of semantic understanding and spatial cognition in humans, this paper proposes JanusVLN—the first dual implicit neural memory framework designed for VLN—which models spatial-geometric memory and visual-semantic memory respectively as fixed-size KV Caches, enabling efficient spatial reasoning from RGB video alone and achieving state-of-the-art performance on the VLN-CE benchmark.
tags:
  - ICLR2026
  - Robotics
  - Vision-Language Navigation
  - Dual Implicit Memory
  - Spatial-Geometric Encoding
  - KV Cache
  - Embodied AI
date: 2026-05-08
content_hash: 3bf6aeed97f2f9aa
---

# JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation

**Conference**: ICLR2026  
**arXiv**: [2509.22548](https://arxiv.org/abs/2509.22548)  
**Code**: [Project Page](https://miv-xjtu.github.io/JanusVLN.github.io/)  
**Area**: Robotics  
**Keywords**: Vision-Language Navigation, Dual Implicit Memory, Spatial-Geometric Encoding, KV Cache, Embodied AI  

## TL;DR
Inspired by the left-brain/right-brain division of semantic understanding and spatial cognition in humans, this paper proposes JanusVLN—the first dual implicit neural memory framework designed for VLN—which models spatial-geometric memory and visual-semantic memory respectively as fixed-size KV Caches, enabling efficient spatial reasoning from RGB video alone and achieving state-of-the-art performance on the VLN-CE benchmark.

## Background & Motivation
Vision-Language Navigation (VLN) requires an agent to navigate in unseen environments following natural language instructions. Multimodal large language model (MLLM)-based approaches have risen rapidly in recent years, yet they share three common bottlenecks:

1. **Limitations of explicit semantic memory**: One line of methods (MapNav, etc.) constructs cognitive maps from textual descriptions, but pure text cannot precisely convey spatial relationships, leading to loss of critical visual, geometric, and contextual information; another line (NaVILA, StreamVLN, etc.) stores historical observation frames, requiring full re-processing of all history at every decision step, incurring severe computational redundancy.
2. **Memory bloat**: The explicit memory of both method families grows exponentially with navigation time, making it difficult for the model to extract key information from massive fragmented memories.
3. **Spatial blindness of 2D encoders**: Visual encoders in existing VLN models almost universally inherit the 2D image-text pretraining paradigm of CLIP, which is adept at capturing high-level semantics but lacks understanding of 3D geometric structure and spatial information—yet navigation is fundamentally a 3D physical interaction task.

## Core Problem
How can one simultaneously address (1) spatial information loss, (2) computational redundancy, and (3) memory bloat in VLN, using only RGB video as input?

## Method

### Overall Architecture
JanusVLN adopts a dual-encoder architecture that decouples visual perception into two pathways: **semantic understanding** and **spatial cognition**:

- **Semantic encoder**: Directly reuses the visual encoder of Qwen2.5-VL to extract semantic features $S_t$.
- **Spatial-geometric encoder**: Introduces VGGT (Visual Geometry Grounded Transformer), a feed-forward 3D visual geometry foundation model pretrained on pixel–3D point cloud pairs, to extract spatial-geometric features $G_t$ from RGB video without any explicit 3D data (depth maps, point clouds, etc.).

### Dual Implicit Memory
The core innovation lies in modeling both types of memory as **fixed-size implicit neural representations**—i.e., historical KV Caches—rather than explicit textual descriptions or raw image frames:

- **Implicit neural representation**: Caches historical KV pairs that have been deeply processed by Transformer attention modules. These KV pairs are not simple raw data storage but high-level semantic abstractions and structured representations distilled by the neural network.
- **Hybrid incremental update strategy**:
    - **Sliding window queue** $M_{sliding}$ (capacity $n$): Stores the KV Caches of the most recent $n$ frames in FIFO order, ensuring the model focuses on the latest context.
    - **Initial window** $M_{initial}$: Permanently retains the KV Caches of the first few frames, leveraging the "Attention Sinks" phenomenon to provide a global anchor for the entire navigation task.
- For each new frame, the model only needs to compute cross-attention between new frame image tokens and the implicit memory to retrieve historical information, without reprocessing historical frames:

$$G_t = \text{Decoder}(\text{CrossAttn}(\text{Encoder}(x_t), \{M_{initial}, M_{sliding}\}))$$

### Spatial-Aware Feature Fusion
After obtaining semantic features $S'_t$ and spatial-geometric features $G'_t$ (shape-aligned via spatial merging), they are fused through a lightweight two-layer MLP projection:

$$F_t = S'_t + \lambda \cdot \text{MLP}(G'_t)$$

where $\lambda = 0.2$ controls the weight of spatial-geometric features. The fused features $F_t$ are fed together with instruction text embeddings into the LLM to predict the next action.

### Training Details
- Backbone: Qwen2.5-VL 7B + VGGT
- Only the LLM and projection layer are fine-tuned (learning rates 2e-5 and 1e-5 respectively); both encoders are frozen
- Initial window: 8 frames; sliding window: 48 frames
- Additional data: 155K trajectories from a ScaleVLN subset + 14K trajectories collected via DAgger

## Key Experimental Results

### R2R-CE Val-Unseen (Primary Metrics: SR / SPL)

| Method | Input | SR↑ | SPL↑ |
|--------|-------|-----|------|
| NaVILA | RGB | 54.0 | 49.0 |
| StreamVLN | RGB | 56.9 | 51.9 |
| **JanusVLN** | **RGB** | **60.5** | **56.8** |
| JanusVLN* (no extra data) | RGB | 52.8 | 49.2 |

- Compared to multi-input methods using panoramic views + odometry + depth, JanusVLN achieves SR gains of 10.5–35.5 using only monocular RGB.
- Compared to depth-dependent methods g3D-LF and NaVid-4D, SR improves by 12.6–16.7.
- JanusVLN* without extra data still surpasses methods relying on additional data by 3.7–18.8 in SPL.

### RxR-CE Val-Unseen

| Method | SR↑ | SPL↑ | nDTW↑ |
|--------|-----|------|-------|
| NaVILA | 49.3 | 44.0 | 58.8 |
| StreamVLN | 52.9 | 46.0 | 61.9 |
| **JanusVLN** | **56.2** | **47.5** | **62.1** |

### Ablation Study (R2R-CE, no extra data)

| Configuration | SR↑ | SPL↑ |
|---------------|-----|------|
| JanusVLN (full) | 52.8 | 49.2 |
| w/o spatial implicit memory | 47.0 | 40.9 |
| w/o semantic implicit memory | 45.5 | 40.0 |
| w/o dual implicit memory | 24.8 | 16.8 |

- Replacing VGGT with DINOv2 or SigLIP 2 yields only marginal gains (SR 47.5/47.9 vs. 47.0), as 2D pretrained encoders are highly redundant with Qwen2.5-VL's representations.
- A randomly initialized VGGT provides no meaningful gain (SR 47.2), confirming that the advantage stems from 3D geometric priors rather than increased parameter count.

### Inference Efficiency

| Memory Type | Inference Time (32 frames) | SR↑ |
|-------------|---------------------------|-----|
| VGGT original (full sequence recomputation) | 1549 ms | 51.2 |
| Cached Memory (KV Cache) | 149 ms | 51.7 |

KV caching reduces inference overhead by 69%–90% while slightly improving performance.

## Highlights & Insights
1. **Paradigm innovation**: The first work to introduce dual implicit neural memory into VLN, replacing bloating explicit memory with fixed-size KV Caches—a fundamentally new memory paradigm.
2. **Elegant 3D prior injection**: VGGT extracts 3D spatial-geometric information from pure RGB video without any additional 3D sensors or data, yet substantially improves spatial reasoning.
3. **Efficient incremental updates**: The hybrid initial-window + sliding-window strategy preserves global anchors while focusing on recent context; inference time scales only linearly with frame count.
4. **Rigorous ablation design**: Replacing encoders with DINOv2/SigLIP 2/randomly initialized VGGT convincingly demonstrates that performance gains stem from 3D geometric priors rather than model capacity.

## Limitations & Future Work
1. The sliding window size (48 frames) is a fixed hyperparameter that may be suboptimal for navigation tasks of varying complexity; adaptive window adjustment warrants exploration.
2. The spatial-geometric feature weight $\lambda = 0.2$ is fixed across all scenarios; a dynamic regulation mechanism could further improve performance.
3. The VGGT encoder is fully frozen; end-to-end fine-tuning or partial fine-tuning of the spatial encoder may unlock additional potential.
4. Evaluation is limited to VLN-CE on Matterport3D scenes; generalization to larger-scale real-world environments remains to be examined.
5. Real-world experiments present only qualitative results, lacking systematic quantitative evaluation.

## Related Work & Insights
- **vs. MapNav** (textual cognitive maps): JanusVLN replaces explicit textual descriptions with implicit KV Caches, avoiding spatial information loss and descriptive redundancy; SR improves by 20.8.
- **vs. NaVILA/StreamVLN** (historical frame storage): Eliminates per-step reprocessing of all historical frames, substantially improving inference efficiency; SR improves by 3.6–10.8.
- **vs. g3D-LF / NaVid-4D** (depth-dependent): Surpasses depth-input methods using only RGB, improving SR by 12.6–16.7 and eliminating reliance on expensive 3D sensors.
- **vs. Uni-NaVid / NaVid** (RGB-only baselines): Achieves substantial gains under identical input conditions, validating the effectiveness of the dual implicit memory paradigm.

The cognitive science analogy of hemispheric specialization provides a clear design philosophy for multi-encoder architectures, which can be generalized to other embodied tasks requiring simultaneous semantic and spatial perception. The Attention Sinks phenomenon (initial frames consistently receiving high attention weights) is validated in navigation scenarios, providing supporting evidence for KV Cache management in long-sequence LLM inference. The approach of using VGGT as a bridge to endow 2D MLLMs with 3D perception is transferable to other 3D interaction tasks such as robotic manipulation and autonomous driving.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Dual implicit memory paradigm is a first in VLN; 3D prior injection is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablations, but real-world experiments are primarily qualitative)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, intuitive analogies, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Establishes a new paradigm, achieves significant SOTA gains, and provides directional guidance for future VLN research)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)
- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](../../CVPR2026/robotics/towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)

<!-- RELATED:END -->
