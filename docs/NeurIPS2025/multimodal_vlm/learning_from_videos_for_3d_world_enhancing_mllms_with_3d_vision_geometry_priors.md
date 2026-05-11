---
title: >-
  [Paper Note] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors
description: >-
  [NeurIPS 2025][Multimodal VLM][MLLM] VG-LLM proposes integrating a 3D visual geometry encoder (VGGT) into multimodal large language models…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "MLLM"
  - "3D Visual Geometry"
  - "Spatial Reasoning"
  - "Video Understanding"
  - "VGGT"
date: 2026-05-08
content_hash: 7d8a46194380e220
---

# Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors

**Conference**: NeurIPS 2025
**arXiv**: [2505.24625](https://arxiv.org/abs/2505.24625)
**Code**: [GitHub](https://lavi-lab.github.io/VG-LLM)
**Area**: 3D Vision / Multimodal Large Language Models
**Keywords**: MLLM, 3D Visual Geometry, Spatial Reasoning, Video Understanding, VGGT

## TL;DR

VG-LLM proposes integrating a 3D visual geometry encoder (VGGT) into multimodal large language models, enabling the extraction and fusion of 3D geometric priors from video input alone—without any explicit 3D data. This approach significantly improves MLLM performance on 3D scene understanding and spatial reasoning tasks, with the 4B model surpassing Gemini-1.5-Pro on VSI-Bench.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved remarkable progress in 2D image and video understanding, yet continue to struggle with 3D spatial reasoning. Existing approaches that adapt MLLMs for 3D scene understanding share a common limitation: **they rely on explicit 3D data as input**.

The core problem chain:
1. Methods such as Video-3D LLM require depth maps or point clouds, while GPT4Scene requires BEV images rendered from reconstructed 3D point clouds.
2. Such 3D data is difficult to obtain in many real-world scenarios.
3. Directly estimating 3D attributes from images introduces estimation errors that degrade performance.
4. Conventional visual encoders process video frames independently into tokens, discarding inter-frame 3D geometric information such as correspondences.

**Core Idea**: Leverage a pretrained 3D visual geometry model (e.g., VGGT) as an auxiliary encoder to implicitly extract 3D geometric priors from video, without requiring any explicit 3D data input.

## Method

### Overall Architecture

VG-LLM augments a standard MLLM with an additional 3D visual geometry encoder branch:
- Input image sequences → 2D visual encoder extracts semantic features + 3D geometry encoder extracts geometric features
- Both feature streams are fused at the patch level → fed into the MLLM backbone to generate responses

### Key Designs

1. **3D Visual Geometry Encoder**:

    - VGGT-1B is adopted as the geometry encoder; VGGT is pretrained on 3D tasks such as point cloud prediction.
    - VGGT consists of three components: a per-image encoder, a cross-frame fusion decoder, and task-specific prediction heads.
    - This work uses only the encoder and fusion decoder (excluding prediction heads) to extract features containing 3D geometric priors.
    - Key capability: VGGT captures inter-frame correspondences, and its latent features can recover 3D scene structure.

2. **Visual Feature Fusion**:

    - The 2D encoder outputs $T_i^{V'} \in \mathbb{R}^{\lfloor h/2p \rfloor \times \lfloor w/2p \rfloor \times c}$ (spatially compressed via Qwen2.5-VL's 2×2 spatial merging).
    - The 3D geometric features $T_i^G$ are transformed into $T_i^{G'}$ using the same spatial merging strategy (concatenating adjacent 2×2 patches followed by a two-layer MLP).
    - Final features: $T_i^S = T_i^{G'} + T_i^{V'}$, using simple additive fusion.
    - The fused features are concatenated with text embeddings and fed into the MLLM backbone.

3. **Multi-Task 3D Scene Understanding**:

    - **3D Visual Grounding**: Given a language query, output a frame index and a 3D oriented bounding box $(x,y,z,w,h,d,\psi,\theta,\phi)$.
    - **3D Dense Captioning**: Given the 3D center coordinates of an object, generate a detailed description of that object.
    - **3D Video Object Detection**: Detect all objects appearing in a video within a unified coordinate system.
    - All tasks are unified under text generation and trained with a next-token prediction objective.

4. **Spatial Reasoning Enhancement Training**:

    - The SPAR-7M dataset is used for spatial reasoning instruction tuning, sampling only 3% (234K samples).
    - The LLaVA-Hound subset of LLaVA-Video-178K is mixed in to preserve general capabilities.
    - The first-frame coordinate system serves as the reference frame; all coordinates are transformed accordingly.

### Loss & Training

- Standard next-token prediction loss (cross-entropy).
- The 2D visual encoder, 3D geometry encoder, and multimodal connector are frozen during training.
- Only the MLLM backbone is trained (via LoRA or full fine-tuning).
- Built upon Qwen2.5-VL (3B and 7B variants) combined with VGGT-1B.
- Training is conducted on 8×H100 GPUs: 9–12 hours for 3D scene understanding and 7–9 hours for spatial reasoning.

## Key Experimental Results

### Main Results

| Task / Benchmark | Metric | VG-LLM-8B | Comparison | Notes |
|---|---|---|---|---|
| ScanRefer (3D Grounding) | Acc@0.25 | **41.6** (57.6*) | Video-3D LLM: 58.1 | *with proposal refinement; no 3D input |
| Scan2Cap (3D Captioning) | C@0.5 | **80.0** | Video-3D LLM: 80.0 | Matches SOTA using only RGB input |
| 3D Video Detection (4 frames) | F1@0.25 | **41.2** | Qwen2.5-VL-7B: 32.5 | +8.7% absolute gain |
| VSI-Bench (Spatial Reasoning) | Avg | **50.7** | Gemini-1.5-Pro: 45.4 | 4B version (47.3) already surpasses |

### Ablation Study

| Configuration | F1 (3D Detection) | Notes |
|---|---|---|
| Qwen2.5-VL-3B (no geometry) | 30.0 | Baseline |
| + VGGT geometry (VG-LLM-4B) | **38.2** | +8.2%; geometric priors are effective |
| Qwen2.5-VL-7B (no geometry) | 32.5 | Stronger baseline |
| + VGGT geometry (VG-LLM-8B) | **41.2** | +8.7%; consistent improvement |

| Fusion Strategy | VSI-Bench Avg | Notes |
|---|---|---|
| 2D visual only | Baseline | No 3D priors |
| Additive fusion (ours) | **50.7** | Simple yet effective |

### Key Findings

- **Without any explicit 3D input**, VG-LLM matches or surpasses methods that require 3D data, demonstrating that 3D geometric priors can be implicitly learned from video.
- 3D geometry enhancement yields the most pronounced gains in **egocentric-to-allocentric transfer** (F1 on 3D video detection improves by 10.7%).
- VG-LLM-4B achieves 47.3% on VSI-Bench, surpassing the best commercial model Gemini-1.5-Pro (45.4%).
- The model is robust to varying frame counts: models trained on 4 frames generalize to 6 frames at inference with negligible performance degradation.
- Only 3% of SPAR-7M data (234K samples) is sufficient to yield substantial spatial reasoning improvements.

## Highlights & Insights

- **Minimal yet effective architecture**: Simply appending a frozen 3D geometry encoder alongside a standard MLLM, with additive feature fusion, yields substantial gains.
- **Implicit 3D modeling via VGGT**: The inter-frame correspondence modeling capability acquired during VGGT pretraining is the core driver—it relieves the MLLM from having to infer 3D structure from raw tokens.
- **Data efficiency**: Only 234K spatial reasoning samples (3% of SPAR-7M) are sufficient to achieve state-of-the-art performance on VSI-Bench.
- **Unified text generation framework**: All 3D tasks (grounding, captioning, detection) are unified under text generation, requiring no task-specific heads.

## Limitations & Future Work

- VGGT-1B introduces additional computational overhead at inference time, particularly for long video sequences.
- The absolute accuracy of 3D visual grounding remains lower than methods that use real 3D data (41.6% vs. 58.1%).
- Additive fusion may not be optimal; cross-attention or gating mechanisms could potentially yield further improvements.
- Using the first frame as the reference coordinate system may be suboptimal for scenes with large-scale motion.

## Related Work & Insights

- **vs. Video-3D LLM**: Video-3D LLM injects 3D coordinates into patch-level visual features and requires explicit depth input; VG-LLM operates on pure video input and replaces explicit geometry with implicit geometric encoding.
- **vs. GPT4Scene**: GPT4Scene requires 3D reconstruction to generate BEV images; VG-LLM requires no reconstruction step.
- **vs. SPAR**: SPAR enhances spatial reasoning through synthetic data augmentation but focuses solely on the data side; VG-LLM introduces geometric capability at the model architecture level.

## Rating

- Novelty: ⭐⭐⭐⭐ Using a 3D visual geometry model as an auxiliary encoder for MLLMs is a natural yet effective idea, though the fusion strategy is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3D scene understanding, spatial reasoning, and general benchmarks across multiple dimensions, with detailed ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, experiments are comprehensively presented, and the structure is well-organized.
- Value: ⭐⭐⭐⭐⭐ Demonstrates that MLLM 3D understanding can be substantially enhanced without explicit 3D data, with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[CVPR 2026\] SpatialStack: Layered Geometry-Language Fusion for 3D VLM Spatial Reasoning](../../CVPR2026/multimodal_vlm/spatialstack_layered_geometry-language_fusion_for_3d_vlm_spatial_reasoning.md)
- [\[NeurIPS 2025\] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models](bridgevla_input-output_alignment_for_efficient_3d_manipulation_learning_with_vis.md)

</div>

<!-- RELATED:END -->
