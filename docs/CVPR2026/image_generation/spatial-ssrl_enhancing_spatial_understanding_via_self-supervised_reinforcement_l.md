---
title: >-
  [Paper Note] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning
description: >-
  [CVPR 2026][Image Generation][Spatial Understanding] This paper proposes Spatial-SSRL, a self-supervised reinforcement learning paradigm that automatically constructs five pretext tasks (patch reordering…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Spatial Understanding"
  - "Self-Supervised Learning"
  - "Reinforcement Learning (RLVR)"
  - "Large Vision-Language Models"
  - "Depth Perception"
date: 2026-05-08
content_hash: 7da4f02504f33d99
---

# Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2510.27606](https://arxiv.org/abs/2510.27606)  
**Code**: [GitHub](https://github.com/InternLM/Spatial-SSRL)  
**Area**: Image Generation
**Keywords**: Spatial Understanding, Self-Supervised Learning, Reinforcement Learning (RLVR), Large Vision-Language Models, Depth Perception

## TL;DR
This paper proposes Spatial-SSRL, a self-supervised reinforcement learning paradigm that automatically constructs five pretext tasks (patch reordering, flip recognition, cropped patch inpainting, depth ordering, and relative 3D position prediction) from standard RGB/RGB-D images. By optimizing LVLMs with GRPO, the method achieves average improvements of 3.89%–4.63% across seven spatial benchmarks without any human annotation or external tools.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) are approaching saturation on tasks such as VQA and image captioning, yet their spatial understanding capabilities remain far below human-level performance. Existing approaches fall into two categories: data-driven SFT (constructing spatial QA pairs for fine-tuning) and RLVR (reinforcement learning with verifiable rewards).

**Limitations of Prior Work**: SFT methods rely on costly human annotation or GPT-4-generated QA pairs and tend to overfit to dataset-specific patterns. Tool-augmented approaches (e.g., using depth estimators or object detectors) involve complex pipelines and high computational costs. Simulation-based methods (rendering 3D scenes) suffer from domain gaps with the real world. RLVR approaches are constrained by specific environments (e.g., 3D scans), limiting data scale and coverage.

**Key Challenge**: Spatial understanding requires large-scale verifiable supervision signals, yet all existing methods incur prohibitive costs to obtain them—either through expensive human annotation, complex toolchains, or dependence on specialized 3D datasets.

**Goal**: Design a zero-annotation, tool-free, and scalable self-supervised scheme that generates verifiable spatial understanding training signals and integrates naturally with the RLVR training paradigm.

**Key Insight**: The inherent structural consistency within images (relative depth, geometric consistency, viewpoint invariance) itself provides deterministic and verifiable supervision signals. Pretext tasks from visual self-supervised learning (SSL) are repurposed as reward functions for RLVR rather than traditional feature pre-training objectives.

**Core Idea**: Classic SSL tasks (jigsaw puzzles, flip detection, etc.) are reformulated as QA prompts with deterministic verification functions for LVLMs, and GRPO is directly applied for post-training.

## Method

### Overall Architecture
Spatial-SSRL consists of two stages: (1) self-supervised task construction—automatically generating QA pairs for five pretext tasks from RGB/RGB-D images (the Spatial-SSRL-81k dataset, with 100% annotation accuracy); and (2) RL training—an SFT cold-start phase to familiarize the model with task formats, followed by GRPO optimization. The five tasks span 2D layout understanding (depth-free) and 3D spatial reasoning (depth-based).

### Key Designs

1. **Shuffled Patch Reordering**:

    - **Function**: The image is divided into an $M \times N$ grid, randomly shuffled, and the model is asked to predict the permutation $\pi^{-1}$ that restores the original arrangement.
    - **Mechanism**: For image $I$, a patch grid $\mathcal{X} = \{x_{i,j}\}$ is constructed and a random permutation $\pi$ is applied to obtain the shuffled image. The ground-truth answer is the inverse permutation $\pi^{-1} = [\pi^{-1}(0), \pi^{-1}(1), \ldots, \pi^{-1}(M \times N - 1)]$. Optionally, one random patch is masked in white to increase difficulty and prevent the model from exploiting boundary artifacts.
    - **Design Motivation**: Recovering the original patch ordering inherently requires understanding global 2D layout consistency and relative positional relationships—skills that transfer directly to understanding object arrangements in real scenes.

2. **Flipped Patch Recognition**:

    - **Function**: One randomly selected patch is flipped horizontally or vertically, and the model must identify the index of the flipped patch and the flip direction.
    - **Mechanism**: The selected patch $\hat{x}_t$ is transformed with equal probability via vertical flip $x_{\text{vert}}(r,c) = x(P_H - 1 - r, c)$ or horizontal flip $x_{\text{horz}}(r,c) = x(r, P_W - 1 - c)$, with the answer being $[t, d]$.
    - **Design Motivation**: Detecting subtle orientation violations requires sensitivity to local geometry, mirror symmetry, and directional cues such as text, faces, and shadows.

3. **Cropped Patch Inpainting**:

    - **Function**: A randomly cropped region is masked in black, and the model must select the correct filling patch from four candidates (including three distractors).
    - **Mechanism**: The distractors are carefully designed—a 90° rotated version, an interior sub-region, and an exterior expanded region—all visually similar to the correct answer, forcing the model to attend to fine-grained texture continuity and semantic consistency.
    - **Design Motivation**: Tests texture-context matching and fine-grained structural reasoning ability.

4. **Regional Depth Ordering**:

    - **Function**: Three regions with clearly separated depths are selected, labeled with numeric indices, presented in shuffled order, and the model is asked to sort them from nearest to farthest.
    - **Mechanism**: Three regions satisfying the following constraints are selected from depth map $D$: intra-region depth range $r(R_i) < r_{\max} = 0.15$ (intra-region consistency), and inter-region separation $d(R_i, R_{i+1}) > d_{\min} = 0.05$ (inter-region discriminability).
    - **Design Motivation**: Sorting requires integrating depth cues, perspective understanding, and ordinal reasoning—foundational capabilities for 3D scene understanding.

5. **Relative 3D Position Prediction**:

    - **Function**: Given the orientation of an object, the model predicts the relative position of another point in the object's coordinate frame (combinations of left/right/front/back).
    - **Mechanism**: A 2D rigid-body transformation maps camera-frame coordinates $(x_2, z_2)$ to the object frame $(\tilde{x}_2, \tilde{z}_2)$, and direction labels $(\tilde{p}_x, \tilde{p}_z)$ are determined by thresholding. Object orientation $\theta$ is sampled uniformly from the four cardinal directions.
    - **Design Motivation**: Requires mental rotation, egocentric coordinate transformation, and depth integration—the highest-order task among the five for spatial understanding.

### Loss & Training
- **Cold-Start SFT**: Fine-tuning for 5 epochs on approximately 3,600 samples (lr=$1 \times 10^{-5}$) to familiarize the model with task formats.
- **GRPO Optimization**: KL regularization weight 0.01, 5 rollouts per sample, temperature 1.0, batch size 128, lr=$1 \times 10^{-6}$, 360 steps.
- **Reward Design**: $r = 0.9 \cdot r_{\text{acc}} + 0.1 \cdot r_{\text{fmt}}$, with accuracy weighted substantially higher than format.
- Think tags are used to guide chain-of-thought output.

## Key Experimental Results

### Main Results (7 Spatial Understanding Benchmarks)

| Model | Spatial457 | 3DSRBench | SpatialEval | QSpatial+ | What'sUp | ViewSpatial | VSI-Bench | Avg |
|------|-----------|-----------|-------------|-----------|----------|-------------|-----------|-----|
| Qwen2.5-VL-3B | 33.70 | 50.30 | 54.65 | 33.66 | 85.85 | 35.38 | 27.84 | 45.91 |
| Spatial-SSRL-3B | 46.07 | 51.72 | 59.59 | 39.60 | 86.71 | 36.62 | 33.49 | 50.54 |
| Δ | **+12.37** | +1.42 | +4.94 | +5.95 | +0.86 | +1.24 | +5.65 | **+4.63** |
| Qwen2.5-VL-7B | 44.67 | 53.39 | 62.37 | 46.53 | 86.95 | 36.83 | 38.08 | 52.69 |
| Spatial-SSRL-7B | 53.34 | 56.53 | 64.03 | 54.46 | 90.61 | 37.81 | 39.29 | 56.58 |
| Δ | **+8.67** | +3.14 | +1.66 | +7.93 | +3.66 | +0.98 | +1.21 | **+3.89** |

Improvements are observed across all seven benchmarks, with the largest gain of +12.37% on Spatial457.

### Reasoning Capability Verification

| Configuration | Avg Accuracy | Notes |
|------|----------|------|
| Qwen2.5-VL-7B (w/o reasoning) | 52.69 | Baseline |
| Qwen2.5-VL-7B (w/ reasoning) | 49.58 | Reasoning hurts! (−3.11) |
| Spatial-SSRL-7B (w/ reasoning) | **56.58** | Reasoning chain is effective (+3.89) |

Enabling chain-of-thought reasoning in the baseline model actually degrades performance (What'sUp: 86.95→70.61), indicating that the base model lacks effective spatial reasoning ability. Spatial-SSRL successfully trains the model to generate effective reasoning chains through RL.

### Key Findings
- 3D reasoning benchmarks benefit most (Spatial457 +12.37%, QSpatial+ +7.93%), validating the contribution of depth-based tasks.
- The finding that enabling CoT reasoning in the base model degrades performance is significant—it demonstrates that spatial reasoning ability requires dedicated training rather than simple prompt engineering.
- On Qwen3-VL-4B, spatial performance improves by +1.29% while general VQA also improves by +1.18%, confirming that the method does not harm general capabilities.
- The Spatial-SSRL-81k dataset achieves 100% annotation accuracy because all answers are derived from deterministic transformations—a standard that noisy-detector-based methods cannot match.
- The cold-start SFT is necessary: direct RL training results in a success rate below 5% for generating correctly formatted outputs.

## Highlights & Insights
- **The combination of SSL and RLVR** is the primary contribution. SSL pretext tasks naturally provide deterministic and verifiable answers, perfectly aligning with the verifiable reward requirement of RLVR. This insight may inspire a substantial body of follow-up work extending other SSL tasks to LVLM post-training.
- **The complementary design of five tasks** covers the full hierarchy from 2D layout to 3D spatial relations: patch reordering (global layout) → flip recognition (local orientation) → inpainting (texture consistency) → depth ordering (3D depth) → 3D position (egocentric coordinate transformation).
- **The ingenuity of distractor design**: in the inpainting task, distractors consisting of rotated versions, interior sub-regions, and exterior expanded regions prevent the model from exploiting low-level features.

## Limitations & Future Work
- The two depth-dependent tasks require RGB-D data, limiting the range of applicable data sources (although depth can be estimated monocularly, this introduces noise).
- The relative weighting of the five tasks is not carefully tuned; the current setup uses equal mixing ratios.
- Evaluation is conducted only on Qwen2.5-VL and Qwen3-VL; generalization to other LVLM architectures (e.g., LLaVA, InternVL) remains unknown.
- While the 81k data scale is already smaller than typical SFT approaches, there is further room to improve RL training efficiency.
- Additional SSL tasks could be explored—such as color channel permutation, frequency-domain transformation prediction, and multi-view consistency verification.

## Related Work & Insights
- **vs. SpatialLadder/SpaceR**: These methods depend on 3D scan datasets and complex pipelines, whereas Spatial-SSRL requires only standard RGB/RGB-D images with a minimal pipeline. In terms of performance, Spatial-SSRL-7B (56.58) surpasses SpaceR-7B (54.54).
- **vs. Jigsaw-R1/Visual Jigsaw**: These methods share a similar SSL+RL philosophy but cover only the jigsaw task. Spatial-SSRL designs five complementary tasks, achieving more comprehensive spatial understanding improvements.
- **vs. SSL4RL**: Focuses exclusively on 2D tasks, whereas Spatial-SSRL covers both 2D and 3D; the depth-based tasks contribute substantial performance gains.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm of using SSL pretext tasks as RLVR rewards is pioneering, and the five-task design is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Seven benchmarks, three base models, and general capability verification are provided, though ablation studies could be more detailed.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Method motivation is clearly articulated, mathematical formalization of task designs is rigorous, and figures are excellent.
- **Value**: ⭐⭐⭐⭐⭐ Zero annotation, scalable, and naturally compatible with RLVR; opens a new direction for improving spatial understanding in LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
