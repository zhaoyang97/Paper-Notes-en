---
title: >-
  [Paper Note] From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors
description: >-
  [ICLR 2026][Robotics][VLA models] This paper proposes FALCON (From Spatial to Action), which injects rich 3D spatial tokens from a spatial foundation model into the Action Head rather than the VLM backbone, achieving strong 3D spatial awareness in VLA models while maintaining flexible modality switching between RGB-only and RGB-D inputs. FALCON achieves state-of-the-art performance on both simulation and real-world tasks.
tags:
  - ICLR 2026
  - Robotics
  - VLA models
  - 3D spatial understanding
  - spatial foundation models
  - modality transferability
  - robotic manipulation
date: 2026-05-08
content_hash: 8e5ec73f972c9905
---

# From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors

**Conference**: ICLR 2026
**arXiv**: [2510.17439](https://arxiv.org/abs/2510.17439)
**Code**: [Available](https://falcon-vla.github.io/)
**Area**: Robotics
**Keywords**: VLA models, 3D spatial understanding, spatial foundation models, modality transferability, robotic manipulation

## TL;DR

This paper proposes FALCON (From Spatial to Action), which injects rich 3D spatial tokens from a spatial foundation model into the Action Head rather than the VLM backbone, achieving strong 3D spatial awareness in VLA models while maintaining flexible modality switching between RGB-only and RGB-D inputs. FALCON achieves state-of-the-art performance on both simulation and real-world tasks.

## Background & Motivation

Most existing VLA models are built upon 2D encoders but must execute manipulation tasks in the 3D physical world, creating a critical spatial reasoning gap. This manifests at three levels:

**Insufficient Spatial Representation**: 2D VLMs lack explicit 3D perception, making it difficult to generalize to scenarios requiring geometric, depth, and spatial relationship reasoning.

**Poor Modality Transferability**: Existing 3D-enhanced methods either depend on specific sensors (point clouds/depth maps) and fail when those sensors are unavailable, or inject weak 3D cues (e.g., pseudo-depth estimation) that are insufficient to capture robust 3D priors.

**Alignment Difficulty**: Concatenating spatial embeddings with text tokens disrupts existing vision-language alignment, and the scarcity of 3D data makes re-alignment difficult, leading to degraded zero-shot generalization.

## Method

### Overall Architecture

FALCON comprises three core components, inspired by the division-of-labor theory in neuroscience (the cerebral cortex for high-level reasoning, the cerebellum for fine motor control):

1. **2D VLM** (cerebral cortex role): Kosmos-2 (~1.6B parameters) processes image and language instructions, outputting semantic action tokens $\hat{\mathbf{t}}_{\text{act}}$.
2. **Embodied Spatial Model (ESM)** (spatial perception module): Built upon the spatial foundation model VGGT, it extracts 3D spatial tokens $\mathbf{T}_{\text{spl}}$, with optional fusion of depth and camera pose.
3. **Spatial-Enhanced Action Head** (cerebellum role): Fuses semantic and spatial representations to generate precise robot actions.

Total parameter count: ~2.9B (VLM 1.6B + ESM 1.0B + Action Head).

### Key Designs

**Embodied Spatial Model (ESM)**:

- Based on the VGGT spatial foundation model, it encodes input images into spatial tokens $\mathbf{T}_{\text{spl}} \in \mathbb{R}^{M \times D_s}$.
- Images are encoded by DINO into visual tokens $\mathbf{T}_{\text{vis}}$, which are concatenated with learnable camera tokens $\mathbf{t}_{\text{cam}}$ and fed into the spatial encoder (cross-attention + self-attention blocks).

**3D Condition Encoding and Injection**:

- **Camera pose** $P \in \mathbb{R}^7$: Encoded via an MLP into a GT camera token $\mathbf{t}_{\text{gt-cam}}$, replacing the learnable camera token.
- **Depth map** $D_t$: Normalized and concatenated with a validity mask, then encoded via 14×14 convolution into $\mathbf{T}_{\text{dpt}}$ and added element-wise to the image tokens.

**Stochastic Conditioning Strategy** (key innovation): During training, depth and pose are randomly injected with probability $p$:

$$(\mathbf{T}_{\text{spl}}, \hat{\mathbf{t}}_{\text{cam}}) = \mathcal{E}_{\text{spl}}(\mathbf{T}_{\text{vis}} + b_d \mathbf{T}_{\text{dpt}}, b_p \mathbf{t}_{\text{gt-cam}} + (1-b_p)\mathbf{t}_{\text{cam}})$$

where $b_d, b_p \sim \text{Bernoulli}(p)$. This ensures the model performs well regardless of whether additional 3D inputs are available.

**Spatial-Enhanced Action Head**:

The core fusion strategy uses element-wise addition:
1. Spatial tokens are compressed into a single vector $\mathbf{t}_{\text{spl}}$ via max-pooling.
2. A lightweight MLP adapter projects the vector into the VLM feature space: $\widetilde{\mathbf{t}}_{\text{spl}} = \mathcal{D}(\mathbf{t}_{\text{spl}})$.
3. Direct addition with the semantic action token: $\mathbf{f}_{\text{fused}} = \hat{\mathbf{t}}_{\text{act}} + \widetilde{\mathbf{t}}_{\text{spl}}$.
4. The fused representation is fed into an action predictor (MLP or LSTM) to generate a 7D action sequence.

### Loss & Training

**Training Objective**:

$$\mathcal{L} = \sum_{i=t}^{t+C-1} \text{MSE}(\hat{a}_{i,\text{pose}}, a_{i,\text{pose}}) + \lambda \cdot \text{BCE}(\hat{a}_{i,\text{gripper}}, a_{i,\text{gripper}})$$

- The first 6 dimensions (pose) use MSE loss; the 7th dimension (gripper) uses BCE loss.
- ESM's spatial reconstruction uses multi-task supervision over depth, point cloud maps, and pose (following VGGT).

**Two-Stage Post-Training**:
- **Stage 1**: All pretrained components are frozen; only lightweight adapters are trained to achieve initial alignment between spatial tokens and the VLA feature space.
- **Stage 2**: The VLM and adapters are jointly fine-tuned while other components remain frozen, allowing the VLM to implicitly incorporate spatial cues.

Training is conducted on 32 A100 GPUs.

## Key Experimental Results

### Main Results

**CALVIN Long-Horizon Manipulation (ABCD→D)**:

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg. Len↑ |
|--------|--------|--------|--------|--------|--------|-----------|
| RT-1 | 84.4 | 61.7 | 43.8 | 32.3 | 22.7 | 2.45 |
| RoboVLM | 96.7 | 93.0 | 89.9 | 86.5 | 82.6 | 4.49 |
| **FALCON** | **97.2** | **93.3** | **90.3** | **88.0** | **84.0** | **4.53** |

**CALVIN Zero-Shot Transfer (ABC→D)**:

| Method | Avg. Len↑ |
|--------|-----------|
| 3D Diffuser Actor (with GT point clouds) | 3.35 |
| RoboVLM | 4.25 |
| **FALCON (RGB-only)** | **4.40** |

**SimplerEnv WidowX Robot**:

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg. |
|--------|-----------|------------|-------------|--------------|------|
| SpatialVLA | 16.7% | 25.0% | 29.2% | 100% | 42.7% |
| **FALCON** | **62.5%** | **41.7%** | 20.8% | 100% | **56.3%** |

**SimplerEnv Google Robot**:

| Method | Pick Coke | Move Near | Open/Close | Drawer Apple | Avg. |
|--------|-----------|-----------|------------|--------------|------|
| RT-2-X (55B) | 78.7% | 77.9% | 25.0% | 3.7% | 46.3% |
| SpatialVLA | 86.0% | 77.9% | 57.4% | 0.0% | 55.3% |
| **FALCON (2.9B)** | **90.7%** | **79.2%** | 39.8% | **41.7%** | **62.9%** |

### Ablation Study

**Spatial Token Injection Location**:

| Injection Strategy | ABCD→D Avg. Len | ABC→D Avg. Len |
|--------------------|-----------------|----------------|
| Inject into VLM (FALCON_VLM-tokens) | 4.00 | 3.79 |
| **Inject into Action Head (FALCON)** | **4.08** | **3.91** |

**Fusion Strategy Comparison (CALVIN ABC→D)**:

| Strategy | Avg. Len↑ |
|----------|-----------|
| Cross-Attention | 3.68 |
| FiLM-Gated | 3.76 |
| **Element-wise Addition** | **3.91** |

**Modality Input Ablation (CALVIN ABC→D)**:

| Configuration | Avg. Len↑ |
|---------------|-----------|
| Kosmos-VLA (RGB-only, no ESM) | 3.48 |
| Kosmos-VLA (RGB-D, point cloud encoder) | 3.98 |
| FALCON (RGB-only) | 3.91 |
| FALCON (RGB-D) | **3.97** |
| FALCON (trained with RGB-D, depth removed at test time) | 3.95 |

### Key Findings

1. **Action Head injection >> VLM injection**: Injecting spatial tokens into the VLM disrupts pretrained semantic representations, degrading generalization (3.91 → 3.79); injecting into the Action Head preserves VLM integrity.
2. **Simplest fusion is optimal**: Element-wise addition outperforms cross-attention and FiLM-Gated with zero additional parameters.
3. **RGB-only surpasses explicit 3D input**: FALCON using only RGB outperforms 3D Diffuser Actor with GT point clouds (4.40 vs. 3.35).
4. **Flexible modality switching**: Adding depth/pose during training and removing it at test time still yields high performance (3.97 → 3.95), and vice versa.
5. **Substantial advantage in real-world spatial understanding**: On tasks requiring perception of object size and height differences, FALCON's success rate far exceeds baselines.
6. **Strong few-shot adaptation**: FALCON outperforms the second-best method by 27% in few-shot settings.

## Highlights & Insights

1. **Apt neuroscience analogy**: The VLM handles high-level semantics (cerebral cortex) while the Action Head integrates spatial information for fine motor control (cerebellum). This design intuition is simple yet highly effective.
2. **Elegant stochastic conditioning**: Randomly toggling depth and pose injection via Bernoulli sampling during training enables a single model to flexibly accommodate multiple sensor configurations, avoiding the need to train separate models for each setup.
3. **Novel application of spatial foundation models**: This work is the first to repurpose 3D reconstruction tokens from the DUSt3R/VGGT family as geometric priors for VLA models, bridging 3D reconstruction and robot control.
4. **RGB-only surpassing GT point clouds**: This demonstrates that the implicit 3D representations learned by spatial foundation models are better suited as policy network inputs than explicit point clouds.

## Limitations & Future Work

1. **Static camera assumption**: The ESM processes images from a static third-person-view camera; applicability to mobile-base robots with egocentric, time-varying viewpoints remains to be validated.
2. **Focus on tabletop manipulation**: Experiments center on tabletop manipulation tasks; navigation and whole-body motion control scenarios are not addressed.
3. **ESM parameter overhead**: ESM accounts for 1B of the 2.9B total parameters; the impact on real-time inference for edge deployment warrants evaluation.
4. **Replaceability of the spatial foundation model**: The current design is based on VGGT; whether future, more capable spatial foundation models can be plug-and-play substituted remains to be verified.
5. **Lack of 3D annotations in Open X-Embodiment pretraining data**: While the stochastic conditioning strategy mitigates this issue, datasets with aligned 3D annotations could potentially yield further performance gains.

## Related Work & Insights

- **vs. SpatialVLA**: SpatialVLA concatenates learnable spatial embeddings into the VLM input, which provides weak signals and disrupts alignment. FALCON injects rich spatial foundation model tokens directly into the Action Head, avoiding the alignment problem.
- **vs. PointVLA/GeoVLA**: These methods consume explicit 3D inputs (point clouds) and fail when sensors are unavailable. FALCON operates on RGB-only and supports optional 3D enhancement.
- **vs. 3D-VLA**: 3D-VLA embeds 3D features into the VLM and requires costly embodied instruction tuning to recover performance. FALCON decouples spatial processing from the VLM.
- **Inspiration**: Spatial foundation models (DUSt3R family) as general-purpose geometric prior injectors can be extended to other downstream tasks requiring 3D understanding, such as navigation and scene understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined design of Action Head injection and ESM stochastic conditioning is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three simulation benchmarks, 11 real-world tasks, and comprehensive ablation studies with exceptional coverage.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation with a well-structured correspondence between three identified limitations and three design contributions.
- Value: ⭐⭐⭐⭐⭐ — Highly practical; deployable with RGB-only inputs while further enhanced when sensors are available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Theory of Space: Can Foundation Models Construct Spatial Beliefs through Active Exploration?](theory_of_space_can_foundation_models_construct_spatial_beliefs_through_active_e.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](../../CVPR2026/robotics/adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](../../ICCV2025/robotics/adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)
- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](../../CVPR2026/robotics/mergevla_crossskill_model_merging_toward_a_general.md)

</div>

<!-- RELATED:END -->
