---
title: >-
  [Paper Note] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation
description: >-
  [NeurIPS 2025][Image Generation][9DoF Pose Control] SceneDesigner introduces a CNOCS map representation combined with a two-stage reinforcement learning training strategy, achieving for the first time precise 9D pose control (position, size, and orientation) over multiple objects, significantly outperforming existing methods in both controllability and generation quality.
tags:
  - NeurIPS 2025
  - Image Generation
  - 9DoF Pose Control
  - CNOCS Representation
  - Multi-Object Generation
  - Reinforcement Learning Fine-tuning
  - ControlNet
date: 2026-05-08
content_hash: 12a8eb4374e174d5
---

# SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation

**Conference**: NeurIPS 2025  
**arXiv**: [2511.16666](https://arxiv.org/abs/2511.16666)  
**Code**: [https://henghuiding.com/SceneDesigner/](https://henghuiding.com/SceneDesigner/) (Project Page)  
**Area**: Controllable Image Generation / 3D-Aware Generation  
**Keywords**: 9DoF Pose Control, CNOCS Representation, Multi-Object Generation, Reinforcement Learning Fine-tuning, ControlNet

## TL;DR
SceneDesigner introduces a CNOCS map representation combined with a two-stage reinforcement learning training strategy, achieving for the first time precise 9D pose control (position, size, and orientation) over multiple objects, significantly outperforming existing methods in both controllability and generation quality.

## Background & Motivation
3D-aware controllable image generation is an important yet insufficiently addressed problem. Existing methods suffer from the following limitations:

**2D Control Dominance**: Most approaches (GLIGEN, InstanceDiffusion) operate only on 2D bounding boxes, failing to capture 3D attributes.

**Insufficient Orientation Control**: LOOSECONTROL uses 3D bounding boxes but lacks orientation information, so a single box may correspond to either the front or back of an object.

**Difficulty with Multi-Object Scenes**: Continuous 3D Words and COMPASS handle only single objects and produce unrealistic visual styles.

**Poor Method Compatibility**: ORIGEN relies on one-step generation models and is incompatible with mainstream multi-step diffusion frameworks.

**Key Insight**: Design an efficient 9D pose encoding representation (CNOCS map), combined with a dedicated dataset and two-stage training, to enable precise multi-object pose control.

## Method

### Overall Architecture
SceneDesigner introduces a ControlNet branch into pretrained Stable Diffusion 3.5, accepting CNOCS maps encoding 9D pose as control conditions. Training proceeds in two stages: the first stage learns basic control capability on the ObjectPose9D dataset, and the second stage applies reinforcement learning fine-tuning to improve generation quality for low-frequency poses. At inference time, Disentangled Object Sampling (DOS) is employed to handle multi-object scenes.

### Key Designs

1. **CNOCS Map (Cuboid Normalized Object Coordinate System)**

    - **Design Motivation**: Traditional NOCS requires precise 3D shapes (CAD models), which is user-unfriendly.
    - **Simplification**: CNOCS approximates object geometry using only a cuboid, preserving key geometric information.
    - **Construction**: (1) For each pixel, find the intersection coordinate on the 3D bounding box surface → (2) Transform from camera coordinate system to object coordinate system → (3) Normalize by bounding box dimensions to $[-1, 1]$ → (4) Map to final pixel values via encoding function $f$.
    - **Variants**: C-CNOCS (constant function, e.g., Euler angles), I-CNOCS (identity function, direct coordinates), S-CNOCS (spherical harmonics). I-CNOCS is selected based on experiments.
    - **Advantage**: Users only need to manipulate a cuboid mesh in 3D space to specify object pose, making the interface intuitive and accessible.

2. **ObjectPose9D Dataset**

    - **Base Data**: Approximately 110K images with accurate pose annotations selected from OmniNOCS (Objectron + Cityscapes subset).
    - **Extended Data**: Large-scale 9D pose annotation of MS-COCO (~65K samples) to enrich object category and scene diversity.
    - **Annotation Pipeline**: Filter suitable objects → estimate orientation via Orient Anything → reconstruct 3D point cloud via MoGe → compute 3D bounding boxes → manual verification.
    - **Total**: 125,486 training samples.

3. **Two-Stage Training Strategy**

    - **Stage 1**: Train the ControlNet branch on ObjectPose9D using flow matching loss (45K iterations) to acquire basic pose control.
    - **Stage 2**: RL fine-tuning (5K iterations) to address poor generation quality for low-frequency poses (e.g., rear views of most animals).
    - **Reward Design**: Location/size reward $r_{ls}$ (IoU via Grounding DINO) + orientation reward $r_o$ (KL divergence via Orient Anything).
    - Randomized truncated backpropagation and gradient checkpointing are used to reduce memory overhead.

4. **Disentangled Object Sampling (DOS)**

    - **Design Motivation**: In multi-object scenes, the model struggles to correctly associate each object with its corresponding pose, leading to under-generation and concept confusion.
    - **Mechanism**: At each denoising step, multiple noise latents are composed and sampled conditioned on either global or individual object conditions; results are merged in latent space via regional masks.
    - Can be combined with user personalization weights (LoRA) to enable customized pose control for reference subjects.

### Loss & Training
- Stage 1: Standard flow matching loss $\|v_\theta(x_t, t, c_p, \{P_i\}) - (\epsilon - x)\|^2$
- Stage 2: Minimize $-\beta r(x, c_p, \{P_i\}) + L_{prior}$, where $L_{prior}$ is the Stage 1 loss used to stabilize training.
- AdamW optimizer, learning rate 5e-6, resolution 512×512, batch size 48, 6× A800 GPUs.

## Key Experimental Results

### Main Results (Pose Alignment Accuracy)

| Benchmark | Method | Acc_ls(%)↑ | mIoU(%)↑ | Abs.Err↓ | Acc@22.5°(%)↑ |
|-----------|--------|-----------|----------|----------|---------------|
| Single-Front | C3DW | 2.02 | 19.61 | 50.01 | 60.32 |
| Single-Front | LOOSECONTROL | 23.89 | 27.12 | 87.26 | 23.08 |
| Single-Front | **SceneDesigner** | **50.20** | **57.21** | **13.23** | **89.47** |
| Single-Back | LOOSECONTROL | 24.36 | 30.49 | 132.26 | 7.05 |
| Single-Back | **SceneDesigner** | **52.56** | **60.66** | **17.47** | **83.33** |
| Multi | LOOSECONTROL | 14.85 | 22.58 | 147.42 | 4.80 |
| Multi | **SceneDesigner** | **47.16** | **52.16** | **23.14** | **80.79** |

### Ablation Study

| Configuration | Acc_ls(%)↑ | mIoU(%)↑ | Abs.Err↓ | Acc@22.5°(%)↑ |
|---------------|-----------|----------|----------|---------------|
| w/o MS-COCO | 41.69 | 50.07 | 74.89 | 24.32 |
| w/o RL Fine-tuning | 43.18 | 50.32 | 43.85 | 52.36 |
| C-CNOCS | 40.45 | 49.86 | 37.86 | 73.70 |
| Pose Embedding | 32.51 | 40.73 | 49.65 | 47.15 |
| Text Description Only | 12.90 | 14.32 | 88.43 | 25.31 |
| **SceneDesigner** | **51.12** | **58.55** | **14.87** | **87.10** |

### Key Findings
- The I-CNOCS map substantially outperforms direct pose embedding injection (Acc@22.5°: 87.1% vs. 47.2%), validating the effectiveness of spatial representation.
- RL fine-tuning improves orientation accuracy from 52.36% to 87.10%, particularly benefiting rear-view pose generation.
- MS-COCO data augmentation raises orientation accuracy from 24.32% to 87.10%, demonstrating the critical importance of category diversity.
- DOS significantly improves all metrics in multi-object scenes (Acc_ls: 36.68% → 47.16%).
- In user studies, SceneDesigner substantially outperforms competing methods in image quality (0.96) and orientation fidelity (0.91).

## Highlights & Insights
- The CNOCS map is an elegant design: replacing precise 3D shapes with cuboids lowers the usage barrier while preserving geometric interpretability.
- The two-stage training strategy directly optimizes for pose control objectives, addressing data imbalance via RL more efficiently than data resampling.
- DOS trades additional inference computation for improved multi-object scene quality, embodying a straightforward and effective design philosophy.
- Integration with DreamBooth/LoRA for personalized pose control enhances practical applicability.

## Limitations & Future Work
- Precise object shape cannot be controlled; only the cuboid bounding box can be manipulated.
- Multi-object scenes remain subject to concept confusion as the number of objects increases, constrained by the base model's capacity.
- DOS introduces additional computational overhead, requiring independent sampling per object.
- The current method supports image generation only; extension to video generation requires further work.

## Related Work & Insights
- Compared to LOOSECONTROL, CNOCS supplies the critical missing orientation information.
- The RL fine-tuning strategy is generalizable to other generative tasks requiring alignment between control conditions and outputs.
- The variant design of CNOCS (constant / identity / spherical harmonic functions) offers a reference design space for representation engineering.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data](osmgen_highly_controllable_satellite_image_synthesis_using_openstreetmap_data.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](../../ICCV2025/image_generation/rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[AAAI 2026\] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](../../AAAI2026/image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)
- [\[NeurIPS 2025\] Text to Sketch Generation with Multi-Styles](text_to_sketch_generation_with_multi-styles.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)

<!-- RELATED:END -->
