---
title: >-
  [Paper Note] RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data
description: >-
  [NeurIPS 2025][3D Vision][Facial Rigging] This paper proposes RigAnyFace (RAF), a scalable facial mesh auto-rigging framework that leverages a 2D supervision strategy to exploit unlabeled neutral meshes for training scale expansion, enabling high-quality FACS blendshape rigging across diverse topologies and disconnected components (e.g., eyeballs).
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Facial Rigging"
  - "Auto-Rigging"
  - "FACS"
  - "Blendshapes"
  - "2D Supervision"
date: 2026-05-08
content_hash: e74be10c7c831be2
---

# RigAnyFace: Scaling Neural Facial Mesh Auto-Rigging with Unlabeled Data

**Conference**: NeurIPS 2025
**arXiv**: [2511.18601](https://arxiv.org/abs/2511.18601)  
**Code**: [GitHub](https://wenchao-m.github.io/RigAnyFace.github.io)  
**Area**: 3D Vision
**Keywords**: Facial Rigging, Auto-Rigging, FACS, Blendshapes, 2D Supervision

## TL;DR

This paper proposes RigAnyFace (RAF), a scalable facial mesh auto-rigging framework that leverages a 2D supervision strategy to exploit unlabeled neutral meshes for training scale expansion, enabling high-quality FACS blendshape rigging across diverse topologies and disconnected components (e.g., eyeballs).

## Background & Motivation

Facial rigging — the process of transforming a static neutral facial mesh into an animatable character — is a critical step widely used in digital character animation and virtual avatar creation. Traditional workflows typically require professional artists to spend tens of hours manually rigging a single facial asset, incurring prohibitively high costs.

**Limitations of Prior Work**:

**Template blendshape dependency**: Most auto-rigging methods rely on transferring blendshapes from a predefined template mesh to the target mesh, leading to reduced accuracy when the target geometry deviates significantly from the template.

**Topology constraints**: Existing methods such as NFR achieve template-free FACS-driven rigging but are restricted to single-connected humanoid faces, and cannot handle meshes with disconnected components (e.g., eyeballs, teeth).

**Scarcity of training data**: High-quality 3D rigging annotations are extremely expensive to obtain; training on limited labeled data constrains model generalization.

**Key Insight**: Given the scarcity of 3D annotated data yet the maturity of 2D facial animation techniques, this paper investigates whether 2D generative models can provide supervision signals for unlabeled meshes. The **Core Idea** is to design a 2D supervision strategy — combining appearance guidance (RGB images) and motion guidance (2D displacement fields) — to substantially expand training data and improve generalization across topologically diverse facial meshes.

## Method

### Overall Architecture

RAF takes a neutral facial mesh $M_0=(V_0, F)$ and a FACS pose vector $A_i$ as input, predicts deformation displacements $\hat{d_i}$, and deforms the neutral mesh into the corresponding FACS pose $\hat{M_i} = (V_0 + \hat{d_i}, F)$. All FACS poses are combined to form a linear blendshape rig. Training proceeds in two stages: Stage 1 trains on a large-scale mixed dataset using only 2D losses; Stage 2 fine-tunes on labeled data with combined 2D and 3D supervision.

### Key Designs

1. **Conditional Diffusion Block**: The deformation network is built on DiffusionNet, which processes mesh surface features by simulating heat diffusion and is triangulation-agnostic. Since the original DiffusionNet does not accept additional conditional inputs, the diffusion block is modified to condition on the FACS vector $A_i$ — concatenating $A_i$ with a global feature $G_0$, broadcasting it to vertex dimensions within each diffusion block, fusing it with block output features, and refining via a small MLP. The core heat diffusion equation is: $h_t(u_0) = (M + tL)^{-1}Mu_0$.

2. **Global Encoder**: The diffusion mechanism in DiffusionNet cannot propagate information across disconnected components. A two-layer lightweight DiffusionNet branch processes the input neutral mesh and produces a single global vector $G_0$ via global average pooling, encoding the spatial context of the entire mesh including disconnected components. Experiments demonstrate that this feature effectively prevents component interpenetration and enables accurate deformation.

3. **2D Supervision Strategy**: Two types of supervision signals are employed — (a) **Appearance supervision**: differentiable rendering is used to generate frontal RGB images and binary masks, with image loss $\mathcal{L}_{img}$ and mask loss $\mathcal{L}_{mask}$; (b) **2D displacement supervision**: an optical-flow-like 2D displacement field $d_i^{2d}$ is defined, representing per-pixel offsets between neutral and posed images. Compared to RGB differences, 2D displacements provide denser feedback for subtle expressions (e.g., jaw lateral shift) and are particularly effective in texturally uniform regions (e.g., cheeks).

4. **2D Supervision Generation**: For unlabeled meshes, posed images are generated using a MegActor-based 2D facial animation diffusion model, and 2D displacements are estimated via the RAFT optical flow model. Both generative models are fine-tuned on a small set of labeled data to improve performance on stylized faces.

### Loss & Training

**Two-stage training**:

- **Stage 1** (coarse): Trained on labeled + unlabeled mixed data using only 2D losses:
$$\mathcal{L}_{s1} = \alpha_1\mathcal{L}_{img} + \alpha_2\mathcal{L}_{mask} + \alpha_3\mathcal{L}_{dis-2d} + \alpha_4\mathcal{L}_{reg}$$

- **Stage 2** (fine): Fine-tuned on labeled data only with combined 2D + 3D losses:
$$\mathcal{L}_{s2} = \alpha_1\mathcal{L}_{img} + \alpha_2\mathcal{L}_{mask} + \alpha_3\mathcal{L}_{mse-3d} + \alpha_4\mathcal{L}_{lmk} + \alpha_5\mathcal{L}_{ec}$$

The model contains only 5.4M parameters and is trained on 8×A100 GPUs for approximately two days. At inference, a complete rig is generated in ~8.7 seconds on an Apple M2 Max CPU and ~3.1 seconds on an Nvidia T4 GPU.

## Key Experimental Results

### Main Results

| Method | MAE (mm) ↓ | MAE Q95 (mm) ↓ | Notes |
|--------|-----------|----------------|-------|
| Deformation Transfer | 2.93 | 8.41 | Requires extra template + correspondences |
| NFR | 2.77 | 7.21 | Humanoid only |
| **RAF (Ours)** | **1.01** | **2.94** | No additional input required |

RAF reduces MAE by 63% relative to NFR and 65% relative to Deformation Transfer across 12 humanoid heads.

### Ablation Study

| Configuration | MAE (mm) ↓ | MAE Q95 (mm) ↓ | Notes |
|---------------|-----------|----------------|-------|
| w/o Global Encoder | 2.14 | 6.64 | Disconnected component interpenetration |
| w/o 2D Loss | 2.08 | 5.84 | No 2D supervision |
| w/o Unrigged Data | 2.01 | 5.81 | No unlabeled data |
| w/o 2D Displacement | 1.95 | 5.89 | No displacement supervision |
| **Full Model** | **1.92** | **5.63** | All components enabled |

### Key Findings

- The Global Encoder reduces the ratio of interpenetrating vertices from 0.377 to 0.166, effectively resolving disconnected component collision issues.
- The 2D displacement loss yields notable improvements for subtle expressions (e.g., Jaw Left), capturing more motion information than appearance loss alone.
- Incorporating unlabeled data enables robust generalization to out-of-distribution facial meshes such as animal faces.
- On in-the-wild samples from ICT FaceKit, Objaverse, and CGTrader, RAF consistently outperforms NFR.

## Highlights & Insights

- The **2D-to-3D distillation** paradigm is particularly elegant: mature 2D facial animation generative models are leveraged to supervise 3D deformation networks, circumventing the bottleneck of scarce 3D annotations.
- The use of 2D displacement fields as supervision signals is insightful — in texturally uniform regions where RGB differences are nearly zero, displacement fields still provide effective gradients.
- The Global Encoder encodes spatial relationships among disconnected components in a compact low-dimensional vector, achieving simplicity and effectiveness.
- With only 5.4M parameters, the model is highly lightweight.

## Limitations & Future Work

- Performance degrades on shell-like meshes that deviate from the training distribution due to insufficient geometric detail.
- When a facial mesh is naturally fragmented into multiple components due to poor discretization, spatial consistency after deformation cannot be guaranteed.
- The current framework only supports FACS linear blendshapes and does not support nonlinear expression spaces.
- 2D supervision generation depends on the quality of MegActor; large style discrepancies may cause failures.

## Related Work & Insights

- **NFR**: The first template-free FACS rigging method, but limited to humanoid, single-connected meshes.
- **DiffusionNet**: The backbone of this work; a triangulation-agnostic surface learning network.
- **MegActor**: A 2D facial animation diffusion model used to generate 2D supervision signals.
- Insight: The paradigm of distilling 2D generative models for 3D tasks may generalize to other 3D deformation problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The 2D supervision strategy is novel, especially the introduction of 2D displacement fields.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations, multi-source evaluation including in-the-wild and non-humanoid tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Direct applicability to the facial animation industry with strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Category-Agnostic Neural Object Rigging](../../CVPR2025/3d_vision/category-agnostic_neural_object_rigging.md)
- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](../../CVPR2025/3d_vision/scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2026\] Lifting Unlabeled Internet-level Data for 3D Scene Understanding](../../CVPR2026/3d_vision/lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)
- [\[CVPR 2025\] MegaSynth: Scaling Up 3D Scene Reconstruction with Synthesized Data](../../CVPR2025/3d_vision/megasynth_scaling_up_3d_scene_reconstruction_with_synthesized_data.md)
- [\[ICCV 2025\] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](../../ICCV2025/3d_vision/deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
