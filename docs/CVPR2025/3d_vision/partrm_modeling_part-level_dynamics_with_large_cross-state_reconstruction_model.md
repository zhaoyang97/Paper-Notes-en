---
title: >-
  [Paper Note] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model
description: >-
  [CVPR 2025][3D Vision][part-level dynamics] PartRM proposes a 4D reconstruction framework based on a large-scale 3D Gaussian reconstruction model. It simultaneously models object appearance, geometry, and part-level motion from multi-view images. By constructing the PartDrag-4D dataset, a multi-scale drag embedding module, and a two-stage training strategy, it achieves state-of-the-art performance in part-level motion learning and can be applied to robot manipulation tasks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "part-level dynamics"
  - "4D reconstruction"
  - "3D Gaussians"
  - "drag interaction"
  - "robot manipulation"
date: 2026-05-08
content_hash: ec6399dbb8e02ca3
---

# PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model

**Conference**: CVPR 2025  
**arXiv**: [2503.19913](https://arxiv.org/abs/2503.19913)  
**Code**: [https://PartRM.c7w.tech/](https://PartRM.c7w.tech/)  
**Area**: 3D Vision  
**Keywords**: part-level dynamics, 4D reconstruction, 3D Gaussians, drag interaction, robot manipulation

## TL;DR

PartRM proposes a 4D reconstruction framework based on a large-scale 3D Gaussian reconstruction model. It simultaneously models object appearance, geometry, and part-level motion from multi-view images. By constructing the PartDrag-4D dataset, a multi-scale drag embedding module, and a two-stage training strategy, it achieves state-of-the-art performance in part-level motion learning and can be applied to robot manipulation tasks.

## Background & Motivation

**Background**: World models need to predict future states based on current observations and actions, where modeling part-level dynamics (such as drawer sliding or door rotating) is crucial for applications like robot manipulation and AR/VR. Existing methods like Puppet-Master achieve drag-controlled object motion generation by fine-tuning large-scale video diffusion models.

**Limitations of Prior Work**: Methods like Puppet-Master suffer from two core defects: (1) The output is only a single-view video, which cannot directly provide the 3D representations required by simulators, introducing extra errors through additional monocular reconstruction models. (2) The diffusion denoising process takes several minutes, failing to meet the demand for rapid trial-and-error generation of manipulation policies.

**Key Challenge**: The gap between 2D video representations and 3D application requirements, as well as the contradiction between generation speed and real-time interaction demands.

**Goal**: To simultaneously model object appearance, geometry, and part-level motion, generating 3D representations renderable from arbitrary viewpoints with fast inference speed.

**Key Insight**: The authors observe that large-scale 3D Gaussian reconstruction models (such as LGM) already possess appearance and geometric priors for static objects. Since part-level motion is naturally correlated with geometry (e.g., a drawer sliding along its surface normal), motion modeling capabilities can be extended on top of these reconstruction models.

**Core Idea**: Extend 4D capabilities on top of a pre-trained large-scale 3D Gaussian reconstruction model, modeling part motion through drag conditions, and employing a two-stage training strategy to avoid catastrophic forgetting.

## Method

### Overall Architecture

Given a single-view observational image and 2D drag interaction information, PartRM first generates multi-view images using a fine-tuned Zero123++. It then propagates a single drag to the entire region of the moving part via the drag propagation module. The multi-view images and drag instructions are fed into an LGM-based U-Net network to output 3D Gaussian representations that represent the deformed state. The whole pipeline adopts a two-stage training strategy: learning motion in the first stage and learning appearance in the second stage.

### Key Designs

1. **PartDrag-4D Dataset**:

    - **Function**: Provide multi-view training data for part-level dynamics.
    - **Mechanism**: Based on the PartNet-Mobility dataset, 738 articulated object meshes across 8 categories are selected. For the movable parts of each object, 6 states are configured between their extreme positions, while other parts' positions are randomized, generating a total of 20,548 states. For each state, Blender is used to render 12 views, and drag points are sampled on the surfaces of moving parts.
    - **Design Motivation**: Existing 4D datasets either lack 3D information or use generic animation data from Objaverse (which contains operations like deformation that violate kinematic dynamics). Therefore, a dataset specifically tailored for articulated kinematics is needed.

2. **Drag Propagation & Multi-Scale Embedding Module**:

    - **Function**: Extend a single drag interaction into drag proposals covering the entire moving part, and embed them into the U-Net across multiple resolution scales.
    - **Mechanism**: In the propagation phase, SAM is used to segment the moving part, and new starting points are sampled on the segmentation mask, maintaining the same direction and intensity as the original drag. In the embedding phase, Fourier encoding and a 3-layer MLP are applied to each drag point to obtain feature embeddings. A multi-scale drag map $M_{t,l}$ matching the output size of the U-Net downsampling blocks is constructed and interacts with the feature maps via concatenation and convolution: $I_{l+1} = O_l + \text{Conv}(M_{t,l} \oplus O_l)$.
    - **Design Motivation**: A single drag condition is ambiguous and can lead to model hallucinations. Multi-scale embedding allows the network to understand drag motion at different granularities—large scales capture fine-grained local details, while small scales capture global motion patterns.

3. **Two-Stage Training Strategy**:

    - **Function**: Prevent catastrophic forgetting of pre-trained appearance and geometric modeling capabilities during fine-tuning.
    - **Mechanism**: The first stage (motion learning) uses a knowledge distillation approach, taking the Gaussian parameters inferred by the pre-trained LGM on target state observations as supervision signals, directly applying an L2 loss to the 14-dimensional parameters of the splatter image. The second stage (appearance learning) uses rendering losses (MSE + LPIPS + alpha MSE) to jointly optimize appearance, geometry, and motion.
    - **Design Motivation**: If only rendering loss is used for supervision (Stage 2), the model tends to exploit loopholes in the loss function rather than truly learning the motion. Learning motion first and then conducting joint optimization achieves a coarse-to-fine training process.

### Loss & Training

Stage 1 uses a pixel-level L2 loss on the splatter image: $\mathcal{L}_1 = \sum \|\mathcal{GS}_i - \mathcal{GS}_j\|_2^2$, where $i, j$ represent corresponding pixels. Stage 2 uses a rendering loss: $\mathcal{L}_2 = L_{\text{mse}} + \lambda_1 L_{\text{lpips}} + \lambda_2 L_{\text{mse}}^{\alpha}$, with $\lambda_1 = \lambda_2 = 1.0$.

## Key Experimental Results

### Main Results

| Method | Setting | PSNR↑ | SSIM↑ | LPIPS↓ | Time |
|------|------|-------|-------|--------|------|
| DiffEditor | Drag-First | 22.34 | 0.9174 | 0.0918 | 128.8s |
| DragAPart | Drag-First | 24.91 | 0.9454 | 0.0567 | 119.4s |
| Puppet-Master | Drag-First | 24.42 | 0.9475 | 0.0528 | 361.5s |
| **PartRM (Ours)** | - | **28.15** | **0.9531** | **0.0356** | **4.2s** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Only Stage 1 | 22.05 | 0.8624 | 0.1274 |
| Only Stage 2 | 25.87 | 0.9387 | 0.0537 |
| Stage 1+2 | **28.15** | **0.9531** | **0.0356** |
| 1 drag | 27.06 | 0.9466 | 0.0452 |
| 5 drags | 27.56 | 0.9483 | 0.0448 |
| 10 drags | **28.15** | **0.9531** | **0.0356** |

### Key Findings

- The two-stage training shows significant improvements over single-stage training: Stage 1+2 outperforms Only Stage 2 by 2.28dB in PSNR, demonstrating that the motion learning stage is crucial for the model to learn correct motion.
- Increasing the number of drags from 1 to 10 improves PSNR from 27.06 to 28.15, indicating that more drag cues provide clearer motion guidance.
- Multi-scale drag embedding (128+32+8) performs better than any single scale, as different scales capture motion information at different granularities.
- PartRM inference takes only 4.2s, which is approximately 86 times faster than Puppet-Master.

## Highlights & Insights

- **Using 3D Gaussians instead of 2D videos as the state representation of world models** naturally supports multi-view rendering and downstream robotic applications, presenting a highly forward-looking design.
- **Drag propagation using SAM segmentation** extends a single interaction into dense motion conditions, cleverly resolving the ambiguity of drag conditioning. This methodology can be transferred to other conditional generation tasks.
- **The knowledge distillation strategy in two-stage training**—using the pre-trained model's own outputs as targets for continuous learning—both preserves generalization capability and accelerates training.

## Limitations & Future Work

- Generalization performance is limited for articulated objects that deviate significantly from the training distribution (such as atypical objects in web-scale data).
- The dataset only contains 8 categories of articulated motions, lacking more complex motion types like soft-body deformation.
- Currently, only single-part motions can be handled at a time, leaving multi-part joint motion scenarios uninvolved.
- Future work could explore extending PartRM to more general object dynamics modeling, incorporating language instructions to achieve more flexible interactions.

## Related Work & Insights

- **vs Puppet-Master**: Puppet-Master achieves drag-controlled generation by fine-tuning video diffusion models, outputting single-view videos. In contrast, PartRM directly outputs 3D Gaussian representations, runs 86x faster, and naturally supports multi-view rendering.
- **vs DragAPart**: DragAPart only performs 2D image-level drag deformation, making it difficult to capture complex motion patterns, and requires an additional 3D reconstruction step.
- **vs L4GM**: L4GM generates dynamic 3D representations from single-view videos, but it is not action-conditional and does not support part-level dynamics.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending large-scale reconstruction models to 4D part dynamics is a novel framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes main results, multiple ablation studies, generalization tests, and robotic applications.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with continuous logic from motivation to methods and experiments.
- Value: ⭐⭐⭐⭐ Holds high application value for articulated object manipulation and 3D world models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LIM: Large Interpolator Model for Dynamic Reconstruction](lim_large_interpolator_model_for_dynamic_reconstruction.md)
- [\[CVPR 2025\] Continuous 3D Perception Model with Persistent State](continuous_3d_perception_model_with_persistent_state.md)
- [\[CVPR 2025\] SUM Parts: Benchmarking Part-Level Semantic Segmentation of Urban Meshes](sum_parts_benchmarking_part-level_semantic_segmentation_of_urban_meshes.md)
- [\[CVPR 2025\] Matrix3D: Large Photogrammetry Model All-in-One](matrix3d_large_photogrammetry_model_all-in-one.md)
- [\[ICLR 2026\] Part-X-MLLM: Part-aware 3D Multimodal Large Language Model](../../ICLR2026/3d_vision/part-x-mllm_part-aware_3d_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
