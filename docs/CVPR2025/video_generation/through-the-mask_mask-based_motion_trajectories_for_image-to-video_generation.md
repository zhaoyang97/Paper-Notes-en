---
title: >-
  [Paper Note] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation
description: >-
  [CVPR 2025][Video Generation][Image-to-Video Generation] This paper proposes Through-The-Mask (TTM), a two-stage compositional I2V framework. By utilizing mask-based motion trajectories as an intermediate representation, it decomposes the image-to-video generation process into "motion generation" and "video generation" stages, achieving SOTA performance in complex multi-object motion scenarios.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Image-to-Video Generation"
  - "Motion Trajectories"
  - "Semantic Segmentation Mask"
  - "Compositional Generation"
  - "Multi-Object Scenarios"
date: 2026-05-08
content_hash: 41847f3577f6e5cb
---

# Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.03059](https://arxiv.org/abs/2501.03059)  
**Code**: [https://guyyariv.github.io/TTM/](https://guyyariv.github.io/TTM/)  
**Area**: Video Generation  
**Keywords**: Image-to-Video Generation, Motion Trajectories, Semantic Segmentation Mask, Compositional Generation, Multi-Object Scenarios

## TL;DR
This paper proposes Through-The-Mask (TTM), a two-stage compositional I2V framework. By utilizing mask-based motion trajectories as an intermediate representation, it decomposes the image-to-video generation process into "motion generation" and "video generation" stages, achieving SOTA performance in complex multi-object motion scenarios.

## Background & Motivation

**Background**: Image-to-video (I2V) generation aims to transform static images into videos based on text descriptions. Current methods (such as DynamiCrafter, ConsistI2V, and AnimateAnything) can generate realistic outputs, but still struggle with multi-object scenarios and complex motion interactions.

**Limitations of Prior Work**: End-to-end I2V models must simultaneously and implicitly reason about object semantics, motion, and appearance. As the number of objects increases, the potential combinations of motion and interaction grow exponentially, making precise generation using a single model challenging. Motion-I2V proposes a two-stage approach using optical flow as an intermediate representation. However, optical flow has three limitations: (1) it represents motion but not semantics; (2) pixel-wise motion prediction is highly redundant for I2V; (3) pixel-level prediction errors in the first stage severely affect the second stage.

**Key Challenge**: An intermediate representation needs to satisfy three properties simultaneously: expressing both motion and semantics, representing interactions between objects, and being robust to signal fluctuations. Optical flow only satisfies the first property and operates at an overly fine granularity. A more suitable intermediate representation is required.

**Goal**: Design a compact yet highly expressive intermediate representation that captures both motion and semantic information at the object level, thereby reducing the prediction difficulty of the first stage and increasing robustness to errors.

**Key Insight**: The authors argue that a "temporally consistent frame-by-frame semantic segmentation mask" (i.e., a mask-based motion trajectory) is the ideal intermediate representation. It naturally contains semantic information (one color per object), motion details (masks moving over time), and operates at the object level, making it robust to pixel-level fluctuations.

**Core Idea**: Use a temporal sequence of segmentation masks as the intermediate motion representation. The first stage generates the mask motion trajectory, and the second stage generates the final video based on the mask trajectory and object-level attention mechanisms.

## Method

### Overall Architecture
The input consists of a reference image $x^{(0)}$ and a text prompt $c$. In the preprocessing stage, an LLM is used to extract the motion description $c_{motion}$ and object-level text prompts $c_{local}$, while Grounding DINO + SAM2 are employed to generate the initial segmentation mask $s^{(0)}$. The first stage (Image-to-Motion) generates the mask trajectory sequence $\hat{s}$ based on the reference image, the initial mask, and the motion prompt. The second stage (Motion-to-Video) generates the final video $\hat{x}$ based on the reference image, mask trajectory, global text, and object-level text.

### Key Designs

1. **Mask-based Motion Trajectory as Intermediate Representation**:

    - Function: Establish an explicit intermediate representation between motion generation and video generation to decompose the complex I2V problem into two simpler sub-problems.
    - Mechanism: Each object is represented by a fixed color in the mask. The mask sequence $s = \{s^{(0)}, ..., s^{(N)}\}$ captures the motion trajectory and semantic identity of each object. The first stage generates the mask sequence based on an LDM in the VAE latent space, taking the encoded reference image $x^{(0)}$ and initial mask $s^{(0)}$ concatenated along the channel dimension as conditions.
    - Design Motivation: Compared to optical flow, mask trajectories operate at the object level. The first stage only needs to predict coarse-grained object motion (displacement, deformation, occlusion) instead of precise pixel-wise movement, significantly reducing prediction difficulty. Even with minor errors in mask prediction, it does not lead to severe pixel-level distortions as seen in optical flow.

2. **Masked Cross-Attention**:

    - Function: Inject object-level text descriptions into matching latent space regions to achieve spatially precise semantic control.
    - Mechanism: Text descriptions for $L$ objects are encoded as $\{e^{(i)}\}_{i=1}^L$. All keys and values are concatenated, and a binary mask $M_{cross} = [M^{(1)}; ...; M^{(L)}]$ is constructed to indicate which object each position belongs to. The attention is computed as $h_{cross} = \sigma(\frac{qk^T}{\sqrt{d}} + \log M_{cross}) v$, ensuring that each latent position only attends to the text description of its corresponding object. This extends the Dense Diffusion method from image generation to the video generation setting.
    - Design Motivation: Global text prompts cannot distinguish between the distinct motions and appearances of multiple objects, necessitating fine-grained object-level control. Mask trajectories naturally provide the spatial locations of objects, which can be leveraged to construct cross-attention masks.

3. **Masked Self-Attention**:

    - Function: Ensure consistency of the same object across different frames while preventing feature interference among different objects.
    - Mechanism: A self-attention mask $M_{self} \in \{0,1\}^{N_{tokens} \times N_{tokens}}$ is constructed, where $M_{self}^{(i,j)} = 1$ if positions $i$ and $j$ belong to the same object, and $0$ otherwise. The attention is computed as $h_{self} = \sigma(\frac{qk^T}{\sqrt{d}} + \log M_{self}) v$. This ensures that each token only attends to the positions of the same object across all frames.
    - Design Motivation: Standard self-attention allows features of different objects to interfere with one another, particularly during object crossover or occlusion. Object-grouped self-attention guarantees the temporal consistency of each individual object.

### Loss & Training
The two stages are trained independently, both using the standard denoising loss of LDMs. During inference, the two stages are chained together. Data preprocessing requires LLM (to extract motion and object descriptions), Grounding DINO (for object detection), and SAM2 (for video segmentation), which are used solely during training data preprocessing. The masked attention mechanism is only applied within the first $K$ blocks.

## Key Experimental Results

### Main Results
Comparison with SOTA methods on single-object and multi-object I2V benchmarks (SA-V-128 benchmark):

| Method | FVD↓ (Single Object) | ViCLIP-T↑ | CF↑ | FVD↓ (Multi-Object) | ViCLIP-T↑ | Motion↑ |
|------|------------|----------|-----|------------|----------|--------|
| VideoCrafter | 1484.18 | 0.209 | 0.966 | 1413.83 | 0.208 | 84.3 |
| DynamiCrafter | 1442.48 | 0.214 | 0.942 | - | - | - |
| ConsistI2V | - | - | - | - | - | - |
| AnimateAnything | - | - | - | - | - | - |
| Motion-I2V (Optical Flow) | - | - | - | - | - | - |
| **TTM (Ours)** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | FVD↓ | ViCLIP-T↑ | Motion↑ | Quality↑ |
|------|------|----------|--------|---------|
| No Intermediate Representation (End-to-End) | High | Low | Low | Medium |
| Optical Flow as Intermediate Representation | Medium | Medium | Medium | Medium |
| Mask Trajectory (Full) | **Lowest** | **Highest** | **Highest** | **Highest** |
| w/o masked cross-attn | Increase | Decrease | - | Decrease |
| w/o masked self-attn | Increase | - | Decrease | Decrease |

### Key Findings
- Using mask trajectories as intermediate representations outperforms optical flow on all metrics, validating the hypothesis that object-level representations are superior to pixel-level representations.
- Masked cross-attention contributes most to text fidelity, ensuring each object adheres to its designated text description.
- Masked self-attention contributes most to temporal consistency, restricting feature interference across different objects.
- The advantage is even more pronounced in multi-object scenarios, as the compositional approach breaks down the complex problem into manageable sub-problems.

## Highlights & Insights
- **Crucial Choice of Intermediate Representation**: The paper clearly argues that a good intermediate representation should satisfy three properties (semantics + motion, object interaction, and robustness), explaining why mask trajectories are more suitable than optical flow—an analytical framework transferable to other task decomposition problems.
- **Object-Level Attention Mechanism**: Extending image-level masked attention from Dense Diffusion to video settings while introducing masked self-attention to preserve temporal consistency is a natural and effective combination.
- **Architecture-Agnostic**: This method is compatible with both U-Net and DiT architectures, offering broad applicability.

## Limitations & Future Work
- The pipeline relies heavily on LLM + Grounding DINO + SAM2 for data preprocessing, introducing additional complexity and potential errors.
- Two-stage inference is slower than end-to-end approaches.
- Mask trajectories cannot represent fine-grained non-rigid deformations (e.g., facial expressions, cloth folds), which still rely on being learned from data in the second stage.
- When objects are completely occluded or new objects appear, the expressiveness of the mask trajectory is limited.

## Related Work & Insights
- **vs Motion-I2V**: The core difference lies in the choice of intermediate representation. Optical flow operates at the pixel level and captures motion without semantics; mask trajectories operate at the object level and encompass both motion and semantics. TTM is thus more robust to first-stage errors.
- **vs AnimateAnything**: AnimateAnything uses an additional mask to constrain motion areas but still performs end-to-end generation. TTM decomposes the process more thoroughly into two stages.
- **vs Dense Diffusion**: TTM extends object-level cross-attention from images (as in Dense Diffusion) to videos and introduces an additional masked self-attention mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using mask trajectories as intermediate representations is intuitive and effective, and masked self-attention is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ A new benchmark, SA-V-128, is proposed, with comprehensive multi-dimensional comparisons and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly justified, and the methodology is rigorously described.
- Value: ⭐⭐⭐⭐ It provides an effective solution for multi-object I2V generation, and the compositional approach can inspire further work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] Pathways on the Image Manifold: Image Editing via Video Generation](pathways_on_the_image_manifold_image_editing_via_video_generation.md)

</div>

<!-- RELATED:END -->
