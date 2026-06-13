---
title: >-
  [Paper Note] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild
description: >-
  [ICLR 2026][Image Generation][single-view 3D human reconstruction] This paper proposes DrPose, which applies direct reward fine-tuning to maximize PoseScore—a metric measuring skeletal consistency between multi-view late…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "single-view 3D human reconstruction"
  - "multi-view diffusion"
  - "direct reward fine-tuning"
  - "pose alignment"
  - "PoseScore"
date: 2026-05-08
content_hash: 42afe862ef185b80
---

# Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild

**Conference**: ICLR 2026
**arXiv**: [2603.02619](https://arxiv.org/abs/2603.02619)  
**Code**: [Project Page](https://seunguk-do.github.io/drpose)  
**Area**: Image Generation
**Keywords**: single-view 3D human reconstruction, multi-view diffusion, direct reward fine-tuning, pose alignment, PoseScore

## TL;DR
This paper proposes DrPose, which applies direct reward fine-tuning to maximize PoseScore—a metric measuring skeletal consistency between multi-view latent images and ground-truth 3D poses—combined with KL regularization to prevent reward hacking. Together with the DrPose15K dataset (15K diverse poses sampled from the Motion-X dataset and animated via the MIMO video generator), DrPose significantly improves 3D human reconstruction quality under challenging poses such as dynamic movements and acrobatics.

## Background & Motivation

**Background**: Multi-view diffusion-based single-view 3D human reconstruction has become the dominant paradigm (e.g., PSHuman, Era3D), following the pipeline: single image → multi-view diffusion generation → explicit/implicit 3D reconstruction. The strong priors of diffusion models make them far superior to early PIFu-based methods in recovering texture details in occluded regions.

**Limitations of Prior Work**: When input images contain dynamic or highly challenging poses (e.g., breakdancing, acrobatics, extreme sports), multi-view diffusion models tend to generate images with severely unnatural human poses. The root cause is that available 3D human training datasets are small and narrowly distributed in pose space (THuman2.1 contains only 2,445 poses; CustomHumans contains only 647), lacking coverage of extreme poses.

**Key Challenge**: Collecting 3D human scan datasets is prohibitively expensive (requiring multi-view capture rigs, subject recruitment, and privacy considerations), making large-scale expansion difficult. In contrast, human motion capture data (e.g., the AIST subset of Motion-X) provides rich and diverse 3D pose sequences. However, a modality gap exists between the two: motion data contains only pose parameters, with no corresponding multi-view images.

**Key Insight**: Rather than relying on expensive 3D human assets, this work leverages motion datasets as pose supervision signals and aligns multi-view diffusion models via direct reward fine-tuning. The key contribution is a differentiable PoseScore reward function that quantifies the consistency between generated multi-view latent images and ground-truth 3D poses.

**Core Idea**: By combining pose supervision from motion datasets with differentiable reward fine-tuning, the multi-view diffusion model learns to generate pose-accurate multi-view images even under challenging pose conditions.

## Method

### Overall Architecture

DrPose comprises three core components:
- **DrPose Algorithm**: Direct reward fine-tuning based on DRTune, training the multi-view diffusion model to maximize PoseScore.
- **DrPose15K Dataset**: 1.5K poses sampled from the AIST subset of Motion-X, each augmented with 9 temporal neighbors to yield 15K poses in total, with corresponding single-view images synthesized via MIMO.
- **3D Reconstruction Pipeline**: The post-trained multi-view diffusion model combined with explicit carving (SMPL-X initialization → differentiable remeshing → appearance fusion).

### Key Designs

1. **PoseScore Differentiable Reward Function**:

    - *Function*: Quantifies the consistency between multi-view latent images $\mathbf{x}_0$ and ground-truth 3D pose $\theta$.
    - *Mechanism*: Both are projected into skeleton image space for comparison. A pretrained U-Net $g_{\text{skel}}$ converts latent images into predicted skeleton maps $\hat{I}_{\text{skel}}$, while a rendering function $\mathcal{R}$ projects GT 3D joints $J(\theta)$ into each camera view to obtain GT skeleton maps $I_{\text{skel}}$.
    - *Reward Formula*: $r(\mathbf{x}_0, \theta) = -\mathbb{E}(\|\hat{I}_{\text{skel}} - I_{\text{skel}}\|)$, where the distance is estimated via BCE + LPIPS.
    - *Design Motivation*: The skeleton map is a 23-channel image (one channel per joint), which preserves precise structural pose information while remaining fully differentiable—enabling gradients to be backpropagated from the reward all the way to the diffusion model parameters.

2. **Direct Reward Fine-Tuning (based on DRTune)**:

    - *Function*: Efficiently fine-tunes the multi-view diffusion U-Net using reward signals.
    - *Mechanism*: Starting from noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, multi-view latent images $x_0$ are generated via DDIM sampling ($T=20$ steps), with gradients retained for only $K=2$ sampled denoising steps (all other steps use stop-grad). The reward loss $\mathcal{L}_{\text{reward}} = 1 - r(\mathbf{x}_0, \theta)$ is then computed and backpropagated.
    - *Design Motivation*: Retaining gradients across all 20 steps is memory-intractable when generating 24 images at 768×768 resolution. Sparse gradient retention with stop-grad is a necessary efficiency optimization.

3. **KL Divergence Regularization**:

    - *Function*: Prevents reward hacking, where reward scores increase while image quality degrades.
    - *Mechanism*: At training step $t \in t_{\text{train}}$, the MSE between the predicted noise of the trainable model $\epsilon_\omega$ and the frozen initial model $\epsilon_{\omega_0}$ is computed: $\mathcal{L}_{\text{KL}} = \mathbb{E}(\|\hat{\epsilon} - \hat{\epsilon}_0\|)$.
    - *Design Motivation*: This constrains the fine-tuned model from deviating too far from the original, preserving image quality while optimizing pose accuracy.

4. **DrPose15K Dataset Construction**:

    - Farthest point sampling is applied to the AIST subset of Motion-X (300K dance poses) to select 1.5K representative poses.
    - Each pose is paired with 9 temporal neighbors to form a pose sequence (as required by MIMO's input format), yielding 15K poses in total.
    - MIMO (a pose-conditioned image-to-video model) animates full-body portrait images according to each pose sequence, producing the corresponding single-view images.
    - Compared to THuman2.1, DrPose15K exhibits 1.73× larger standard deviation in SMPL-X joint positions, indicating significantly greater pose diversity.

### Loss & Training

- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{reward}} + w_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$
- DDIM sampling with $T=20$ steps; $K=2$ training steps with gradient retention; maximum early-stop timestep $m=8$
- $w_{\text{KL}} = 0.01$
- Base model: PSHuman (768×768, 6-view normal maps + RGB)
- Trained on a single NVIDIA H200, batch size 2 with 2-step gradient accumulation, for 18K iterations
- $g_{\text{skel}}$ in PoseScore is pretrained on THuman2.1 + CustomHumans (~3K scans)

## Key Experimental Results

### Main Results (Geometry Quality, Table 1)

| Method | THuman2.1 CD↓ | CustomHumans CD↓ | MixamoRP CD↓ |
|--------|--------------|-----------------|-------------|
| ECON | 101.65 | 126.14 | 166.54 |
| SiTH | 63.30 | 71.94 | 158.27 |
| PSHuman | 52.96 | 52.22 | 137.28 |
| **Ours (PSHuman)** | **42.05** | **44.13** | **126.53** |

### Appearance Quality (Table 2)

| Method | THuman2.1 PSNR↑ | CustomHumans PSNR↑ | MixamoRP PSNR↑ |
|--------|-----------------|-------------------|---------------|
| PSHuman | 18.39 | 18.91 | 17.59 |
| **Ours** | **20.86** | **19.19** | **17.66** |

### Ablation Study
- Base model ablation: consistent gains are observed when using either Era3D or PSHuman as the backbone; PSHuman is selected for its superior facial quality.
- Validation of $g_{\text{skel}}$ in PoseScore: achieves PSNR=22.48 and SSIM=0.93 on the THuman2.1 test set, reliably predicting skeleton maps from latent images.

### Key Findings
- **Largest gains on the challenging pose benchmark MixamoRP**: CD decreases from 137.28 to 126.53 (↓7.8%), demonstrating the value of DrPose under extreme poses.
- **Consistent improvements on standard benchmarks**: CD on THuman2.1 decreases from 52.96 to 42.05 (↓20.6%), indicating that DrPose benefits not only challenging poses but also general ones.
- **Qualitative in-the-wild results** (Fig. 8, 9): On real internet images featuring dancing, skateboarding, yoga, and similar scenarios, the post-trained DrPose model generates multi-view images with noticeably more natural poses.

## Highlights & Insights
- **Effective use of motion data**: 3D human scan data is costly and scarce, whereas motion capture data is abundant. By combining a differentiable PoseScore reward with a video generator for image synthesis, the pose diversity of motion data is indirectly transferred to the multi-view diffusion model—without requiring any 3D human assets.
- **Skeleton maps as an intermediate representation**: The consistency problem between latent space and 3D pose space is reformulated as a 2D skeleton map comparison, ensuring differentiability while avoiding the complexity of 3D spatial alignment. The 23-channel design (one channel per joint) is more precise than rendering a complete skeleton image directly.
- **Lightweight post-training paradigm**: Post-training requires only a single H200 GPU and 18K iterations, with no architectural changes to the base model. The approach is plug-and-play compatible with various I2MV models such as Era3D and PSHuman.

## Limitations & Future Work
- Requires well-segmented input images; imperfect segmentation leads to floating geometry artifacts at boundary regions.
- High memory consumption: computing PoseScore requires iterative denoising to generate 24 images at 768×768 resolution, and KL regularization requires storing the frozen initial U-Net.
- Images in DrPose15K are synthesized by MIMO, so their quality is bounded by MIMO's generation capability, potentially introducing domain shift.
- Only the 6-view configuration is validated; scalability to more views or higher resolutions remains unexplored.
- The MixamoRP benchmark is constructed using commercial models, limiting reproducibility.

## Related Work & Insights
- **vs. PSHuman/Era3D**: DrPose serves as a plug-and-play post-training module directly improving these base models, yielding consistent gains on both.
- **vs. DRTune**: This work adapts the general direct reward fine-tuning framework to the 3D human domain; the key innovation lies in designing the domain-specific PoseScore reward function.
- **vs. RLHF/GRPO-style methods**: DrPose adopts a differentiable reward + direct backpropagation approach (similar to DRTune), requiring neither multi-trajectory sampling nor group normalization, resulting in faster convergence.
- **vs. data augmentation methods**: The approach is complementary to larger-scale motion datasets (e.g., the full Motion-X collection) or more capable video generators, offering further room for improvement.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces direct reward fine-tuning to the 3D human domain; PoseScore is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks + a newly proposed challenging pose benchmark (MixamoRP) + qualitative in-the-wild validation.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ Effectively addresses the quality bottleneck of 3D human reconstruction under dynamic poses; the lightweight, post-training paradigm is scalable and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function](diffusion_fine-tuning_via_reparameterized_policy_gradient_of_the_soft_q-function.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[ICCV 2025\] DreamDance: Animating Human Images by Enriching 3D Geometry Cues from 2D Poses](../../ICCV2025/image_generation/dreamdance_animating_human_images_by_enriching_3d_geometry_cues_from_2d_poses.md)
- [\[ICLR 2026\] RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation](refany3d_3d_asset-referenced_diffusion_models_for_image_generation.md)

</div>

<!-- RELATED:END -->
