---
title: >-
  [Paper Note] Generative Omnimatte: Learning to Decompose Video into Layers
description: >-
  [CVPR 2025][3D Vision][Video layer decomposition] Generative Omnimatte fine-tunes a video inpainting diffusion model (Casper) to learn the joint removal of objects and their associated effects (shadows, reflections). Combined with a trimask conditioning mechanism and omnimatte optimization, it achieves high-quality video layer decomposition and disoccluded region completion without assuming static backgrounds or requiring camera poses.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Video layer decomposition"
  - "Omnimatte"
  - "Video diffusion models"
  - "Object effect removal"
  - "Video editing"
date: 2026-05-08
content_hash: 71502f1774a8962e
---

# Generative Omnimatte: Learning to Decompose Video into Layers

**Conference**: CVPR 2025  
**arXiv**: [2411.16683](https://arxiv.org/abs/2411.16683)  
**Code**: [https://gen-omnimatte.github.io](https://gen-omnimatte.github.io) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Video layer decomposition, Omnimatte, Video diffusion models, Object effect removal, Video editing

## TL;DR
Generative Omnimatte fine-tunes a video inpainting diffusion model (Casper) to learn the joint removal of objects and their associated effects (shadows, reflections). Combined with a trimask conditioning mechanism and omnimatte optimization, it achieves high-quality video layer decomposition and disoccluded region completion without assuming static backgrounds or requiring camera poses.

## Background & Motivation

1. **Background**: The Omnimatte method decomposes video into semantically meaningful RGBA layers, with each layer containing an object and its associated effects (such as shadows and reflections). Existing methods like Omnimatte and OmnimatteRF rely on strict assumptions: static backgrounds or precise camera poses and depth estimation.
2. **Limitations of Prior Work**: (a) The static background assumption is frequently violated in real-world videography (e.g., handheld camera shake, dynamic backgrounds); (b) accurate pose/depth estimation is required but often imprecise, leading to blurry backgrounds; (c) the lack of generative priors prevents the completion of severely occluded dynamic regions; (d) video inpainting models can only restore regions inside the mask and cannot remove associated effects outside the mask (e.g., the shadow of an object).
3. **Key Challenge**: Object effects (e.g., shadows) extend beyond the object mask boundary, but inpainting models are designed to "preserve the area outside the mask and fill in the area inside," which contradicts the need for effect removal.
4. **Goal**: To achieve video layer decomposition including dynamic region completion and effect association, without relying on static backgrounds or pose estimation.
5. **Key Insight**: Pre-trained video generative models have already learned the correlation between objects and their effects in their internal representations (e.g., self-attention weights show that shadow regions attend to object regions). This capability can be activated through fine-tuning on a small amount of data.
6. **Core Idea**: To fine-tune a video inpainting model to add object effect removal capabilities (Casper), using a trimask to distinguish among "remove/preserve/potentially modify" regions, and then reconstruct RGBA omnimatte layers via post-optimization.

## Method

### Overall Architecture
A two-stage pipeline: (1) Use the Casper model (fine-tuned from Lumiere inpainting) to generate a clean background video $\mathcal{I}_{bg}$ and $N$ single-object "solo" videos $\mathcal{I}_i$, where each solo video preserves only one object and its associated effects; (2) perform test-time optimization on each pair $(\mathcal{I}_i, \mathcal{I}_{bg})$ to reconstruct the foreground RGBA omnimatte layers $\mathcal{O}_i$.

### Key Designs

1. **Trimask Conditioning Mechanism**:
    - **Function**: Provides a three-level regional indicator of "remove/preserve/potentially modify" to the diffusion model.
    - **Mechanism**: Traditional inpainting uses a binary mask (0 = inpaint, 1 = preserve), but object effect removal requires a third state—"background regions that might need modification (e.g., erasing shadows)." The trimask defines three values: $\mathcal{M}=0$ (object region to be removed), $\mathcal{M}=1$ (object region to be preserved), and $\mathcal{M}=0.5$ (background region that may contain effects to be removed). To generate the background, all objects are marked to be removed. To generate the $i$-th solo video, object $i$ is marked to be preserved, while other objects are marked to be removed.
    - **Design Motivation**: Binary masks cannot express the semantic ambiguity of "this region's content may need to be either changed or preserved" (e.g., distant shadow regions). Trimask resolves this ambiguity. Ablation studies demonstrate that models trained with a trimask are significantly more accurate in multi-object scenes than those trained with a binary mask.

2. **Casper Object Effect Removal Model**:
    - **Function**: Removes a specified object and all of its associated visual effects (shadows, reflections, water splashes, etc.).
    - **Mechanism**: Fine-tuned based on the Lumiere video inpainting model. Key designs include: (a) keeping the RGB values of the removed region (instead of setting them to zero) so that the model can infer associated effects based on the object's appearance inside the mask; (b) using a carefully curated dataset of four types (31 Omnimatte scenes, 15 tripod-captured videos, 569 Kubric synthetic videos, and 1024 Object-Paste videos), which is small in total size but covers diverse scenarios such as shadows, reflections, and water effects; (c) employing 256-step DDPM sampling during inference, with temporal multidiffusion processing to handle long videos.
    - **Design Motivation**: Since the self-attention of pre-trained video models already links objects with their effects (validated by visualization in Fig. 5), a small amount of targeted training data is sufficient to activate this capability.

3. **Omnimatte Optimization**:
    - **Function**: Reconstructs RGBA foreground layers from RGB solo videos and background videos.
    - **Mechanism**: Optimizes an alpha network (a U-Net that generates a smooth alpha map from $\mathcal{I}_i$ and the error map $\Delta_i = |\mathcal{I}_i - \mathcal{I}_{bg}|$) and foreground RGB pixel values. The loss function includes reconstruction loss $\mathcal{L}_{recon} = \|\mathcal{I}_i - \alpha_i \mathcal{I}_{i,fg} - (1-\alpha_i)\mathcal{I}_{bg}\|_2$, alpha sparsity regularization (L0 + L1), and mask supervision loss (gradually decayed). Optimization is performed first at a base resolution of 128px, then upsampled to 640×384 for continued optimization, followed by detail transfer.
    - **Design Motivation**: Decoupling into two steps (RGB effect removal followed by RGBA optimization) avoids the need for massive RGBA datasets that would be required to directly train a diffusion model to output alpha channels. Optimizing foreground RGB instead of directly using input video pixels produces cleaner foreground layers.

### Loss & Training
Casper Training: Fine-tunes the Lumiere inpainting base model with four types of data in an approximately 50:50 real/synthetic ratio. BLIP-2 is used to generate text descriptions for target videos. Data augmentation includes horizontal flipping, temporal flipping, and random cropping. Omnimatte Optimization: Uses the loss $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{sparsity}\mathcal{L}_{sparsity} + \lambda_{mask}\mathcal{L}_{mask}$, where the weight of the mask supervision is progressively decreased across iterations, allowing the alpha map to expand freely from the mask initialization to the effect regions.

## Key Experimental Results

### Main Results

| Method | Movie PSNR↑ | Movie LPIPS↓ | Kubric PSNR↑ | Kubric LPIPS↓ | Avg PSNR↑ | Avg LPIPS↓ |
|------|------|------|------|------|------|------|
| Omnimatte | 21.76 | 0.239 | 26.81 | 0.207 | 24.29 | 0.223 |
| OmnimatteRF | 33.86 | 0.017 | 40.91 | 0.028 | 37.38 | 0.023 |
| ObjectDrop | 28.05 | 0.124 | 34.22 | 0.083 | 31.14 | 0.104 |
| ProPainter | 27.44 | 0.114 | 34.67 | 0.056 | 31.06 | 0.085 |
| **Ours** | 32.69 | 0.030 | **44.07** | **0.010** | **38.38** | **0.020** |

With an average PSNR of 38.38, it outperforms OmnimatteRF's 37.38 (+1.0 dB), and its LPIPS of 0.020 is superior to OmnimatteRF's 0.023.

### Ablation Study

| Configuration | Effect |
|------|------|
| Removing Omnimatte training data | Loss of real shadow/reflection association capabilities |
| Removing Tripod data | Degraded performance on water reflection effects |
| Removing Kubric data | Decreased capability in multi-object scenes |
| Removing Object-Paste data | Lower background fidelity and degraded inpainting quality |
| Replacing Trimask with Binary mask | Erroneous removal of objects that should be preserved in multi-object scenes |
| Without optimizing foreground RGB (directly using input pixels) | Background color bleeding in foreground layers |

### Key Findings
- The four types of training data each make irreplaceable contributions; a total of less than 2000 videos is sufficient to fine-tune a high-performing Casper model.
- Trimask is critical for multi-object scenes—a binary mask model cannot distinguish between objects that "should be preserved" and "should be removed."
- This method does not require camera poses or depth estimation, yet outperforms OmnimatteRF (which requires poses) by 3 dB in PSNR on Kubric synthetic scenes.
- Self-attention in pre-trained video models indeed associates objects with their effects (query tokens in shadow regions highly attend to the object regions).
- The method can handle scenarios that are completely intractable for existing methods, such as dynamic backgrounds, extreme occlusions, and water-surface effects.

## Highlights & Insights
- **Leveraging implicit semantic understanding in pre-trained video models for object-effect association**: Instead of designing complex lighting or physical models to detect effects like shadows, this method exploits the object-effect association priors already learned by large-scale generative video models. Activating this capability with a small amount of fine-tuning data is a powerful paradigm that can be extended to other video analysis tasks requiring semantic understanding.
- **Three-level regional design of Trimask**: This simple yet effective design overcomes the fundamental limitation of inpainting models being unable to handle out-of-mask effects. The extension from binary to ternary masks might seem straightforward, but it precisely defines the input space for the task.
- **Two-stage decoupled design avoids modifying the diffusion model's output space**: Keeping the RGB output space intact makes fine-tuning highly cost-effective, while obtaining RGBA through post-optimization is a highly practical and realistic engineering strategy.

## Limitations & Future Work
- The current training data does not include physical deformation effects (e.g., objects bouncing on a spring, bending rods), making the model unable to handle such cases.
- In the presence of multiple visually similar objects (e.g., a flock of penguins), effect association might get confused—this might require stronger instance segmentation cues.
- Casper may mistakenly associate unrelated dynamic background elements (e.g., wave details) with the foreground object, though this can be mitigated by user-specified preservation masks.
- The base resolution is restricted to 128px (inherited from Lumiere) and relies on SSR upsampling, which can lead to detail loss.
- Inference is relatively slow (approx. 12 minutes for an 80-frame video, in addition to post-optimization time).

## Related Work & Insights
- **vs Omnimatte/OmnimatteRF**: These methods rely on static backgrounds or pose estimation to associate effects via motion cues. This work replaces motion priors with generative priors, relaxing input constraints.
- **vs ObjectDrop**: ObjectDrop is an image-level method; processing frame-by-frame lacks temporal consistency. The video diffusion model in this work naturally maintains temporal coherence.
- **vs Video Inpainting (ProPainter/Lumiere)**: Inpainting models cannot remove effects outside the mask (like shadows); the proposed trimask and fine-tuning strategy resolves this core limitation.

## Supplementary Analysis
- Casper is fine-tuned on Lumiere inpainting using fewer than 2,000 training videos, yet achieves excellent effect removal capabilities, demonstrating the extremely strong inherent effect-correlation priors of video diffusion models.
- The proportions of the four database sources in the training data (approx. 25% each) are meticulously balanced to respectively cover real effect relationships, water surface effects, multi-object scenarios, and inpainting quality.
- Demonstrated applications cover diverse editing scenarios such as object removal, layer replacement, motion retiming, and foreground stylization, proving that omnimatte decomposition serves as a versatile intermediate representation for video editing.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Introducing video generative priors to the omnimatte problem is a completely fresh approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive quantitative, qualitative, ablation, and application evaluations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Thorough motivation analysis, where each design choice is clearly justified.
- **Value**: ⭐⭐⭐⭐⭐ Significantly lowers the input requirements for video layer decomposition, offering immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EUGens: Efficient, Unified, and General Dense Layers](../../NeurIPS2025/3d_vision/eugens_efficient_unified_and_general_dense_layers.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [\[CVPR 2025\] PhysAnimator: Physics-Guided Generative Cartoon Animation](physanimator_physics-guided_generative_cartoon_animation.md)
- [\[ICLR 2026\] Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation](../../ICLR2026/3d_vision/lyra_generative_3d_scene_reconstruction_via_video_diffusion_model_self-distillat.md)

</div>

<!-- RELATED:END -->
