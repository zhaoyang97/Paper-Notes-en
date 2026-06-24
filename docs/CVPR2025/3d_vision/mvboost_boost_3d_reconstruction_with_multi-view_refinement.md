---
title: >-
  [Paper Note] MVBoost: Boost 3D Reconstruction with Multi-View Refinement
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] MVBoost proposes a framework to boost 3D reconstruction by generating pseudo-ground-truth data through a multi-view refinement strategy. It elegantly combines the high precision of multi-view generative models with the consistency advantages of 3D reconstruction models, achieving SOTA single-image-to-3D reconstruction performance on the GSO dataset (PSNR 18.561, CD 0.101).
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "Multi-view Refinement"
  - "Pseudo-ground-truth Generation"
  - "Gaussian Splatting"
  - "LoRA Fine-tuning"
date: 2026-05-08
content_hash: 23ca2833bf898b5b
---

# MVBoost: Boost 3D Reconstruction with Multi-View Refinement

**Conference**: CVPR 2025  
**arXiv**: [2411.17772](https://arxiv.org/abs/2411.17772)  
**Code**: [https://github.com/Piggy-ch/MVBoost](https://github.com/Piggy-ch/MVBoost)  
**Area**: 3D Computer Vision  
**Keywords**: Single-image 3D Reconstruction, Multi-view Refinement, Pseudo-ground-truth Generation, Gaussian Splatting, LoRA Fine-tuning

## TL;DR
MVBoost proposes a framework to boost 3D reconstruction by generating pseudo-ground-truth data through a multi-view refinement strategy. It elegantly combines the high precision of multi-view generative models with the consistency advantages of 3D reconstruction models, achieving SOTA single-image-to-3D reconstruction performance on the GSO dataset (PSNR 18.561, CD 0.101).

## Background & Motivation

1. **Background**: Generating 3D assets from a single image is a core task in 3D vision. Feed-forward methods (e.g., LRM, LGM, GRM) train 3D reconstruction networks using four-view ground truths and rely on multi-view diffusion models to generate inputs during inference. Although Score Distillation Sampling (SDS) methods (e.g., DreamFusion) produce realistic visual effects, they require hours of optimization and suffer from the Janus problem.

2. **Limitations of Prior Work**: (a) Scarcity of high-quality 3D datasets—public datasets like Objaverse suffer from poor texture quality and high redundancy; (b) Multi-view diffusion models may produce cross-view inconsistent outputs during inference; (c) Domain inconsistency exists between using 3D dataset ground truths during training and using diffusion-generated multi-views during inference.

3. **Key Challenge**: Multi-view generative models excel at generating high-precision images from individual views but lack consistency across different views, whereas 3D reconstruction models guarantee multi-view consistency but fall short in terms of fidelity/precision. As both types of models have complementary strengths, the critical challenge is how to combine their advantages.

4. **Goal**: How to train high-fidelity feed-forward 3D reconstruction models using 2D image datasets without relying on high-quality 3D datasets?

5. **Key Insight**: The authors observe that consistency can be preserved via reconstruction while precision is enhanced via generation—namely, first utilizing a 3D reconstruction model to obtain a consistent coarse 3D model, rendering multi-view images from it, and then employing a diffusion model to refine these images to generate high-quality pseudo-ground-truths.

6. **Core Idea**: Converting arbitrary 2D image datasets into high-quality multi-view training data through a "generation → reconstruction → rendering → refinement" pipeline, and fine-tuning feed-forward reconstruction models via LoRA to achieve SOTA performance.

## Method

### Overall Architecture
Given a single-view image, multi-view images are first generated using a multi-view diffusion model (Era3D) and then fed into a large-scale 3D reconstruction model (LGM) to produce a consistent 3D Gaussian Splatting (3DGS) representation. Next, images are rendered from specific camera poses of this 3D representation. Forward diffusion (noising) and reverse denoising refinement are applied to these rendered images to yield a high-precision, highly consistent pseudo-ground-truth multi-view dataset. Finally, this dataset is used to fine-tune the reconstruction model with LoRA, accompanied by an input view optimization step to align the output with the user input.

### Key Designs

1. **Multi-View Refinement Strategy**:

    - **Function**: Generating pseudo-ground-truth multi-view data that balance high precision and high consistency.
    - **Mechanism**: A reconstruction model is first used to obtain a consistent 3DGS $\theta$, from which multi-view images $x^\pi$ are rendered. Controlled noise is added to the rendered images: $x_t^\pi = \alpha_t x^\pi + \sigma_t \epsilon$ (with noise strength $s$). Then, a multi-view diffusion model is utilized to denoise these images conditioned on the original input image, producing refined multi-view images $C_\uparrow^\pi = \mathcal{G}(X_t^\pi; c, t)$, where $t = sT$. The key parameter $s$ controls the intensity of the refinement—if $s$ is too small, the refinement effect is weak; if $s$ is too large, new inconsistencies may be introduced.
    - **Design Motivation**: Direct multi-view generation using diffusion models lacks consistency constraints (e.g., VFusion3D). The proposed method adopts a two-stage process of reconstruction prior to refinement, leveraging 3D reconstruction to ensure consistent structure while utilizing the diffusion model to enhance texture fidelity.

2. **Boosting Reconstruction Model**:

    - **Function**: Fine-tuning the feed-forward reconstruction model using the refined pseudo-ground-truth data.
    - **Mechanism**: Starting with LGM, LoRA is applied only to its cross-view self-attention components to obtain the boosted model $\mathcal{R}_{\phi^*}$. During training, the input consists of the original (unrefined) multi-view images $C^\pi$, while the supervision signal is the refined multi-view images $C_\uparrow^\pi$. The loss function $\mathcal{L}$ is a combination of MSE + LPIPS, comparing the rendered views to the refined views. The training data is obtained from a pipeline of 100k+ ChatGPT-generated prompts → text-to-image models → multi-view refinement, eliminating any dependency on existing 3D datasets.
    - **Design Motivation**: LoRA fine-tuning is stable and parameter-efficient, requiring only around one day of training on 8 A100 GPUs, which, to the authors' knowledge, is the lowest training cost among comparable methods.

3. **Input View Optimization**:

    - **Function**: Precisely aligning the generated 3D assets with the user input image.
    - **Mechanism**: This is a post-processing stage. First, the optimal camera pose $\pi_{opt}$ that minimizes the LPIPS loss is searched across all possible poses. A learnable weight matrix $W$ is then applied to the 3DGS, optimizing only the view corresponding to the optimal pose to align with the input image while freezing other views. This enhances the fidelity of the input view without compromising the quality of other views.
    - **Design Motivation**: Aligning 3D reconstruction results with the input image is a critical metric for quality assessment. Post-processing optimization significantly reduces the LPIPS of the input view (from 0.108 to 0.002).

### Loss & Training
- **Training Loss**: A weighted sum of MSE + LPIPS, supervising the difference between rendered views and refined multi-views.
- **Input View Optimization Loss**: Pure LPIPS loss, optimizing only the perceptual similarity between the render at the optimal pose and the input image.
- Training data is synthesized entirely by the ChatGPT + text-to-image + multi-view refinement pipeline, featuring zero 3D dataset dependency.
- Training takes approximately one day on 8 A100 GPUs.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CD↓ | F-Score↑ |
|------|-------|-------|--------|-----|----------|
| OpenLRM | 16.728 | 0.785 | 0.208 | 0.148 | 0.656 |
| VFusion3D | 17.416 | 0.846 | 0.155 | 0.161 | 0.637 |
| CRM | 17.435 | 0.800 | 0.195 | 0.124 | 0.731 |
| InstantMesh | 16.796 | 0.786 | 0.207 | 0.123 | 0.737 |
| LGM | 17.148 | 0.776 | 0.220 | 0.170 | 0.635 |
| **MVBoost** | **18.561** | **0.859** | **0.131** | **0.101** | **0.798** |

### Ablation Study

| Refinement Strength s | Refined 2D Quality PSNR↑ | Reconstruction Model PSNR↑ |
|-----------|-------------------|----------------|
| original (w/o refinement) | 17.811 | 17.851 |
| 0.50 | 17.760 | 17.764 |
| 0.90 | 18.270 | 18.021 |
| **0.95** | **19.132** | **18.093** |
| 1.00 | 18.583 | 18.053 |

### Key Findings
- **Optimal refinement strength is $s=0.95$**: If too low, it fails to improve quality effectively; if too high ($s=1.0$, i.e., complete regeneration), it introduces new inconsistencies. This suggests that moderate noise addition and denoising provide the best trade-off between precision and consistency.
- **Core value of multi-view refinement**: The reconstruction model trained on the original multi-view dataset achieves a PSNR of 17.851, which improves to 18.093 when trained on the refined data, demonstrating that the pseudo-ground-truth data is of higher quality than the raw generated outputs.
- **Input view optimization yields remarkable gains**: LPIPS is reduced from 0.108 to 0.002, though this is a post-processing step and does not affect the primary results.
- **Compatibility across 3D representations**: Although this training utilizes 3DGS, it also outperforms mesh-based methods on geometric quality metrics (CD and F-Score).

## Highlights & Insights
- **The design philosophy of "ensuring consistency via reconstruction and improving precision via generation" is highly practical**. Instead of pursuing perfect multi-view generation directly, it leverages the complementary advantages of two distinct model categories. This perspective is highly transferable to other scenarios involving complementary imperfect models.
- **Zero dependency on 3D datasets is a key highlight**. The training data is entirely sourced from the text → image → multi-view → refinement pipeline, which can theoretically scale infinitely to any domain. This offers a new paradigm for addressing the 3D data scarcity bottleneck.
- **The LoRA fine-tuning strategy is exceptionally cost-effective** (8 A100 GPUs × 1 day) while significantly boosting the base model's performance. The decision to only fine-tune the cross-view self-attention components is both stable and efficient.

## Limitations & Future Work
- The pipeline heavily relies on multiple pre-trained models (text-to-image, multi-view diffusion, 3D reconstruction); bottlenecks in any individual model can propagate to the final output.
- The refinement process still depends on the multi-view diffusion model, making it unable to correct systemic biases inherent to the model itself (such as hallucinations regarding specific objects/categories).
- Evaluation is limited to the GSO dataset; validation on more complex scenes (e.g., humans, indoor environments) is currently lacking.
- Input view optimization is executed on a single view only; multi-view inputs have not yet been addressed.

## Related Work & Insights
- **vs LGM**: LGM serves as the base model for MVBoost. By fine-tuning LGM with refined data, MVBoost improves its PSNR from 17.148 to 18.561 and reduces CD from 0.170 to 0.101.
- **vs VFusion3D**: VFusion3D also generates multi-view data using video diffusion models but lacks explicit consistency constraints. MVBoost ensures consistency through its "reconstruction before refinement" workflow.
- **vs SDS Methods (DreamFusion)**: SDS-based methods require hour-long optimizations and suffer from the Janus problem, whereas MVBoost is a feed-forward method that features rapid inference without the Janus issue.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of "reconstruction + refinement" to generate pseudo-ground-truths is novel, although individual components already exist.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is conducted only on GSO, meaning scene diversity is limited.
- Writing Quality: ⭐⭐⭐⭐ Clarify of methodology explanation is good, and the framework diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Provides a training paradigm independent of 3D datasets with extremely low training costs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement](imfine_3d_inpainting_via_geometry-guided_multi-view_refinement.md)
- [\[CVPR 2025\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)
- [\[CVPR 2025\] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation](sharp-it_a_multi-view_to_multi-view_diffusion_model_for_3d_synthesis_and_manipul.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](../../ICCV2025/3d_vision/boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)
- [\[CVPR 2025\] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](murre_sfm_guided_depth_reconstruction.md)

</div>

<!-- RELATED:END -->
