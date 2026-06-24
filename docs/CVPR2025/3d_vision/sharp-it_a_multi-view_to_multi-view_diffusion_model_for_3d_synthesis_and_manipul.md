---
title: >-
  [Paper Note] Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation
description: >-
  [CVPR 2025][3D Vision][3D Enhancement] Proposes Sharp-It, a multi-view to multi-view diffusion model that enhances low-quality object outputs from 3D generative models like Shap-E into high-quality multi-view images via 2D diffusion, reducing the FID to 6.60 and supporting appearance editing in just 10 seconds.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Enhancement"
  - "Multi-view Diffusion"
  - "Shap-E"
  - "Geometric Refinement"
  - "3D Editing"
date: 2026-05-08
content_hash: 18901aa61d1ceccd
---

# Sharp-It: A Multi-view to Multi-view Diffusion Model for 3D Synthesis and Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2412.02631](https://arxiv.org/abs/2412.02631)  
**Code**: [Project Page](https://yiftachede.github.io/Sharp-It/)  
**Area**: 3D Vision  
**Keywords**: 3D Enhancement, Multi-view Diffusion, Shap-E, Geometric Refinement, 3D Editing

## TL;DR

Proposes Sharp-It, a multi-view to multi-view diffusion model that enhances low-quality object outputs from 3D generative models like Shap-E into high-quality multi-view images via 2D diffusion, reducing the FID to 6.60 and supporting appearance editing in just 10 seconds.

## Background & Motivation

3D content generation faces a trade-off between quality and controllability:
- **Multi-view reconstruction methods**: Generate multi-view images first and then reconstruct 3D; high quality but poor controllability, and prone to the Janus problem.
- **Native 3D generative models** (e.g., Shap-E): Directly generate 3D representations; highly controllable (editing/controlled generation) but limited by resolution, resulting in low output quality and coarse geometry.

Core Idea: Instead of replacing the low-quality output, **enhance** it—starting from multi-view renderings of the low-quality 3D asset, a diffusion model is used to inject fine geometric and texture details.

## Method

### Overall Architecture

1. Generate low-quality 3D objects using Shap-E → Render 6-view images.
2. The Sharp-It diffusion model conditions on the low-quality multi-view views + text prompt to enhance all views in one pass.
3. The enhanced multi-view images are reconstructed into high-quality 3D models using methods like InstantMesh.

### Key Designs

#### Key Design 1: Multi-view Conditioning Architecture

- **Function**: Enable the diffusion model to simultaneously accept low-quality input views and text guidance.
- **Mechanism**: Based on the Zero123++ architecture, the UNet input is expanded from 4 channels to 8 channels—4 channels of latent noise + 4 channels of VAE-encoded Shap-E multi-view images. In the cross-attention layers, the original image embeddings are replaced with text prompts to provide appearance control.
- **Design Motivation**: The 6 views are arranged in a $3\times2$ grid (total resolution of $960\times640$). Self-attention in the model naturally enables cross-view feature sharing (cross-view attention). The 8-channel design adapts conditioning injection methods from image editing, but scales it to multi-view 3D-consistent enhancement.

#### Key Design 2: Encoder-Paired Dataset Construction

- **Function**: Provide high-quality and low-quality paired training data.
- **Mechanism**: High-quality 3D objects from Objaverse are encoded into the latent space using the Shap-E encoder and then decoded to obtain corresponding low-quality versions. Rendering 6 views $\times$ 3 HDR illuminations yields 180K pairs of training data. Instances with encoding failures (over-degradation) and excessively thin objects are filtered out, and BLIP-2 is used to generate text captions.
- **Design Motivation**: It leverages the "lossy compression" property of the Shap-E encoder to naturally construct degraded-real pairs without needing hand-crafted degradation methods. Multi-light augmentation prevents the model from overfitting to a single lighting condition (as validated by ablation studies).

#### Key Design 3: Cross-view Self-Attention Consistency

- **Function**: Ensure 3D consistency across different views after enhancement.
- **Mechanism**: The model automatically learns cross-view correspondences via self-attention layers on the $3\times2$ grid. Attention map visualizations show that a query point on a wheel in one view receives the highest attention weights in other views, and can also identify semantically similar parts (other wheels).
- **Design Motivation**: Since the inputs are already 3D-consistent (rendered from the same 3D model), the diffusion model can focus on adding details rather than establishing consistency from scratch, substantially simplifying the task.

### Loss & Training

Standard v-prediction diffusion training loss is used, with a CFG drop probability of 0.1, trained for 500K steps on a single A6000 GPU.

## Key Experimental Results

### Main Results: Comparison of 3D Object Enhancement Quality

| Method | FID↓ | CLIP↑ | DINO↑ | Runtime |
|------|------|-------|-------|---------|
| GaussianDreamer | 50.89 | 0.81 | 0.82 | 6min |
| MVEdit | 44.87 | 0.83 | 0.77 | 1min |
| MVDream w/ SDEdit | 28.71 | 0.81 | 0.83 | 10sec |
| Zero123++ w/ SDEdit(R) | 19.13 | 0.87 | 0.89 | 10sec |
| **Sharp-It** | **6.60** | **0.90** | **0.92** | **10sec** |

The FID of 6.60 outperforms all other methods by a large margin, while achieving the highest CLIP and DINO similarities, indicating that the outputs are closest to real high-quality objects.

### Ablation Study

| Ablation Setting | Effect |
|--------|------|
| w/o text prompt | Degraded enhancement quality and controllability |
| w/o diverse lighting | Poorer generalization to different lighting conditions |
| **Ours** | **Best performance** |

### Key Findings

- Sharp-It preserves the color and structural consistency of the input objects, whereas other methods tend to deviate from the original source.
- Supports appearance editing: Enhancing the same Shap-E output with different text prompts can modify the material or style.
- Can be combined with the controllable generation features of Shap-E (e.g., skeleton control) to achieve controlled high-quality 3D generation.

## Highlights & Insights

1. **Enhancement Over Replacement**: Utilizes the coarse but roughly correct geometry of the low-quality 3D model as a prior, allowing the 2D diffusion model to focus solely on detail synthesis.
2. **Encoder as Degradation**: Cleverly exploits the information loss of the Shap-E encoder to synthesize paired training data without needing hand-crafted degradation functions.
3. **10-Second High-Quality 3D**: The pipeline of Shap-E generation followed by Sharp-It enhancement achieves a win-win in both quality and speed.

## Limitations & Future Work

- It is tightly coupled with Shap-E as a backbone; the upper limit of Shap-E's generative capacity bounds Sharp-It's performance.
- Cannot effectively enhance objects where the Shap-E encoding fails severely.
- Six views may be insufficient to fully capture all details of highly complex objects.
- Future improvements could extend this approach to stronger 3D generators (e.g., 3DShape2VecSet).

## Related Work & Insights

- **Shap-E**: The 3D backbone of Sharp-It, offering low-quality but highly controllable implicit 3D generation.
- **Zero123++**: The foundational 2D architecture for Sharp-It, serving as a multi-view image generative model.
- **InstantMesh**: A feed-forward framework used to reconstruct 3D meshes from enhanced multi-view inputs.

## Rating

⭐⭐⭐⭐ — Clean design intuition (enhancement $\neq$ generation), clever data construction, with impressive empirical results (FID of 6.60 in just 10 seconds). The main limitation is its tight coupling with Shap-E.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [\[CVPR 2025\] SplatFlow: Multi-View Rectified Flow Model for 3D Gaussian Splatting Synthesis](splatflow_multi-view_rectified_flow_model_for_3d_gaussian_splatting_synthesis.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)
- [\[CVPR 2025\] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)

</div>

<!-- RELATED:END -->
