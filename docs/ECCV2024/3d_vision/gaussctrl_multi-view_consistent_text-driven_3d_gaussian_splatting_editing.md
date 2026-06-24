---
title: >-
  [Paper Note] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing
description: >-
  [ECCV 2024][3D Vision][3D Editing] GaussCtrl is proposed, which utilizes depth-conditioned ControlNet editing and an attention alignment module to achieve multi-view consistent text-driven 3DGS scene editing, supporting editing of all viewpoints at once and requiring only a single 3D model update.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Editing"
  - "3D Gaussian Splatting"
  - "Diffusion Models"
  - "Multi-view Consistency"
  - "ControlNet"
date: 2026-05-08
content_hash: f29ba2445f203266
---

# GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing

**Conference**: ECCV 2024  
**arXiv**: [2403.08733](https://arxiv.org/abs/2403.08733)  
**Code**: [https://gaussctrl.active.vision/](https://gaussctrl.active.vision/)  
**Area**: 3D Vision  
**Keywords**: 3D Editing, 3D Gaussian Splatting, Diffusion Models, Multi-view Consistency, ControlNet

## TL;DR

GaussCtrl is proposed, which utilizes depth-conditioned ControlNet editing and an attention alignment module to achieve multi-view consistent text-driven 3DGS scene editing, supporting editing of all viewpoints at once and requiring only a single 3D model update.

## Background & Motivation

**Background**: NeRF and 3DGS have achieved high-quality 3D reconstruction and novel-view rendering, but the editing capability on these 3D representations is still under-explored. Instruct-NeRF2NeRF (IN2N) pioneered the "3D editing via 2D editing" paradigm—editing rendered images frame-by-frame using a 2D diffusion model, and then iteratively updating the 3D model.

**Limitations of Prior Work**: The core problem of IN2N and subsequent methods is **multi-view inconsistency**—2D diffusion models process each image independently without guaranteeing geometric and appearance consistency, leading to blurriness, artifacts, and issues like "faces popping up on the back" in 3D editing. Moreover, the iterative optimization scheme results in slow convergence.

**Key Challenge**: 2D diffusion models naturally lack 3D geometric awareness, yet 3D editing relies on 2D editing results—how can 3D consistency constraints be injected during the 2D editing stage?

**Goal**: Explicitly enforce multi-view consistency across the editing process of all view images, enabling editing of all images at once and updating the 3D model only once.

**Key Insight**: Leverage the consistency information already provided by 3DGS itself (depth maps are naturally geometrically consistent) and a cross-view attention mechanism (unifying text editing $\rightarrow$ appearance consistency) as a dual approach.

**Core Idea**: Depth map conditioned editing to guarantee geometric consistency + attention latent code alignment to guarantee appearance consistency = multi-view consistent 3D editing.

## Method

### Overall Architecture

The pipeline of GaussCtrl consists of four steps:
1. Render all training-view images and corresponding depth maps from the reconstructed 3DGS.
2. Use DDIM Inversion to invert each image into latent noise.
3. Use ControlNet (conditioned on depth) coupled with an attention alignment module to denoise and generate edited images based on the edited text prompt.
4. Retrain the 3DGS using the edited images to obtain the edited 3D model.

### Key Designs

1. **Depth-Conditioned Image Editing**: Utilizing the depth map $\mathcal{D}$ rendered from the 3DGS as the condition for ControlNet, ensuring that edited images preserve the original geometric structure. The core pipeline is:

    - DDIM Inversion: Encodes the original image $\mathcal{I}$ into $z^0$ via ControlNet's VAE, and iteratively inverts it into noise $z^T$:
   
    $$z^{t+1} = \sqrt{\alpha_{t+1}} \frac{z^t - \sqrt{1-\alpha_t} \cdot \epsilon^t}{\sqrt{\alpha_t}} + \sqrt{1-\alpha_{t+1}} \epsilon^t$$
   
    - Editing Denoising: Replaced with the edited prompt $\hat{p}_e$ and denoised via classifier-free guidance:
   
    $$\epsilon^t = \epsilon_\emptyset^t + \omega \cdot (\epsilon_p^t - \epsilon_\emptyset^t)$$
   
   Design Motivation: Depth maps derived from the same 3DGS are naturally geometrically consistent across views. Conditioning the editing on these depth maps avoids geometric incongruence. DDIM inversion ensures that the initial latent codes inherit the consistent color and geometry of original images.

2. **Attention-Based Latent Code Alignment Module**: While depth conditions guarantee geometric consistency, each view is still edited independently, which can cause appearance inconsistencies (color discrepancies, anomalies in challenging views). The module unifies appearance by blending self-attention and cross-view attention:

    $$\text{AttnAlign}_e = \lambda \cdot \text{Attn}_{e,e} + (1-\lambda) \cdot \frac{1}{N_r} \sum_{i=1}^{N_r} \text{Attn}_{e,i}$$

   Where the attention operation is defined as:
   
    $$\text{Attn}_{i,j} = \text{Softmax}\left(\frac{W_q(z_i) W_k(z_j)^\top}{\sqrt{c}}\right) W_v(z_j)$$

    - Self-attention $\text{Attn}_{e,e}$: Preserves the uniqueness of each edited image.
    - Cross-view attention $\text{Attn}_{e,i}$: Aligns appearance to $N_r$ reference views.
    - $\lambda = 0.6$ and $N_r = 4$ randomly sampled reference views.
   
   Design Motivation: Prior research suggests that key-value pairs in self-attention within diffusion models dictate the appearance of generated images. Injecting the K/V pairs of reference views unifies the appearance across all views.

3. **Optional Semantic Segmentation Mask**: Employs Language-based SAM (Lang SAM) to generate masks, filtering background areas when editing regional objects to enhance editing quality.

### Loss & Training

- Edited images are directly used to retrain 3DGS (using NeRFStudio's splatfacto model).
- Based on Stable Diffusion v1.5 and its corresponding ControlNet.
- All images are preprocessed to $512 \times 512$ resolution.
- Editing a scene takes approximately 9 minutes (NVIDIA RTX A5000, 24GB VRAM).
- The alignment module simultaneously replaces all self-attentions in both the U-Net and ControlNet blocks.

## Key Experimental Results

### Main Results (CLIP Directional Similarity + Editing Time)

| Scene | IN2N CLIPdir | IN2N(GS) CLIPdir | ViCA-NeRF CLIPdir | **Ours CLIPdir** | Ours Time |
|------|:---:|:---:|:---:|:---:|:---:|
| Bear Statue | 0.1019 | 0.1165 | 0.1104 | **0.1388** | ~9min |
| Dinosaur | 0.1466 | 0.1490 | 0.0723 | **0.1584** | ~9min |
| Garden | **0.3027** | 0.1663 | 0.2903 | 0.2891 | ~9min |
| Stone Horse | 0.1654 | 0.1947 | 0.1926 | **0.2268** | ~9min |
| Fangzhou | 0.1598 | 0.2032 | 0.1809 | 0.1887 | ~9min |
| Face | 0.1332 | 0.1357 | 0.1119 | **0.1503** | ~9min |

Editing time comparison: IN2N ~1.5h, IN2N(GS) ~13.5min, ViCA-NeRF ~38.5min, **Ours ~9min**

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| (b) Instruct Pix2Pix Single Edit | Fails completely on challenging views (backside #4, 6, 7, 8) and exhibits artifacts in frontal views. |
| (c) ControlNet + Random Noise | Geometrically consistent but style deviates from the original; incorrectly generates frontal features on back views. |
| (d) ControlNet + Inverted Latent (W/O Alignment) | Style is markedly improved, but challenging views still suffer from artifacts and "face-on-the-back" issues. |
| **(e) + AttnAlign (Full Method)** | Artifacts are substantially mitigated, appearance is unified, and semantically correct views are maintained even under challenging perspectives. |

### Key Findings

- CLIPdir is optimal on 4 out of 6 scenes, achieving the fastest editing speed (~9min vs. ~1.5h for IN2N, a 10x acceleration).
- The CLIPdir metric does not fully reflect edit quality—it measures global text-to-image similarity while disregarding local details (the paper shows counterexamples where CLIPdir is high but visual quality is poor).
- 360-degree scenes highlight the method's advantages more than forward-facing scenes due to extreme changes in viewpoint.
- DDIM Inversion + depth condition forms the baseline of consistency, while AttnAlign further eliminates appearance inconsistency and semantic anomalies.

## Highlights & Insights

- **Paradigm shift of "edit all views at once"**: Unlike prior methods (e.g., IN2N) which require an iterative loop of single-frame editing and 3D model updating, this work achieves batch editing followed by a one-time 3D update, speeding up the process by 10x.
- **Ingenious utilization of 3DGS depth info**: 3DGS inherently provides multi-view consistent depth maps, making their use as a ControlNet condition an elegant design choice.
- **Deep understanding of the attention mechanism**: The insight that K/V in self-attention determines appearance is effectively leveraged. Introducing cross-view attention fundamentally establishes appearance communication across views.
- **Analysis of the limitations of the CLIPdir metric**: It points out the inadequacy of this metric and provides counterexamples, offering critical insights for valuation methodologies in the field.

## Limitations & Future Work

- **Inability to alter geometric structures**: The depth condition preserves original geometry, thereby preventing edits that require substantial geometric changes (e.g., turning a bear into a giraffe)—though the paper notes that methods like IN2N share this limitation.
- **Dependence on ControlNet capability**: If ControlNet is unfamiliar with certain concepts (e.g., "Hulk"), editing will fail.
- **Reference view selection strategy**: Currently, 4 reference views are randomly sampled, which may not be optimal; coverage-based selection could yield better results.
- **Imperfect evaluation metrics**: CLIPdir is insufficient for evaluating 3D edit quality, calling for improved benchmarks.

## Related Work & Insights

- IN2N pioneered the "2D editing $\rightarrow$ 3D editing" paradigm, yet its iterative nature inherently causes inconsistency.
- ViCA-NeRF attempts to blend reference view projections to mitigate inconsistency, but this introduces blurriness.
- Works such as Prompt-to-Prompt and MasaCtrl reveal the control of K/V pairs over appearance in attention mechanisms.
- The methodology in this work can be extended to tasks requiring temporal/spatial consistency, such as video editing and 4D scene editing.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the combination of depth conditions and attention alignment is not entirely collection, its design in the context of 3D editing is refined and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation covers 360-degree and forward-facing scenes, various editing types, detailed ablation studies, and consistency visualizations (10-view comparison).
- Writing Quality: ⭐⭐⭐⭐⭐ Rich illustrations, exceptionally clear problem formulation, progressively structured ablation designs, and a candid discussion of metric limitations.
- Value: ⭐⭐⭐⭐ 10x acceleration coupled with superior quality offers significant practical value, setting a new baseline for 3D editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DEgo: 3D Editing on the Go!](3dego_3d_editing_on_the_go.md)
- [\[ECCV 2024\] DATENeRF: Depth-Aware Text-based Editing of NeRFs](datenerf_depth-aware_text-based_editing_of_nerfs.md)
- [\[ECCV 2024\] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation](dreamview_injecting_view-specific_text_guidance_into_text-to-3d_generation.md)
- [\[ECCV 2024\] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)

</div>

<!-- RELATED:END -->
