---
title: >-
  [Paper Note] SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons
description: >-
  [CVPR 2026][Segmentation][Vector layer construction] This paper proposes SemLayer, a generative-model-based pipeline that recovers semantically structured, layered representations from flattened vector icons. The approach reframes segmentation as a colorization task via a diffusion model, follows with semantic amodal completion of occluded regions, and applies integer linear programming (ILP) to determine layer ordering, achieving segmentation gains of +5.0 mIoU and +16.7 PQ.
tags:
  - CVPR 2026
  - Segmentation
  - Vector layer construction
  - semantic segmentation colorization
  - amodal completion
  - icon editing
  - diffusion models
date: 2026-05-08
content_hash: b6cc2da9d45b2031
---

# SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons

**Conference**: CVPR 2026
**arXiv**: [2603.24039](https://arxiv.org/abs/2603.24039)
**Code**: [https://xxuhaiyang.github.io/SemLayer/](https://xxuhaiyang.github.io/SemLayer/)
**Area**: Segmentation / Vector Graphics
**Keywords**: Vector layer construction, semantic segmentation colorization, amodal completion, icon editing, diffusion models

## TL;DR

This paper proposes SemLayer, a generative-model-based pipeline that recovers semantically structured, layered representations from flattened vector icons. The approach reframes segmentation as a colorization task via a diffusion model, follows with semantic amodal completion of occluded regions, and applies integer linear programming (ILP) to determine layer ordering, achieving segmentation gains of +5.0 mIoU and +16.7 PQ.

## Background & Motivation

1. **State of the Field**: Vector icons are a cornerstone of modern design workflows. Designers typically organize semantically meaningful graphical elements across multiple editable layers. However, icons are frequently "flattened" upon publication and distribution, merging all layers into a single composite path and discarding the original semantic layer hierarchy.

2. **Limitations of Prior Work**: Once semantic structure is lost, downstream operations such as recoloring, animation, and local editing become extremely difficult, forcing designers to manually re-segment and reconstruct icons. Existing methods such as SAM perform poorly on highly abstract black-and-white icons due to the absence of texture, shadow, and color cues, while optimization-based methods tend to generate excessively fragmented layers.

3. **Root Cause**: The high level of abstraction in icons means that conventional visual understanding cues—texture, shading, and depth—are almost entirely absent. At the same time, recovering complete geometry, including occluded regions, and correctly inferring stacking order remain essential requirements.

4. **Paper Goals**: To recover an editable, semantically layered representation from flattened single-path or composite-path vector icons.

5. **Starting Point**: Leverage the rich shape priors embedded in generative models (diffusion models) to compensate for the scarcity of icon-domain data and the absence of visual features.

6. **Core Idea**: Reframe semantic segmentation as a colorization task—using a diffusion model to colorize black-and-white icons such that different semantic components become visually separable—then employ a diffusion model for amodal completion of occluded regions, and finally apply ILP to determine layer ordering.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) **Semantic-aware Generative Segmentation**: a monochrome icon is fed into a diffusion model to generate a colorized version in which distinct colors correspond to distinct semantic components; binary masks $\{V_1, ..., V_K\}$ are extracted via color thresholding. (2) **Amodal Layer Completion**: each semantic component's complete shape, including occluded portions, is recovered using a diffusion model to yield $\{A_1, ..., A_K\}$. (3) **Layer Order Optimization**: ILP determines the stacking order of components, which are subsequently vectorized back into SVG format.

### Key Designs

1. **Semantic-aware Generative Segmentation (Segmentation as Colorization)**

    - **Function**: Decompose monochrome or two-color icons into semantically meaningful components.
    - **Mechanism**: Conventional segmentation methods (e.g., SAM) fail on abstract icons due to the absence of color and texture cues. The paper reframes segmentation as a colorization task—assigning distinct colors to different semantic components while preserving structural integrity. The approach is built on the EasyControl framework, employing a conditional LoRA on a diffusion Transformer, with binary contour maps as control-encoded conditions and text prompts guiding colorization. Training uses a flow-matching objective: $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\epsilon} \|v_\theta(z_n, t, z_c) - (\epsilon - x_0)\|_2^2$. At inference, each color channel in the output is thresholded to produce independent masks. The training set comprises 8,567 icon–colorization pairs sourced from real SVGs and GPT-4o+gpt-image-1 synthesis.
    - **Design Motivation**: Colorization is a task better suited to generative models than segmentation—diffusion models possess strong priors for understanding shape semantics and color assignment, elegantly circumventing the difficulties that conventional segmentation methods face on icons.

2. **Amodal Layer Completion**

    - **Function**: Recover the complete shape of each semantic component, including regions occluded by other components.
    - **Mechanism**: Built on fine-tuning the pix2gestalt latent diffusion model. The input consists of the occluded image and the visible region mask; high-level semantic conditioning is provided via CLIP image embeddings, while geometric conditioning is supplied by concatenating VAE-encoded occluded patches with the mask. Training employs a fragmented-visibility strategy: when a component is occluded and split into multiple disconnected fragments $\{V^{(i)}\}$, each fragment is used independently as input, but all fragments supervise the recovery of the same complete shape (many-to-one completion). A post-inference IoU merging step ($\tau=0.7$) consolidates multiple completion results belonging to the same object. The completion dataset SemLayer-Completion contains 50,000 training triplets.
    - **Design Motivation**: Amodal completion models trained on natural images cannot be directly applied to black-and-white icons due to the large domain gap, necessitating dedicated fine-tuning for the icon style.

3. **Layer Order Optimization (ILP Formulation)**

    - **Function**: Determine the stacking order of completed components.
    - **Mechanism**: Binary variables $x_{ij}$ encode whether component $i$ lies above component $j$, subject to anti-symmetry and transitivity constraints. Two pixel-level coverage variables are introduced: $y_i=1$ indicates that the extra region $E_i = A_i \setminus I$ is covered by an upper layer (desirable), and $z_i=1$ indicates that the visible region $V_i$ is incorrectly occluded (undesirable). The objective $\max_{x,y,z} \sum_i y_i - \lambda \sum_i z_i$ trades off between rewarding correct occlusion coverage and penalizing erroneous occlusion, with $\lambda=1$.
    - **Design Motivation**: Layer order determination is a combinatorial optimization problem for which ILP provides an exact solution, and the objective metrics—occlusion consistency versus visibility preservation—are clearly defined.

### Loss & Training

The segmentation model is trained from scratch for 40,000 steps (lr $1 \times 10^{-4}$, CFG scale 4.5), with 25 inference steps at resolution $512 \times 512$. The completion model is fine-tuned for 50,000 steps (lr $1 \times 10^{-5}$), with 50 inference steps at resolution $256 \times 256$. All experiments are conducted on 8 A100 GPUs. Vectorization uses potrace with a curve-reuse strategy to maximize retention of original Bézier segments.

## Key Experimental Results

### Main Results

Segmentation performance comparison (48-icon real SVG test set):

| Method | mIoU (%) | PQ (%) | Completion mIoU (%) | Completion CD ↓ |
|--------|----------|--------|---------------------|-----------------|
| gpt-image-1 | 25.4 | 6.20 | 60.9 | 71.4 |
| SAM2 | 51.1 | 26.2 | 69.2 | 61.7 |
| SAM2* (fine-tuned) | 79.3 | 59.4 | 80.7 | 49.1 |
| **SemLayer (Ours)** | **84.3** | **76.1** | **85.2** | **46.6** |

Completion model comparison (using fixed segmentation input from this paper):

| Method | mIoU (%) ↑ | CD ↓ |
|--------|-----------|------|
| gpt-image-1 | 10.7 | 98.6 |
| MP3D | 70.5 | 79.4 |
| MP3D-finetuned | 75.3 | 68.9 |
| **SemLayer (Ours)** | **85.2** | **46.6** |

### Ablation Study

Refined segmentation metrics:

| Configuration | mIoU_Refined (%) | PQ_Refined (%) |
|---------------|-----------------|----------------|
| gpt-image-1 | 57.2 | 39.3 |
| SAM2 | 62.2 | 37.8 |
| SAM2* | 85.3 | 78.0 |
| **SemLayer** | **86.4** | **78.3** |

### Key Findings

- **Segmentation-as-colorization significantly outperforms direct segmentation**: gains of +5.0 mIoU and +16.7 PQ over fine-tuned SAM2*.
- **gpt-image-1 performs poorly on icon segmentation**: mIoU of only 25.4%, demonstrating that general-purpose generative models struggle to understand the semantic structure of icons.
- **Domain adaptation of the completion model is critical**: the generic MP3D model achieves mIoU=70.5%; fine-tuning raises this to 75.3%, while the dedicated icon completion training in this paper reaches 85.2%.
- **The fragmented-visibility training strategy is effective**: many-to-one completion training enables the model to recover complete shapes from individual fragments.
- **The end-to-end pipeline produces directly editable layered SVGs**: supporting local recoloring, rotation, scaling, and simple animation.

## Highlights & Insights

- **The paradigm shift of segmentation-as-colorization is particularly elegant**: when conventional segmentation methods fail in a specific domain, the key insight is to ask "what task formulation is more amenable to generative models?" Colorization is a more natural task for diffusion models—this insight is transferable to other domains where direct segmentation is difficult.
- **The data construction pipeline is practical**: combining real SVGs from LayerPeeler with GPT-4o/gpt-image-1 synthesis, the authors constructed a segmentation dataset of 8,567 training samples and 50,000 completion triplets at modest manual cost.
- **The ILP formulation for layer ordering is intuitively clear**: the objective function—rewarding correct occlusion coverage and penalizing erroneous visibility occlusion—is concise and elegant.

## Limitations & Future Work

- **Only black-and-white line icons are handled**: colored and filled icons are not yet covered, though the authors note that color itself serves as a strong semantic cue, making extension relatively straightforward.
- **Highly entangled or heavily occluded icons may fail**: the paper acknowledges failure cases (Fig. 9).
- **The test set contains only 48 icons**: the evaluation scale is limited and may not represent all icon styles.
- **Stochasticity of generative models**: multiple runs and averaging are needed to stabilize results.

## Related Work & Insights

- **vs. LayerPeeler**: LayerPeeler provides a source of existing layered SVG data but lacks a segmentation method; SemLayer builds a complete segmentation–completion–ordering pipeline on top of its data.
- **vs. SAM2**: Even after fine-tuning, SAM2 still suffers from fragmentation and alignment issues because its design assumes rich visual cues; the colorization paradigm avoids these problems.
- **vs. optimization-based vectorization methods**: Differentiable rendering methods such as DiffVG achieve high visual fidelity but generate excessively fragmented layers and lack semantic consistency.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight of reframing segmentation as colorization is highly creative, and the three-stage pipeline design is clean and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐ Quantitative evaluation on 48 test icons is somewhat limited, though qualitative visualizations sufficiently demonstrate the results.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear, with four identified challenges each addressed by a corresponding solution.
- Value: ⭐⭐⭐⭐ The work has practical application value for design tooling, and the dataset and methodology can lay a foundation for vector graphics understanding.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semanticguided_modalityaware_segmentation_for.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[ICCV 2025\] LayerAnimate: Layer-level Control for Animation](../../ICCV2025/segmentation/layeranimate_layer-level_control_for_animation.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)

<!-- RELATED:END -->
