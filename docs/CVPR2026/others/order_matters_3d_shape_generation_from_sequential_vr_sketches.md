---
title: >-
  [Paper Note] Order Matters: 3D Shape Generation from Sequential VR Sketches
description: >-
  [CVPR 2026][VR sketching] This paper proposes VRSketch2Shape, a framework that, for the first time, models the temporal stroke order of VR sketches. Through a sequence-aware BERT encoder combined with a diffusion-based 3…
tags:
  - "CVPR 2026"
  - "VR sketching"
  - "3D shape generation"
  - "stroke order"
  - "diffusion model"
  - "sketch-to-shape"
date: 2026-05-08
content_hash: a54827bf348b4b51
---

# Order Matters: 3D Shape Generation from Sequential VR Sketches

**Conference**: CVPR 2026
**arXiv**: [2512.04761](https://arxiv.org/abs/2512.04761)  
**Authors**: Yizi Chen, Sidi Wu, Tianyi Xiao, Nina Wiedemann, Loic Landrieu (ETH Zurich, LIGM/ENPC/IP Paris)
**Code**: [VRSketch2Shape](https://chenyizi086.github.io/VRSketch2Shape_website/)  
**Area**: Others
**Keywords**: VR sketching, 3D shape generation, stroke order, diffusion model, sketch-to-shape

## TL;DR
This paper proposes VRSketch2Shape, a framework that, for the first time, models the temporal stroke order of VR sketches. Through a sequence-aware BERT encoder combined with a diffusion-based 3D generator (SDFusion), the framework generates high-fidelity 3D shapes from ordered VR sketches. The work also contributes a multi-category dataset comprising 20k synthetic and 900 real sketches.

## Background & Motivation

Creating high-quality 3D content is a core requirement in architecture and industrial design. Traditional CAD tools (e.g., Blender) have steep learning curves and are ill-suited for rapid ideation and early-stage exploration. While text-guided 3D generation has advanced, natural language remains too ambiguous to precisely specify complex geometry.

VR sketching enables users to explore and iterate ideas directly in 3D space, eliminating the perspective ambiguity and occlusion inherent to 2D sketching. However, existing VR sketch-to-shape methods face three key challenges:

- **Data scarcity**: The only public benchmark, 3DVRChair, contains merely 1,005 sketch–shape pairs restricted to the chair category.
- **Geometric misalignment**: Hand-drawn sketches naturally contain perspective and depth perception errors, resulting in imperfect alignment with target shapes.
- **Loss of temporal information**: Existing methods treat VR sketches as unordered point clouds, discarding critical signals such as stroke order and stroke length—signals that encode important information about connectivity, structure, and design intent.

Core insight: **stroke order matters**. Humans draw global outlines before adding details; this coarse-to-fine ordering encodes rich structural priors.

## Method

### Overall Architecture
VRSketch2Shape consists of three components: (1) a synthetic sketch generation pipeline, (2) a sequence-aware sketch encoder, and (3) a diffusion-based 3D shape generator.

### Synthetic Sketch Generation Pipeline (Training-Free)
This pipeline automatically generates temporally ordered VR sketches from arbitrary 3D meshes, producing 20,838 samples in approximately 10 hours:

1. **Salient point extraction**: 2,048 points are uniformly sampled on the mesh surface; high-curvature and structurally salient regions are retained via Sharp Edge Sampling (SES) with a curvature threshold of 15.
2. **Stroke recovery**: EMAP is used to fit Bézier splines (maximum degree 2, minimum segment length 12); near-linear redundant points are pruned (cosine distance threshold 0.04), and strokes whose endpoint distance is less than 2% of the shape extent are merged.
3. **Stroke ordering**: A connectivity graph is constructed based on endpoint spatial proximity and traversed depth-first; nearest connections are skipped with 10% probability to introduce randomness.

### Real Sketch Collection
A VR drawing interface developed in Unity incorporates a **surface snapping mechanism** that projects drawn points onto the 3D model surface along the shortest path. Fifteen participants drew 900 sketches (300 chairs, 200 tables, 200 cabinets, 200 airplanes), each taking approximately 15 minutes, totaling roughly 225 person-hours.

### Sequence-Aware Sketch Encoder

**Sketch tokenization**: The sketch is represented as an ordered stroke sequence with special tokens SEP (stroke end) and EoS (end of sketch):
$$\mathcal{S} = [p_1^1, \cdots, p_{n_1}^1, \text{SEP}, \cdots, p_1^S, \cdots, p_{n_S}^S, \text{SEP}, \text{EoS}]$$

**Spatial embedding**: Fourier feature encoding ($L=10$ frequencies) is applied to each 3D coordinate $(x, y, z)$; the concatenated features are mapped to $D=256$ dimensions via a 2-layer MLP:
$$E_{\text{spa}}(p) = \text{MLP}_{\text{spa}}([\Phi_{\text{spa}}(x), \Phi_{\text{spa}}(y), \Phi_{\text{spa}}(z)])$$

**Sequential embedding**: Sinusoidal encoding followed by linear projection is applied separately to stroke index $s$ and point index $i$:
$$E_{\text{stroke}}(s) = \text{Lin}_{\text{stroke}}(\Phi_{\text{seq}}(s)), \quad E_{\text{point}}(i) = \text{Lin}_{\text{point}}(\Phi_{\text{seq}}(i))$$

**Final token embedding**: The three embeddings are summed: $E(p_i^s) = E_{\text{spa}}(p_i^s) + E_{\text{stroke}}(s) + E_{\text{point}}(i)$

**BERT encoder**: 6 Transformer layers, 8 attention heads, feedforward width ratio of 1, dropout 0.1.

**Key differences from SketchBERT**: (i) spatial Fourier features instead of raw coordinates; (ii) stroke separators as learnable tokens rather than concatenated one-hot flags; (iii) continuous Fourier encoding replacing fixed lookup tables, enabling flexible handling of variable-length sketches.

### Data Augmentation
- **Stroke dropout**: 15% of strokes are randomly masked.
- **Point dropout**: 30% of points within remaining strokes are randomly masked.
- **Stroke swap**: 20% of strokes are randomly reordered.

### Diffusion-Based 3D Shape Generation
SDFusion (a latent diffusion model) is used to generate 3D shapes. Ground-truth 3D shapes are voxelized and encoded into a compact latent representation by a pretrained 3D VQ-VAE. A U-Net predicts denoised latent vectors conditioned on the BERT encoder output. At inference, 100-step DDIM sampling generates 3D shapes from random noise. The VQ-VAE is frozen while the U-Net and sketch encoder are jointly optimized end-to-end.

## Key Experimental Results

### Table 2: Quantitative Main Results — Sketch-to-Shape Generation

| Method | 3DVRChair (chair only) | | VRSketch2Shape (chair) | | VRSketch2Shape (all) | |
|---|---|---|---|---|---|---|
| | F-score ↑ | CD ↓ | F-score ↑ | CD ↓ | F-score ↑ | CD ↓ |
| LAS-Diffusion⋆ | 26.1 | 66.0 | 37.0 | 51.1 | 40.2 | 27.1 |
| Luo et al. | 26.6 | 35.5 | 42.2 | 13.4 | 48.8 | 13.0 |
| **VRSketch2Shape (ours)** | **31.1** | **25.8** | **64.3** | **4.0** | **69.8** | **4.8** |

VRSketch2Shape achieves substantial improvements across all settings:
- CD reduced by **27%** on 3DVRChair (25.8 vs. 35.5)
- F-score improved by **43%** and CD reduced by **63%** on the full VRSketch2Shape dataset (69.8 vs. 48.8; 4.8 vs. 13.0)

### Table A-1: DDIM Steps and Speed–Accuracy Trade-off

| DDIM Steps | F-score ↑ | CD×1000 ↓ | Inference Time (s/sample) |
|---|---|---|---|
| 10 | 69.24 | 5.04 | 2.26 |
| 25 | 69.70 | 4.82 | 3.06 |
| 50 | 69.74 | 4.89 | 4.47 |
| 100 | 69.80 | 4.78 | 6.33 |

As few as 10 DDIM steps achieves near-optimal performance at **3×** faster inference, making the approach suitable for interactive design scenarios.

### Key Findings from Ablation Study (Zero-Shot, Chair Subset)
- **Removing sequential information** (w/o ordering): Significant performance drop, confirming that "order indeed matters."
- **Removing data augmentation**: Noticeable degradation; simple augmentations effectively improve robustness.
- **Removing synthetic pretraining**: Model collapses to trivial solutions; 200 real sketches alone are far insufficient.
- **Replacing with SketchBERT encoder**: Large accuracy drop, demonstrating the importance of 3D-specific design choices.
- **Sketch as point cloud** (PointNet++): Notable degradation, confirming that sequence modeling—not merely the diffusion generator—drives the core improvement.
- **Sketch as image** (VGG + multi-view rendering): Occlusion causes geometric information loss, leading to significantly worse results.

### Few-Shot Synthetic-to-Real Transfer
Fine-tuning with only 50 real sketches per category approaches optimal performance. The model already performs strongly in the zero-shot (no fine-tuning) setting, validating the effectiveness of the synthetic sketch pipeline.

### Partial Sketch Shape Completion
Retaining only the first 50% of sketch points achieves performance close to that of full sketches, reflecting the human drawing habit of "outlining first, then adding details."

## Highlights & Insights

- **First temporal modeling of VR sketches**: Elevates sketches from unordered point clouds to ordered stroke sequences, demonstrating the critical role of stroke order in shape generation.
- **Training-free synthetic pipeline**: Purely geometry-driven heuristics automatically generate temporally ordered sketches—20k+ samples in 10 hours—effectively replacing costly manual annotation.
- **Strong generalization**: The model transfers effectively from synthetic training data to real sketches in both zero-shot and few-shot settings; it also produces reasonable outputs for non-snapped sketches, freehand sketches, and unseen categories.
- **Cross-modal shape completion**: By leveraging temporal drawing priors, the model infers complete 3D shapes from partial sketches, potentially accelerating interactive design workflows.
- **End-to-end single-stage training**: Unlike methods requiring multi-stage training or cross-modal alignment, the sketch encoder and diffusion model are jointly optimized.

## Limitations & Future Work

- **SDF resolution constraints**: The frozen 3D VQ-VAE ($64^3$ SDF resolution) limits fine-grained geometric detail reconstruction, and generated results are sometimes overly smooth.
- **Limited category diversity in training**: Training covers only 4 ShapeNet categories (chair, table, cabinet, airplane); the model may degrade to training-category shape priors on substantially different unseen categories (e.g., trucks, beds).
- **Inference latency**: 100-step DDIM requires ~6.6 seconds per sample, with 99% of time spent on latent denoising; reducing steps helps but real-time performance remains a bottleneck.
- **Dependence on surface snapping**: Real sketch collection relies on snapping tools to ensure geometric alignment; accuracy degrades somewhat in non-snapped settings.

## Related Work & Insights

- **2D sketch → 3D**: Methods learning deterministic mappings or diffusion models (Doodle Your 3D, LAS-Diffusion) are limited by single-view ambiguity and occlusion.
- **VR sketch → 3D**: Luo et al. and Chen et al. encode VR sketches as unordered point clouds using PointNet++, discarding stroke temporal order; VRSketch2Gaussian uses 3D Gaussians but similarly ignores ordering.
- **Sketch synthesis**: CLIPasso/DiffSketcher and similar methods optimize parameterized curves but require pretrained model guidance; the proposed purely geometry-driven heuristic pipeline requires no training.
- **Sketch encoding**: SketchBERT models sequences in 2D sketches, but direct extension to 3D yields poor results; the proposed Fourier feature + learnable separator design is better suited to 3D settings.
- **3D generation**: Latent diffusion models such as SDFusion perform well under text/image conditioning; this work is the first to extend such models to sequential VR sketch conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic introduction of stroke temporal order into VR sketch-to-3D shape generation; problem formulation is clear and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two datasets, multiple baselines, comprehensive ablations (design choices, sketch formats, DDIM steps), and evaluations covering zero-shot, few-shot, partial sketch, non-snapped, freehand, and unseen category settings.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, richly illustrated, rigorous method description, with thorough exposition of the synthetic pipeline and encoder design.
- Value: ⭐⭐⭐⭐ — Provides a complete data + model + pipeline solution for VR sketch-driven 3D design; open-sourced dataset and code can advance the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Condition Matters in Full-head 3D GANs](../../ICLR2026/others/condition_matters_in_full-head_3d_gans.md)
- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[AAAI 2026\] Higher-Order Responsibility](../../AAAI2026/others/higher-order_responsibility.md)

</div>

<!-- RELATED:END -->
