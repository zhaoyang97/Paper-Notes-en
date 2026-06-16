---
title: >-
  [Paper Note] From One to More: Contextual Part Latents for 3D Generation
description: >-
  [ICCV 2025][3D Vision][Part-level 3D generation] This paper proposes CoPart, a framework that represents 3D objects via contextual part latents and fine-tunes pretrained diffusion models with a mutual guidance strategy…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Part-level 3D generation"
  - "diffusion models"
  - "mutual guidance"
  - "3D VAE"
  - "part editing"
date: 2026-05-08
content_hash: 4e900e8db6031e3f
---

# From One to More: Contextual Part Latents for 3D Generation

**Conference**: ICCV 2025
**arXiv**: [2507.08772](https://arxiv.org/abs/2507.08772)  
**Code**: [Project Page](https://copart3d.github.io)  
**Area**: 3D Vision
**Keywords**: Part-level 3D generation, diffusion models, mutual guidance, 3D VAE, part editing

## TL;DR

This paper proposes CoPart, a framework that represents 3D objects via contextual part latents and fine-tunes pretrained diffusion models with a mutual guidance strategy, enabling high-quality part-level 3D generation along with support for part editing, articulated object generation, and small-scale scene generation.

## Background & Motivation

The 3D-native latent diffusion paradigm faces three core challenges:

**Single-latent representation**: Most methods encode an entire 3D object into a single latent code, ignoring the multi-part nature of complex objects and resulting in loss of fine-grained detail.

**Neglect of part independence**: 3D artists typically create objects part by part, yet holistic representations fail to capture inter-part relationships.

**Insufficient global control**: Existing methods rely on global conditioning (text/image) and lack fine-grained local controllability.

CoPart adopts a **bottom-up** strategy: directly learning the distribution of parts and jointly generating coherent parts, rather than generating a whole object and then segmenting it.

## Method

### Part Representation Encoding

Each 3D part is represented by a **hybrid part latent**:
- **Geometry tokens** $\mathbf{L}_{3D} = \mathcal{E}_{3D}(P, Q) \in \mathbb{R}^{T \times D}$: encoded by a 3D part VAE from surface-sampled points and normals.
- **Image tokens** $\mathbf{L}_{2D} = \mathcal{E}_{2D}(O_k) \in \mathbb{R}^{T \times D}$: encoded by an image VAE from multi-view renderings of each part.

### Mutual Guidance

Synchronizes the diffusion processes across different parts and different modalities:

**Cross-part synchronization** (Cross-Part Attention):

$$\mathcal{G}^{p'} = \text{Attention}(\mathcal{G}^p, \{\mathcal{G}^i\}_{i=1}^N)$$

**Cross-modality synchronization** (Cross-Modality Attention):

$$\mathcal{G}^{p'} = \mathcal{G}^p + \text{LN}(\text{Attention}(\mathcal{G}^p, \{\mathcal{F}_k^p\}_{k=1}^v))$$

$$\mathcal{F}_k^{p'} = \mathcal{F}_k^p + \text{LN}(\text{Attention}(\mathcal{F}_k^p, \mathcal{G}^p))$$

where LN denotes a zero-initialized linear layer to ensure training stability.

### 3D Bounding Box Condition Encoding

Each bounding box is treated as a hexahedral mesh and encoded into the geometry latent space via the pretrained 3D VAE:

$$\mathbf{L}_{box}^p = \mathcal{E}_{3D}(\mathbf{P}_{box}^p, \mathbf{Q}_{box}^p)$$

The encoded box is injected into the 3D denoiser via cross-attention, and simultaneously rendered as a 2D wireframe image and injected into the 2D denoiser via ControlNet.

### Loss & Training

Standard denoising loss:

$$Loss_{3D} = \frac{1}{N} \sum_{p=1}^N \mathbb{E} \|\epsilon_{3d}^p - \mathcal{N}_{3d}(\mathbf{L}_{3D}^{p,t}, \mathbf{L}_{2D}^{p,t}, t)\|_2^2$$

## Key Experimental Results

### Main Results

Comparison with state-of-the-art 3D generators:

| Method | CLIP(N-T) | CLIP(I-T) | ULIP-T | Part-CLIP(N-T) | Time |
|--------|-----------|-----------|--------|----------------|------|
| Shap-E | 0.155 | 0.161 | 0.105 | 0.088 | 3s |
| Trellis | 0.207 | 0.236 | 0.175 | 0.127 | 10s |
| Rodin | 0.204 | 0.242 | 0.179 | 0.143 | - |
| **CoPart** | 0.201 | **0.239** | 0.174 | **0.161** | 65s |

CoPart achieves substantial gains on part-aware metrics while remaining competitive with state-of-the-art methods on overall quality.

### User Study

| Method | Overall Preference | Part Preference |
|--------|--------------------|-----------------|
| Rodin | 33.3% | 25.5% |
| PartGen | 11.8% | 13.7% |
| **CoPart** | **54.9%** | **60.8%** |

Users strongly prefer CoPart's generated results, with a particularly pronounced advantage in part quality.

## Highlights & Insights

1. **Mutual guidance mechanism**: Achieves inter-part and cross-modality information exchange via attention, which is more efficient than explicit 2D–3D projection.
2. **Bounding box encoding innovation**: Treats bounding boxes as meshes and encodes them with the 3D VAE, naturally mapping spatial layout into the geometry latent space.
3. **PartVerse dataset**: 91k parts / 12k objects / 175 categories, addressing the gap in large-scale 3D part data.
4. **Multi-application support**: Part editing, articulated object generation, and small-scale scene generation are all supported without additional training.

## Limitations & Future Work

- GPU memory constraints limit the maximum number of parts to $N=8$.
- Part ordering ambiguity is alleviated by bounding box conditioning but not fully resolved.
- A generation time of 65s is significantly slower than holistic methods.
- The PartVerse dataset construction pipeline (automated segmentation + manual post-processing) incurs substantial annotation cost.

## Related Work & Insights

- CLAY, CraftMan: 3D-native diffusion.
- SALAD, DiffFacto: Part-level generation (limited to PartNet categories).
- PartGen, Part123: Top-down part generation approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (part-level 3D diffusion + mutual guidance)
- Technical Depth: ⭐⭐⭐⭐⭐ (elegant and complete framework design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (quantitative evaluation + user study + ablation)
- Value: ⭐⭐⭐⭐⭐ (part editing addresses a genuine practical need)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)
- [\[ICCV 2025\] BokehDiff: Neural Lens Blur with One-Step Diffusion](bokehdiff_neural_lens_blur_with_one-step_diffusion.md)
- [\[CVPR 2026\] Learning Hierarchical Hyperbolic Mixture Model for Part-aware 3D Generation](../../CVPR2026/3d_vision/learning_hierarchical_hyperbolic_mixture_model_for_part-aware_3d_generation.md)
- [\[CVPR 2026\] More Natural, More Real: Object-aware Gaussian Splatting for 3D Visual Decoding from Human Brain](../../CVPR2026/3d_vision/more_natural_more_real_object-aware_gaussian_splatting_for_3d_visual_decoding_fr.md)

</div>

<!-- RELATED:END -->
