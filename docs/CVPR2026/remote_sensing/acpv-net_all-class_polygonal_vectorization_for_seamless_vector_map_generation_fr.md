---
title: >-
  [Paper Note] ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery
description: >-
  [CVPR 2026][Remote Sensing][Polygonal Vectorization] ACPV-Net is the first framework that generates topologically consistent all-class polygonal vector maps from aerial imagery in a single pass…
tags:
  - "CVPR 2026"
  - "Remote Sensing"
  - "Polygonal Vectorization"
  - "Vector Map Generation"
  - "Planar Partition"
  - "Conditional Diffusion"
  - "Topological Consistency"
  - "Aerial Imagery"
date: 2026-05-08
content_hash: 1a2c98645c7eb158
---

# ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery

**Conference**: CVPR 2026
**arXiv**: [2603.16616](https://arxiv.org/abs/2603.16616)
**Code**: [HeinzJiao/ACPV-Net](https://github.com/HeinzJiao/ACPV-Net)
**Area**: Remote Sensing
**Keywords**: Polygonal Vectorization, Vector Map Generation, Planar Partition, Conditional Diffusion, Topological Consistency, Aerial Imagery

## TL;DR

ACPV-Net is the first framework that generates topologically consistent all-class polygonal vector maps from aerial imagery in a single pass, employing a semantically supervised conditional diffusion model for vertex heatmap generation and proposition-driven PSLG reconstruction to ensure zero gaps and zero overlaps.

## Background & Motivation

**Importance of vector base maps**: Topographic vector maps form the core of national geospatial data infrastructure and are widely used in cadastral management and land-use planning, requiring adjacent polygons to share precise boundaries with zero gaps and zero overlaps.

**Fundamental flaws of existing methods**: Current polygonalization methods (DeepSnake, FFL, TopDiG, HiSup, GCP) are all single-class designs that require per-class inference followed by stitching, which inevitably introduces topological inconsistencies such as duplicate boundaries, gaps, and overlaps.

**Five key technical challenges**: (i) Semantic-geometric heterogeneity (raster discrete semantics vs. vector continuous geometry); (ii) strict alignment between semantic regions and geometric boundaries; (iii) weak/ambiguous visual cues (shadows, occlusions, ambiguous boundaries); (iv) cartographic conventions (vertex sampling density, simplification strategies) are hard to encode explicitly; (v) global topological reconstruction goes beyond single-class geometry.

**Lack of evaluation benchmarks**: Existing datasets either provide only single-class vector annotations (WHU-Building) or only multi-class raster masks (LoveDA, ISPRS), with no public benchmark supporting all-class polygonal vectorization with global topological consistency evaluation.

**Limitations of discriminative methods**: Existing discriminative vertex detection methods produce broad or band-like responses under weak visual cues and lack the ability to learn cartographic conventions.

**Insufficiency of conditional diffusion pipelines**: Existing conditional diffusion pipelines (e.g., ControlNet) inject external conditions but do not apply explicit semantic supervision to the conditioning branch, failing to guarantee semantic-geometric alignment.

## Method

### Overall Architecture

ACPV-Net consists of two tightly coupled components: (1) **Semantically Supervised Conditioning (SSC)** — a diffusion model reconstructs vertex Gaussian mixture heatmaps in latent space, with the conditioning branch explicitly supervised by a semantic segmentation loss to ensure semantically guided vertex generation; (2) **Proposition-driven topological reconstruction** — starting from the semantic mask $\hat{M}$ and vertex heatmap $\hat{Y}$, a PSLG algorithm deterministically reconstructs vector maps satisfying all ACPV constraints.

### Key Designs

1. **Distributional Vertex Modeling**

    - **Function**: Encodes polygon vertices from aerial images as Gaussian mixture heatmaps $y \in [0,1]^{H \times W}$, encoded into latent space $z_0 = \mathcal{E}(y)$ via a pretrained VAE, where a diffusion model performs denoising reconstruction.
    - **Mechanism**: Transforms the discrete point set problem into a continuous distribution generation problem. The probabilistic modeling capability of diffusion models enables inference of sharp, compact vertex peaks under weak visual cues and automatic learning of cartographic conventions (e.g., sampling density along smooth boundaries).
    - **Design Motivation**: Comparative experiments show that a purely discriminative decoder (ViTPose baseline) produces broad band-like responses in weak-cue regions, while diffusion-based reconstruction achieves smaller FWHM, lower Area@0.5, and higher Sharpness, validating the advantage of generative modeling.

2. **Semantically Supervised Conditioning (SSC)**

    - **Function**: The semantic encoder $S_\psi(I)$ provides conditioning features at the same scale as the latent space, with a lightweight segmentation head applying a semantic segmentation loss $\mathcal{L}_{\text{seg}}$ so that the conditioning branch itself learns discriminative semantics aligned with downstream vertex generation.
    - **Mechanism**: Unlike generic conditional diffusion pipelines that merely "inject" conditioning signals, SSC transforms the conditioning branch into an "active guiding signal," forcing vertex generation to fall on category-consistent boundaries.
    - **Design Motivation**: Ablation experiments (No-SSC) show that removing $\mathcal{L}_{\text{seg}}$ causes the vertex-boundary alignment rate V2B@2 to drop sharply from 0.78 to 0.38, with numerous false-positive vertices appearing inside homogeneous regions rather than on boundaries.

3. **Proposition-driven PSLG Reconstruction**

    - **Function**: Deterministically reconstructs a planar straight-line graph (PSLG) satisfying all ACPV constraints from $(\hat{M}, \hat{Y})$.
    - **Mechanism**: First proves the sufficient condition of Proposition 1 — if every edge in the PSLG lies on a label-transition boundary and the vertex set consists entirely of geometric key points, then the polygon partition reconstructed accordingly satisfies all ACPV constraints. Implementation proceeds in two steps: (1) **Over-dense PSLG construction** — extracts all label-transition pixels from the multi-class mask and connects them into edges, yielding a superset covering all valid boundary positions; (2) **Vertex-guided subset selection** — extracts discrete vertex peaks from the heatmap, projects them onto the PSLG, retains anchor and key points, and simplifies redundant vertices.
    - **Design Motivation**: Topological consistency is guaranteed by constructive proof rather than heuristic post-processing, fundamentally avoiding gaps and overlaps.

### Loss Function & Training Strategy

Unified loss function:

$$\mathcal{L}_{\text{SSC}} = \lambda_\epsilon \mathbb{E}\|\epsilon - \epsilon_\theta(\cdot)\|_1 + \lambda_0 \mathbb{E}\|z_0 - \hat{z}_0\|_1 + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}(\hat{M}, M)$$

The first two terms are the noise prediction L1 loss and the latent-space reconstruction L1 loss (standard diffusion objectives), and the third term is the semantic segmentation loss. All three are jointly trained end-to-end. The VAE encoder/decoder remains frozen and does not participate in training.

## Key Experimental Results

### Table 1: Global Topological Consistency on Deventer-512

| Method | Gap ↓ | Inter-Overlap ↓ | Intra-Overlap ↓ | Shared-Edge ↑ |
|:---|:---:|:---:|:---:|:---:|
| DeepSnake (CVPR'20) | 12.41 | 68.86 | 51.16 | 38.73 |
| FFL (CVPR'21) | 5.48 | 29.17 | 0.08 | 9.20 |
| TopDiG (CVPR'23) | 8.47 | 13.57 | 0.00 | 10.81 |
| HiSup (ISPRS'23) | 5.43 | 4.50 | 0.00 | 25.73 |
| GCP (TGRS'25) | 8.75 | 10.39 | 41.91 | 20.25 |
| **ACPV-Net (Ours)** | **0.00** | **0.00** | **0.00** | **100.00** |

ACPV-Net is the only method achieving zero gaps, zero overlaps, and 100% shared-edge consistency.

### Table 2: Core Category Comparison on Deventer-512 (Building / Road)

| Category | Method | IoU ↑ | C-IoU ↑ | PoLiS ↓ | MTA ↓ | N-ratio →1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Building | HiSup | 81.22 | 70.60 | 2.23 | 42.59 | 1.55 |
| Building | **Ours** | **82.08** | **77.24** | **1.76** | **39.39** | **1.00** |
| Road | HiSup | 73.92 | 57.36 | 4.97 | 44.84 | 2.00 |
| Road | **Ours** | **76.01** | **68.22** | **4.44** | **43.85** | **1.07** |

ACPV-Net comprehensively outperforms the single-class best baseline across all five categories (IoU, C-IoU, PoLiS, topological fidelity), with N-ratio close to 1.0, indicating extremely high vertex efficiency.

### Table 3: Single-Class Polygonalization on WHU-Building

| Method | IoU ↑ | C-IoU ↑ | PoLiS ↓ | MTA ↓ | N-ratio →1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| HiSup | 87.63 | 67.15 | 1.40 | 35.27 | 1.93 |
| **ACPV-Net** | **88.50** | **81.45** | **1.38** | **34.85** | **1.07** |

Without any architectural modification, ACPV-Net can be applied to single-class scenarios and achieves state-of-the-art results on WHU-Building, with C-IoU dramatically improving from 67.15 to 81.45.

## Highlights & Insights

1. **Pioneering task definition**: The first formal definition of the ACPV task with six strict constraints (a)–(f), theoretically characterizing the complete requirements for vector base map generation.
2. **Constructive topological guarantee**: Through the sufficient condition of Proposition 1 and a deterministic PSLG algorithm, topological consistency is guaranteed by design rather than relying on heuristic post-processing — an elegant theoretical contribution.
3. **Ingenious SSC mechanism design**: The conditioning branch is upgraded from passive injection to active semantic guidance, improving V2B alignment from 0.46 to 0.85 and effectively eliminating false-positive vertices in homogeneous regions.
4. **Versatility**: The same architecture handles both multi-class (Deventer-512) and single-class (WHU-Building) scenarios without modification, with good cross-region generalization.
5. **Benchmark contribution**: The first ACPV evaluation benchmark Deventer-512 is released, containing ~2k patches, 84k+ instances, and a unified evaluation protocol, filling a data gap in this field.

## Limitations & Future Work

1. **Limited dataset scale**: Deventer-512 contains only ~2k patches from a single source (Deventer, Netherlands) with only 5 land-cover categories, potentially limiting generalization to more complex scenes (e.g., high-density urban areas, industrial parks).
2. **Diffusion model inference efficiency**: Latent-space diffusion denoising requires iterative sampling, which may become an inference speed bottleneck in practical deployment; inference time is not reported in the paper.
3. **Fixed radius τ limitation**: Vertex projection onto the PSLG uses a fixed radius τ, which may cause vertex mismatching or omission in complex dense regions.
4. **Curved boundary representation**: All boundaries are approximated by piecewise linear segments, which may require many vertices for faithful representation of curved water bodies and natural boundaries; N-ratio performance on the water category is slightly weaker.
5. **Incremental updates unexplored**: Practical production workflows require incremental updates to existing vector maps rather than full-map reconstruction, which the current framework does not support.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First formal definition of the ACPV task; the SSC + proposition-driven reconstruction framework design is highly original
- **Experimental rigor**: ⭐⭐⭐⭐ — Comprehensive multi-class/single-class evaluation with thorough ablations, but the dataset scale is small and inference efficiency analysis is missing
- **Writing quality**: ⭐⭐⭐⭐⭐ — Rigorous problem definition, formal mathematical exposition, complete constructive proofs, and extremely clear writing
- **Impact**: ⭐⭐⭐⭐⭐ — Fills the gap in all-class topological consistency for remote sensing vectorization, with both theoretical and practical value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)
- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[CVPR 2026\] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_vision-language_instruction_from_offline_teacher_to_prompt-tuned_ne.md)
- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[NeurIPS 2025\] Mass Conservation on Rails – Rethinking Physics-Informed Learning of Ice Flow Vector Fields](../../NeurIPS2025/remote_sensing/mass_conservation_on_rails_--_rethinking_physics-informed_learning_of_ice_flow_v.md)

</div>

<!-- RELATED:END -->
