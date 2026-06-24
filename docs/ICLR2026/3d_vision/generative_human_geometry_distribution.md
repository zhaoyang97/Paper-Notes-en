---
title: >-
  [Paper Note] Generative Human Geometry Distribution
description: >-
  [ICLR 2026][3D Vision][Geometry Distribution] The authors upgrade "Geometry Distribution" from representing a single object to a "generative model scalable to datasets." By replacing network weights with 2D feature maps and using the SMPL template as the source distribution for Flow Matching instead of a Gaussian, this work enables large-scale 3D human generation with geometry distributions for the first time, achieving a 57% improvement in geometry quality over the SOTA.
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Geometry Distribution"
  - "Flow Matching"
  - "SMPL"
  - "3D Human Generation"
  - "Clothing Details"
  - "Feature Map Representation"
date: 2026-05-08
content_hash: 93202bdbd15caa55
---

# Generative Human Geometry Distribution

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=YsQM7sQl0j](https://openreview.net/forum?id=YsQM7sQl0j)  
**Code**: TBD  
**Area**: 3D Vision / 3D Human Generation  
**Keywords**: Geometry Distribution, Flow Matching, SMPL, 3D Human Generation, Clothing Details, Feature Map Representation  

## TL;DR
The authors upgrade "Geometry Distribution" from representing a single object to a "generative model scalable to datasets." By replacing network weights with 2D feature maps and using the SMPL template as the source distribution for Flow Matching instead of a Gaussian, this work enables large-scale 3D human generation with geometry distributions for the first time, achieving a 57% improvement in geometry quality over the SOTA.

## Background & Motivation
- **Background**: 3D human geometry generation must simultaneously capture high-frequency clothing wrinkles and accurately model pose-dependent "cloth-body" interactions. Existing representations have limitations: NeRF-based methods focus on rendering with coarse geometry limited by resolution/speed; Signed Distance Functions (SDF) struggle with thin structures and oversmoothing; Point clouds and voxels compromise between memory and quality; Tri-planes are limited by resolution in capturing details.
- **Limitations of Prior Work**: The recently proposed **Geometry Distributions** (Zhang et al. 2025) models a single 3D shape as a "probability distribution of surface points," mapping from a Gaussian sample to target geometry via a flow network. This allows infinite point sampling and high fidelity. However, it faces two fatal issues: ① Geometry information is **stored in the flow network weights**, requiring one network per shape, which causes memory explosion and prevents scaling to a generative model. ② Learning a velocity field from a Gaussian to a single shape is feasible, but doing so for thousands of shapes in a dataset is extremely inefficient.
- **Key Challenge**: Single geometry distributions are high-fidelity but **not scalable or generative**. To enable generation, one must model a "distribution-of-distributions," but existing works can only handle coarse shapes. Modeling the distribution of high-fidelity geometry distributions remains an open problem.
- **Goal**: Construct the first generative method for geometry distributions to model human geometry at scale, supporting tasks like pose-conditional random generation and novel pose synthesis for a given avatar.
- **Key Insight**: **Use 2D feature maps instead of network weights to encode each geometry distribution** (making the representation compressible and scalable) and **use the SMPL template distribution instead of Gaussian as the source distribution for Flow Matching** (making the source closer to the target to shorten the flow path). This is wrapped in a two-stage generative framework (compress into latent feature maps, then train a generative model on the latents).

## Method

### Overall Architecture
The method follows a two-stage approach, analogous to the "compress then generate" paradigm in modern image/3D generation. **Stage 1 (Conditional Distribution Encoding)**: An auto-decoder compresses each human geometry distribution into a compact 2D feature map $z_{T|S}\in\mathbb{R}^{C\times H\times W}$, from which high-fidelity geometry can be sampled via a denoising process. **Stage 2 (Geometry Generation)**: A flow/U-Net model is trained on the latent space of these feature maps, guided by SMPL vertex maps (with optional image/text conditions) to generate new feature maps (i.e., new human geometry distributions).

```mermaid
flowchart LR
    A[SMPL Template Distribution ΦS] --> B[Construct Training Pairs x0'-x1 + Distance Normalization]
    B --> C[Auto-Decoder Encoding<br/>Geometry → 2D Feature Map z_T|S]
    C --> D[Denoising Network uθ<br/>Condition: x0', Dec(z)(x0')]
    D --> E[High-Fidelity Point Cloud Geometry]
    C --> F[Stage 2 U-Net<br/>Generate Feature Maps on Latent]
    G[SMPL Vertex Map / Image / Text Condition] --> F
    F --> C
```

### Key Designs

**1. SMPL as Source Distribution + Pair Construction: Short-path flow from "Template → Geometry".** Original geometry distributions learn a velocity field starting from a Gaussian $\mathcal{N}(0,1)$, which involves a long path and slow convergence. The insight here is to use the **SMPL template shape distribution $\Phi_S$** as the source, making it naturally closer to the target geometry $\Phi_T$. The optimization objective becomes $\arg\min_\theta \mathbb{E}_{x_0\sim\Phi_S,x_1\sim\Phi_T}\|u_\theta(x_t,t)-(x_1-x_0)\|$, where $x_t=(1-t)x_0+x_1$. Since Flow Matching approximates conditional optimal transport, the authors **explicitly construct short-range training pairs**: sparse points $\{x_0\}_S$ are sampled on the SMPL template and points $\{x_1\}_T$ on the target geometry. For each $x_1$, the nearest SMPL point $x_0'=\arg\min_{x_0}\|x_1-x_0\|_2$ is paired to avoid learning irrelevant long-distance paths. To prevent holes in loose clothing areas where multiple $x_1$ share the same $x_0'$, Gaussian noise $\mathcal{N}(0,\sigma)$ is added to $x_0'$ to inject stochasticity and diversity.

**2. Distribution Normalization to Dense Displacement Fields: Eliminating spatial sampling imbalance.** Directly learning the mapping from SMPL to human geometry results in imbalanced spatial supervision because points only lie on the surface. Changes in pose/shape make sampling unstable in certain regions. The authors **subtract $x_0'$ from both source and target**: the source becomes a zero-centered Gaussian $\mathcal{N}(0,1)$ (with $\sigma=1$), and the target becomes a **normalized dense displacement field** $\Delta x = x_1 - x_0'$. While this subtraction removes the absolute position $x_0'$, it is re-injected as a conditioning signal to scale hidden features. Building the flow in a "normalized dense space" significantly improves training efficiency.

**3. Auto-Decoder + UV Feature Map Encoding: Learning latents to align with human priors.** Instead of an auto-encoder, an **auto-decoder** is used: the geometry of each sample is directly encoded into a learnable 2D feature map $z_{T|S}\in\mathbb{R}^{C\times H\times W}$. A **UNet-style decoder $\mathrm{Dec}_\phi$** upsamples this to a higher resolution, concatenated with SMPL vertex positions rendered as UV maps. Bilinear sampling is used at the UV coordinates of $x_0'$ on the high-resolution map to obtain the per-point latent $\mathrm{Dec}_\phi(z_{T|S})(x_0')$ as a condition for the denoising network. The denoising network $u_\theta$ also takes $x_0'$ concatenated with normals and canonical coordinates to provide clothing orientation and body part semantics.

**4. Two-stage Generation Framework.** Once all latents $\{z_{T|S}\}$ are learned, a generative model (U-Net) is trained in this latent space. For **pose-conditional generation**, SMPL vertex positions are rendered as UV maps and injected into the U-Net. For **novel pose synthesis**, an additional frontal normal map is used to indicate avatar identity, with features extracted via DINO-ViT and integrated via cross-attention. Points are synthesized directly on the deformed human body, allowing for pose-dependent wrinkles rather than static details from "canonical generation + skinning."

## Key Experimental Results

### Main Results (Pose-conditional FID, THuman2)

| Method | Raw Geometry FID ↓ | Enhanced Rendering FID ↓ |
|------|--------------------|--------------------------|
| ENARF* | 223.72 | 223.72 |
| GNARF* | 166.62 | 166.62 |
| EVA3D* | 60.37 | 60.37 |
| E3Gen | 65.32 | 28.12 |
| GetAvatar | 56.07 | 22.77 |
| gDNA | 42.90 | 17.43 |
| **Ours** | **16.16** | **16.16** |

The Raw Geometry FID improves by **57%** (42.9 → 16.2) compared to the SOTA (gDNA). Ours' raw geometry even outperforms others' results after "enhanced rendering" by **7%** (17.4 → 16.2).

### Ablation Study

Geometry distribution formula comparison (Chamfer Distance ↓):

| Setting | Single | Dataset |
|------|--------|---------|
| Zhang et al. (Gaussian Source) | 0.0083 | 0.0101 |
| w/o Pairs (Naive Eq.2) | 0.0040 | 0.0706 |
| w/o DistNorm ($\mathcal{N}(x_0',\sigma)$ w/o norm) | 0.0020 | 0.0071 |
| **Ours** | 0.0032 | **0.0032** |

Architecture comparison (Surface Distance ↓):

| Model | Surface Distance ↓ |
|------|--------------------|
| VecSet (auto-encoder) | 0.0018 |
| FeatureMap | 0.0014 |
| **Ours (auto-decoder)** | **0.0012** |

### Key Findings
- **w/o DistNorm works for single shapes but fails on datasets**: While decentralized Gaussian centers help focus on local details for a fixed pose, they severely hinder convergence on multi-pose datasets. Distribution normalization is key to "scalability."
- **Sparse sampling of training pairs is crucial**: Directly finding the nearest point on a dense SMPL mesh causes holes (undersampling) in normal maps; sparse sampling followed by nearest-neighbor search distributes the mapping load better.
- **Auto-decoder > Auto-encoder**: Converting independent embeddings into feature maps improves reconstruction, but the learnable latents of an auto-decoder provide the highest precision.
- **Pose Awareness**: Synthesizing points directly on the deformed body allows the model to generate pose-dependent wrinkles. It remains robust even when provided with a feature map that does not match the target pose.

## Highlights & Insights
- **High-fidelity "distribution-of-distributions"**: This work successfully upgrades single-shape geometry distributions to a generative framework, surpassing previous limitations that restricted such methods to coarse shapes.
- **Source distribution replacement**: Changing the Flow Matching source from uninformative Gaussian to a prior-rich SMPL template transforms a long-range transmission problem into a short-range displacement field regression, benefiting both efficiency and quality.
- **Elegant migration of representation**: Moving geometry from "network weights" to "2D feature maps" solves memory and scalability issues while aligning with mature latent paradigms and conditioning mechanisms in image/3D generation.
- **Optimization of geometry over rendering**: By focusing on the geometry itself rather than relying on rendering tricks, the raw geometry quality significantly exceeds previous benchmarks.

## Limitations & Future Work
- **Non-uniform surface sampling**: The number of target points $x_1$ associated with each SMPL point $x_0'$ varies. Although mitigated by oversampling, better training pair strategies are needed.
- **Constraint of training diversity**: While generalizing well to body shapes, the model **cannot generate clothing styles entirely absent from the training set**.
- **UV Seam Artifacts**: Discontinuous UV segments can cause seams in random generation; using UV partitions aligned with real garment patterns could be a solution.

## Related Work & Insights
- **Origins**: Geometry Distributions (Zhang et al. 2025) is the direct predecessor; this work generalizes it from "single shape, stored in weights" to "datasets, stored in feature maps."
- **2D/UV Representation**: Aligns with works using UV maps for 3D objects (e.g., Yan et al. 2024), which are more memory-efficient than tri-planes and better suited for human priors.
- **Human Reconstruction/Generation**: Unlike ICON/ECON (reconstruction from normals), E3Gen (Gaussian Splatting), or gDNA (implicit functions), this method learns directly from 3D data and supports infinite sampling without relying on rendering.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to elevate geometry distributions to a generative framework. The combination of "distribution-of-distributions + SMPL source + feature map carrier" is novel and solves a real problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid evaluation across two tasks, multiple baselines, and comprehensive ablations. Could be improved with more diverse datasets or extensive quantitative task scaling.
- **Writing Quality**: ⭐⭐⭐⭐ Logical progression of motivation, clear formulas, and good visualizations.
- **Value**: ⭐⭐⭐⭐ The 57% geometry improvement and the "geometry-first" approach are highly valuable for 3D digital humans and virtual try-ons.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Human Geometry Distribution for 3D Animation Generation](../../CVPR2026/3d_vision/human_geometry_distribution_for_3d_animation_generation.md)
- [\[ICLR 2026\] G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](g4splat_geometry-guided_gaussian_splatting_with_generative_prior.md)
- [\[ICLR 2026\] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows](h2oflow_grounding_human-object_affordances_with_3d_generative_models_and_dense_d.md)
- [\[ICLR 2026\] GOOD: Geometry-guided Out-of-Distribution Modeling for Open-set Test-time Adaptation in Point Cloud Semantic Segmentation](good_geometry-guided_out-of-distribution_modeling_for_open-set_test-time_adaptat.md)
- [\[ICLR 2026\] HoloPart: Generative 3D Part Amodal Segmentation](holopart_generative_3d_part_amodal_segmentation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[CVPR 2026\] Human Geometry Distribution for 3D Animation Generation](../../CVPR2026/3d_vision/human_geometry_distribution_for_3d_animation_generation.md)
- [\[ICLR 2026\] G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior](g4splat_geometry-guided_gaussian_splatting_with_generative_prior.md)
- [\[ICLR 2026\] GOOD: Geometry-guided Out-of-Distribution Modeling for Open-set Test-time Adaptation in Point Cloud Semantic Segmentation](good_geometry-guided_out-of-distribution_modeling_for_open-set_test-time_adaptat.md)
- [\[ICLR 2026\] H2OFlow: Grounding Human-Object Affordances with 3D Generative Models and Dense Diffused Flows](h2oflow_grounding_human-object_affordances_with_3d_generative_models_and_dense_d.md)
- [\[ICLR 2026\] HoloPart: Generative 3D Part Amodal Segmentation](holopart_generative_3d_part_amodal_segmentation.md)

</div>

<!-- RELATED:END -->
