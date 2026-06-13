---
title: >-
  [Paper Note] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer
description: >-
  [NeurIPS 2025][Image Generation][3D appearance transfer] This paper proposes GuideFlow3D, a training-free 3D appearance transfer framework that alternately injects differentiable guidance losses (part-aware appearance lo…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "3D appearance transfer"
  - "rectified flow"
  - "universal guidance"
  - "structured latent"
  - "part-aware loss"
date: 2026-05-08
content_hash: c122010dd2e1cc66
---

# GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer

**Conference**: NeurIPS 2025
**arXiv**: [2510.16136](https://arxiv.org/abs/2510.16136)  
**Code**: [Project Page](https://sayands.github.io/guideflow3d)  
**Area**: Image Generation
**Keywords**: 3D appearance transfer, rectified flow, universal guidance, structured latent, part-aware loss

## TL;DR
This paper proposes GuideFlow3D, a training-free 3D appearance transfer framework that alternately injects differentiable guidance losses (part-aware appearance loss + self-similarity loss) into the sampling process of a pretrained rectified flow model, enabling robust texture and geometric detail transfer between objects with significant geometric discrepancies.

## Background & Motivation
Transferring appearance (texture + fine geometric details) from one 3D object to another has broad applications in gaming, AR, and digital content creation. Existing methods struggle when the input object and the appearance object differ significantly in geometry:

- **2D style transfer → 3D lifting**: Applies 2D style transfer on multi-view images before 3D reconstruction, but inter-view geometric inconsistencies introduce artifacts.
- **Direct application of 3D generative models**: Rectified flow models such as Trellis can generate high-quality 3D assets, but are constrained by training-time conditioning signals and data distributions, generalizing poorly to appearance transfer—especially under large geometric discrepancies.
- **ControlNet-based methods** (e.g., TEXTure, EasiTex): Rely on specific training setups and conditioning modalities, limiting generalizability.
- **Pure optimization methods**: Directly optimizing the latent space to match an appearance target deviates from the data distribution modeled by the generative network, producing unnatural results.

The core motivation is: can the inductive biases of a pretrained 3D generative model be leveraged to achieve flexible appearance transfer through inference-time guidance, without retraining?

## Core Problem
How to robustly transfer the texture and fine geometric details of an appearance object onto an input object while preserving its global geometric structure—particularly when the two objects differ substantially in geometry (e.g., chair→bed, giraffe→furniture)?

## Method

### Overall Architecture
GuideFlow3D builds upon the structured latent (SLat) representation and rectified flow generative model of Trellis, controlling the generation process at inference time by alternating between flow steps and guided optimization steps.

### 1. Structured Latent Representation
A 3D object $\mathcal{O}$ is encoded as a structured latent:

$$\mathbf{z} = \{(z_i, p_i)\}_{i=1}^{L}, \quad z_i \in \mathbb{R}^C, \quad p_i \in \{0, 1, \ldots, N-1\}^3$$

- $p_i$: active voxel positions (intersecting the object surface), delineating coarse structure
- $z_i$: latent vector for each voxel, capturing fine geometry and texture features
- Key design: **$p_i$ is fixed** (preserving global geometry); only the generation of $z_i$ is guided

### 2. Guidance Objective Functions

**(a) Part-aware appearance loss $\mathcal{L}_{\text{appearance}}$** (applicable when the appearance object has a mesh):

$$\mathcal{L}_{\text{appearance}} = \frac{1}{L_q} \sum_{i=1}^{L_q} \| \tilde{z}_i^q - z_m^a \|_2^2$$

- Geometric features from PartField are used for co-segmentation clustering, establishing part-level correspondences between the input and appearance objects (e.g., chair back↔chair back, chair leg↔chair leg)
- $m$ denotes the index of the matched voxel in the appearance object via part clustering
- Ensures localized texture and geometric correspondence

**(b) Self-similarity loss $\mathcal{L}_{\text{structure}}$** (applicable when the appearance object is only an image or text):

$$\mathcal{L}_{\text{structure}} = -\frac{1}{L_q} \sum_{i=1}^{L_q} \log \frac{\sum_{j \in \mathcal{C}_q(i), j \neq i} \exp(\text{sim}_{ij})}{\sum_{j \in \mathcal{C}'_q(i)} \exp(\text{sim}_{ij})}$$

- A geometry-clustering-based contrastive loss: voxel features within the same part are encouraged to be similar (positives), while those from different parts are encouraged to differ (negatives)
- Promotes local consistency without enforcing global homogenization

### 3. Guided Rectified Flow Sampling
The reverse process of standard rectified flow is:

$$\hat{\mathbf{z}}_t = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{c})$$

GuideFlow3D injects guidance gradients at each step:

$$\hat{\mathbf{z}}_t = \mathbf{z}_t + \Delta t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t \mid \mathbf{c}) + \nabla_{\mathbf{z}_t} \mathcal{L}(\mathbf{z} \mid \mathbf{c})$$

- The condition $\mathbf{c}$ can be an image or text
- From a Bayesian perspective, the rectified flow models the prior $P(\mathcal{O})$ and likelihood $P(\mathbf{c}|\mathcal{O})$, while the guidance term models additional constraints
- This extends universal guidance from diffusion models to arbitrary rectified flow models

### 4. Conditioning Flexibility
- **Image + Mesh condition**: Uses $\mathcal{L}_{\text{appearance}}$ to transfer texture and geometric details
- **Image-only condition**: Uses $\mathcal{L}_{\text{structure}}$ (Trellis can first generate a mesh from the image)
- **Text condition**: Uses $\mathcal{L}_{\text{structure}}$ to transfer texture only

## Loss & Training

### Training-Free
GuideFlow3D is a fully **training-free** framework requiring no fine-tuning or retraining of the generative model. All appearance transfer control is injected at inference time through guidance.

### Implementation Details
- **Base model**: Pretrained Trellis models (`trellis-image-large` for image conditioning, `trellis-text-large` for text conditioning), using their default configurations
- **Part features**: PartField is used to compute a part feature field for each mesh; voxel coordinates $p_i$ are used to query per-voxel part features
- **Sampling steps**: Rectified flow sampling and single-instance optimization are alternated for a total of 300 steps
- **Optimizer**: AdamW, learning rate $5 \times 10^{-4}$
- **Hardware**: Single NVIDIA RTX 4090 GPU
- **Runtime**: 96 seconds (baseline Trellis: 78 seconds; ~23% overhead)
- All conditioning types (image/text) use the same optimization settings

### Evaluation Rendering
- All assets are rendered in Blender with smooth area lighting
- Each object is rendered from 4 viewpoints (fixed radius 2, pitch 30°, yaw starting at 45° with 90° increments)
- All meshes are placed in canonical pose to ensure alignment
- Metrics are computed per viewpoint and per object, then averaged

## Key Experimental Results

### Dataset
- Input meshes: Procedurally generated simple geometries (simple)
- Appearance objects: ABO dataset (complex), ~8K 3D models across 55 categories
- 4 experimental settings: simple-complex intra-/inter-category, complex-complex intra-/inter-category
- 250 input–appearance pairs per setting

### Evaluation Protocol
Traditional encoder-based metrics (PSNR, SSIM, LPIPS, FID, etc.) require ground truth and cannot handle dissimilar geometries. Therefore, a **GPT-based ranking system** is adopted, evaluating six dimensions: Style Fidelity, Structure Clarity, Style Integration, Detail Quality, Shape Adaptation, and Overall Quality (lower rank is better). A user study confirms high agreement between GPT rankings and human preferences.

### Main Results (simple-complex intra-category, image conditioning)

| Method | Fidelity↓ | Clarity↓ | Overall↓ |
|--------|-----------|----------|----------|
| UV Nearest Neighbor | 4.12 | 3.84 | 4.33 |
| MambaST | 4.94 | 3.55 | 4.87 |
| Cross Image Attention | 3.56 | 3.48 | 3.59 |
| EasiTex | 3.18 | 4.30 | 3.81 |
| Trellis | 2.51 | 2.58 | 2.62 |
| **GuideFlow3D (Ours)** | **1.89** | **2.41** | **2.12** |

### Text Conditioning Results (simple-complex intra-category)

| Method | Fidelity↓ | Clarity↓ | Overall↓ |
|--------|-----------|----------|----------|
| Trellis | 2.01 | 1.89 | 2.39 |
| **GuideFlow3D (Ours)** | **1.54** | **1.63** | **1.95** |

- GuideFlow3D achieves the best rankings across all settings (intra-/inter-category, simple/complex) and both conditioning modalities
- In-the-wild experiments demonstrate robust transfer across semantic categories (animals→furniture, furniture→vehicles, etc.)

### Runtime
- GuideFlow3D: 96 seconds (NVIDIA RTX 4090 GPU)
- Trellis baseline: 78 seconds
- ~23% additional overhead for substantial quality improvement

### Ablation Study
Ablations are conducted on the simple-complex intra-category image-conditioning setting:

| Variant | Fidelity↓ | Clarity↓ | Overall↓ |
|---------|-----------|----------|----------|
| (i) w/o flow + global feat. | 4.52 | 4.51 | 4.50 |
| (ii) w/o flow + SLat spatial NN matching | 3.58 | 3.62 | 3.63 |
| (iii) w/ flow + K-means on SLat (no PartField) | 2.57 | 2.65 | 2.66 |
| (iv) w/ flow + $\mathcal{L}_{\text{structure}}$ (image cond.) | 2.17 | 2.05 | 2.03 |
| (v) w/ flow + $\mathcal{L}_{\text{appearance}}$ (image cond.) | **1.23** | **1.08** | **1.06** |

Key findings:
1. **Global features are insufficient**: Global latents via min/max/avg pooling fail to capture semantic correspondences
2. **Unstructured NN matching is insufficient**: Nearest-neighbor matching directly in SLat space improves fidelity but lacks robust semantic alignment
3. **PartField vs. K-means on SLat**: Semantically aware PartField segmentation significantly outperforms K-means on SLat features, demonstrating that part-aware semantic information is critical for accurate part correspondence
4. **Two losses are complementary**: $\mathcal{L}_{\text{appearance}}$ yields stronger fidelity, while $\mathcal{L}_{\text{structure}}$ provides better alignment and adaptability

### Scene Editing Application
- Scene-level editing capability is validated on ScanNet indoor scenes
- Per-object CAD mesh annotations are used to select appearance objects for each semantic category in the scene
- Multiple objects in the scene can be selectively re-stylized while preserving the spatial layout
- Demonstrates potential for interactive scene customization

### Limitations of Traditional Metrics
- DINOv2, CLIP Score, DreamSim, etc. require ground truth or assume geometric similarity
- These metrics cannot reflect true transfer quality when input and appearance objects differ substantially in geometry
- For instance, CLIP Score assigns higher scores to the Trellis baseline under text conditioning, because text typically describes shapes different from the input geometry

## Highlights & Insights
1. **Training-free**: Appearance transfer is achieved entirely through inference-time guidance injection, without modifying generative model parameters
2. **Geometric robustness**: Global geometry is preserved by fixing voxel positions $p_i$; part-aware losses handle large geometric discrepancies
3. **Unified multi-modal framework**: The same framework supports three appearance representations—mesh, image, and text
4. **Principled formulation**: Universal guidance is extended to rectified flow via Bayesian formulation, yielding a theoretically grounded framework
5. **Generality and scalability**: The method generalizes to different diffusion/flow models and guidance functions
6. **Evaluation innovation**: A GPT-based multi-dimensional ranking evaluation system is proposed and validated against human judgments via user study

## Limitations & Future Work
1. **Not real-time**: The optimization-based approach (96-second inference) is unsuitable for real-time applications; future work could train a self-supervised feed-forward model for acceleration
2. **Dependency on external models**: Relies on Trellis (SLat encoding/decoding) and PartField (part features); failures in these models cascade to the final output
3. **Requires clean meshes**: Assumes noise-free input meshes, limiting applicability to noisy inputs such as scanned data
4. **Limited scope of main experiments**: Main experiments focus on the furniture category (ABO dataset); although in-the-wild results suggest broader generalization, systematic evaluation is lacking
5. **Absence of traditional metric comparisons**: Complete reliance on GPT-based evaluation may overlook certain objective quality differences

## Related Work & Insights

| Method | Training Required | Geometric Robustness | Multi-modal Support | Part-aware | Output Representation |
|--------|------------------|---------------------|--------------------|-----------|-----------------------|
| StyleGaussian | Yes | Weak | Style only | No | Rendering only |
| TEXTure | SDS distillation | Moderate | Text | No | Texture |
| EasiTex | ControlNet | Weak (large geometric deviation) | Image | No | Texture |
| Trellis | No additional training | Weak | Image/Text | No | Mesh/3DGS/NeRF |
| Cross Image Attention | No | Weak (2D→3D artifacts) | Image | No | Depends on lifting method |
| **GuideFlow3D** | **No** | **Strong** | **Mesh+Image+Text** | **Yes** | **Mesh/3DGS/NeRF** |

**Additional Insights:**
- **3D extension of universal guidance**: The idea of universal guidance from Bansal et al. for 2D diffusion is extended to 3D rectified flow models, opening a new direction for controllability in 3D generation—any differentiable objective can be injected at inference time
- **Part-aware correspondence via PartField**: Using PartField's geometric co-segmentation to establish inter-object part correspondences is an elegant solution to the correspondence problem under large geometric discrepancies
- **Position–feature decoupling in structured latent space**: The design of fixing $p_i$ while optimizing $z_i$ elegantly achieves "preserve geometry, modify appearance," a principle transferable to other 3D editing tasks
- **GPT-as-evaluator paradigm**: In generative task evaluation without ground truth, GPT ranking combined with user study validation is a methodology worth broader adoption

## Rating
- Novelty: ⭐⭐⭐⭐ (Extending universal guidance to 3D rectified flow is a novel idea; the part-aware loss design is creative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple settings, multiple baselines, in-the-wild demonstrations, user study, and ablation study are all covered)
- Writing Quality: ⭐⭐⭐⭐ (Clear mathematical derivations, rich illustrations, and well-motivated method design)
- Value: ⭐⭐⭐⭐ (A training-free 3D appearance transfer framework with strong practicality and scalability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](rectified-cfg_for_flow_based_models.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](../../ICML2026/image_generation/offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)

</div>

<!-- RELATED:END -->
