---
title: >-
  [Paper Note] PhysX-3D: Physical-Grounded 3D Asset Generation
description: >-
  [NeurIPS 2025][3D Vision][Physical-property 3D generation] PhysX proposes the first end-to-end physical-property-driven 3D asset generation paradigm…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Physical-property 3D generation"
  - "3D dataset"
  - "physical annotation"
  - "articulated object modeling"
  - "embodied AI"
date: 2026-05-08
content_hash: a302c56389f22ef1
---

# PhysX-3D: Physical-Grounded 3D Asset Generation

**Conference**: NeurIPS 2025
**arXiv**: [2507.12465](https://arxiv.org/abs/2507.12465)
**Code**: [Project Page](https://physx-3d.github.io/)
**Area**: 3D Vision
**Keywords**: Physical-property 3D generation, 3D dataset, physical annotation, articulated object modeling, embodied AI

## TL;DR

PhysX proposes the first end-to-end physical-property-driven 3D asset generation paradigm, comprising PhysXNet (the first 3D dataset with systematic annotations across five physical dimensions—absolute scale, material, functional affordance, kinematics, and functional description—covering 26K+ objects) and PhysXGen (a dual-branch feed-forward generation framework that injects physical knowledge into a pretrained 3D structural latent space).

## Background & Motivation

Recent years have witnessed remarkable progress in 3D asset generation, yet existing methods **focus almost exclusively on geometry and texture, neglecting physical properties**. Real-world objects, however, inherently possess rich physical and semantic characteristics:

**Absolute Scale**: the true physical dimensions of an object

**Material**: material name, Young's modulus, Poisson's ratio, density

**Functional Affordance**: per-part priority for touching/grasping interactions

**Kinematics**: joint type, range of motion, motion direction, parent–child part relationships

**Functional Description**: basic, functional, and motion descriptive text

Coverage across existing datasets is highly fragmented (see Table 1 for comparison):
- PartNet-Mobility contains only 2.7K objects with kinematic annotations alone
- ABO provides material and scale annotations but only at the object level (not part level)
- Objaverse is large-scale but contains no physical annotations whatsoever

This absence of physical properties severely hinders the practical deployment of 3D assets in simulation, robotics, and embodied AI. The core motivation of PhysX is to **establish a complete physical 3D asset pipeline spanning upstream data annotation to downstream generative modeling**.

## Method

### Overall Architecture

PhysX consists of two core components:
- **PhysXNet dataset**: 26K+ physical 3D objects, plus PhysXNet-XL with 6 million procedurally augmented objects
- **PhysXGen generative model**: a dual-branch feed-forward framework built upon the pretrained TRELLIS 3D generative model

### Key Designs

1. **Human-in-the-Loop Annotation Pipeline**: organized into two stages:

    - **Initial Data Acquisition**: Each part is first rendered via alpha compositing (the target part in red, all others in gray) to maximize visual clarity and minimize occlusion interference. GPT-4o then performs automatic annotation to obtain basic physical attributes (material, density, part name, affordance, functional description). Human annotators review and correct the VLM outputs.
    - **Kinematic Parameter Determination**: For all parts with constrained motion (excluding free or rigid connections), the contact region between each child–parent mesh pair is computed. Plane fitting is applied to obtain candidate motion-axis directions, and candidate positions are generated (for revolute joints, K-means is additionally used to determine the axis position). Human annotators then select the best candidate and finalize the kinematic parameters.

   During preprocessing, overly fine-grained parts in PartNet are merged (area $\leq 0.2$ or face count $\leq 100$ with area $\leq 0.06$), and merging results are manually verified. Kinematic types include five categories: A (free), B (prismatic/translational joint), C (revolute joint), D (hinge joint), E (rigid connection), and the composite type CB.

2. **PhysXGen Dual-Branch Generation Architecture**: consists of two stages:

    - **Physical 3D VAE**: A physical VAE encoder $\mathcal{E}_{phy}$ and decoder $\mathcal{D}_{phy}$ are constructed to encode physical attributes—scale $P_{dim} \in \mathbb{R}^{N \times 1}$, affordance $P_{aff} \in \mathbb{R}^{N \times 1}$, density $P_\rho \in \mathbb{R}^{N \times 1}$, and kinematic parameters $P_{mov} \in \mathbb{R}^{N \times 11}$—concatenated as $P_{phy} \in \mathbb{R}^{N \times 14}$, together with CLIP-encoded functional descriptions $P_{sem} \in \mathbb{R}^{N \times 768 \times 3}$, into a physical latent space $P_{plat} \in \mathbb{R}^{N \times 8}$. The structural branch employs pretrained DINOv2 feature encoding. A key design is the residual connection establishing an information pathway from $\mathcal{D}_{phy}$ to $\mathcal{D}_{aes}$, exploiting the correlation between physical and structural representations.

    - **Physical Latent Generation**: A Transformer-based diffusion model trained with a conditional flow matching (CFM) objective. The physical branch uses 14 Transformer blocks (fewer than the 24 blocks in the structural branch to reduce computational cost). The structural branch provides guidance to the physical branch via learnable skip-connection layers. The total loss is $\mathcal{L}_{diff} = \mathcal{L}_{aes} + \mathcal{L}_{phy}$.

3. **PhysXNet-XL Procedural Augmentation**: Starting from PhysXNet, intra-class and cross-class combination rules are applied to procedurally generate 6 million+ physical 3D objects. Intra-class combination covers 9 categories including cabinets, tables, and bottles; cross-class combination identifies drawers and doors as modular components that can be flexibly integrated. Structural and physical consistency is enforced throughout.

### Loss & Training

**VAE Loss:**

$$\mathcal{L}_{vae} = \mathcal{L}_{aes}^{color} + \mathcal{L}_{aes}^{geometry} + \mathcal{L}_{phy} + \mathcal{L}_{sem} + \mathcal{L}_{kl} + \mathcal{L}_{reg}$$

- $\mathcal{L}_{aes}^{color}$: L2 + LPIPS
- $\mathcal{L}_{aes}^{geometry}$: mask + normal + depth
- $\mathcal{L}_{phy}$ / $\mathcal{L}_{sem}$: normalized L2
- AdamW optimizer, lr = 1e-4, 8× A100 GPUs
- 24K training / 1K validation / 1K test

## Key Experimental Results

### Main Results

**PhysXGen vs. baseline methods:**

| Method | PSNR↑ | CD↓ | F-Score↑ | Abs. Scale↓ | Material↑ | Affordance↑ | Motion COV↑ | Motion MMD↓ | Description↑ |
|--------|-------|-----|----------|-------------|-----------|-------------|-------------|-------------|--------------|
| TRELLIS | 24.31 | 13.2 | 76.9 | – | – | – | – | – | – |
| TRELLIS+PhysPre | 24.31 | 13.2 | 76.9 | 13.21 | 8.63 | 7.23 | 0.24 | 0.12 | 6.55 |
| **PhysXGen** | **24.53** | **12.7** | **77.3** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | **10.11** |

**Comparison with GPT baseline (TRELLIS+PartField+GPT-4o):**

| Method | Abs. Scale↓ | Material↑ | Affordance↑ | Motion COV↑ | Motion MMD↓ | Description↑ |
|--------|-------------|-----------|-------------|-------------|-------------|--------------|
| TRELLIS+PartField+GPT | 8.81 | 7.95 | 6.73 | 0.09 | 0.24 | **14.31** |
| **PhysXGen** | **7.24** | **13.01** | **11.30** | **0.33** | **0.08** | 10.11 |

PhysXGen achieves gains of 24%, 64%, 28%, and 72% over the GPT baseline on absolute scale, material, kinematics, and affordance, respectively.

### Ablation Study

**Effect of the dual-branch architecture:**

| Dep-VAE | Dep-Diff | PSNR↑ | Abs. Scale↓ | Material↑ | Affordance↑ | Motion COV↑ | Description↑ |
|---------|---------|-------|-------------|-----------|-------------|-------------|--------------|
| ✗ | ✗ | 24.31 | 13.21 | 8.63 | 7.23 | 0.24 | 6.55 |
| ✗ | ✓ | 24.31 | 12.01 | 10.69 | 8.95 | 0.26 | 7.71 |
| ✓ | ✗ | 24.32 | 10.57 | 9.86 | 9.32 | 0.28 | 7.54 |
| ✓ | ✓ | **24.53** | **7.24** | **13.01** | **11.30** | **0.33** | **10.11** |

### Key Findings

- **Physical–structural correlation is central**: joint modeling of the two modalities yields substantial improvements in physical attributes while also enhancing geometric quality (CD reduced from 13.2 to 12.7)
- The dual-branch design in both the VAE and the diffusion model is indispensable; removing either branch significantly degrades physical attribute generation quality
- GPT-4o performs better on functional descriptions (benefiting from its language capabilities), but is substantially inferior to end-to-end learning on structured physical attributes
- Absolute scale prediction is challenged by long-tail distributions spanning 1–1000 cm, where neither linear nor logarithmic normalization yields satisfactory results
- Kinematics is the most challenging attribute, requiring simultaneous accurate prediction of discrete part hierarchies and continuous motion parameters

## Highlights & Insights

- **Addressing a critical gap**: this work is the first to systematically define and annotate a complete physical-property spectrum for 3D objects, offering substantial value to the embodied AI community
- **Scalable annotation pipeline design**: the combination of GPT-4o and human verification is both efficient and reliable, and is directly reusable for new datasets
- **Exploitation of physical–structural correlation** is a well-motivated design choice: physical attributes (e.g., material → density → motion characteristics) are inherently correlated with geometric shape, making joint modeling natural and effective
- The 6-million-scale procedural augmentation of PhysXNet-XL provides a viable path toward large-scale physical 3D data

## Limitations & Future Work

- Physical attribute generation may produce spatially inconsistent artifacts (e.g., discontinuous material or affordance predictions across adjacent regions)
- Regression-based prediction of kinematic parameters struggles to accurately determine part counts and parent–child hierarchical relationships
- The dataset is constrained by PartNet's indoor/CAD model distribution, lacking outdoor and real-scan data
- Functional description relies on CLIP encodings, whose non-invertibility limits the ability to decode embeddings back into text
- Only four physical property types are used for generation; finer-grained quantities such as friction coefficients are not included

## Related Work & Insights

- **TRELLIS** serves as the foundation for structured 3D generation; PhysXGen extends it by superimposing a physical branch onto its latent space
- Compared to PartNet-Mobility, PhysXNet represents a qualitative leap in both annotation dimensions and scale
- The "part-level visual isolation + VLM annotation" strategy in the annotation pipeline generalizes to other scenarios requiring fine-grained annotation
- The work has direct downstream value for robotic manipulation and physics-based simulation

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to define the physical-property 3D generation problem; dataset contribution is particularly notable
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comparisons against multiple baselines and ablations validate core design choices, though the test set is limited to indoor CAD models
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and the annotation pipeline is described in thorough detail
- **Value**: ⭐⭐⭐⭐⭐ The dataset provides significant impetus for the embodied AI and robotics communities; the research direction is of great importance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](../../ICCV2025/3d_vision/gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[ICCV 2025\] DSO: Aligning 3D Generators with Simulation Feedback for Physical Soundness](../../ICCV2025/3d_vision/dso_aligning_3d_generators_with_simulation_feedback_for_physical_soundness.md)
- [\[NeurIPS 2025\] Hybrid Physical-Neural Simulator for Fast Cosmological Hydrodynamics](hybrid_physical-neural_simulator_for_fast_cosmological_hydrodynamics.md)
- [\[NeurIPS 2025\] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation](sofar_language-grounded_orientation_bridges_spatial_reasoning_and_object_manipul.md)
- [\[NeurIPS 2025\] Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation](cue3d_quantifying_the_role_of_image_cues_in_single-image_3d_generation.md)

</div>

<!-- RELATED:END -->
