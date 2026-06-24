---
title: >-
  [Paper Note] SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing
description: >-
  [CVPR 2025][3D Vision][Simulation-ready avatars] SimAvatar proposes the first fully simulation-ready text-driven 3D avatar generation framework. By representing the body, clothing, and hair as layered representations consisting of SMPL meshes, clothing meshes, and hair strands, and attaching 3D Gaussians to learn the appearance, the method is able to leverage diffusion model priors for realistic textures while directly interfacing with physical/neural simulators to produce re…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Simulation-ready avatars"
  - "layered representation"
  - "hair strand simulation"
  - "clothing simulation"
  - "text-driven generation"
date: 2026-05-08
content_hash: ce3a4bd1d49d8787
---

# SimAvatar: Simulation-Ready Avatars with Layered Hair and Clothing

**Conference**: CVPR 2025  
**arXiv**: [2412.09545](https://arxiv.org/abs/2412.09545)  
**Code**: [Project Page](https://nvlabs.github.io/SimAvatar)  
**Area**: 3D Vision / Digital Human Generation  
**Keywords**: Simulation-ready avatars, layered representation, hair strand simulation, clothing simulation, text-driven generation

## TL;DR

SimAvatar proposes the first fully simulation-ready text-driven 3D avatar generation framework. By representing the body, clothing, and hair as layered representations consisting of SMPL meshes, clothing meshes, and hair strands, and attaching 3D Gaussians to learn the appearance, the method is able to leverage diffusion model priors for realistic textures while directly interfacing with physical/neural simulators to produce realistic dynamics.

## Background & Motivation

Text-driven 3D avatar generation has made significant progress, but existing methods face a fundamental contradiction:

1. **Single-layer representation methods** (GAvatar, TADA, etc.) treat hair, body, and clothing as a unified geometry and use linear blend skinning (LBS) for animation. They cannot simulate individual areas independently, and the dynamics of loose clothing and hair are unrealistic—for example, a skirt naturally splits into two pieces during leg raising.

2. **Layered but non-optimizable/non-simulatable** methods (HumanLiff, TELA, etc.) use implicit representations like NeRF to model each layer separately. Although convenient for optimizing with diffusion models, the meshes extracted from NeRF are noisy and cannot be directly used by existing clothing/hair simulators.

The key challenge lies in connecting the two representations: simulators require clean, non-watertight meshes or specially designed hair strands (which are topologically fixed and difficult to optimize), while generative pipelines require implicit representations (which can be optimized under the supervision of noise from diffusion models, but are difficult to convert into simulatable formats). SimAvatar's key insight: adopt appropriate representations for different body parts and use 3D Gaussians as a bridge.

## Method

### Overall Architecture

SimAvatar consists of three steps: (1) using three text-conditioned generative models to generate clothing meshes, SMPL body shapes, and hair strands separately; (2) attaching 3D Gaussians to the three layers of geometry and optimizing appearance via SDS; (3) driving the clothing and hair strands using physical simulators during animation, and transferring the motion to the Gaussians. The body is driven by LBS, clothing is driven by the HOOD neural simulator, and hair is driven by a physical simulator.

### Key Designs

#### Key Design 1: Text-conditioned Clothing Diffusion Model

- **Function**: Generate clean, simulatable non-watertight clothing meshes from text prompts.
- **Mechanism**: First train a VAE to learn the clothing geometry latent space—encoding 10,000 uniformly sampled points into a $Z \in \mathbb{R}^{512 \times 16}$ vector, where the decoder represents the non-watertight mesh using UDF (Unsigned Distance Field). Then, train a text-conditioned diffusion model in the latent space, injecting text embeddings extracted by BERT via cross-attention. During inference, denoise from noise to a latent code, which is then decoded into a mesh. The training data contains ~20,000 paired meshes and text annotations.
- **Design Motivation**: Directly optimizing non-watertight clothing meshes with SDS cannot change the topology and easily produces noisy meshes. A learned generative model ensures the cleanliness and diversity of the output meshes.

#### Key Design 2: Part-wise Customization of 3D Gaussians

- **Function**: Model high-fidelity appearance on simulation-ready geometry while ensuring motion consistency.
- **Mechanism**: (a) **Mesh Gaussians** (body/clothing): Each Gaussian is bound to a mesh face. Its position/rotation/scale are defined in the face's local coordinate system and transformed to the global system via $\hat{\mu}_i(\theta) = kR(\theta) \cdot p_i + P(\theta)$. Color and opacity are queried from an implicit field $\mathcal{F}_\phi$ (separate implicit fields for body and clothing prevent texture entanglement). (b) **Hair Gaussians**: Each line segment is assigned a Gaussian, with position $\mu_i = (l_i + l_{i+1})/2$, scaling long along the hair strand direction and ultra-thin ($\gamma=0.001$) in the radial direction, and rotation computed from the direction of the hair strand.
- **Design Motivation**: Gaussians are flexible enough to be driven by meshes/hair strands and possess exceptional capability for appearance modeling. Part-wise customization ensures the unique structure of each geometry is respected.

#### Key Design 3: Hair Opacity Regularization and Phong Shading

- **Function**: Prevent hair strand breakage and baked-in shadows.
- **Mechanism**: (a) Hair regularization $L_{hair} = \frac{1}{N_s N_l} \sum_{i=1}^{N_s} \sum_{j=2}^{N_l} (o_{i,j-1} - o_{i,j})$, encouraging Gaussians closer to the scalp to have higher opacity than distant ones, allowing the optimization process to prune excess hair strands without causing breakage. (b) A Phong shading model randomly samples point light source positions/colors to generate shading, encouraging Gaussians to learn pose-independent albedo rather than baked-in shadows.
- **Design Motivation**: The high variance of SDS optimization easily leads to transparent Gaussians in the middle of hair strands (breakage). De-lighting ensures correct dynamic lighting effects like wrinkles during animation.

### Loss & Training

The final loss is $L = L_{SDS} + \lambda_{hair} L_{hair}$, where $L_{SDS}$ is the Score Distillation Sampling loss (using a pretrained text-to-image diffusion model) and $\lambda_{hair} = 1.0$.

## Key Experimental Results

### User Study: Preference Comparison with SOTA Methods

| Metric | vs TADA | vs Fantasia3D | vs GAvatar |
|------|---------|--------------|-----------|
| Appearance Preference (% selecting SimAvatar) | 89.55% | 100% | 87.03% |
| Motion Preference (% selecting SimAvatar) | 91.87% | 100% | 94.47% |

### Qualitative Comparison

| Aspect | SimAvatar | GAvatar/TADA |
|------|-----------|-------------|
| Loose clothing (skirt) | Physically realistic swinging | Unnatural splitting |
| Hair dynamics | Smooth hair strand floating | Fixed to follow the body |
| Texture quality | High fidelity, rich details | Blurry or artifacts |

### Key Findings

- Users overwhelmingly prefer SimAvatar in both appearance and motion (87-100%).
- Realizes physically realistic dynamics of skirts, long hair, etc., in text-driven avatars for the first time.
- The clothing diffusion model covers common types such as T-shirts, outerwear, shorts, skirts, etc.
- Phong shading effectively prevents the shadow baking problem.

## Highlights & Insights

1. **First fully simulation-ready text-driven avatar**: Truly bridges the representation gap between "generation" and "simulation".
2. **3D Gaussians as a bridge representation**: Maintains the physical correctness of simulated geometry while offering the optimization flexibility required by diffusion models.
3. **Layered + part-wise strategy**: Custom solutions are adopted for different physical attributes of the body, clothing, and hair.

## Limitations & Future Work

- Hair and clothing generative models are constrained by the diversity of training data.
- Clothing and hair are currently simulated sequentially, failing to handle scenarios requiring joint simulation, such as hoods.
- Accessories and shoes remain entangled with the body/clothing layer.
- Future work can explore joint simulation and fully disentangled avatar generation.

## Related Work & Insights

- **GAvatar**: Primitive-based 3DGS avatar; cannot be simulated due to single-layer representation.
- **TADA**: Mesh + adaptive subdivision; high quality but likewise single-layer.
- **HOOD**: Neural clothing simulator, directly integrated and used by SimAvatar.
- **HAAR**: Text-conditioned hair strand diffusion model, called by SimAvatar to generate initial hair strands.
- **Insight**: Decoupling generation and physical simulation at the representation level allows each to play to its strengths.

## Rating

⭐⭐⭐⭐ — For the first time, it unifies the flexibility of text-driven generation with the realism of physics simulation into a single framework. The layered design is reasonable, and the overwhelming advantage in the user study proves the effectiveness of the method. It plays an important leading role in the field of avatar generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](../../CVPR2026/3d_vision/physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2025\] MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[CVPR 2025\] Volumetric Surfaces: Representing Fuzzy Geometries with Layered Meshes](volumetric_surfaces_representing_fuzzy_geometries_with_layered_meshes.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](../../ICCV2025/3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)

</div>

<!-- RELATED:END -->
