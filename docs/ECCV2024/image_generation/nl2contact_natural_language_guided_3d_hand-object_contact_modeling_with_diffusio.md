---
title: >-
  [Paper Note] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model
description: >-
  [ECCV 2024][Image Generation] This paper proposes NL2Contact, which is the first to leverage natural language descriptions for the controllable modeling of 3D hand-object contact maps. It generates hand poses and contact areas from text using a staged diffusion model, and constructs ContactDescribe, the first hand-object contact dataset with fine-grained linguistic descriptions.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 8acd561de0e8b00a
---

# NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2407.12727](https://arxiv.org/abs/2407.12727)  
**Area**: Image Generation

## TL;DR

This paper proposes NL2Contact, which is the first to leverage natural language descriptions for the controllable modeling of 3D hand-object contact maps. It generates hand poses and contact areas from text using a staged diffusion model, and constructs ContactDescribe, the first hand-object contact dataset with fine-grained linguistic descriptions.

## Background & Motivation

Hand-object contact modeling is crucial for applications such as animation, VR/AR, and robotic grasping. Existing methods (e.g., ContactOpt, S2Contact) rely on geometric constraints to infer contact from point clouds, but suffer from two core problems:

**Uncontrollable**: These methods cannot specify or control the contact patterns. The generated results often tend toward "full-grasp" modes where all fingers contact the object, which deviates from actual human usage habits (e.g., using scissors only requires two fingers).

**Lack of Semantics**: Existing intent labels (verbs, object affordance) are too coarse-grained to precisely describe contact patterns.

To address this, this paper is the first to introduce the new task of guiding 3D hand-object contact modeling with natural language, leveraging the expressive power of language to achieve more precise contact control.

## Method

### Overall Architecture

NL2Contact consists of three core modules:

1. **Text-to-Hand-Object Fusion Module**: Fuses cross-modal features of text, hand pose, and object point clouds.
2. **Staged Latent Diffusion Module**: A two-stage diffusion model that first generates the hand pose and then the contact map.
3. **Contact Optimization**: Iteratively optimizes hand pose parameters using the generated contact maps.

The input consists of an initial hand pose $\widetilde{\mathcal{H}}$, an object point cloud $\mathcal{O} \in \mathbb{R}^{2048 \times 3}$, and a text description $\mathbb{T}$. The output is the contact probability maps on both the hand and the object.

### Key Designs

**ContactDescribe Dataset Construction**:

- Based on the ContactPose dataset (2,300 grasping instances, 25 types of everyday objects, 50 participants).
- Multi-level (coarse-to-fine) language descriptions are designed: high-level describes the grasping action, mid-level describes the grasp types and finger states, and low-level specifies the location of contacting joints.
- Assisted by ChatGPT, diverse natural language descriptions were generated, with 5 different descriptions for each grasping instance, totaling 11,500 descriptions.

**Text-to-Hand Fusion**:
- VPoser is used to encode hand pose features $f_\theta^g \in \mathbb{R}^{64}$ and BERT is used to extract text token embeddings $f_{text} \in \mathbb{R}^{768 \times n}$.
- Fuses text, object, and hand pose features through two cascaded multi-head attention modules.

**Staged Diffusion**:
- **Stage 1 (Hand Pose Diffusion)**: Conditioned on the Text-to-Hand embedding, it denoises and generates the hand pose within the VPoser latent space.
- **Stage 2 (Contact Diffusion)**: Freezing the parameters of Stage 1, and conditioned on the generated hand pose and text-object fused features, it uses a PointNet encoder to encode the contact map into a latent space of $x_c^0 \in \mathbb{R}^{32 \times 32}$ and then denoises it to generate the contact map using a U-Net.

### Loss & Training

Hand pose diffusion loss: $\mathcal{L}_{\text{pose}} = \mathbb{E}[\|\epsilon - G_\theta^1(\mathbf{x}_h^t | t, f_h)\|_2^2]$

Contact diffusion loss: $\mathcal{L}_{\text{contact}} = \mathbb{E}[\|\epsilon - G_\theta^2(\mathbf{x}_c^t | t, f_c)\|_2^2]$

Contact optimization loss: $\mathcal{L}_{opt} = \|\mathcal{C}'_H - \hat{\mathcal{C}}_H\|_2^2 + \lambda_o \|\mathcal{C}'_O - \hat{\mathcal{C}}_O\|_2^2 + \lambda_{pen}\mathcal{L}_{pen}$

where $\lambda_O=5$, $\lambda_{pen}=3$, and the penetration loss $\mathcal{L}_{pen}$ penalizes hand-object interpenetrations.

## Key Experimental Results

### Main Results

**Grasp Pose Optimization Experiments (ContactDescribe + HO3D)**:

| Method | MPJPE↓ | Inter.↓ | Cover.↑ | Pr.↑ | Re.↑ |
|------|--------|---------|---------|------|------|
| Perturbed Pose | 79.9 | 8.4 | 2.3% | 9.9% | 11.5% |
| ContactNet | 45.2 | 15.6 | 18.4% | 31.6% | 47.6% |
| ContactOpt | 25.1 | 12.8 | 19.7% | 38.7% | 54.8% |
| S2Contact | 29.4 | 12.2 | 22.2% | 42.5% | 56.1% |
| **NL2Contact** | **21.7** | **7.1** | **30.5%** | **49.2%** | **59.9%** |

**Grasp Generation Experiments**:

| Method | Inter.↓ | Cover.↑ | Diversity↑ | SD↓ |
|------|---------|---------|------------|-----|
| GrabNet | 15.50 | 99% | 2.06 | 2.34 |
| GraspTTA | 7.37 | 76% | 1.43 | 5.34 |
| ContactGen | 9.96 | 97% | 5.04 | 2.70 |
| **NL2Contact** | **5.89** | **99%** | **5.91** | **2.31** |

### Ablation Study

Generational experiments on the HO3D dataset demonstrate that NL2Contact still achieves a near-optimal penetration volume on unseen objects (4.39 vs. S2Contact's 3.52) while having the lowest MPJPE (8.4mm), which proves the robust generalization capability of hand-centric contact descriptions.

### Key Findings

- Compared to ContactOpt, NL2Contact reduces MPJPE by 4.4mm, while simultaneously **reducing** the penetration volume (whereas other methods tend to increase penetration).
- Language guidance endows contact generation with controllability, enabling the generation of specific finger contact patterns consistent with the descriptions.
- Jointly achieves the lowest penetration, highest coverage, and highest diversity in grasp generation.

## Highlights & Insights

1. **Pioneering Task Definition**: Natural language guided 3D hand-object contact modeling, where linguistic descriptions provide more precise control than verbs or affordance labels.
2. **LLM-Assisted Annotation**: Cleverly utilizes ChatGPT to generate diverse natural language descriptions from structured prompts, avoiding the high cost and inconsistent quality of manual annotation.
3. **Reasonable Staged Design**: Generates hand poses first and then contact maps, decoupling the cross-modal complexity of text-to-3D interaction.
4. **High Practicality**: Contact map generation takes only about 3 seconds per instance, and training takes around 11 hours on a single V100 GPU.

## Limitations & Future Work

- Relies on initial hand pose inputs; optimization can be challenging for severely erroneous initial poses.
- The ContactDescribe dataset is limited in scale (25 types of objects); generalization to more object categories needs further validation.
- The ambiguity of natural language descriptions may introduce uncertainty into the contact generation.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty: ★★★★★ — Pioneering the language-guided contact modeling task + first fine-grained contact language dataset
- Technical Quality: ★★★★ — Reasonable staged diffusion design, effective cross-modal fusion
- Experiments: ★★★★ — Multi-task validation (optimization + generation), multi-dataset evaluation
- Writing Quality: ★★★★ — Clear structure, intuitive illustrations

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)

</div>

<!-- RELATED:END -->
