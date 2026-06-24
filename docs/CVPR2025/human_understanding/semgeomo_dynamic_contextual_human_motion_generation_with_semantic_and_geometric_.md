---
title: >-
  [Paper Note] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance
description: >-
  [CVPR 2025][Human Understanding][Dynamic contextual human motion generation] This paper proposes SemGeoMo, which leverages an LLM-based automatic annotator to provide semantic guidance, combined with hierarchical geometric guidance at both affordance and joint levels. This two-stage framework achieves high-quality human-object interaction generation under dynamic contextual environments, while simultaneously outputting the corresponding textual descriptions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Dynamic contextual human motion generation"
  - "semantic-geometric guidance"
  - "Affordance Map"
  - "LLM annotator"
  - "human-object interaction"
date: 2026-05-08
content_hash: b2a9ab868fbb986f
---

# SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance

**Conference**: CVPR 2025  
**arXiv**: [2503.01291](https://arxiv.org/abs/2503.01291)  
**Code**: [Project Page](https://4dvlab.github.io/project_page/semgeomo/)  
**Area**: Human Understanding  
**Keywords**: Dynamic contextual human motion generation, semantic-geometric guidance, Affordance Map, LLM annotator, human-object interaction

## TL;DR

This paper proposes SemGeoMo, which leverages an LLM-based automatic annotator to provide semantic guidance, combined with hierarchical geometric guidance at both affordance and joint levels. This two-stage framework achieves high-quality human-object interaction generation under dynamic contextual environments, while simultaneously outputting the corresponding textual descriptions.

## Background & Motivation

Dynamic contextual motion generation aims to generate human interaction motions that adapt to real dynamic environments, which is crucial for applications such as robotics, VR/AR, etc. Existing methodologies suffer from two main types of limitations:

1. **Text-driven joint generation methods**: Simultaneously generating both human and object motions leads to an excessively large search space, resulting in sub-optimal generation quality and a lack of fine-grained control.
2. **Contextual motion generation methods**: Most only handle static environments (e.g., fixed furniture). The few methods addressing dynamic targets (such as OMOMO) lack semantic guidance from text and fail to fully exploit fine-grained geometric representations.

The core challenge lies in how to construct effective semantic guidance (understanding "how to interact") and geometric guidance (ensuring precise contact and avoiding penetration), and integrate them seamlessly.

## Method

### Overall Architecture

A two-stage conditional diffusion framework is proposed: (1) SemGeo hierarchical guidance generation: Under text and point cloud conditions, a dual-branch Transformer jointly generates affordance-level and joint-level interaction cues; (2) SemGeo-guided motion generation: Utilizes the geometric cues and semantic information from the first stage to guide detailed human motion generation (based on Motion ControlNet + MDM).

### Key Designs

#### 1. LLM-based Automatic Interaction Text Annotator

- **Function**: Automatically generates coarse-to-fine interaction text descriptions from 4D point clouds, eliminating the need for manual annotation.
- **Mechanism**: A two-step process: (a) Extracting bounding boxes and movement trajectories from point clouds, then combining them with predefined action/category lists to generate coarse-grained descriptions using a LoRA-fine-tuned LLaMA; (b) Combining predicted hand joint positions to calculate contact details, enabling the LLM to divide the interaction into three phases and generate fine-grained descriptions (e.g., "left hand contacts the bottom left of the box").
- **Design Motivation**: LLMs possess common-sense knowledge of interaction processes and can reason out plausible interaction methods. The coarse-to-fine annotation strategy progressively increases description granularity, allowing semantic guidance to function effectively at different levels.

#### 2. Dual-branch Transformer for Hierarchical Geometric Guidance Generation

- **Function**: Jointly generates the affordance map and joint positions to capture coarse-to-fine interaction geometric cues.
- **Mechanism**: Under the guidance of CLIP text features $F_{text}$ and BPS point cloud features $F_{pc}$, a conditional diffusion model is used to generate parallel outputs via two branches: JointTransformer and AffordanceTransformer. The AffordanceTransformer models the close relationship between affordance and point cloud geometry using cross-attention, and finally feeds the affordance information back to the joint branch for refinement via mutual cross-attention.
- **Design Motivation**: Decoupling contact geometry generation from motion generation reduces the learning difficulty for a single model. While affordance provides coarse-grained "where to contact" information, joint positions offer precise spatial localization, making them complementary.

#### 3. SemGeo Condition Module and Motion ControlNet

- **Function**: Effectively fuses multi-level semantic and geometric conditional guidance to generate full-body motions.
- **Mechanism**: Uses LongCLIP (capable of handling long text) to extract fine-grained text features $F'_{text}$, concatenates point cloud features with the affordance map, passes them through an MLP and Temporal Transformer to extract spatio-temporal features $F$, and then integrates joint and affordance features using mutual cross-attention. These conditions are fed into Motion ControlNet (with frozen MDM weights). During sampling, refinement is performed using a classifier-guided joint loss $L_{joint}$ and foot stability loss $L_{foot}$.
- **Design Motivation**: The ControlNet architecture allows leveraging the motion priors of the pre-trained MDM, while multi-level conditioning provides a dual guarantee of semantic plausibility and geometric accuracy.

### Loss & Training

- Stage 1: $\mathcal{L} = \mathbb{E}_{x^0,t}\|\hat{x}_\theta(x^t,t,c) - x^0\|_1$ ($L_1$ reconstruction loss)
- Stage 2 Sampling Guidance: $L_{joint} = \frac{1}{J}\sum|J_{pred} - J'_h|_2 \cdot \text{Mask}$ (contact joint constraint); $L_{foot}$ penalizes foot floating, sliding, and acceleration.

## Key Experimental Results

### Main Results: FullBodyManipulation Dataset

| Method | HandJPE↓ | MPJPE↓ | $C_{prec}$↑ | $C_{rec}$↑ | FID↓ | R-score↑ |
|------|----------|--------|------------|----------|------|----------|
| SceneDiff | 95.38 | 19.84 | 0.64 | 0.19 | 1.64 | 0.59 |
| OMOMO | 33.18 | 18.06 | 0.77 | 0.71 | 1.98 | 0.38 |
| CHOIS | 31.68 | 17.12 | 0.76 | 0.58 | 2.27 | 0.49 |
| **Ours (GT text)** | **27.84** | **16.62** | **0.84** | **0.74** | **1.17** | **0.66** |

### Ablation Study Highlights

- Semantic guidance (textual descriptions) significantly contributes to both contact accuracy and motion quality.
- Fine-grained LLM annotations further improve performance compared to coarse-grained annotations.
- Dual-branch joint generation outperforms generating affordance and joint positions separately.

### Key Findings

- SemGeoMo achieves state-of-the-art (SOTA) performance across three human-object interaction datasets.
- The method demonstrates generalization capabilities to unseen objects, human-human interactions, and deformable objects.
- Simultaneous generation of motion and text descriptions enhances the interpretability of interactions.
- The LLM annotator approaches or even matches the performance of manual human annotation.

## Highlights & Insights

1. **Completeness of Dual Semantic and Geometric Guidance**: Text provides the common sense of "what to do," while affordance and joint positions provide precise constraints on "where to do it."
2. **LLM as a Source of Common Sense for Interaction**: Utilizing the reasoning capabilities of LLMs to automatically generate annotations reduces manual costs while providing rich semantics.
3. **Coarse-to-Fine Hierarchical Design**: Step-by-step refinement from affordance to joints and finally to full-body motion reduces the learning difficulty of generation.

## Limitations & Future Work

- Dependence on multiple pre-trained models (LLaMA, CLIP, LongCLIP, MDM) increases system complexity.
- Currently evaluated primarily on tabletop manipulation scenarios; large-scale full-body interaction scenarios require more data.
- The quality of the LLM annotator is constrained by the coverage of the fine-tuning data.

## Related Work & Insights

- The concept of using LLMs as automatic annotators can be extended to other motion datasets that lack textual annotations.
- The dual-branch Transformer architecture for joint generation of multiple intermediate representations can be applied to other hierarchical tasks.

## Rating

⭐⭐⭐⭐ — The framework is clearly designed, and the hierarchical guidance methodology is well-structured. The LLM annotator is a highly practical innovation. Consistently achieving SOTA results across three datasets demonstrates the effectiveness of the proposed method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionReFit: Dynamic Motion Blending for Versatile Motion Editing](motionrefit_motion_editing.md)
- [\[CVPR 2025\] D3-Human: Dynamic Disentangled Digital Human from Monocular Video](d3-human_dynamic_disentangled_digital_human_from_monocular_video.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)
- [\[CVPR 2025\] Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](lost_in_translation_found_in_context_sign_language_translation_with_contextual_c.md)
- [\[CVPR 2025\] Learning Affine Correspondences by Integrating Geometric Constraints](learning_affine_correspondences_by_integrating_geometric_constraints.md)

</div>

<!-- RELATED:END -->
