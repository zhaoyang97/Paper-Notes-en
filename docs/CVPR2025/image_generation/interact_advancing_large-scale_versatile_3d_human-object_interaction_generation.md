---
title: >-
  [Paper Note] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation
description: >-
  [CVPR 2025][Image Generation][Human-Object Interaction] This paper Mojo presents the InterAct benchmark, which consolidates and standardizes 21.81 hours of 3D human-object interaction data (expanded to 30.70 hours). Through a unified optimization framework, it corrects motion capture artifacts and augments the data, defining six generation tasks and a unified modeling approach to achieve SOTA performance across multiple HOI generation tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Human-Object Interaction"
  - "Motion Generation"
  - "Dataset"
  - "Interaction Correction and Augmentation"
  - "Multi-Task Learning"
date: 2026-05-08
content_hash: 1b46b0c32218f2cd
---

# InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation

**Conference**: CVPR 2025  
**arXiv**: [2509.09555](https://arxiv.org/abs/2509.09555)  
**Code**: [https://github.com/wzyabcas/InterAct](https://github.com/wzyabcas/InterAct)  
**Area**: Image Generation  
**Keywords**: Human-Object Interaction, Motion Generation, Dataset, Interaction Correction and Augmentation, Multi-Task Learning

## TL;DR
This paper Mojo presents the InterAct benchmark, which consolidates and standardizes 21.81 hours of 3D human-object interaction data (expanded to 30.70 hours). Through a unified optimization framework, it corrects motion capture artifacts and augments the data, defining six generation tasks and a unified modeling approach to achieve SOTA performance across multiple HOI generation tasks.

## Background & Motivation

1. **Background**: Large-scale human motion capture datasets have driven the progress of human motion generation. However, modeling and generating dynamic 3D Human-Object Interaction (HOI) still face challenges, primarily limited by the quality and scale of datasets.
2. **Limitations of Prior Work**: Existing HOI datasets suffer from three key issues: (a) limited and inconsistent datasets—different datasets utilize different human representations, coordinate systems, and annotation formats, making consolidation difficult; (b) coarse and incomplete annotations—lacking detailed textual descriptions of interaction details and involved body parts; (c) ubiquitous artifacts—problems such as contact penetration, floating contact, inaccurate hand poses, and motion jittering.
3. **Key Challenge**: The scarcity of high-quality HOI data limits the ability of generative models to learn realistic interaction dynamics, while directly consolidating existing datasets faces the dual obstacles of inconsistent representations and low quality.
4. **Goal**: (a) How to uniformly consolidate heterogeneous HOI datasets? (b) How to automatically repair MoCap artifacts? (c) How to expand the data volume without collecting new data? (d) How to design a unified multi-task HOI generation framework?
5. **Key Insight**: Leveraging the "interaction invariance" principle—meaning that the contact relationship between human and object should remain consistent even under minor variations in human motion—to augment data.
6. **Core Idea**: Constructing the first large-scale unified HOI benchmark, expanding data through data correction and contact invariance-based augmentation, and achieving SOTA across six HOI generation tasks using a unified multi-task learning framework.

## Method

### Overall Architecture
InterAct consists of two components: data collection/construction and method design. Data construction integrates 21.81 hours of interaction data across seven source datasets, unifying the human representation into a marker-based representation. Textual descriptions are generated via a two-stage annotation (human + GPT-4), followed by a three-step optimization pipeline (whole-body correction $\rightarrow$ hand correction $\rightarrow$ interaction augmentation) to improve quality and expand the duration to 30.70 hours. Method design defines six tasks and employs a transformer-based diffusion model for joint multi-task modeling.

### Key Designs

1. **Marker-based Unified Human Representation**:

    - **Function**: Unify representations of different human models (SMPL-H, SMPL-X) to facilitate interaction modeling.
    - **Mechanism**: Select specific vertices on the human body surface as markers instead of using joint nodes or rotation parameters. Marker correspondences between different models are established by indexing marker points on the SMPL-H surface and then finding the nearest corresponding points on the SMPL-X. Markers are located on the body surface and directly participate in interactions, keeping errors within $1\text{ cm}$.
    - **Design Motivation**: Joint nodes lie inside the body and do not directly participate in contact interactions; the SMPL rotation representation is less intuitive than Cartesian coordinates; the marker representation both unifies different models and is naturally suited for contact modeling.

2. **Stage-wise Interaction Correction and Augmentation Optimization Framework**:

    - **Function**: Repair artifacts in MoCap data and generate synthetic data to augment the dataset.
    - **Mechanism**: Perform gradient optimization in three steps: (1) **Whole-body Correction**: Optimize parent bodies and object poses using reconstruction loss to stay close to original data, while adding contact and penetration losses to minimize artifacts; (2) **Hand Correction**: Individually optimize hand poses using a contact promotion loss $E_{\text{cont}} = \sum_i c_i \sum_j d_j[i]$ to guide the hand to fit the object, combined with joint limits constraints to maintain naturalness; (3) **Interaction Augmentation**: Apply random displacements to object trajectories and optimize human motions to preserve contact consistency using a weighted distance loss $E_{\text{align}} = \sum_{i,j,k} \frac{1}{(\hat{D}_{jk}+\epsilon)^2} |\hat{D}_{jk} - D_{jk}|^2$, followed by filtering out low-quality results.
    - **Design Motivation**: Hand correction is decoupled from whole-body correction because hand parameters are abundant in the SMPL representation but contribute minimally to the overall loss, so separate processing balances them better; augmentation relies on "interaction invariance"—e.g., when walking while holding a box, minor variations in gait do not affect the hand-box contact.

3. **Unified Multi-Task HOI Generation Modeling**:

    - **Function**: Uniformly handle five kinematic generation tasks (text/motion/object/human conditional generation and interaction prediction) with a single model.
    - **Mechanism**: Unify the HOI sequence representation $\langle h, o \rangle$ for each task (containing marker coordinates, velocities, signed distance vectors to the object, foot-ground contact labels, and object motion + BPS geometric encoding), and introduce an additional human-object relationship feature $\eta$ (the vector from the human marker to the nearest point on the object) as an auxiliary output for multi-task learning. A transformer-based diffusion model is utilized, jointly regressing motion and contact features during training.
    - **Design Motivation**: A unified representation avoids designing separate models for each task; jointly modeling the contact relationship $\eta$ forces the model to learn the spatial relationships of the interactions, improving consistency.

### Loss & Training
Data correction employs gradient descent optimization, comprising reconstruction, contact, penetration, smoothness, and prior losses. The generative model utilizes a diffusion denoising loss, optionally incorporating classifier guidance based on contact prediction. Textual encoding utilizes an interaction-aware encoder trained with Sentence-BERT + InfoNCE contrastive learning.

## Key Experimental Results

### Main Results

**Text-conditioned Interaction Generation:**

| Configuration | R-Prec Top1 ↑ | FID ↓ | MM Dist ↓ | Diversity |
|------|--------------|-------|-----------|-----------|
| Ground Truth | 0.852 | 0.000 | 2.810 | 11.489 |
| Baseline (w/o HOI awareness) | 0.733 | 3.192 | 4.950 | 11.192 |
| +Contact Modeling | 0.730 | 1.997 | 4.752 | 11.501 |
| +HOI-aware Object Encoding | 0.737 | 1.837 | 4.631 | 11.369 |
| +HOI-aware Text Encoding | 0.784 | 1.570 | 4.414 | 11.409 |
| **+Guidance (Full)** | **0.784** | **1.567** | **4.412** | 11.518 |

**Motion-conditioned Interaction Generation:**

| Method | FID ↓ | Multimodality ↑ | Diversity |
|------|-------|----------------|-----------|
| HOI-Diff | 3.566 | 5.321 | 10.989 |
| **Ours** | **2.161** | **5.792** | **11.291** |

### Ablation Study

| Data Version | Penetration ↓ | Contact Ratio | User Preference |
|---------|--------------|---------------|---------|
| BEHAVE Original | 0.017 | 0.048 | 22.3% |
| BEHAVE Corrected | 0.016 | 0.071 | 39.7% |
| BEHAVE Corrected + Augmented | 0.016 | 0.069 | 38.0% |
| OMOMO Original | 0.009 | 0.071 | 23.9% |
| OMOMO Corrected | 0.007 | 0.131 | 39.4% |

| Human Representation | Penetration ↓ | Contact Ratio |
|---------|--------------|---------------|
| SMPL | 0.030 | 0.025 |
| Joint | 0.027 | 0.032 |
| **Marker** | **0.025** | **0.028** |

### Key Findings
- Interaction correction significantly enhances data quality: the contact ratio increases by 47%-85%, and user preference increases from 22-24% to 39-40%.
- The augmented data demonstrates a quality close to that of corrected data, validating the effectiveness of the interaction invariance principle.
- The HOI-aware text encoder contributes most significantly: R-Precision increases from 0.737 to 0.784, and FID decreases from 1.837 to 1.570.
- The Marker representation outperforms both SMPL and Joint representations in penetration and contact metrics.
- In interaction prediction tasks, larger models and scaling data consistently improve performance (validating the scaling law).

## Highlights & Insights
- **"Interaction invariance" data augmentation** is an elegant insight: generating new data by displacing the object and optimizing the human body to preserve contact consistency not only increases diversity but also guarantees the physical plausibility of the interactions. This concept is transferable to other scenarios requiring contact consistency, such as hand-object interactions.
- **The marker-based representation** balances representation unification and interaction modeling requirements: marker points located on the surface are naturally suited for computing contact distances and penetration detection, proving superior to internal joint nodes for HOI tasks.
- **Decoupling hand and whole-body correction** is a valuable strategy: when scales of different components in high-dimensional parameter spaces vary significantly, stage-wise optimization achieves a better balance.

## Limitations & Future Work
- Object displacements in data augmentation are uniformly random; smarter displacement strategies (e.g., based on scene semantics) could generate more meaningful variants.
- Currently, only single-human-single-object interactions are supported; multi-person or multi-object scenarios are not covered.
- Physical simulation (interaction imitation tasks) and kinematic generation remain isolated; there is still room for end-to-end physically consistent generation.
- Textual descriptions are primarily rewritten by GPT-4, which may introduce biases.

## Related Work & Insights
- **vs InterDiff**: InterDiff first introduced diverse dynamic object interactions, but is limited by data scale. InterAct comprehensively outperforms it via large-scale data and multi-task learning.
- **vs OMOMO**: OMOMO uses two-stage generation (hands first, then the whole body), which works well for hand-dominant interactions but is less suited for whole-body interactions. InterAct's single-stage multi-task learning is more versatile.
- **vs PhysHOI**: Physical simulation methods guarantee physical plausibility but suffer from monotonous interaction patterns. InterAct demonstrates that utilizing corrected data can boost the simulation success rate from 84.4% to 90.7%.

## Rating
- Novelty: ⭐⭐⭐⭐ The data construction pipeline and interaction-invariance augmentation are innovative, but the multi-task learning framework is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across six tasks, including quantitative metrics, user studies, ablation studies, and physical simulation verification.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though some details require referring to the supplementary materials due to the large volume of content.
- Value: ⭐⭐⭐⭐⭐ As the largest-scale 3D HOI benchmark to date, it fundamentally drives advancement in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OneHOI: Unifying Human-Object Interaction Generation and Editing](../../CVPR2026/image_generation/onehoi_unifying_human-object_interaction_generation_and_editing.md)
- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[CVPR 2025\] FoundHand: Large-Scale Domain-Specific Learning for Controllable Hand Image Generation](foundhand_large-scale_domain-specific_learning_for_controllable_hand_image_gener.md)
- [\[CVPR 2025\] RORem: Training a Robust Object Remover with Human-in-the-Loop](rorem_training_a_robust_object_remover_with_human-in-the-loop.md)

</div>

<!-- RELATED:END -->
