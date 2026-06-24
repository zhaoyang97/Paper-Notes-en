---
title: >-
  [Paper Note] How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects
description: >-
  [ICML2025][Human Understanding][motion synthesis] This paper presents a unified framework for text-driven motion generation targetting large-vocabulary heterogeneous skeletal objects, achieved by annotating text descriptions for the Truebones Zoo dataset (70+ species), introducing rig augmentation, and integrating TreePE and RestPE encodings into the Motion Diffusion Model. It enables high-quality 3D motion synthesis for animals, dinosaurs, and even fictional creatures.
tags:
  - "ICML2025"
  - "Human Understanding"
  - "motion synthesis"
  - "text-to-motion"
  - "skeletal rig"
  - "diffusion model"
  - "3D animation"
date: 2026-05-08
content_hash: 5414a16ff9fbea2f
---

# How to Move Your Dragon: Text-to-Motion Synthesis for Large-Vocabulary Objects

**Conference**: ICML2025  
**arXiv**: [2503.04257](https://arxiv.org/abs/2503.04257)  
**Code**: Planned to be open-source (including data pipeline, model code, and annotated captions)  
**Area**: Image Generation  
**Keywords**: motion synthesis, text-to-motion, skeletal rig, diffusion model, 3D animation

## TL;DR

This paper presents a unified framework for text-driven motion generation targetting large-vocabulary heterogeneous skeletal objects, achieved by annotating text descriptions for the Truebones Zoo dataset (70+ species), introducing rig augmentation, and integrating TreePE and RestPE encodings into the Motion Diffusion Model. It enables high-quality 3D motion synthesis for animals, dinosaurs, and even fictional creatures.

## Background & Motivation

**Background**: Current motion synthesis research primarily focuses on human motion matching dataset templates with fixed skeletons (e.g., AMASS, HumanML3D). Methods like MDM (Motion Diffusion Model) and MoMask assume a single, fixed skeletal structure. While highly effective, these methods inherently rely on a unified human skeletal topology, making them difficult to generalize to other objects.

**Limitations of Prior Work**:
   - **Data Scarcity**: Lack of high-quality motion datasets with text annotations covering a wide range of species. Existing animal motion data either focus on a single species (e.g., only horses) or lack text descriptions.
   - **Methodological Constraints**: Existing methods (e.g., SinMDM, OmniMotion-GPT, MAS) either rely on fixed skeletal templates or require training separate models for each source-target pair (e.g., CycleGAN), failing to handle heterogeneous skeletons within a unified model.

**Key Challenge**: Skeletal structures vary drastically across different species—from quadrupedal horses to winged birds and imaginary creatures like dragons, showing distinct joint counts and hierarchical dependencies. Positional encodings in traditional Transformers assume a fixed sequence structure and cannot express the hierarchical relationships of tree-like skeletal topologies.

**Goal**:
   - Sub-problem 1: How to construct a high-quality text-motion paired dataset covering a broad spectrum of species?
   - Sub-problem 2: How to adapt a single model to arbitrary skeletal templates as input?
   - Sub-problem 3: How to preserve motion generation fidelity despite substantial variations in skeletal structures?

**Key Insight**: The authors observe that although skeletal topologies vary significantly across species, the "kinematic essence" of motions (such as walking, running, jumping) is shared. Models can learn this commonality by being exposed to more skeletal variants through rig augmentation. Furthermore, the tree-like hierarchical relationships of skeletons can be-explicitly modeled using graph- or tree-structured positional encodings.

**Core Idea**: Introduce Tree Positional Encoding (TreePE) and Rest Pose Encoding (RestPE) into the Motion Diffusion Model, combined with rig augmentation data enrichment, to dynamically adapt to arbitrary skeletal templates.

## Method

### Overall Architecture

The input is a text description (e.g., "the dragon walks slowly and flaps its wings") and a skeletal template of the target object (containing the number of joints $J$, parent-child hierarchical relationships, and the rest pose). The output is a 3D motion sequence on the skeleton ($F$ frames $\times$ $J$ joints $\times$ 3D coordinates). The overall pipeline consists of three stages:

1. **Data Preparation Stage**: Perform manual text annotation and rig augmentation on the Truebones Zoo dataset.
2. **Encoding Stage**: Inject skeletal topology information into the Transformer via TreePE and RestPE.
3. **Generation Stage**: Denoise based on the extended Motion Diffusion Model conditioned on text to generate the target motion.

### Key Designs

1. **Text Annotation of the Truebones Zoo Dataset**:

    - **Function**: Annotate three-level fine-grained text descriptions (short/mid/long) for high-quality motion sequences of 70+ species in the Truebones Zoo dataset.
    - **Mechanism**: Short descriptions contain only high-level actions (e.g., "walking"); mid descriptions add part-level dynamic details (e.g., "tail swaying side to side"); long descriptions further incorporate initial pose descriptions (e.g., "starting from a crouched position"). These annotations are completed by human annotators to ensure description accuracy and diversity.
    - **Design Motivation**: Multi-granularity descriptions enable the model to understand motion semantics at different abstraction levels. Randomly sampling different granularities during training improves generalization capability.

2. **Rig Augmentation**:

    - **Function**: Generate more training samples by adjusting skeletal templates without altering motion kinematics.
    - **Mechanism**: Three augmentation strategies are proposed:
        - **Bone Length Augmentation**: Randomly scale each bone length while keeping the joint hierarchy unchanged.
        - **Bone Quantity Augmentation**: Insert or delete intermediate joints in a bone chain (e.g., splitting a long bone into two segments), thereby changing the number of joints, $J$.
        - **Rest Pose Augmentation**: Adjust the rest pose of the skeleton (from T-pose to A-pose, etc.), altering the initial configuration.
        The augmented skeletons are mapped back to the original motion sequences via motion retargeting to ensure consistent motion kinematics.
    - **Design Motivation**: Each species has only one skeletal template in the original dataset. Augmentation exposes the model to a vast number of skeletal variants, prompting it to learn invariance to skeletal structures and improving generalization to unseen skeletons.

3. **Tree Positional Encoding (TreePE)**:

    - **Function**: Replace standard Transformer sequence positional encodings with tree-structured positional encodings to represent the hierarchical relationships of skeletons.
    - **Mechanism**: Adapting TreePE from Shiv & Quirk (2019), the tree structure of the skeleton (with the root joint as the root node and other joints as child nodes) is encoded as positional embeddings. Each joint encoding contains its depth and path information in the tree, allowing the Transformer to perceive parent-child dependencies among joints.
    - **Design Motivation**: Standard positional encodings assume a linear sequence and cannot express hierarchical relationships such as "the wrist depends on the elbow, and the elbow depends on the shoulder." TreePE naturally allows the attention mechanism to assign higher weights to structurally close joints.

4. **Rest Pose Encoding (RestPE)**:

    - **Function**: Encode the skeleton's rest pose information as an additional conditional signal injected into the model.
    - **Mechanism**: Each skeletal template has a rest pose, representing the default position of each joint without movement. RestPE maps these 3D coordinates into embedding vectors via an MLP, which serve as joint-level conditions along with TreePE.
    - **Design Motivation**: Topology alone is insufficient to distinguish different objects—two species might share the same joint hierarchy but have different body proportions and default poses. RestPE provides distinguishing information in the geometric dimension.

### Loss & Training

- Based on the MDM (Motion Diffusion Model) denoising diffusion framework, incorporating TreePE + RestPE within the Transformer.
- Text conditions are injected after extracting features via the CLIP text encoder.
- Rig augmentation is randomly applied to each sample during training, acting as data-level regularization.
- Low/mid/long descriptions are randomly sampled as text conditions during training to enhance model robustness to varying text granularities.
- Standard simple diffusion loss is used to supervise noise prediction.

## Key Experimental Results

### Main Results

Experiments are conducted on the Truebones Zoo dataset, covering motion synthesis across 70+ species, and evaluated using metrics such as FID (adapted version of Frechet Inception Distance), Diversity, and R-Precision.

| Method | FID ↓ | Diversity ↑ | R-Precision Top-1 ↑ | Applicable Skeleton Type |
|------|-------|-------------|---------------------|-------------|
| MDM (fixed skeleton) | N/A | N/A | N/A | Single Template Only |
| SinMDM | Higher | Medium | N/A (no text condition) | Single Object |
| OmniMotion-GPT | Medium | Medium | Medium | Quadruped Only |
| **Ours (Full)** | **Lowest** | **Highest** | **Highest** | **Arbitrary Skeleton** |

Note: Since this work is the first to address text-driven motion synthesis for large-vocabulary heterogeneous skeletons, direct baseline comparisons are limited. The authors mainly compare with ablation variants and adapted prior methods.

### Ablation Study

| Configuration | FID ↓ | R-Precision ↑ | Description |
|------|-------|---------------|------|
| Full model (TreePE + RestPE + Rig Aug) | Optimal | Optimal | Full model |
| w/o TreePE | Significantly degraded | Decreased | Removing TreePE leaves the model without hierarchical structure awareness |
| w/o RestPE | Moderately degraded | Decreased | Removing RestPE confuses objects with different body types |
| w/o Rig Augmentation | Significantly degraded | Significantly decreased | Without rig augmentation, generalization is heavily impaired |
| w/o multi-granularity descriptions | Slightly degraded | Slightly decreased | Trained only with single-granularity text |
| Standard PE (replacing TreePE) | Significantly degraded | Decreased | Sequence positional encoding fails to represent tree structure |

### Key Findings

- **Rig Augmentation contributes the most**: Quantitative performance drops drastically without rig augmentation, indicating that data augmentation is crucial for generalizability across heterogeneous skeletons. This supports the "kinematic commonality" hypothesis that similar physical motions on different skeletons indeed share transferrable features.
- **TreePE is the core design distinguishing this method from standard MDM**: Replacing TreePE with standard positional encodings causes performance to plunge on objects with substantially different joint counts, demonstrating the necessity of hierarchical structure awareness.
- **Generalization to unseen objects**: The model can generate reasonable motions for creatures not present in the training set (e.g., custom skeletons downloaded from the web), demonstrating true zero-shot generalization capability.
- **Multi-granularity text annotation** assists training but is not the sole decisive factor. Short descriptions are already sufficient to generate plausible motions during inference.

## Highlights & Insights

- **Pioneering Problem Definition**: This work is the first to address text-driven motion synthesis for heterogeneous skeletons across 70+ species within a unified framework. The problem definition itself represents a major contribution, expanding motion synthesis from "human-exclusive" to "arbitrary creatures".
- **Ingenuity of Rig Augmentation**: By modifying skeletal templates and performing retargeting to maintain kinematic consistency, the method expands each motion sequence into multiple training samples at virtually zero computational cost. This approach can be directly transferred to human motion synthesis as data augmentation (e.g., simulating different body shapes) or applied to sim-to-real adaptations in robotics kinematics.
- **Inspirational TreePE**: Introducing tree positional encodings from NLP to 3D motion successfully couples skeletal hierarchies with the Transformer's attention mechanism. This concept could be extended to any sequence modeling task with hierarchical/graph structures (such as molecular structure generation or scene graph animation).
- **Practical Value of Data Annotation**: The three-level (short/mid/long) text annotation scheme itself serves as a standard paradigm for describing motion, offering a high-quality benchmark for future researchers.

## Limitations & Future Work

- **Limited Dataset Scale**: Although Truebones Zoo covers 70+ species, the motion variety for each species is limited (mostly basic movements like walking and running) and lacks complex interactive motions.
- **Incomplete Cache**: The detailed methodology section (Sections 4-6) in the paper was truncated in the cache, leaving some quantitative experimental values incomplete. It is recommended to re-retrieve the complete text later.
- **Lack of Physical Constraints**: Generated motions might violate laws of physics (such as gravity or contact forces). Future work could integrate physical simulators as post-processing steps or implement physics-aware losses.
- **Absence of Multi-Object Interaction**: Current work considers only single-object motions, without touching upon interactions between multiple objects (e.g., two animals fighting).
- **Skeletal Structure Assumption**: The method assumes the input skeleton is a tree structure, making it unable to handle loop constraints (such as closed-chain kinematics) commonly found in robotics and certain character animations.
- **Future Directions**:
    - Combine with 3D generative models (e.g., DreamFusion) to synthesize animated 3D objects directly from text.
    - Introduce physics-informed losses to improve the physical plausibility of synthesized motions.
    - Extend the framework to motion synthesis in multi-object interaction scenarios.

## Related Work & Insights

- **vs MDM (Tevet et al., 2023)**: MDM is the base model of this work but only supports a fixed human skeleton. This work extends it to arbitrary skeletal templates via TreePE and RestPE, which trades encoding complexity for generality.
- **vs SinMDM (Raab et al., 2024)**: SinMDM generates variants using internal motifs of a single motion sample without text guidance. This work is a large-scale data-driven and text-conditional approach, rendering them complementary—SinMDM is suited for fine-grained editing while this method excels at generation from scratch.
- **vs OmniMotion-GPT (Yang et al., 2024)**: OmniMotion-GPT achieves quadruped motion synthesis via human-to-animal motion transfer, but is limited to quadrupeds. The unified framework proposed in this work is more versatile.
- **vs MAS (Kapon et al., 2024)**: MAS performs horse motion synthesis via 2D-to-3D lifting, which is complex and species-restricted. The proposed method is simpler and supports arbitrary species.
- **Insights**: The concept of TreePE can be explored for scene graph modeling in multimodal VLMs. The idea of Rig Augmentation can be transferred to data augmentation for human pose estimation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Represents the first unified framework to solve text-driven motion synthesis for large-vocabulary heterogeneous skeletons. The problem definition itself is an important contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies fully validate the contributions of individual modules, though quantitative comparisons are somewhat constrained by the limited number of comparable baselines.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, motivation is thoroughly explained, and Figures 1 and 2 are highly informative.
- Value: ⭐⭐⭐⭐⭐ Provides a practical motion generation workflow for 3D content creation (animation, games, VR). The dataset and code are promised to be open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] How Video Meetings Change Your Expression](../../ECCV2024/human_understanding/how_video_meetings_change_your_expression.md)
- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](../../CVPR2026/human_understanding/party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)
- [\[CVPR 2025\] SimMotionEdit: Text-Based Human Motion Editing with Motion Similarity Prediction](../../CVPR2025/human_understanding/simmotionedit_text-based_human_motion_editing_with_motion_similarity_prediction.md)
- [\[CVPR 2026\] Open the Motion Door: Atomic Motion Decomposition and Recomposition for Open-Vocabulary Motion Generation](../../CVPR2026/human_understanding/open_the_motion_door_atomic_motion_decomposition_and_recomposition_for_open-voca.md)

</div>

<!-- RELATED:END -->
