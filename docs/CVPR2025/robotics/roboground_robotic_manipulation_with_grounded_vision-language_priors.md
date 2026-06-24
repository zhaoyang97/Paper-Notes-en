---
title: >-
  [Paper Note] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors
description: >-
  [CVPR 2025][Robotics][Vision-Language Grounding] Proposes RoboGround, a two-stage framework: first, a Grounded VLM (GLaMM) generates segmentation masks of target objects and placement areas from images and text instructions; second, Grounded Perceiver utilizes these masks as intermediate representations to guide the robotic policy network in execution. This achieves a 60-100% relative improvement on complex semantic manipulation tasks.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Vision-Language Grounding"
  - "Segmentation Mask"
  - "Robotic Manipulation"
  - "Grounded Perceiver"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: b20b8bc5d0aedb9e
---

# RoboGround: Robotic Manipulation with Grounded Vision-Language Priors

**Conference**: CVPR 2025  
**arXiv**: [2504.21530](https://arxiv.org/abs/2504.21530)  
**Code**: [https://robo-ground.github.io](https://robo-ground.github.io)  
**Area**: Robotic Manipulation  
**Keywords**: Vision-Language Grounding, Segmentation Mask, Robotic Manipulation, Grounded Perceiver, Zero-shot Generalization

## TL;DR

Proposes RoboGround, a two-stage framework: first, a Grounded VLM (GLaMM) generates segmentation masks of target objects and placement areas from images and text instructions; second, Grounded Perceiver utilizes these masks as intermediate representations to guide the robotic policy network in execution. This achieves a 60-100% relative improvement on complex semantic manipulation tasks.

## Background & Motivation

**Background**: Language-conditioned robotic manipulation policies (e.g., RT-1, GR-1) directly feed text instructions and images into end-to-end networks to predict actions. Such approaches perform moderately well on simple instructions ("pick up the cup") but exhibit insufficient generalization when facing instructions requiring semantic reasoning (e.g., "place the red one next to the blue bowl").

**Limitations of Prior Work**: The semantic complexity in text instructions (appearance descriptions, spatial relations, common sense reasoning) cannot be effectively understood by purely end-to-end networks. The model needs to simultaneously resolve "which object" and "how to operate," but pure language conditioning does not provide sufficient visual grounding.

**Key Challenge**: VLMs possess powerful semantic understanding capabilities but are poor at low-level action prediction; policy networks excel at precise control but lack semantic understanding. An intermediate representation with appropriate information density is required between the two.

**Goal**: To identify the optimal intermediate representation—segmentation masks—between VLM semantic understanding and policy network execution, and to design effective fusion mechanisms.

**Key Insight**: Segmentation masks provide pixel-level shape information of objects (richer than points/bounding boxes) and are sufficiently compact to be directly concatenated into image channels.

**Core Idea**: Generating segmentation masks via VLM as intermediate representations + performing mask-aware feature extraction with Grounded Perceiver = significantly enhanced semantic manipulation capabilities.

## Method

### Overall Architecture

A two-stage pipeline: Stage 1 processes images and text instructions using GLaMM (based on CLIP + LLM + pixel decoder) to output target object masks $M_o$ and target placement area masks $M_p$. Stage 2 concatenates these masks with the images and feeds them into ViTMAE to extract visual features, uses Grounded Perceiver to extract key region information via mask-guided attention, and employs a Transformer decoder to predict arm and gripper actions.

### Key Designs

1. **Grounded Perceiver**:

    - **Function**: Compresses 196 patch tokens into $3 \times k$ semantically focused tokens.
    - **Mechanism**: Designs three sets of learnable queries: global query $Q_g$ (capturing the overall scene), target object query $Q_o$ (focusing on the target), and placement area query $Q_p$ (focusing on the placement location). During attention computation, attention matrices are mask-filled using the segmentation masks—$Q_o$ is constrained to attend only to patches covered by $M_o$, and $Q_p$ to patches covered by $M_p$.
    - **Design Motivation**: Although simple channel concatenation can deliver mask information, it fails to explicitly focus the policy network's attention on target regions. Ablation studies show a synergistic effect between the two approaches: concatenation alone achieves 26%/30%, Perceiver alone achieves 22%/30%, while combining them yields 32%/32%.

2. **Multimodal Instruction Data Generation Pipeline**:

    - **Function**: Automatically constructs large-scale, diverse training data.
    - **Mechanism**: Scaled from RoboCasa to 3,526 objects across 176 categories. Three instruction types are automatically generated: (1) appearance instructions—GPT-4 extracts object attributes (color/shape/material) and CLIP embeddings sample distractors based on similarity; (2) spatial instructions—rule-based generation of relative position descriptions ($\pm 30^{\circ}$ tolerance); (3) common sense instructions—GPT-4 generates daily scene tasks.
    - **Design Motivation**: 24K demonstrations $\times$ 4.7 diverse instructions/demonstration = 112K training pairs, ensuring policy network generalization across varying semantic complexities.

3. **Masks as Intermediate Representations (vs. Points/Bounding Boxes)**:

    - **Function**: Provides the richest visual grounding information.
    - **Mechanism**: Images, target masks, and placement masks are mapped to 3 channels via a linear layer and fed into ViTMAE. Masks are binary, with each pixel explicitly marking whether it belongs to the target region.
    - **Design Motivation**: Ablation comparison demonstrates that Masks > Bounding Boxes > Point prompts (success rates of 42% > 38% > 32%), as masks preserve precise shape and size information of objects.

### Loss & Training

VLM fine-tuning utilizes segmentation loss $\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{DICE}$ and text autoregressive cross-entropy. The policy network is trained with $\mathcal{L}_{total} = \text{SmoothL1}(\hat{a}_{arm}, a_{arm}) + \text{BCE}(\hat{a}_{gripper}, a_{gripper})$. After VLM fine-tuning, the mIoU increases from 13.2% (zero-shot) to 48.2%.

## Key Experimental Results

### Main Results

Simulation pick-and-place tasks (contact rate/success rate):

| Method | Easy | Appearance Reasoning | Spatial Reasoning | Common Sense Reasoning |
|------|------|---------|---------|---------|
| ACT | 47.3/18.3 | 18.5/3.8 | 17.5/3.5 | 15.3/2.8 |
| GR-1 | 85.3/42.8 | 49.5/13.8 | 54.5/16.3 | 43.0/11.5 |
| **RoboGround** | **89.0/43.3** | **78.5/30.5** | **81.0/33.5** | **76.3/30.0** |

Zero-shot generalization (unseen object instances / unseen categories):

| Setting | w/ Mask | w/o Mask | Gain |
|------|--------|---------|------|
| Unseen Instances (Appearance) | 75.5/29.5 | 38.0/11.5 | +100% |
| Unseen Categories (Appearance) | 68.5/14.3 | 27.5/5.3 | +170% |

### Ablation Study

| Intermediate Representation | Easy Success Rate | Appearance Success Rate |
|---------|-----------|-----------|
| None | 24% | 12% |
| Point Prompt | 32% | 26% |
| Bounding Box | 38% | 30% |
| **Segmentation Mask** | **42%** | **32%** |
| GT Mask (Upper Bound) | 68% | 48% |

### Key Findings
- **Masks are the optimal intermediate representation**: Success rate is 4-12% higher than boxes and 6-10% higher than points. Shape information is particularly critical for grasping.
- **Large gap between GT masks and predicted masks**: 68% vs 42% (easy), indicating that VLM mask quality remains a major bottleneck.
- **Contact rate $\gg$ Success rate**: 89% contact but only 43% success, indicating that precise grasping after contact remains a challenge.
- **Huge impact of data diversity**: Training only on simple data yields only a 6% success rate on reasoning tasks, which rises to 30% after incorporating diverse instructions.
- **Synergy between Grounded Perceiver and channel concatenation**: Both perform similarly when used individually, but combining them yields a significant performance boost.

## Highlights & Insights

- **Hierarchical experiments on intermediate representations**: Systematically compares performance gradients from points $\rightarrow$ bounding boxes $\rightarrow$ masks $\rightarrow$ GT masks, clearly showing the positive correlation between visual grounding information density and manipulation performance, serving as a quantitative reference for future work.
- **Attention masking mechanism in Grounded Perceiver**: Uses object masks to guide attention queries to focus solely on target regions, exploiting spatial priors more effectively than simple concatenation.
- **Scalability of data generation**: The hybrid data generation pipeline using GPT-4 + CLIP + rules scales to new scenes and objects at low cost.

## Limitations & Future Work

- **Grasping precision bottleneck**: The massive gap between contact rate (89%) and success rate (43%) indicates that precise grasp posing remains a major challenge, which could be integrated with specialized grasping models such as AnyGrasp.
- **Single-step mask extraction**: Masks are extracted once at the beginning of the episode, which fails to handle dynamic scenes or changes in object positions during multi-step tasks.
- **Simulation-only experiments**: Not validated on physical robots; sim-to-real domain gaps exist.
- **Limited VLM mask quality**: After fine-tuning, the mIoU is only 48.2%; the upper-bound performance with GT masks is significantly higher than with predicted masks.
- **Insufficient diversity in placement regions**: Diversity of placement targets in the dataset is overlooked, limiting placement precision.

## Related Work & Insights

- **vs GR-1**: GR-1 uses pure language conditioning, achieving only 11-16% success rates on complex reasoning tasks. RoboGround boosts this to 30-33% through mask grounding, proving the necessity of visual grounding.
- **vs End-to-End VLA (e.g., OpenVLA)**: End-to-end methods fail to provide fine-grained visual grounding, whereas RoboGround’s two-stage design provides stronger semantic understanding while maintaining flexibility.
- **vs SoM/Set-of-Mark**: SoM marks objects using bounding boxes, whereas RoboGround uses segmentation masks to provide more precise shape details.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of masks as intermediate representations + Grounded Perceiver is effective and convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive ablation comparisons among multiple representations, zero-shot generalization, and data diversity analysis are highly thorough.
- Writing Quality: ⭐⭐⭐⭐ Structurally clear with detailed descriptions of the data generation pipeline.
- Value: ⭐⭐⭐⭐ Provides systematic experimental evidence for selected intermediate representations between VLMs and policy networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[CVPR 2026\] INSIGHT Bench: Towards Grounded IN-SItu Guidance for Robotic Manipulation](../../CVPR2026/robotics/insight_bench_towards_grounded_in-situ_guidance_for_robotic_manipulation.md)
- [\[CVPR 2026\] Learning Surgical Robotic Manipulation with 3D Spatial Priors](../../CVPR2026/robotics/learning_surgical_robotic_manipulation_with_3d_spatial_priors.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
