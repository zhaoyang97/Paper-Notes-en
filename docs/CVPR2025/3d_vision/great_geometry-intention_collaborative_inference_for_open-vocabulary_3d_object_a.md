---
title: >-
  [Paper Note] GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding
description: >-
  [CVPR 2025][3D Vision][open-vocabulary affordance] This paper proposes the GREAT framework, which fine-tunes InternVL using a Multi-Head Affordance Chain-of-Thought (MHACoT) to reason about the object's geometric attributes and latent interaction intentions in interaction images, forming an affordance knowledge dictionary. It injects this knowledge into point cloud and image features through a Cross-Modal Adaptive Fusion Module (CMAFM) to achieve open-vocabulary 3D object aff…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "open-vocabulary affordance"
  - "chain-of-thought"
  - "MLLM (InternVL)"
  - "point cloud"
  - "cross-modal fusion"
  - "PIADv2 dataset"
date: 2026-05-08
content_hash: c0697711c54ca137
---

# GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding

**Conference**: CVPR 2025  
**arXiv**: [2411.19626](https://arxiv.org/abs/2411.19626)  
**Code**: [Project Page](https://yawen-shao.github.io/GREAT/)  
**Area**: 3D Vision  
**Keywords**: open-vocabulary affordance, chain-of-thought, MLLM (InternVL), point cloud, cross-modal fusion, PIADv2 dataset

## TL;DR

This paper proposes the GREAT framework, which fine-tunes InternVL using a Multi-Head Affordance Chain-of-Thought (MHACoT) to reason about the object's geometric attributes and latent interaction intentions in interaction images, forming an affordance knowledge dictionary. It injects this knowledge into point cloud and image features through a Cross-Modal Adaptive Fusion Module (CMAFM) to achieve open-vocabulary 3D object affordance grounding. Additionally, it constructs PIADv2, the largest 3D affordance dataset to date (15K images + 38K point clouds).

## Background & Motivation

**Background**: 3D object affordance grounding aims to locate regions on objects that support specific interactions (e.g., the handle of a cup can be "grasped"), acting as a bridge between robot perception and manipulation. Existing methods guide 3D affordance grounding by introducing external interaction priors via images or texts.

**Limitations of Prior Work**:
1. **Limited Semantic Space**: Existing methods (such as IAGNet, LASO, OpenAD) rely on affordance categories present in the training set and fail to generalize to unseen affordances (e.g., training on "grasp" but testing on "pour").
2. **Neglected Invariant Geometry**: Different objects sharing the same affordance often share geometric attributes (e.g., pourable objects usually feature a spout-like structure), which remains unexploited.
3. **Lack of Analogical Reasoning**: While humans can associate one interaction with other potential interactions, models lack such brainstorming capabilities.

**Key Challenge**: The dynamic and diverse nature of affordances makes purely data-driven pattern matching difficult to generalize to open-vocabulary scenarios.

**Goal**: Locate 3D object affordance regions under arbitrary instructions, with a particular focus on generalizing to unseen objects and unseen affordances.

**Key Insight**: Mimic human cognition—using the world knowledge of MLLMs to perform multi-step reasoning (CoT), excavating geometric attributes and interaction intentions to form transferable affordance knowledge.

**Core Idea**: Fine-tune an MLLM with MHACoT to mine object geometric attributes (why this part supports interaction) and interaction intentions (how else it can support interactions), and inject the reasoned knowledge into point cloud features to achieve open-vocabulary affordance grounding.

## Method

### Overall Architecture

1. **Feature Extraction**: ResNet18 extracts image features $\mathbf{F}_i$, and PointNet++ extracts point cloud features $\mathbf{F}_p$.
2. **MHACoT Reasoning**: Fine-tune InternVL to perform four-step CoT reasoning on interaction images.
3. **Knowledge Encoding and Integration**: RoBERTa encodes reasoning texts $\rightarrow$ cross-attention associates object knowledge with affordance knowledge.
4. **CMAFM**: Injects geometric knowledge into point cloud features and fuses intention knowledge into image features.
5. **Decoder**: Fuses features to output pointwise affordance heatmaps.

### Key Designs

#### 1. Multi-Head Affordance Chain-of-Thought (MHACoT)

Performs four-step reasoning across two "heads":

**Object-Head (Geometric Reasoning)**:
- Prompt 1: "Point out the part of the object in the image interacting with the human" $\rightarrow$ Locate the interaction region.
- Prompt 2: "Explain from a geometric structure perspective why this part can support interaction" $\rightarrow$ Extract geometric attributes.

**Affordance-Head (Intention Reasoning/Brainstorming)**:
- Prompt 3: "Describe the interaction process between the object and the human" $\rightarrow$ Fine-grained interaction description.
- Prompt 4: "List two other common ways this object interacts with humans" $\rightarrow$ Analogical reasoning of latent affordances.

Fine-tuning strategy: Only train standard LoRA adapters (rank=16) and freeze the backbone parameters of InternVL, for 10 epochs, with lr=4e-5.

#### 2. Knowledge Encoding and Integration

- RoBERTa encodes Object-Head output as geometric knowledge $\mathbf{T}_o \in \mathbb{R}^{N_o \times C}$
- RoBERTa encodes Affordance-Head output as intention knowledge $\mathbf{T}_a \in \mathbb{R}^{N_a \times C}$
- Cross-attention associates the two types of knowledge: $\bar{\mathbf{T}}_o = f_\delta(f_m(\mathbf{T}_o, \mathbf{T}_a))$, $\bar{\mathbf{T}}_a = f_\delta(f_m(\mathbf{T}_a, \mathbf{T}_o))$

#### 3. Cross-Modal Adaptive Fusion Module (CMAFM)

Injects geometric knowledge $\bar{\mathbf{T}}_o$ into the deepest layer of the PointNet++ encoder:
- Cross-attention: Point cloud features act as query, while geometric knowledge acts as key/value.
- Feature re-representation + pooling expansion $\rightarrow$ concatenation fusion $\rightarrow$ 1×1 convolution.
- Intention knowledge $\bar{\mathbf{T}}_a$ is directly reshaped and then concatenated with image features for fusion.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

- Focal loss: For handling positive/negative sample imbalance.
- Dice loss: For optimizing region-level overlap.
- Direct supervision on pointwise heatmaps without using affordance category labels.

## Key Experimental Results

### Main Results (PIADv2 dataset)

| Method | Seen AUC ↑ | Unseen Obj AUC ↑ | Unseen Aff AUC ↑ |
|------|-----------|-------------------|-------------------|
| Baseline | 87.04 | 72.74 | 58.09 |
| IAGNet | 89.03 | 73.03 | 62.29 |
| LASO | 90.34 | 73.32 | 64.07 |
| **GREAT** | **91.99** | **79.57** | **69.81** |

- Unseen Affordance split: GREAT's aIoU (12.05) increases by **44.0%** compared to LASO (8.37).
- Unseen Object split: AUC increases by 8.5%, and aIoU increases by 25.6%.

### Ablation Study

| Configuration | Seen AUC | Unseen Obj AUC | Unseen Aff AUC |
|------|----------|----------------|----------------|
| **Full GREAT** | **91.99** | **79.57** | **69.81** |
| ✗ AffCoT | 90.88 | 74.58 | 67.18 |
| ✗ ObjCoT | 90.13 | 75.87 | 64.69 |
| ✗ CMAFM | 89.52 | 78.42 | 63.00 |
| ✗ MLLM Fine-tuning | 88.75 | 77.83 | 66.49 |

- AffCoT contributes the most to Unseen Object (removing it drops AUC by 5 points).
- ObjCoT contributes the most to Unseen Affordance (removing it drops AUC by 5 points and aIoU by 3.2 points).
- CMAFM has the greatest impact on Unseen Affordance aIoU (dropping from 12.05 to 6.24).

### Key Findings

1. Moving from Seen to Unseen Object, and then to Unseen Affordance, the performance of all methods gradually declines, validating the difficulty of the open-vocabulary (OV) setting.
2. The brainstorming of AffCoT enables the model to reason analogically about affordances unseen during training.
3. ObjCoT directs the model to focus on geometric attributes rather than object categories, improving cross-object generalization.
4. Attention visualization: With ObjCoT, the model focuses on the knife edge or spout, whereas without ObjCoT, it attends to the entire object.
5. MLLM fine-tuning is necessary—the original InternVL does not understand the concept of affordance.

## Highlights & Insights

1. **CoT Reasoning for Affordance**: For the first time, Chain-of-Thought reasoning is introduced to the 3D affordance task to mimic the human cognitive process.
2. **Ingenious Dual-Head Design**: Object-Head addresses "why it can support interaction" (geometric level), and Affordance-Head addresses "how else it can support interactions" (intention level), which are mutually complementary and reinforcing.
3. **Significant Dataset Contribution**: PIADv2 (15K images + 38K point clouds + 24 affordances + 43 object categories) is current largest 3D affordance dataset.
4. **Independence of Affordance Category Labels**: The loss function directly supervises the heatmaps, making it naturally tailored for open-vocabulary scenarios.
5. **High Visual Interpretability**: Attention maps clearly demonstrate the impact of different CoT steps on the model's focus areas.

## Limitations & Future Work

1. MLLM inference requires running four prompts for each interaction image, resulting in low efficiency.
2. PointNet++ and ResNet18 are relatively basic feature extractors; employing stronger backbones might yield better results.
3. The images and point clouds in PIADv2 are not paired one-to-one (sampled from different instances), which may introduce pairing noise.
4. Only LoRA fine-tuning was performed on InternVL, without exploring other MLLMs (e.g., LLaVA, GPT-4V).
5. The prompt design relies on manual engineering, leaving automated prompt optimization unexplored.

## Related Work & Insights

1. **IAGNet** (Yang et al., 2023): For the first time used 2D interaction images to guide 3D affordance, acting as the direct baseline for GREAT.
2. **LASO** (Li et al., 2024): Segments afforded regions using text-conditioned affordance queries.
3. **InternVL** (Chen et al., 2024): The base MLLM fine-tuned in GREAT.
4. **Chain-of-Thought** (Wei et al., 2022): The origin of CoT reasoning.

**Insights**: Leveraging MLLMs for "knowledge mining" rather than directly predicting the final task represents an effective paradigm for utilizing large models. CoT reasoning can supply explainable intermediate reasoning steps for computer vision tasks. Affordance understanding forms a critical link between perception and manipulation, which is of paramount significance for the robotics field.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The design of MHACoT dual-head reasoning and geometry-intention collaborative inference is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three-split evaluation, detailed ablation studies, attention visualization, and dataset contribution.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, rich illustrations, and complete method descriptions.
- **Value**: ⭐⭐⭐⭐ — Open-vocabulary affordance holds direct application value for robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2026\] QueryMe: Query-Driven Open-Vocabulary 3D Object Affordances Grounding from Multimodal Evidence](../../CVPR2026/3d_vision/queryme_query-driven_open-vocabulary_3d_object_affordances_grounding_from_multim.md)

</div>

<!-- RELATED:END -->
