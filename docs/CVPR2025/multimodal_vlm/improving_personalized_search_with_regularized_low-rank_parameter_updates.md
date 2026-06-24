---
title: >-
  [Paper Note] Improving Personalized Search with Regularized Low-Rank Parameter Updates
description: >-
  [CVPR 2025][Multimodal VLM][Personalized Retrieval] This paper proposes POLAR, which applies a **rank-1 LoRA update** with regularization to the value matrix of the **last layer** of the CLIP text encoder. With only a few samples, it learns personalized concepts while retaining general knowledge, outperforming previous text-inversion-based methods by 4% to 22% on the DeepFashion2 and ConCon-Chi benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Personalized Retrieval"
  - "LoRA Fine-Tuning"
  - "Text Encoder"
  - "Catastrophic Forgetting"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: 2f2acb277d9598b6
---

# Improving Personalized Search with Regularized Low-Rank Parameter Updates

**Conference**: CVPR 2025  
**arXiv**: [2506.10182](https://arxiv.org/abs/2506.10182)  
**Code**: None  
**Area**: Multimodal VLMs  
**Keywords**: Personalized Retrieval, LoRA Fine-Tuning, Text Encoder, Catastrophic Forgetting, Vision-Language Models

## TL;DR

This paper proposes POLAR, which applies a **rank-1 LoRA update** with regularization to the value matrix of the **last layer** of the CLIP text encoder. With only a few samples, it learns personalized concepts while retaining general knowledge, outperforming previous text-inversion-based methods by 4% to 22% on the DeepFashion2 and ConCon-Chi benchmarks.

## Background & Motivation

Personalized vision-language retrieval (PerVL) aims to enable pre-trained dual-encoder models (such as CLIP) to recognize new personalized concepts (e.g., "my dog Fido") and retrieve these concepts in different contexts (e.g., "Fido catching a frisbee"). Existing methods are mainly based on **textual inversion**: learning a pseudo-text token to represent the new concept, which is inserted into the query text. This approach has two key limitations: (1) the pseudo-token affects the entire encoding process, easily disrupting the general knowledge in the language encoder; (2) the expressiveness of the concept is limited to a single input token. The core contradiction lies in: **how to learn personalized concepts with very few samples without forgetting the general knowledge of the model?** Key Insight of this work: rather than inserting pseudo-tokens at the input, it is better to directly apply minimal, regularized low-rank updates to the internal parameters of the model, injecting personalized information in the final stage of the encoding process.

## Method

### Overall Architecture

POLAR (PersOnalized Low-rank Adaptation for Retrieval) learns a rank-1 LoRA update on the value projection matrix in the last attention layer of the CLIP text encoder. A fixed vocabulary token (e.g., "sks") is used as a placeholder for the personalized concept. During training, an MSE loss is used to pull the text embeddings close to the image embeddings, while $L_2$ regularization constrains the update magnitude to preserve general knowledge. Multi-concept queries are achieved by directly summing the LoRA parameters of each concept.

### Key Designs

1. **Rank-1 Value LoRA Update**:
    - Function: Learning personalized concepts with minimal parameters
    - Mechanism: Learn a low-rank update $V'_{L,c} = V_L + B_{L,c} A_{L,c}$ for the value matrix $V_L$ in the last layer of the text encoder, where $B \in \mathbb{R}^{d \times 1}$ and $A \in \mathbb{R}^{1 \times d}$. Each concept only requires storing $2d$ parameters
    - Design Motivation: Rank-1 reflects the essential need of "learning a concept with very few samples," minimizing interference with the baseline representation. The value matrix (rather than Q/K) is chosen because experiments show that the value matrix has the most direct impact on the final representation in the last layer

2. **Structured Regularization Strategy**:
    - Function: Preventing catastrophic forgetting of general knowledge
    - Mechanism: Two constraints: (1) Apply $L_2$ regularization $\mathcal{L}_{\text{reg}} = |B_{L,c}|^2$ to $B_{L,c}$ to control the update magnitude; (2) Constrain $\|A_{L,c}\|_2 = 1$ so that $A$ only learns the directional information of "when to activate," while the update magnitude is fully controlled by the regularized $B$. The total loss is $\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{reg}}$
    - Design Motivation: Leverage the structure of the rank-1 decomposition—$A \cdot x$ can be understood as detecting the similarity of the input to the personalized direction, while $B$ controls the direction and magnitude of the update. When $BA = 0$, the encoder degrades to the original CLIP, so regularizing $B$ directly controls the degree of deviation

3. **Multi-Concept Parameter Merging**:
    - Function: Supporting queries referencing multiple personalized concepts (e.g., "Fido playing with Rex's frisbee")
    - Mechanism: Directly sum the LoRA updates of multiple concepts $V'_{L,c_1+c_2} = V'_{L,c_1} + V'_{L,c_2}$, which is equivalent to constructing a joint rank-2 update
    - Design Motivation: Parameter addition is the simplest and most effective merging strategy, leveraging the composability of low-rank updates

### Loss & Training

- **MSE Loss**: Pull together the normalized text and image embeddings: $\mathcal{L}_{\text{MSE}} = \frac{1}{N_c} \sum_i \left(\frac{\psi'_T(q_i)}{\|\psi'_T(q_i)\|_2} - \frac{\psi_I(I_i^c)}{\|\psi_I(I_i^c)\|_2}\right)^2$
- Train for 500 iterations with a learning rate of 0.001 using the Adam optimizer, converging within 50 epochs
- Backpropagation goes only through the last layer; the personalization process takes less than 1 second on a V100 GPU
- $\lambda=0.35$ on ConCon-Chi, and $\lambda=0.1$ on DeepFashion2

## Key Experimental Results

### Main Results (DeepFashion2, 5 training images)

| Method | Architecture | Context mRR | Context r@5 | Concept mRR | Concept mAP |
|------|------|------------|------------|------------|------------|
| PALAVRA | ViT-B/32 | 28.4 | 39.2 | - | - |
| SEARLE | ViT-B/32 | 21.90 | 27.15 | 25.97 | 12.74 |
| **POLAR (Ours)** | ViT-B/32 | **34.82** | **44.88** | **59.26** | **28.75** |
| SEARLE | ViT-L/14 | 27.62 | 34.12 | 32.07 | 16.17 |
| **POLAR (Ours)** | ViT-L/14 | **40.72** | **51.31** | **65.96** | **35.07** |

### Ablation Study

| Configuration | Context mRR | Concept mAP | VLM cap r@10 | Description |
|------|------------|------------|-------------|------|
| Last-layer Value only (r=1) | **51.64** | **68.71** | 52.62 | Optimal configuration |
| r=2 | 52.31 | 66.07 | 52.78 | Parameters doubled but no significant improvement |
| r=16 | 51.67 | 67.93 | 52.62 | More parameters do not help |
| All layers | 43.23 | 63.77 | 52.45 | Updating early layers disrupts general knowledge |
| Layer 1 only | 44.69 | 64.66 | 52.18 | Injecting personalized information too early performs poorly |
| Q matrix| 16.65 | 10.91 | 51.84 | Q/K updates perform very poorly |
| Prompt Tuning (1 tok) | 31.77 | 58.95 | **30.84** | Severe catastrophic forgetting |
| Textual Inversion | 42.45 | 64.71 | N/A | Context queries are weaker than POLAR |

### Key Findings

- **The location of parameter updates is crucial**: Updating only the last layer yields the best results. Updating early layers disrupts the general representations built during the encoding process
- **The Value matrix is the optimal target**: Updates to the Q and K matrices are almost ineffective (mRR of only 16%), and Output and MLP are also inferior to Value
- **VLM Caption metrics reveal forgetting**: Although Prompt Tuning is strong in concept retrieval, its VLM caption r@10 drops drastically from 52.69 to 30.84, showing severe forgetting of general knowledge; POLAR maintains 52.62, virtually unchanged
- **rank-1 is sufficient**: Increasing the rank does not bring obvious benefits, reflecting the low complexity of the "learning a single concept" task

## Highlights & Insights

- **Minimally designed strategy yields optimal results**: The "minimal update" strategy of rank-1, a single layer, and a single matrix surprisingly outperforms more complex configurations
- **Regularization leverages the LoRA structure**: Separating the roles of $A$ and $B$, where $A$ selectively activates and $B$ controls the update magnitude, provides an elegant geometric interpretation
- **New evaluation metric**: VLM Caption recall fills the gap in evaluating general knowledge preservation
- **Extremely fast personalization**: Completed in <1 second on a V100 GPU, significantly outperforming textual inversion methods that require backpropagating through the entire encoder

## Limitations & Future Work

- Only validated on the CLIP architecture; larger VLMs (e.g., LLaVA) have not been tested
- $\lambda$ needs to be tuned on the validation set, with different values used across different datasets
- Multi-concept merging may cause interference as the number of concepts increases (rank accumulation might saturate)
- Joint updates on the image encoder side have not been explored

## Related Work & Insights

- Inspired by parameter fine-tuning strategies in personalized image generation (DreamBooth, Custom Diffusion), but finding that retrieval tasks require a more conservative update strategy
- Related in concept to Perfusion (rank-1 U-net updates + key-locking), but requiring a different design for discriminative tasks
- Validates the effectiveness of LoRA at extremely low ranks ($r=1$), which holds reference value for other personalization tasks

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce parameter updates (instead of textual inversion) to personalized retrieval, with a design that is simple and effective
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies (ranks, layers, parameter types, regularization), though limited to two datasets
- Writing Quality: ⭐⭐⭐⭐⭐ Distinctly explained methodological motivations and design choices, with rigorous ablation logic
- Value: ⭐⭐⭐⭐ The 4% to 22% improvement is significant, and the method is highly lightweight and production-ready

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Vision Graph Prompting via Semantic Low-Rank Decomposition](../../ICML2025/multimodal_vlm/vision_graph_prompting_via_semantic_low-rank_decomposition.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](../../ICCV2025/multimodal_vlm/sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[CVPR 2025\] Dynamic Updates for Language Adaptation in Visual-Language Tracking](dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](../../AAAI2026/multimodal_vlm/bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)

</div>

<!-- RELATED:END -->
