---
title: >-
  [Paper Note] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios
description: >-
  [AAAI 2026][Multimodal VLM][Tactile perception] SToLa proposes the first Mixture-of-Experts (MoE)-based touch-language framework, which employs a dynamic routing mechanism to manage the modality gap between tactile and l…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Tactile perception"
  - "multimodal reasoning"
  - "mixture of experts"
  - "touch-language model"
  - "commonsense reasoning"
date: 2026-05-08
content_hash: cdb22f0509421551
---

# SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios

**Conference**: AAAI 2026
**arXiv**: [2505.04201](https://arxiv.org/abs/2505.04201)
**Code**: [Project Page](https://cocacola-lab.github.io/SToLa-Page/)
**Area**: Multimodal VLM
**Keywords**: Tactile perception, multimodal reasoning, mixture of experts, touch-language model, commonsense reasoning

## TL;DR

SToLa proposes the first Mixture-of-Experts (MoE)-based touch-language framework, which employs a dynamic routing mechanism to manage the modality gap between tactile and linguistic inputs. The work also introduces TactileBench, an open-ended tactile commonsense reasoning dataset covering 8 physical properties and 4 interaction characteristics. With only 7B parameters, SToLa achieves state-of-the-art performance on the PhysiCLeAR benchmark, surpassing the 13B Octopi model.

## Background & Motivation

### State of the Field
Touch is a fundamental sensory modality through which humans interact with the physical world, and is particularly indispensable in scenarios involving visual occlusion. In robotics and artificial intelligence, tactile sensing has been widely recognized as a critical modality for robot-environment interaction. Recent work has begun to integrate tactile signals with large language models (LLMs) to leverage their reasoning capabilities for tactile commonsense reasoning.

### Limitations of Prior Work

**Modality gap overlooked**: Existing touch-language models (e.g., Octopi, Touch-LLM) treat tactile signals as a simple "sub-modality" of language—mapping tactile data into a representation space similar to text via a tactile encoder, and then forcing both modalities to share a single Transformer architecture. This approach ignores the critical fact that even after being projected into a shared space, tactile and linguistic representations remain semantically distinct (analogous to the brain having dedicated neural pathways for tactile processing).

**Scarcity of open-ended tactile data**: Existing datasets (e.g., PhysiCLeAR) are limited in scope—covering only three physical properties (hardness, roughness, and bumpiness) and employing template-based question-answering formats. This stands in sharp contrast to real-world scenarios where question formats are unpredictable, severely limiting model generalization.

### Root Cause
The core challenge lies in how to effectively fuse two modalities with significant semantic discrepancy—tactile and language—within a unified framework while simultaneously handling the diverse, open-ended tactile reasoning problems encountered in real-world settings.

### Core Idea
The paper leverages the dynamic routing property of MoE to assign different expert networks to tokens from different modalities, enabling the model to **adaptively** distinguish and manage tactile tokens versus text tokens, rather than naively sharing all parameters. A broader open-ended tactile commonsense reasoning dataset is also constructed to support this goal.

## Method

### Overall Architecture
SToLa consists of three core components: a **tactile encoder** (processing raw tactile data), a **touch-language adapter** (bridging the modality gap), and an **LLM with MoE layers** (dynamically managing multimodal tokens). A two-stage progressive training strategy is adopted.

### Key Designs

1. **Input Unification**:

    - **Tactile signal unification**: Static tactile images and dynamic temporal sequences are handled uniformly—a single image is treated as a one-frame video.
    - **Multi-sensor support**: Both GelSight and GelSight Mini sensor configurations are supported under the same processing pipeline.
    - **Temporal aggregation**: Tactile video input $X_{touch} \in \mathbb{R}^{N \times H \times W \times C}$ is processed frame-independently by the encoder to produce frame-level token sequences, which are then average-pooled into a video-level representation $\mathcal{Z}' \in \mathbb{R}^{P \times C}$.
    - **Tactile-text concatenation**: The adapter projects tactile tokens into the LLM embedding dimension, which are then concatenated with text tokens before being fed into the LLM.
    - **Design Motivation**: Inspired by ViFi-CLIP, average pooling implicitly captures temporal patterns, enabling unified processing of diverse tactile input formats.

2. **MoE Module (Core Contribution)**:

    - **FFN replacement**: Within each Transformer block of the LLM, the standard feed-forward network is replaced with a MoE layer.
    - **Shared self-attention + expert routing**: Each MoE block retains a shared self-attention layer (applicable to both modalities), augmented with a router and FFN-based expert networks.
    - **Routing formulation**: $\mathcal{P}(\mathbf{x}) = Softmax(Top\text{-}k(x \cdot W_r, k))$, where $W_r \in \mathbb{R}^{D \times K}$ denotes the router weights.
    - **MoE output**: $\text{MoE}(x) = \sum_{i=1}^{K} \mathcal{P}_i(x) \cdot E_i(x)$, computed as the weighted sum of activated expert outputs.
    - **Design Motivation**: Different experts develop distinct preferences for tactile vs. text tokens, enabling modality-aware dynamic knowledge allocation.

3. **Two-Stage Progressive Training**:

    - **Stage I (Tactile Token Adaptation)**:
        - Only the touch-language adapter is trained; the tactile encoder and LLM are frozen.
        - Trained on tactile-language pairs from Touch100k.
        - MoE layers are **not used** at this stage.
        - Objective: enable the LLM to comprehend tactile input content.
        - Loss: cross-entropy loss $\mathcal{L}_{ce} = -\mathbb{E}\left[\log\pi_\theta(\mathcal{Y}_i|\mathcal{V},\mathcal{T}_{<i})\right]$
    - **Stage II (End-to-End MoE Fine-tuning)**:
        - The tactile encoder and word embedding layer are frozen; the adapter and LLM are fine-tuned.
        - Self-attention layers are fine-tuned via LoRA; FFN layers are converted from dense to sparse MoE via **sparse upcycling**.
        - Key detail: FFN weights from Stage I are replicated to initialize all $K$ experts.
        - Trained on the PhysiCLeAR dataset and the authors' own tactile instruction data.

### Loss & Training
- Total loss for Stage II: $\mathcal{L}_{total} = \mathcal{L}_{ce} + \mathcal{L}_{aux}$
- Auxiliary load-balancing loss: $\mathcal{L}_{aux} = \alpha \cdot K \cdot \sum_{i=1}^{K} \mathcal{F}_i \cdot \mathcal{G}_i$
    - $\mathcal{F}_i$: fraction of tokens dispatched to expert $E_i$
    - $\mathcal{G}_i$: routing probability assigned to expert $E_i$
    - Prevents token collapse to a small subset of experts, ensuring balanced expert utilization.
- Backbone LLM: Vicuna-7B v1.5
- Tactile encoder: TLV-Link (pre-aligned with the language modality)
- Training hardware: 1 × Nvidia A100-80G GPU, batch size 16

## Key Experimental Results

### Main Results

| Model | PhysiCLeAR CIDEr | PhysiCLeAR METEOR | TactileBench METEOR | TactileBench GPT-4 | TactileBench DeepSeek-R1 |
|------|------------------|-------------------|---------------------|--------------------|-----------------------|
| Touch-LLM (7B) | - | - | 17.92 | 6.88 | 7.06 |
| Octopi-7B | 138.60 | 77.63 | 21.47 | 6.91 | 7.17 |
| Octopi-13B | 141.20 | 77.79 | 28.83 | 7.85 | 7.97 |
| **SToLa (7B, Ours)** | **195.03** | **82.58** | **30.27** | **8.02** | **8.12** |

### Ablation Study

| Configuration | PhysiCLeAR CIDEr | PhysiCLeAR METEOR | TactileBench GPT-4 | Note |
|------|------------------|-------------------|--------------------|----- |
| SToLa (Full) | 195.03 | 82.58 | 8.02 | All components included |
| w/o MoE | 176.79 | 81.55 | 7.44 | MoE removed; CIDEr drops by 18.24 |
| w/o LoRA | 166.71 | 80.39 | 7.95 | LoRA removed; CIDEr drops by 28.32 |
| w/o Stage I | 172.52 | 80.55 | 7.72 | Stage I skipped; significant performance drop |

### Fine-Grained PhysiCLeAR Sub-task Results

| Model | Attribute Comparison | Attribute Superlative | Attribute-Object Matching | Attribute Scene Reasoning | Object Attribute Description (Combined) |
|------|---------|------------|-------------|------------|----------------------|
| Octopi-7B | 48.10 | 74.67 | 44.39 | 69.57 | 47.37 |
| Octopi-13B | 55.06 | 84.00 | 60.43 | 67.39 | 55.26 |
| **SToLa** | **62.28** | 74.86 | 57.32 | **69.80** | 48.72 |

### Key Findings

1. **7B surpasses 13B**: SToLa (7B) outperforms Octopi-13B on PhysiCLeAR CIDEr by 53.83 points (195.03 vs. 141.20), achieving comprehensive superiority with fewer parameters.
2. **MoE is the core contribution**: Removing the MoE module results in a CIDEr drop of 18.24 points, confirming the critical role of dynamic expert routing for multimodal management.
3. **Progressive training is indispensable**: Skipping Stage I leads to a significant performance degradation, demonstrating the necessity of first adapting the LLM to tactile inputs before introducing MoE.
4. **Experts develop modality preferences**: Routing distribution visualizations show that different experts develop clear selection preferences for tactile versus text tokens; tactile tokens tend to favor experts 3 and 4, while text tokens prefer expert 2 in shallower layers.
5. **Advantage in open-ended reasoning**: On TactileBench, which employs free-form question answering, SToLa improves over Octopi-13B by 0.17 points in GPT-4 scoring (8.02 vs. 7.85), demonstrating stronger generalization to open-ended scenarios.

## Highlights & Insights

1. **First exploration of MoE for touch-language models**: This work pioneering introduces MoE into the tactile domain, demonstrating that dynamic expert routing is more effective than simple parameter sharing.
2. **Elegant sparse upcycling design**: Multiple experts are initialized by copying the dense FFN weights from Stage I, preserving acquired knowledge while introducing diversity.
3. **TactileBench fills a critical gap**: The dataset covers 8 physical properties (hardness, roughness, weight, texture, etc.) and 4 interaction characteristics (graspability, bendability, etc.) with free-form question answering.
4. **Hierarchical cognitive design**: TactileBench distributes data according to a three-level hierarchy: basic attribute understanding (50%) → tactile interaction perception (30%) → commonsense-driven reasoning (20%).
5. **In-depth routing visualization**: PCA-based token routing visualizations clearly illustrate how the model dynamically manages tokens from different modalities.

## Limitations & Future Work

1. **Computational constraints**: Due to resource limitations, only a 7B LLM is used; scaling to 13B was not explored, leading to underperformance on certain PhysiCLeAR sub-tasks relative to Octopi-13B.
2. **MoE designed only at the modality level**: Expert allocation is not considered from a task-level or modality-task joint perspective, and more optimal designs may exist.
3. **Frozen tactile encoder**: The effect of end-to-end tactile encoder training is not explored.
4. **Limited sensor coverage**: Only the GelSight series is supported; generalization to other tactile sensors (e.g., BioTac, DIGIT) has not been validated.
5. **TactileBench relies on GPT-4 generation**: Question-answer pairs in the dataset are generated by GPT-4, which may introduce annotation bias.
6. **Limited object diversity**: TactileBench covers only 14 object categories, and diversity warrants further expansion.

## Related Work & Insights

- **Octopi**: A Vicuna-based touch-language model that introduces the PhysiCLeAR benchmark; serves as the primary baseline in this work.
- **Touch-LLM**: Aligns tactile embeddings with image embeddings via contrastive learning, but does not support interleaved processing of temporal tactile signals.
- **Switch Transformer / GLaM**: Seminal applications of MoE in language models, demonstrating the efficiency advantages of sparse activation.
- **MoE-LLaVA / Uni-MoE**: Applications of MoE in vision-language models; this work extends the same paradigm to the touch-language domain.
- **Insights**: The dynamic routing property of MoE is naturally suited to handling the heterogeneity of multimodal inputs—tokens from different semantic spaces should be processed by different experts, a principle that can be generalized to broader modality combinations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](../../CVPR2026/multimodal_vlm/seeing_through_touch_tactile_localization.md)
- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)
- [\[NeurIPS 2025\] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/mm-opera_benchmarking_open-ended_association_reasoning_for_large_vision-language.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)

</div>

<!-- RELATED:END -->
