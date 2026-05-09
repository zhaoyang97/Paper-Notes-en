---
title: >-
  [Paper Note] One-Shot Knowledge Transfer for Scalable Person Re-Identification
description: >-
  [ICCV 2025][Human Understanding][Knowledge Transfer] This paper proposes OSKT (One-Shot Knowledge Transfer), which distills teacher model knowledge into a compact intermediate representation termed a "weight chain," enabling the generation of student models of arbitrary sizes for person re-identification with a single round of computation.
tags:
  - ICCV 2025
  - Human Understanding
  - Knowledge Transfer
  - Weight Chain
  - Person Re-Identification
  - Model Compression
  - One-Shot Computation
date: 2026-05-08
content_hash: a59a96f12afbf931
---

# One-Shot Knowledge Transfer for Scalable Person Re-Identification

**Conference**: ICCV 2025
**arXiv**: [2511.06016](https://arxiv.org/abs/2511.06016)
**Code**: N/A
**Area**: Person Re-Identification / Model Compression
**Keywords**: Knowledge Transfer, Weight Chain, Person Re-Identification, Model Compression, One-Shot Computation

## TL;DR

This paper proposes OSKT (One-Shot Knowledge Transfer), which distills teacher model knowledge into a compact intermediate representation termed a "weight chain," enabling the generation of student models of arbitrary sizes for person re-identification with a single round of computation.

## Background & Motivation

- Deploying person re-identification (ReID) models on edge devices requires compact models adapted to diverse resource constraints.
- Conventional compression methods (e.g., knowledge distillation, pruning) require independent training pipelines for each target model size, resulting in computational costs that scale linearly with the number of target models.
- Existing approaches such as weight selection and learngene either lack the flexibility to generate densely sampled model sizes or are computationally inefficient.
- **Core Problem**: Can one obtain student models of varying sizes suited to diverse resource scenarios through only a single round of computation?

## Method

### Overall Architecture

OSKT operates in two stages: (1) distilling a weight chain from the teacher model; and (2) expanding the weight chain into student models of arbitrary sizes without additional computation. The framework is compatible with both CNN (ResNet) and ViT architectures.

### Key Designs

1. **Unified CNN/ViT Representation**: Convolutional filters in CNNs and weight matrix rows in ViTs are uniformly abstracted as "rows," while corresponding channels/columns are abstracted as "columns." Teacher model parameters are denoted as $\{\mathcal{F}_l \in \mathbb{R}^{N_l \times N_{l-1} \times O_l}, 1 \leq l \leq L\}$, where $N_l$ is the output feature dimension and $O_l$ is the number of operation parameters ($K \times K$ for convolutional layers and $1$ for fully connected layers).

2. **Weight Chain Design**: The weight chain retains the same depth as the teacher model but with substantially reduced width, parameterized as $\{\mathcal{F}^C_l \in \mathbb{R}^{M_l \times N_{l-1} \times O_l}\}$, where $M_l \ll N_l$. Key design principles include:

    - Normalization layer parameters are shared with the teacher model, enabling multiple $(\gamma, \beta)$ pairs to reuse core feature dimensions.
    - The design is grounded in three core insights: (a) efficient knowledge transfer leveraging teacher weights; (b) neural identity transformation — merging identical rows and summing the corresponding columns preserves network functionality; (c) the weight chain serves as a bridge between the teacher and student models.

3. **Weight Chain Refinement**:

    - **Initialization**: Weight rows at each layer of the teacher model are clustered, and cluster centers are used to initialize the corresponding rows of the weight chain (Euclidean distance for CNN; cosine distance for ViT).
    - **Progressive Refinement**: The teacher model and the smallest student model (S-Student, with width equal to the weight chain width) are jointly trained, with gradients back-propagated through the weight chain.
    - Refinement loss: $\mathcal{L}_{ref} = \frac{1}{L}\sum_l \frac{1}{M_l}\sum_j \sum_{k \in \mathcal{I}_l^{(j)}}(\mathcal{F}_{l,k} - \mathcal{F}^C_{l,j})^2$
    - Total loss: $\mathcal{L} = \mathcal{L}_T + \mathcal{L}_S + \alpha \mathcal{L}_{ref}$, where both teacher and student use ID loss + Triplet loss.
    - In ViT, $\alpha$ is a progressive coefficient $\alpha = \frac{iter}{n\_iter}$; in CNN, $\alpha$ is fixed at $1$.

4. **Student Model Generation (O(1) Operation)**: Given a target of $C_l$ rows at layer $l$ (where $M_l \leq C_l \leq N_l$), refined rows of the weight chain are stacked to $C_l$ rows in proportion to the teacher's row counts, the column weights of the subsequent layer are merged accordingly (via summation), and $(\gamma, \beta)$ pairs of the normalization layers are averaged. **No additional computation is required.**

### Loss & Training

- Teacher model: $\mathcal{L}_T = \mathcal{L}_{id}(\boldsymbol{p}^T) + \mathcal{L}_{tri}(\boldsymbol{f}^T)$ (ID loss + hard triplet loss)
- S-Student: $\mathcal{L}_S = \mathcal{L}_{id}(\boldsymbol{p}^S) + \mathcal{L}_{tri}(\boldsymbol{f}^S)$ (gradients back-propagated through the weight chain)
- Refinement loss: $\mathcal{L}_{ref}$ (MSE to tighten cluster centers)
- Layers with residual connections are jointly clustered to align feature dimensions.

## Key Experimental Results

### Main Results (Tables)

**ResNet50 → Student Models (Market1501, Single-Scene):**

| Method | Res-50-S1 mAP | Res-50-S1 R1 | Res-50-S3 mAP | Res-50-S3 R1 | Res-50-S5 mAP | Res-50-S5 R1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Scratch | 30.6 | 53.3 | 47.4 | 70.9 | 60.7 | 80.9 |
| WTSel | 48.5 | 72.3 | 63.0 | 81.5 | 84.1 | 93.0 |
| KD++ | 41.5 | 65.7 | 59.9 | 80.4 | 71.4 | 86.9 |
| DepGraph | 61.3 | 80.7 | 83.2 | 92.7 | 86.3 | 94.3 |
| **OSKT** | **75.7** | **89.4** | **84.7** | **93.3** | **87.6** | **94.5** |

**ViT-B/ViT-S → Student Models (Market1501, Single-Scene):**

| Method | ViT-S-S1 | ViT-S-S2 | ViT-B-S1 | ViT-B-S2 |
|:---:|:---:|:---:|:---:|:---:|
| Scratch | 13.9/23.9 | 18.3/31.2 | 22.2/37.4 | 21.9/36.3 |
| DepGraph | 56.5/74.1 | 69.0/84.2 | 15.3/30.1 | 81.5/91.7 |
| **OSKT** | **74.2/87.1** | **77.2/89.0** | **81.6/91.9** | **82.9/92.4** |

### Ablation Study (Tables)

**Key Component Ablation (Market1501 + MSMT17→CUHK03):**

| Setting | Single-Scene Res-S1 | Single-Scene Res-S2 | Cross-Scene Res-S1 | Cross-Scene Res-S2 |
|:---:|:---:|:---:|:---:|:---:|
| Scratch | 30.6/53.3 | 41.5/65.1 | 6.1/4.8 | 9.1/6.7 |
| (a) Random Teacher | 54.7/76.5 | 58.2/78.9 | 28.2/28.2 | 30.8/30.9 |
| (b) Random Clustering | 55.1/76.2 | 59.9/80.2 | 31.4/33.8 | 34.9/35.4 |
| (e) w/o Refinement | 52.2/73.3 | 59.8/79.3 | 21.6/20.9 | 29.4/29.0 |
| (f) w/o $\mathcal{L}_T$ | 62.7/81.2 | 64.0/81.5 | 34.2/35.1 | 35.4/37.3 |
| **OSKT** | **75.7/89.4** | **77.6/90.6** | **45.7/47.3** | **49.2/52.2** |

### Key Findings

- OSKT yields the largest gains on the smallest student models: on Res-50-S1, mAP surpasses DepGraph by **14.4 points** (75.7 vs. 61.3).
- Cross-scene transfer performance is substantial: on MSMT17→CUHK03, Res-50-S1 mAP exceeds DepGraph by 14.4 points.
- Weight refinement is critical: removing the refinement step (setting e) results in a significant performance drop (75.7→52.2).
- The framework integrates well with lightweight ReID architectures (OSNet, MSINet), consistently yielding improvements.
- Advantages are maintained under few-shot settings (10%/30%/50% identities).

## Highlights & Insights

- The concept of the "weight chain" as a knowledge carrier is novel — it is essentially a clustering-based weight sharing scheme that implicitly trains all intermediate models by "lifting both ends of a bead chain."
- Student model generation is an O(1) operation, truly realizing "compute once, deploy anywhere."
- The mathematical property of neural identity transformation (merging identical rows and summing corresponding columns) guarantees functional equivalence.
- The unified framework for both CNN and ViT architectures offers significant practical engineering value.

## Limitations & Future Work

- The width of the weight chain (i.e., the minimum student model size) must be specified in advance, constraining the lower bound of generated model sizes.
- The choice of clustering distance metric is architecture-sensitive (Euclidean for CNN, cosine for ViT); swapping the metrics degrades performance.
- Validation is currently limited to ReID tasks; generalization to other vision tasks (detection, segmentation) remains unexplored.
- Averaging normalization parameters is an approximation that may introduce errors under extreme compression ratios.

## Related Work & Insights

- Compared to weight reuse methods such as Net2Net, Weight Selection, and Learngene, OSKT demonstrates superior capability in generating densely sampled model sizes and greater knowledge transfer efficiency.
- The clustering-based refinement strategy for the weight chain may inspire broader research on using model weights as knowledge carriers.
- The framework has significant implications for rapid model customization in edge computing scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The weight chain concept is highly original; the idea of generating models of arbitrary sizes from a single computation is groundbreaking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across CNN/ViT, single-scene/cross-scene, few-shot settings, and lightweight architecture compatibility.
- **Writing Quality**: ⭐⭐⭐⭐ The three core insights are articulated clearly, though the dense mathematical notation raises the reading threshold slightly.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a practical pain point in edge deployment; the framework is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ICCV 2025\] Controllable and Expressive One-Shot Video Head Swapping](controllable_and_expressive_one-shot_video_head_swapping.md)
- [\[NeurIPS 2025\] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](../../NeurIPS2025/human_understanding/k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [\[CVPR 2026\] Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification](../../CVPR2026/human_understanding/vision-language_attribute_disentanglement_and_reinforcement_for_lifelong_person_.md)

</div>

<!-- RELATED:END -->
