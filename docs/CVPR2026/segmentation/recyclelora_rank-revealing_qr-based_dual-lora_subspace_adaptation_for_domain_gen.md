---
title: >-
  [Paper Note] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Domain Generalized Semantic Segmentation] This paper proposes RecycleLoRA, which employs Rank-Revealing QR (RRQR) decomposition to systematically "recycle" subspace structures from pretrained Vi…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Domain Generalized Semantic Segmentation"
  - "LoRA"
  - "RRQR Decomposition"
  - "Dual Adapter"
  - "Subspace Structure"
date: 2026-05-08
content_hash: d20d81ef0ea358ad
---

# RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.28142](https://arxiv.org/abs/2603.28142)
**Code**: [https://github.com/chanseul01/RecycleLoRA.git](https://github.com/chanseul01/RecycleLoRA.git)
**Area**: Semantic Segmentation / Domain Generalization / Parameter-Efficient Fine-Tuning
**Keywords**: Domain Generalized Semantic Segmentation, LoRA, RRQR Decomposition, Dual Adapter, Subspace Structure

## TL;DR

This paper proposes RecycleLoRA, which employs Rank-Revealing QR (RRQR) decomposition to systematically "recycle" subspace structures from pretrained Vision Foundation Model weights. By initializing a primary adapter from minor directions and a secondary adapter from major directions, the method substantially improves LoRA representational diversity and parameter utilization efficiency, achieving state-of-the-art performance on both synthetic-to-real and real-to-real domain generalized semantic segmentation benchmarks (average mIoU of 68.95 / 72.10).

## Background & Motivation

1. **Background**: Domain Generalized Semantic Segmentation (DGSS) aims to enable models to maintain robust performance on unseen target domains. With the advent of Vision Foundation Models (VFMs) such as DINOv2 and CLIP, the focus of DGSS has shifted from data augmentation toward efficiently adapting the rich multi-domain knowledge encoded in VFMs.
2. **Limitations of Prior Work**:
   - Existing SVD-based methods (e.g., SoMA) achieve reasonable results by attending to minor singular value directions, yet SVD prioritizes variance preservation and is not necessarily the most effective decomposition for downstream adaptation;
   - SoMA adapts only minor directions while completely freezing major directions, limiting the model's capacity to handle complex novel tasks;
   - Many LoRA methods suffer from **representational redundancy** among basis vectors, resulting in low parameter utilization efficiency (effective rank far below the target rank).
3. **Key Challenge**: How to simultaneously address "better exploitation of VFM subspace structures" and "enhanced LoRA representational diversity."
4. **Goal**: (1) Identify a decomposition strategy more suitable for VFM adaptation than SVD; (2) eliminate representational redundancy among LoRA basis vectors; (3) fully leverage both major and minor directions in pretrained weights.
5. **Key Insight**: RRQR greedily selects the most informative columns from the original weight matrix via column pivoting, naturally guaranteeing directional independence and structural diversity.
6. **Core Idea**: Decompose VFM weights via RRQR, initialize a primary adapter from minor directions and a secondary adapter from major directions, constructing a complementary dual-adapter structure without additional regularization.

## Method

### Overall Architecture

Given the pretrained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ of a VFM (DINOv2-Large), RRQR decomposition is applied to each linear layer to obtain an orthogonal matrix $\mathbf{Q}$ and a permutation matrix $\mathbf{P}$. The decomposition results are used to initialize the primary adapter (minor directions) and secondary adapter (major directions), with Mask2Former serving as the segmentation head. At inference, both adapters are merged into the original weights, **introducing no additional inference latency**.

### Key Designs

1. **RRQR Decomposition Initialization Strategy**:

   - **Function**: Provides structurally diverse, directionally independent initialization for LoRA adapters.
   - **Mechanism**: The weight matrix is factorized as $\mathbf{W}_0 \mathbf{P} = \mathbf{Q}\mathbf{R}$. At each step, RRQR selects the column with the largest norm after orthogonal projection, inherently minimizing redundancy. The columns of $\mathbf{Q}$ supply orthogonal basis directions, while $\mathbf{P}$ records the importance ordering. The primary adapter matrix $\mathbf{B}$ is initialized to the last $r$ columns of $\mathbf{Q}$ (minor directions), and $\mathbf{A}$ adopts a sparse initialization setting selected column indices to 1.
   - **Design Motivation**: Unlike SVD, which seeks new globally variance-maximizing orthogonal bases, RRQR directly selects from the original weight columns, preserving local structural information and dimensional correspondence. Post-training analysis confirms that the column norms at sparse initialization positions are on average 1.22× higher (up to 1.63×) than those at zero-initialized positions, indicating that the initialization bias is maintained throughout training.

2. **Complementary Dual-Adapter Structure**:

   - **Function**: Enables the primary and secondary adapters to learn representations from distinct subspaces, achieving complementarity without additional regularization.
   - **Mechanism**: The primary adapter uses RRQR minor directions (rank=32, lr=1e-4); the secondary adapter uses major directions (rank=4, lr=5e-5). A key finding is that initialization directions are intrinsically linked to optimal learning rates — major directions encode core VFM knowledge and require more conservative updates, while minor directions provide a safer learning space amenable to more aggressive adaptation.
   - **Design Motivation**: Grassmann distance analysis confirms that after training, the two adapters maintain near-orthogonal subspaces (similarity far lower than that of dual adapters with Kaiming initialization). PCA visualizations further reveal complementary feature modification patterns — the primary adapter focuses on foreground objects while the secondary adapter covers background regions.

3. **Effective Rank Enhancement**:

   - **Function**: Improves the representational capacity of LoRA under a limited parameter budget.
   - **Mechanism**: RRQR's greedy selection ensures directional independence among basis vectors, directly increasing effective rank. At rank=16, RecycleLoRA achieves a Rank Efficiency of 0.850 versus 0.611 for SoMA; at rank=32, the values are 0.770 and 0.650, respectively. Cosine similarity heatmaps also show that inter-row similarity of $\mathbf{A}$ and inter-column similarity of $\mathbf{B}$ in RecycleLoRA are substantially lower than those of SoMA.
   - **Design Motivation**: Higher effective rank implies that each low-rank component captures more independent and distinctive features, directly benefiting domain generalization performance.

### Loss & Training

- Standard semantic segmentation losses are employed; the VFM backbone is frozen, and only the dual adapters are trained.
- To preserve the initial output of pretrained weights, a residual matrix (original weights minus the initial adapter values) is constructed and frozen.
- Learning rates for the primary and secondary adapters are 1e-4 and 5e-5 (learning rate multipliers of 1.0 and 0.5), respectively.

## Key Experimental Results

### Main Results

**Synthetic-to-Real Generalization (GTAV → Real Domains)**

| Method | Backbone | →Citys. | →BDD | →Map. | Avg. |
|--------|----------|---------|------|-------|------|
| SoMA (CVPR'25) | DINOv2-L | 71.82 | 61.31 | 71.67 | 68.27 |
| MFuser (CVPR'25) | EVA02-L | 70.19 | 63.13 | 71.28 | 68.20 |
| **RecycleLoRA** | DINOv2-L | **73.01** | **61.77** | **72.07** | **68.95** |

**Real-to-Real Generalization (Cityscapes → BDD/Map.)**

| Method | →BDD | →Map. | Avg. |
|--------|------|-------|------|
| SoMA | 67.02 | 76.45 | 71.74 |
| MFuser | 65.81 | 77.93 | 71.87 |
| **RecycleLoRA** | 66.65 | **77.54** | **72.10** |

### Ablation Study

| Configuration | Params | →Citys. | →BDD | →Map. | Avg. |
|---------------|--------|---------|------|-------|------|
| Sub Adapter only | 1.6M | 70.64 | 60.56 | 71.11 | 67.44 |
| Main Adapter only | 12.6M | 72.92 | 61.22 | 71.75 | 68.63 |
| Main + Sub (full) | 14.2M | **73.01** | **61.77** | **72.07** | **68.95** |

### Key Findings

- **The Main Adapter alone already surpasses all existing SOTA** (68.63 vs. SoMA 68.27), demonstrating the standalone efficacy of the RRQR initialization strategy.
- The Sub Adapter provides an additional gain of 0.32 mIoU with only 1.6M extra parameters, confirming the complementarity of the dual-adapter design.
- Learning rate analysis validates the hypothesis: the Sub Adapter (major directions) performs best at the lower learning rate of 5e-5 (↑1.81), while the Main Adapter (minor directions) performs best at the standard rate of 1e-4, with a lower rate causing a drop of 1.99 mIoU.
- The method also generalizes effectively to the EVA02-L backbone (66.35 avg), demonstrating its versatility.

## Highlights & Insights

- **The insight of replacing SVD with RRQR is particularly elegant**: SVD seeks new variance-maximizing bases, whereas RRQR directly selects the most informative directions from the original columns, preserving local structural information. This "recycling" paradigm is more suited to LoRA adaptation than "reconstruction."
- **The intrinsic link between initialization direction and learning rate** is a broadly transferable finding — major directions require conservative updates while minor directions can sustain aggressive adaptation, a principle that can guide hyperparameter design in other PEFT methods.
- **Complementary learning without additional regularization**: relying solely on initialization and learning rate differences, the two adapters operate in near-orthogonal subspaces, avoiding the complexity of explicit orthogonality constraints.

## Limitations & Future Work

- Validation is limited to DINOv2 and EVA02; additional VFMs (e.g., SAM, SigLIP) remain untested.
- The computational overhead of RRQR decomposition itself is not discussed in the paper.
- The rank selection for the Sub Adapter (4 vs. 2) is set manually; adaptive rank allocation could be explored.
- The method has not been combined with text-based information from VLM approaches (e.g., MFuser), which may yield further performance gains.

## Related Work & Insights

- **vs. SoMA**: Both methods use matrix decomposition to initialize LoRA, but SoMA applies SVD and attends only to minor directions, whereas RecycleLoRA uses RRQR to leverage both major and minor directions, achieving 40% higher effective rank.
- **vs. PiSSA**: PiSSA initializes from SVD principal directions; the secondary adapter in RecycleLoRA is analogous but uses RRQR, and is further complemented by the primary adapter initialized from minor directions.
- **vs. Rein**: Rein refines features layer-by-layer with learnable tokens, whereas RecycleLoRA operates through LoRA adapters, offering greater parameter efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of replacing SVD with RRQR is novel, though the dual-adapter structure extends existing paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Synthetic-to-real, real-to-real, multi-source training, multiple backbones, and detailed ablations are all covered.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly derived; analyses (effective rank, Grassmann distance, PCA visualization) are rich and convincing.
- **Value**: ⭐⭐⭐⭐ — Makes a substantive contribution to understanding PEFT initialization strategies; the method is concise and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[CVPR 2026\] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)
- [\[AAAI 2026\] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation](../../AAAI2026/segmentation/do_we_need_perfect_data_leveraging_noise_for_domain_generalized_segmentation.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)

</div>

<!-- RELATED:END -->
