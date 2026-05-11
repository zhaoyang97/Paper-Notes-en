---
title: >-
  [Paper Note] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation
description: >-
  [CVPR2026][Medical Imaging][Continual Test-Time Adaptation] This paper proposes the SPEGC framework, which combines semantic-prompt-enhanced feature representations with a differentiable graph clustering solver to refine raw similarity matrices into higher-order structural representations. These representations guide the adaptation of medical image segmentation models to continuously shifting target domains, effectively mitigating error accumulation and catastrophic forgetting.
tags:
  - CVPR2026
  - Medical Imaging
  - Continual Test-Time Adaptation
  - Graph Clustering
  - Semantic Prompt
  - Optimal Transport
  - Domain Shift
  - Retinal/Polyp Segmentation
date: 2026-05-08
content_hash: 959c61e475018dd6
---

# SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation

**Conference**: CVPR2026
**arXiv**: [2603.11492](https://arxiv.org/abs/2603.11492)
**Code**: [Jwei-Z/SPEGC-for-MIS](https://github.com/Jwei-Z/SPEGC-for-MIS)
**Area**: Medical Image Segmentation
**Keywords**: Continual Test-Time Adaptation, Graph Clustering, Semantic Prompt, Optimal Transport, Domain Shift, Retinal/Polyp Segmentation

## TL;DR

This paper proposes the SPEGC framework, which combines semantic-prompt-enhanced feature representations with a differentiable graph clustering solver to refine raw similarity matrices into higher-order structural representations. These representations guide the adaptation of medical image segmentation models to continuously shifting target domains, effectively mitigating error accumulation and catastrophic forgetting.

## Background & Motivation

**Domain shift challenges in clinical deployment**: Variations in acquisition devices, operators, and scanning protocols cause significant performance degradation in pre-trained models when deployed in new target domains, rendering them unsuitable for direct clinical use.

**CTTA better reflects real-world conditions**: Conventional TTA assumes a static target domain, whereas real clinical data arrives as a continuously streaming distribution. Continual Test-Time Adaptation (CTTA) is therefore more practically relevant.

**Existing CTTA methods rely on unreliable supervision signals**: Entropy minimization and pixel-level/instance-level signal-based methods tend to produce misleading gradients under severe domain shift, triggering a vicious cycle of self-reinforcing error accumulation.

**Limited expressiveness of prompt-based methods**: Methods that freeze the backbone and learn only lightweight prompts in the input space leave core parameters unchanged, resulting in a limited performance ceiling.

**Local features are sensitive to noise**: Under domain shift, local features of unlabeled test samples are highly susceptible to noise and style variations, making directly computed similarity matrices unreliable.

**Lack of higher-order structural supervision**: Existing methods do not fully exploit intra-data cluster-level structural information to guide adaptation, preventing decision boundaries from being dynamically adjusted.

## Method

### Overall Architecture

SPEGC consists of two core modules: **Semantic Prompt Feature Enhancement (SPFE)** and a **Differentiable Graph Clustering Solver (DGCS)**. The pipeline proceeds as follows:

1. A ResNet backbone extracts local features; MC Dropout estimates uncertainty and samples low-uncertainty foreground nodes.
2. SPFE injects global contextual information into local features via decoupled commonality/heterogeneity prompt pools.
3. Enhanced features are enqueued to construct a pseudo mini-batch, from which a global similarity matrix is computed.
4. DGCS reformulates edge sparsification as an optimal transport problem and end-to-end refines the similarity matrix.
5. The refined structural representation jointly guides model adaptation via a graph consistency loss and a clustering loss.

### Key Designs

**SPFE — Semantic Prompt Feature Enhancement**:

- Attentive pooling aggregates node features into a global query $\hat{q}_i$.
- **Heterogeneity prompt pool** $P_{HE}$: Standard Softmax attention retrieves domain-specific information matching the query, capturing class-discriminative patterns.
- **Commonality prompt pool** $P_{CO}$: Reverse attention (negative match scores truncated by ReLU) retrieves cross-domain shared semantics that do not match the query, preserving core discriminative knowledge.
- Both prompts are added as decoupled contextual biases to the original node features: $V_i^* = V_i + p_{CO}(i) + p_{HE}(i)$

**DGCS — Differentiable Graph Clustering Solver**:

- Learnable projections $W_q, W_k$ compute the global similarity matrix $S$ without Softmax, preserving high-confidence signals.
- A directed edge similarity matrix $S'$ is constructed based on node density $D(v_i)$ and Sigmoid gating.
- Core insight: A spanning forest with $Z$ connected components contains exactly $k = V - Z$ edges, which sets the global sparsification budget.
- Edge selection is formulated as a binary optimal transport problem and solved iteratively via the Sinkhorn algorithm to obtain the entropy-regularized transport plan $\Gamma^*$.
- The second column of $\Gamma^*$ is reshaped into the refined edge similarity matrix $S^\star$.

### Loss & Training

$$L = L_G + \lambda L_C$$

- **Graph consistency loss** $L_G$: For any two nodes that are structurally similar in $S^\star$, their semantic predictions are enforced to be consistent (KL divergence with stop-gradient).
- **Clustering loss** $L_C$: Constrains the commonality prompt pool by encouraging the commonality prompts of all images within a batch to be close in semantic space (cosine distance), explicitly preserving cross-domain shared knowledge.
- $\lambda = 0.2$

## Key Experimental Results

### Datasets & Setup

- **Retinal fundus segmentation** (OD/OC): Five public datasets (RIM-ONE, REFUGE, ORIGA, REFUGE-Test, Drishti-GS), evaluated under cross-domain settings.
- **Polyp segmentation**: Four public datasets (BKAI-IGH, CVC-ClinicDB, ETIS, Kvasir).
- Backbone: ResNet-50 + ResUNet-50, ImageNet pre-trained.
- Online single-sample adaptation, label-free, on a single NVIDIA 3090 GPU.

### Main Results

| Method | OD/OC Avg. DSC | Polyp Avg. DSC |
|--------|:-:|:-:|
| No Adapt | 72.75 | 71.49 |
| SAR (ICLR'23) | 73.44 | 69.21 |
| VPTTA (CVPR'24) | 73.40 | 73.40 |
| NC-TTT (CVPR'24) | 79.23 | 75.44 |
| GraTa (AAAI'25) | 78.66 | 76.24 |
| TTDG (CVPR'25) | 82.88 | 76.20 |
| **SPEGC (Ours)** | **84.37** | **78.27** |

### Ablation Study

| Configuration | Avg. DSC |
|---------------|:-:|
| No Adapt (baseline) | 72.75 |
| + Graph clustering | 74.64 |
| + MC Dropout uncertainty sampling | 76.52 |
| + Heterogeneity prompt only (unconstrained) | 75.39 (↓) |
| + Commonality prompt only + $L_C$ | 81.07 |
| + Commonality + heterogeneity prompts (full) | **84.37** |

### Key Findings

- **Structure-driven adaptation outperforms entropy minimization**: Entropy-based methods such as SAR even fall below the No Adapt baseline on polyp segmentation due to overconfident erroneous predictions caused by "camouflaged targets"; SPEGC avoids this pitfall by relying on intra-data structure.
- **Superior long-term CTTA stability**: Over five rounds of continual adaptation, SPEGC achieves the highest average DSC (83.10%) with only 1.27% performance degradation, demonstrating robustness against both catastrophic forgetting and error accumulation.
- **Commonality prompts are critical**: Adding the heterogeneity prompt alone actually degrades performance (75.39 < 76.52), indicating that unconstrained prompts introduce noise; the commonality prompt combined with the clustering loss yields a substantial gain of 4.55%.
- **Efficiency–performance trade-off in feature pool size**: A pool size of 7 achieves the highest DSC (85.24%) but increases FLOPs to 21.7G; a pool size of 3 (84.37%, 5.8G FLOPs) represents the optimal balance.

## Highlights & Insights

- Introducing graph clustering into CTTA and replacing unreliable pixel-level/entropy signals with higher-order structural information is a novel and principled approach.
- The decoupled design of commonality/heterogeneity prompt pools is elegant: reverse attention captures cross-domain shared knowledge while standard attention retrieves domain-specific information.
- Formulating edge sparsification as an optimal transport problem and solving it via Sinkhorn enables end-to-end differentiable graph clustering.
- SPEGC comprehensively outperforms state-of-the-art methods on two medical segmentation benchmarks; long-term CTTA experiments thoroughly validate its robustness against catastrophic forgetting and error accumulation.

## Limitations & Future Work

- The similarity matrix computation in DGCS has $O(V^2)$ complexity, and FLOPs grow sharply as the feature pool increases (reaching 120G at pool size 15), limiting scalability.
- The number of clusters $Z$ is a manually set hyperparameter that requires tuning across different tasks.
- Validation is limited to ResNet-50/ResUNet-50; stronger backbones (e.g., ViT/Swin) and larger-scale datasets have not been evaluated.
- Only the online single-sample adaptation scenario is explored; mini-batch arrival settings are not investigated.
- The commonality prompt pool relies on the clustering loss, which assumes that sequentially arriving data share core semantics — an assumption that may not hold under extreme domain shift.

## Related Work & Insights

- **Clustering-based segmentation**: Yu et al. reformulate cross-attention as a clustering solver; Liang et al. propose recurrent cross-attention for iterative clustering; Ding et al. extend clustering to 3D volumetric data. However, these methods operate as static in-domain post-processing and cannot exploit dynamic graph structures to guide adaptation.
- **CTTA methods**: SAR (entropy filtering), DomainAdaptor (BN statistics), VPTTA (visual prompts + BN alignment), NC-TTT (noise estimation), GraTa (gradient alignment), TTDG (graph matching + pre-trained priors). SPEGC is most closely related to TTDG, but whereas TTDG relies on source-domain prototype alignment, SPEGC derives all adaptation signals purely from the internal structure of the target data.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of decoupled prompts and optimal-transport-based graph clustering is new to the CTTA literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multi-domain cross-evaluation, long-term CTTA, ablation studies, hyperparameter analysis, and t-SNE visualization.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, and well-motivated problem formulation.
- Value: ⭐⭐⭐⭐ — Practically meaningful for clinical medical imaging deployment, though computational cost remains a barrier to real-world adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](../../ICCV2025/medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)

</div>

<!-- RELATED:END -->
