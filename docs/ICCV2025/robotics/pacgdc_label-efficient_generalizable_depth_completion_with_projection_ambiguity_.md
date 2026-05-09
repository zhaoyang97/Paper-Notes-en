---
title: >-
  [Paper Note] PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency
description: >-
  [ICCV 2025][Robotics][Depth Completion] This paper proposes PacGDC, which exploits the inherent shape and position ambiguities in 2D-to-3D projection to synthesize large quantities of pseudo-geometric data—using multiple depth foundation models as scale manipulators—thereby achieving generalizable depth completion with minimal annotation cost, attaining state-of-the-art performance in both zero-shot and few-shot settings.
tags:
  - ICCV 2025
  - Robotics
  - Depth Completion
  - Label-Efficient
  - Projection Ambiguity
  - Data Synthesis
  - Generalizability
date: 2026-05-08
content_hash: bff0a1bfeb4f97e4
---

# PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency

**Conference**: ICCV 2025
**arXiv**: [2507.07374](https://arxiv.org/abs/2507.07374)
**Code**: [https://github.com/Wang-xjtu/PacGDC](https://github.com/Wang-xjtu/PacGDC)
**Area**: Robotics
**Keywords**: Depth Completion, Label-Efficient, Projection Ambiguity, Data Synthesis, Generalizability

## TL;DR

This paper proposes PacGDC, which exploits the inherent shape and position ambiguities in 2D-to-3D projection to synthesize large quantities of pseudo-geometric data—using multiple depth foundation models as scale manipulators—thereby achieving generalizable depth completion with minimal annotation cost, attaining state-of-the-art performance in both zero-shot and few-shot settings.

## Background & Motivation

**Background**: Depth completion aims to infer dense metric depth maps from paired images and sparse depth measurements. Existing methods (NLSPN, CFormer, etc.) perform well within their training domains but generalize poorly across domains. Recent generalizable depth completion methods (G2-MonoDepth, SPNet, OMNI-DC) attempt to address this issue but rely on large-scale dense metric depth annotations.

**Limitations of Prior Work**: Collecting large-scale dense depth annotations is extremely time-consuming and labor-intensive, requiring specialized equipment such as LiDAR and RGB-D sensors, which severely limits the training data coverage of generalizable models.

**Key Challenge**: Generalizable depth completion requires training data that covers as broad a real-world distribution as possible (diverse scene semantics, scales, and sparsity patterns), yet acquiring diverse annotated data at scale is prohibitively expensive.

**Goal**: To maximize training data coverage with minimal annotation cost, enabling depth completion models to generalize to unseen domains.

**Key Insight**: The authors observe that 2D-to-3D projection is inherently ambiguous—the same 2D image can correspond to multiple distinct 3D geometric scenes. This ambiguity is decomposed into **shape ambiguity** (the same 2D object can correspond to different 3D shapes) and **position ambiguity** (the same 3D shape can occupy different sizes and positions). Notably, the two inputs to depth completion—images providing shape cues and sparse depth providing positional cues—align naturally with these two types of ambiguity.

**Core Idea**: The paper leverages the "scale inaccuracy" of depth foundation models as a feature rather than a bug, treating these models as scale manipulators to synthesize large quantities of pseudo depth labels that are shape-consistent but scale-diverse, thereby greatly expanding the geometric diversity of training data.

## Method

### Overall Architecture

PacGDC is a data synthesis pipeline that does not modify the model architecture at inference time. The input consists of a small set of annotated triplets $\mathcal{T} = \{I, p, d\}$ (image, sparse depth, dense depth), and the output is a large collection of synthesized pseudo-triplets $\hat{\mathcal{T}}$. The pipeline proceeds as follows:

1. Multiple depth foundation models generate pseudo dense depth maps from images.
2. Interpolation and relocation strategies augment geometric diversity.
3. Pseudo sparse depth is sampled from the pseudo dense depth maps.
4. The synthesized data trains a generalizable depth completion model built on the SPNet framework.

### Key Designs

1. **Theoretical Foundation of Projection Ambiguity and Consistency**:

    - **Function**: Establishes a theoretical framework for leveraging projection ambiguity to enhance data diversity.
    - **Mechanism**: Under the pinhole camera model, $d_i P^{-1} [u_i, v_i, 1]^T = [x_i, y_i, z_i]^T$. Applying a scaling factor $\alpha_i$ to the depth does not alter the 2D pixel location but produces a new 3D geometry $\hat{d}_i = \alpha_i d_i$. This implies that depth scale manipulation can generate infinitely many valid 3D geometries. Shape consistency (geometric shapes aligned with image semantics) and position consistency (sparse depth constraining spatial position) together ensure the quality of synthesized data.
    - **Design Motivation**: Rather than treating projection ambiguity as a problem, this work exploits it as the core mechanism for data augmentation.

2. **Pseudo-Label Synthesis via Depth Foundation Models**:

    - **Function**: Leverages monocular depth estimation foundation models (DepthAnything, DepthPro) to generate pseudo depth labels that are shape-consistent but scale-diverse.
    - **Mechanism**: Foundation models robustly predict semantically consistent dense depth $\hat{d} = \mathcal{R}(I)$ from image $I$, but their predicted scales are typically inaccurate—a property that is deliberately exploited here. Predictions from multiple foundation models are combined via interpolation and relocation: $\hat{d} = \theta(\sum_{t=1}^{L} \lambda^t \mathcal{R}^t(I) + (1 - \sum_{t=1}^{L} \lambda^t) d)$, where $\lambda^t$ are random interpolation coefficients and $\theta$ is a random relocation factor.
    - **Design Motivation**: A single model produces predictions at only one scale. By combining multiple models with random interpolation and relocation, the geometric distribution coverage is maximized while maintaining shape consistency.

3. **Extension to Unlabeled Data**:

    - **Function**: Incorporates unannotated images (e.g., 390K images from SA1B) to further expand semantic and scene diversity.
    - **Mechanism**: When $\sum \lambda^t = 1$, the formula reduces to a weighted combination of pure foundation model predictions, requiring no ground-truth annotation $d$. Complete pseudo-triplets are formed from images, pseudo dense depths, and sampled pseudo sparse depths.
    - **Design Motivation**: A core insight of PacGDC is that pseudo data can effectively train generalizable models even without accurate scale, since the model learns geometric alignment rather than scale priors—enabling unlabeled data to contribute meaningful training signal.

### Loss & Training

Training follows the SPNet framework with a standard depth regression loss $\min_{\mathcal{F}} |\mathcal{F}(I, p) - d|$. In the zero-shot stage, the AdamW optimizer is used with a batch size of 192, an initial learning rate of 0.0002, and cosine decay over 100 epochs. In the few-shot stage, the model is fine-tuned from zero-shot pretrained weights at 1/10 of the original learning rate. No additional computational overhead is introduced at inference time.

## Key Experimental Results

### Main Results

Zero-shot depth completion (uniform sampling at 10%/1%/0.1%, averaged across 6 datasets):

| Method | Avg. RMSE↓ | Avg. MAE↓ | Notes |
|--------|-----------|----------|-------|
| NLSPN | 9284 | 6701 | Fully supervised, poor generalization |
| CFormer | 6408 | 4503 | Fully supervised |
| G2MD | 2387 | 923 | Generalizable method |
| SPNet | 2271 | 791 | Generalizable method |
| OMNI-DC | 2847 | 1310 | Generalizable method |
| **PacGDC (Ours)** | **1966** | **731** | Best; −13.4% RMSE vs. SPNet |

Few-shot depth completion (KITTI 64-beam LiDAR):

| # Training Samples | Method | RMSE↓ | MAE↓ |
|-------------------|--------|-------|------|
| 1 | ImprovingDC | 1358 | 337 |
| 1 | **PacGDC** | **1078** | **250** |
| 100 | SparseDC | 1203 | 325 |
| 100 | **PacGDC** | **911** | **229** |
| 1000 | SparseDC | 1049 | 263 |
| 1000 | **PacGDC** | **830** | **220** |

### Ablation Study

Ablation of synthesis strategies based on SPNet-Tiny (zero-shot, averaged across 6 datasets with uniform sampling):

| Configuration | RMSE↓ | MAE↓ | Notes |
|--------------|-------|------|-------|
| SPNet baseline | 2484 | 990 | Baseline |
| +DepthAnything (P=0) | 2463 | 956 | Foundation model predictions only |
| +DepthAnything (P=0.5) | 2344 | 889 | 50% interpolation probability |
| +DepthAnything (P=1.0) | 2330 | 889 | Full interpolation |
| +Relocation | 2277 | 857 | With relocation |
| +DepthPro | 2241 | 854 | Multiple foundation models |
| +SA1B unlabeled data | 2143 | 792 | Full pipeline |

### Key Findings

- **Every synthesis component contributes**: From baseline to the full pipeline, RMSE is cumulatively reduced by 13.7%.
- **Interpolation probability P=1.0 is optimal**: Fully using interpolated depth labels outperforms partial mixing, underscoring that maximizing data diversity is key.
- **Few-shot vs. fully supervised**: Using only 1,000 annotated samples surpasses certain fully supervised methods (e.g., S2D, TWISE) trained on 86K samples.
- Even with a single training sample, the model achieves meaningful performance (RMSE 1078), demonstrating the strong regularization effect of pretrained weights.
- The method is robust across different sparsity patterns (uniform sampling, VIO, LiDAR) and diverse scenes (indoor/outdoor/synthetic).

## Highlights & Insights

- **Turning a limitation into an asset**: The scale inaccuracy of depth foundation models is conventionally viewed as a deficiency; this paper repurposes it as the core tool for generating diverse training data—a counterintuitive and inspiring insight.
- **Theoretically clear and elegantly decomposed**: Decomposing projection ambiguity into orthogonal shape and position dimensions, each naturally corresponding to one of the two inputs in depth completion (image and sparse depth), yields a compelling theoretical framework.
- **Zero inference overhead**: All innovations reside in the data synthesis stage; the original model efficiency (126.6 images/s) is fully preserved at inference, which is highly valuable for practical deployment.
- **Transferable idea**: Exploiting the uncertainty or diversity of foundation model predictions to augment training data could generalize to other annotation-intensive perception tasks.

## Limitations & Future Work

- The approach relies on SPNet as the underlying framework; its effectiveness on other completion architectures has not been thoroughly verified (only preliminary validation on G2MD is provided).
- The shape consistency assumption of depth foundation models may break down in extreme scenarios (e.g., highly reflective or transparent surfaces).
- Only two depth foundation models are used; incorporating additional models may yield further improvements.
- The training dataset is large (745K samples), resulting in non-trivial training costs.
- Integration with self-supervised depth estimation methods remains unexplored.

## Related Work & Insights

- **vs. SPNet**: PacGDC is built upon the SPNet framework, with its core innovation on the data side rather than the model side, demonstrating the effectiveness of a "data over model" philosophy in generalizable learning.
- **vs. OMNI-DC**: OMNI-DC focuses on model improvement via multi-resolution depth guidance, while PacGDC focuses on data synthesis—the two approaches are complementary.
- **vs. pseudo-label methods**: Conventional pseudo-label methods aim to improve pseudo-label quality, whereas PacGDC deliberately pursues pseudo-label diversity, inverting the traditional objective.
- This paper offers a practical solution for rapid deployment of robotic perception systems, enabling generalizable model training with only a small amount of real annotated data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical analysis of projection ambiguity is novel and elegant; repurposing foundation model scale inaccuracy as a data augmentation advantage is highly inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Zero-shot evaluation across 7 datasets with multiple sparsity patterns, few-shot experiments, and detailed ablation studies—exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical exposition is clear and figures are intuitive, though the dense notation requires careful cross-referencing in places.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the annotation cost bottleneck in practical deployment, with immediate applicability to depth perception in robotics and autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency](../../CVPR2026/robotics/test-time_ego-exo-centric_adaptation_for_action_anticipation_via_multi-label_pro.md)
- [\[ICCV 2025\] EvolvingGrasp: Evolutionary Grasp Generation via Efficient Preference Alignment](evolvinggrasp_evolutionary_grasp_generation_via_efficient_preference_alignment.md)
- [\[ICCV 2025\] CombatVLA: An Efficient Vision-Language-Action Model for Combat Tasks in 3D Action Role-Playing Games](combatvla_an_efficient_vision-language-action_model_for_combat_tasks_in_3d_actio.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](../../NeurIPS2025/robotics/pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)

</div>

<!-- RELATED:END -->
