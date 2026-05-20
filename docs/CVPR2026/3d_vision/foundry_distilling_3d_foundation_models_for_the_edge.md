---
title: >-
  [Paper Note] Foundry: Distilling 3D Foundation Models for the Edge
description: >-
  [CVPR 2026][3D Vision][Foundation model distillation] This paper proposes the Foundation Model Distillation (FMD) paradigm and the Foundry framework. Through a compress-and-reconstruct objective…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Foundation model distillation"
  - "3D point cloud"
  - "SuperToken"
  - "representation space compression"
  - "edge deployment"
date: 2026-05-08
content_hash: d0214ed343cff074
---

# Foundry: Distilling 3D Foundation Models for the Edge

**Conference**: CVPR 2026
**arXiv**: [2511.20721](https://arxiv.org/abs/2511.20721)  
**Code**: None  
**Area**: 3D Vision / Model Compression
**Keywords**: Foundation model distillation, 3D point cloud, SuperToken, representation space compression, edge deployment

## TL;DR

This paper proposes the Foundation Model Distillation (FMD) paradigm and the Foundry framework. Through a compress-and-reconstruct objective, the student model learns a set of learnable SuperTokens to compress the basis vectors of the teacher's latent space. The resulting single distilled model retains generality across classification, segmentation, and few-shot tasks, while reducing FLOPs from 478G to as low as 137G.

## Background & Motivation

**Background**: Foundation models pre-trained via self-supervised learning (SSL) have become powerful general-purpose feature extractors, demonstrating particular strength in the 3D point cloud domain (e.g., Point-BERT, Point-JEPA), with broad applications in robotics, autonomous driving, and AR/VR. These models acquire strong generalization to diverse downstream tasks through pre-training on large-scale unlabeled data.

**Limitations of Prior Work**: These foundation models are extremely large (hundreds of millions of parameters with quadratic attention complexity) and cannot be deployed on edge devices such as robots or AR headsets. Even modern GPUs may run out of memory when processing medium-scale point clouds of 300K points. Although existing knowledge distillation (KD) methods can produce efficient student models, they yield "expert models" that excel at specific tasks but lose the task-agnostic generality that is central to foundation models.

**Key Challenge**: Standard knowledge distillation trains students on task-specific logits, producing students that inherit only the teacher's behavior on that task and lack cross-task transferability. This contradicts the core value of foundation models—universal representation capability. An ideal distillation approach should preserve the teacher's entire representation space rather than only its outputs on a specific task.

**Goal**: To design a novel distillation paradigm that compresses large SSL foundation models into compact, efficient, and faithful surrogate models while preserving their universal representation capability.

**Key Insight**: Rather than directly performing feature mimicry of the teacher's embeddings, the method employs an information bottleneck to force the student to learn compact basis vectors of the teacher's latent space—first compressing into a small number of SuperTokens, then reconstructing the teacher's complete token-level representations from them.

**Core Idea**: Replace "mimicry" with "compress-and-reconstruct," so that the student learns not a specific output of the teacher but a set of basis vectors that can efficiently represent the teacher's entire latent space.

## Method

### Overall Architecture

Foundry's distillation proceeds in three steps: (1) **Teacher forward pass**—a frozen pre-trained teacher processes the input point cloud and produces target representations $\mathbf{Y} \in \mathbb{R}^{c \times d}$; (2) **Student compression and reconstruction**—the DSO module compresses $c$ tokens into $s \ll c$ SuperTokens, which are processed by a lightweight student encoder, and the CAU module then reconstructs the teacher's complete representations $\hat{\mathbf{Y}} \in \mathbb{R}^{c \times d}$; (3) **Distillation optimization**—minimizing $\mathcal{L}_{distillation} = \text{SmoothL1}(\hat{\mathbf{Y}}, \mathbf{Y})$.

### Key Designs

1. **Dynamic Supertoken Optimization (DSO)**:

    - **Function**: Compresses $c$ input tokens into $s$ learnable SuperTokens.
    - **Mechanism**: Maintains a set of randomly initialized learnable SuperTokens $\mathbf{S} \in \mathbb{R}^{s \times d}$ as a collection of basis vectors in the latent space. A hard assignment matrix is computed via cross-attention (SuperTokens as queries, input tokens as keys/values): $\text{CAM}_{j,i} = 1$ when $i = \arg\max_k \frac{\mathbf{q}_k \cdot \mathbf{k}_j}{\sqrt{d}}$. Each SuperToken then aggregates the mean of value vectors assigned to it: $\mathbf{S}_{updated} = \frac{\text{CAM}^T \mathbf{V}}{\text{sum}(\text{CAM}^T, \text{axis}=1)}$. Gumbel-Softmax is used to ensure differentiability.
    - **Design Motivation**: Unlike static K-Means clustering, learnable SuperTokens can adapt to the distillation objective through end-to-end training, learning truly information-dense latent basis vectors. Semantic grouping is performed before positional encoding is added, ensuring that SuperTokens perform feature compression based on content rather than position.

2. **Cross-Attention Upsampling (CAU)**:

    - **Function**: Reconstructs the teacher's complete token-level representations from the compressed SuperTokens.
    - **Mechanism**: Reuses the assignment matrix CAM computed during the DSO stage as a routing mechanism. Each original token position retrieves the updated representation of its corresponding SuperToken via CAM, applies a residual connection with the original input token, and is then mapped to the teacher's representation dimension via an MLP: $\hat{\mathbf{Y}} = \text{MLP}(\mathbf{T} + \text{CAM} \cdot \mathbf{S}_{encoder\_out})$.
    - **Design Motivation**: The residual connection is critical—it reintroduces local high-frequency detail that may be lost during compression, ensuring high-fidelity reconstruction. Reusing CAM avoids additional computational overhead.

3. **Gated Compression (optional)**:

    - **Function**: Enables dynamic, on-demand computation budget control at inference time.
    - **Mechanism**: A 2-layer MLP gating network predicts a fusion probability $\pi_i$ for each input token. Only tokens with $\pi_i > r$ (a user-defined threshold) are compressed via DSO; the remaining tokens bypass compression and are fed directly into the student encoder alongside the SuperTokens. A regularization term $\mathcal{L}_{gate} = -\lambda_{gate} \sum_i \pi_i$ encourages greater compression during training.
    - **Design Motivation**: Different deployment scenarios have different accuracy–speed requirements. The gating mechanism allows flexible trade-offs between accuracy and computation by adjusting the threshold $r$ at inference time, without retraining.

### Loss & Training

- Core distillation loss: $\mathcal{L}_{distillation} = \text{SmoothL1}(\hat{\mathbf{Y}}, \mathbf{Y})$
- Gated variant: $\mathcal{L} = \mathcal{L}_{distillation} + \mathcal{L}_{gate}$
- Training is conducted on ShapeNet55 for 150 epochs; the student encoder is initialized from teacher weights, with optional freezing.
- All teachers are Point-JEPA models with ViT-S architecture.

## Key Experimental Results

### Main Results (General Model vs. Expert Model)

| Method | ShapeNet55 Classification Acc | ShapeNetPart Segmentation mIoU_C / mIoU_I |
|--------|------|------|
| Teacher (Point-JEPA) | 90.54 | 83.91 / 85.73 |
| Foundry (general, 16 SuperTokens) | 89.87 | 81.87 / 84.82 |
| Expert-Classification (KD) | 75.09 | — |
| Expert-Segmentation (KD) | — | 61.88 / 65.72 |

### Ablation Study on SuperToken Mechanism

| Method | ShapeNet55 Acc |
|------|---------|
| Foundry (learnable DSO + CAU) | 89.68 |
| KMeans-Student (static clustering) | 76.08 |
| FPS-Student (pre-sampling) | 87.56 |

### Key Findings

- **FMD general model outperforms expert distillation**: A single general student maintains high performance on both tasks (89.87% / 81.87%), whereas expert students collapse on their own target tasks (classification expert: 75.09%; segmentation expert: 61.88%), demonstrating the superiority of the FMD paradigm.
- **Learnable SuperTokens substantially outperform static alternatives**: DSO surpasses K-Means by 13.6% (89.68 vs. 76.08), establishing the necessity of end-to-end basis vector learning.
- **Extreme compression remains effective**: With only 1 SuperToken, Foundry achieves 91.8% in 10-shot classification, approaching the teacher's 96.1%.
- **Edge deployment is feasible**: FLOPs are reduced from 478G to 137–178G ($s$=1–16), and latency decreases from 0.09s to 0.05–0.06s. On a 6GB GPU processing large scenes of 300K points, both the teacher and ToMe run out of memory, while Foundry requires only 4.0GB.
- Distillation loss is strongly correlated with downstream accuracy (clear inverse relationship); returns diminish at $s \leq 4$, indicating that very few SuperTokens suffice to adequately span the teacher's latent space.

## Highlights & Insights

- **Paradigm innovation: from task-specific KD to representation distillation**: This is the paper's most significant contribution. Conventional KD produces task experts, whereas FMD produces miniature surrogates of foundation models. This paradigm difference is meaningful for all scenarios requiring edge deployment of foundation models.
- **Information bottleneck design of compress-and-reconstruct**: Forcing the student to reconstruct teacher representations through an extremely narrow SuperToken bottleneck captures the structure of the latent space more effectively than direct L2 feature mimicry. This idea is transferable to the distillation of 2D image foundation models (e.g., DINOv2, SAM).
- **Compatibility with existing token compression methods**: Foundry's FMD framework can be combined with ToMe, PiToMe, and PatchMerger to achieve further performance gains.

## Limitations & Future Work

- Validation is limited to a single teacher (Point-JEPA ViT-S); generalization to other 3D foundation models (e.g., Point-MAE, PointGPT) and larger architectures (ViT-B/L) remains unknown.
- The approach covers only the 3D point cloud domain; extension to 2D image and video foundation models is an important future direction.
- Although the gating mechanism provides inference-time flexibility, its performance is sensitive to the choice of $\lambda_{gate}$ during training.
- The hard assignment in DSO may limit representation quality, as incorrectly assigned tokens cannot be corrected (though the residual connection in CAU partially mitigates this).
- End-to-end performance evaluation on real edge devices (e.g., Jetson, mobile phones) has not been conducted.

## Related Work & Insights

- **vs. TinyCLIP / CLIP-KD**: These methods distill CLIP's cross-modal alignment capability (a specific capability), whereas Foundry distills the entire representation space of an SSL model (a universal capability)—a fundamentally different philosophy.
- **vs. ToMe / PiToMe**: ToMe merges tokens online at inference time to accelerate computation, whereas Foundry performs offline distillation to create an entirely new student model. The two approaches are complementary—experiments show that distilling a ToMe student within the FMD framework yields the best results.
- **vs. 3DLST**: 3DLST also employs learnable supertokens, but targets inference acceleration for a specific segmentation task rather than creating a general surrogate model. Foundry elevates this idea to a distillation objective.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes the FMD paradigm and clearly articulates its essential distinction from conventional KD and direct feature mimicry.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6+1 datasets, general vs. expert comparisons, multiple SuperToken counts, gated variants, computational analysis, and large-scene testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, experimental structure is well-organized, and differentiation from related work is clear.
- Value: ⭐⭐⭐⭐⭐ Directly advances edge deployment of 3D foundation models; the FMD paradigm has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](nanosd_edge_efficient_foundation_model_for_real_time_image_restoration.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] Global-Aware Edge Prioritization for Pose Graph Initialization](global-aware_edge_prioritization_for_pose_graph_initialization.md)

</div>

<!-- RELATED:END -->
