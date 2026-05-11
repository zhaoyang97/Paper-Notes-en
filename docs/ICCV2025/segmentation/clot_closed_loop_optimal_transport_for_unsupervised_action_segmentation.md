---
title: >-
  [Paper Note] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation
description: >-
  [ICCV 2025][Segmentation][Unsupervised Action Segmentation] This paper proposes Closed Loop Optimal Transport (CLOT), a framework that jointly solves three OT problems through a three-level cyclic feature learning pipeli…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Unsupervised Action Segmentation"
  - "Optimal Transport"
  - "Closed-Loop Learning"
  - "Encoder-Decoder"
  - "Sliced Wasserstein Distance"
date: 2026-05-08
content_hash: b3982c62a4d8e7a7
---

# CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.03539](https://arxiv.org/abs/2507.03539)
**Code**: [https://github.com/elenabbbuenob/CLOT](https://github.com/elenabbbuenob/CLOT)
**Area**: Video Understanding / Action Segmentation
**Keywords**: Unsupervised Action Segmentation, Optimal Transport, Closed-Loop Learning, Encoder-Decoder, Sliced Wasserstein Distance

## TL;DR

This paper proposes Closed Loop Optimal Transport (CLOT), a framework that jointly solves three OT problems through a three-level cyclic feature learning pipeline (frame embeddings → segment embeddings → cross-attention refined frame embeddings), establishing an explicit feedback loop between frame-level and segment-level representations to substantially improve boundary detection and clustering quality in unsupervised action segmentation.

## Background & Motivation

Unsupervised action segmentation aims to annotate video frames with action categories without supervision, with important applications in sports, surveillance, and robotics. Existing methods fall into two categories:

**Classical pipeline methods** (CTE, VTE, etc.): learn frame representations first and then cluster, lacking feedback between representation learning and clustering.

**OT-based methods** (TOT, ASOT, etc.): leverage optimal transport to jointly learn action representations and pseudo-labels, enabling feedback via self-training.

Among these, **ASOT** achieves the best performance by making no assumption on action order and obtaining temporally consistent segmentations between frames and action labels via Gromov-Wasserstein OT. However, ASOT suffers from two issues: (a) an implicit segment-length prior that hinders detection of short-duration actions; and (b) the absence of **explicit feedback between frame-level and segment-level representations**, causing learned clusters to misalign with true segment boundaries.

On the other hand, **HVQ** improves short-action detection through hierarchical vector quantization, but its fixed codebook lacks the feedback mechanism of OT-based methods and generalizes poorly. CLOT is designed to combine the strengths of both: preserving the feedback capability of OT while strengthening segment-level consistency.

## Method

### Overall Architecture

CLOT comprises a three-level architecture (see Fig. 2 in the original paper):
- **Level 1**: MLP encoder + feature dispatching mechanism → frame embeddings $F$ + pseudo-labels $\mathbf{T}$ (via OT_1)
- **Level 2**: Parallel decoder → segment embeddings $S$ + pseudo-labels $\mathbf{T}_S$ (via OT_2)
- **Level 3**: Frame-segment cross-attention → refined frame embeddings $F_R$ + pseudo-labels $\mathbf{T}_R$ (via OT_3)

The three levels form a closed loop: frames → segments → refined frames, each level imposing OT constraints to generate pseudo-labels, enabling multi-granularity cyclic optimization.

### Key Designs

1. **Feature Dispatching Encoder**: An MLP maps input frame features $X \in \mathbb{R}^{N \times D}$ to frame embeddings $F \in \mathbb{R}^{N \times d}$. The feature dispatching mechanism is based on a learnable similarity function $\phi(A,F) = \sigma(\beta + \alpha \cdot \frac{A \cdot F}{\|A\|\|F\|})$, where $A$ denotes learnable action cluster embeddings. Each frame is assigned to the most relevant cluster via attention weights and its representation is updated as $f_i' = f_i + \frac{1}{K}\sum_{k=0}^K \phi(A_k, f_i) \cdot A_k$. This enables frame embeddings to dynamically adapt to the currently learned cluster structure, yielding more organized representations.

2. **Parallel Decoder**: Adopts a query-based attention mechanism inspired by DETR, using learnable queries $Q \in \mathbb{R}^{K' \times d_{dec}}$ as segment prototypes. Frame embeddings are decoded into segment embeddings $S \in \mathbb{R}^{K' \times d}$ ($K' \leq K$) via multi-head cross-attention and self-attention. Unlike autoregressive decoders, the parallel decoder predicts all segments simultaneously, avoiding error accumulation.

3. **Cross-Attention Refinement**: Injects segment-level structural information back into frame embeddings: $F_R = F + \text{softmax}(\frac{FS^\top}{\tau \cdot \sqrt{d}})S$. This is the key to closing the loop—allowing frame representations to be structurally adjusted according to segment embeddings, ensuring that frame-level details are aligned with the temporal segmentation process.

4. **Sliced Wasserstein Distance**: Introduced as a complement to cosine distance in the cost matrix. SWD projects high-dimensional distributions onto one-dimensional subspaces via random projections: $\text{SWD}_p(x_i, a_j) = (\frac{1}{M}\sum_{m=1}^M d(R_{\theta_m\#}x_i, R_{\theta_m\#}a_j))^{1/p}$. The cost matrix is defined as $\mathbf{C}_{ij}^{sw} = 1 + \text{SWD}(x_i, a_j) - \mathbf{C}_{i,j}^k$, combining the SWD with the visual cost. Using $p=1$ yields a closed-form solution, ensuring computational efficiency.

### Loss & Training

The OT formulation is unbalanced, integrating two sub-problems, KOT and GW: $\min_\mathbf{T} \alpha \mathcal{F}_{GW} + (1-\alpha) \mathcal{F}_{KOT} - \lambda KL(\mathbf{T}^\top \mathbf{1}_n \| \nu)$, where the KL divergence penalty allows flexible non-uniform label assignment. The training objective is the sum of cross-entropy losses across all three levels: $\mathcal{L}_{train} = \mathcal{L}(\mathbf{T}, \mathbf{P}) + \mathcal{L}(\mathbf{T}_S, \mathbf{P}_S) + \mathcal{L}(\mathbf{T}_R, \mathbf{P}_R)$, where $P_{ij} = \text{softmax}(FA^\top / \tau)_{ij}$.

## Key Experimental Results

### Main Results (Activity-level, Hungarian Matching)

| Dataset | Metric | ASOT | HVQ | **CLOT** | Gain |
|--------|------|------|-----|---------|------|
| Breakfast | MoF | 56.1 | 54.4 | **60.1** | +4.0 |
| Breakfast | F1 | 38.3 | 39.7 | **40.1** | +0.4 |
| YTI | MoF | 52.9 | 50.3 | **54.4** | +1.5 |
| 50Salads(Eval) | F1 | 53.6 | - | **63.2** | +9.6 |
| 50Salads(Eval) | mIoU | 30.1 | - | **38.8** | +8.7 |
| DA | F1 | 68.0 | - | **72.6** | +4.6 |

### Ablation Study

| Configuration | Breakfast MoF | 50Salads(Eval) F1 | DA F1 | Note |
|------|-------------|------------------|-------|------|
| **CLOT** | **60.1** | **63.2** | **72.6** | Full model |
| w/o SWD | 59.8 | 52.7 | 62.3 | Removing SWD causes large F1 drop |
| w/o FD | 51.9 | 51.9 | 68.3 | Removing feature dispatching drops MoF by 8% |
| w/o Decoder | 58.0 | 52.7 | 68.4 | No segment-level OT, F1 drops noticeably |
| w/o Refinement | 59.7 | 52.6 | 68.2 | No cross-attention refinement, closed loop broken |

### Key Findings

- **Closed-loop refinement is the most critical component**: removing the Decoder or Refinement drops 50Salads F1 from 63.2 to 52.7, indicating that segment-level feedback is essential for boundary detection.
- **Feature dispatching contributes most on Breakfast** (MoF drops 8.2), where the diversity of activity categories and temporal complexity make structured representations especially important.
- **SWD yields significant gains on 50Salads and DA** (F1 drops of 10.5 and 10.3, respectively), validating its more robust distance measurement in high-dimensional spaces.
- Under video-level evaluation, CLOT achieves MoF 66.3 on Breakfast (vs. 63.3 for ASOT) and F1 69.7 on 50Salads(Eval) (vs. 58.9 for ASOT), with even more pronounced improvements.

## Highlights & Insights

- The closed-loop "frames → segments → refined frames" three-level cycle is the core contribution, enabling representations at different granularities to mutually reinforce each other rather than propagating information unidirectionally.
- The feature dispatching mechanism realizes "soft-clustering-guided representation updates," offering greater flexibility compared to hard clustering.
- Using Sliced Wasserstein Distance in place of cosine distance for constructing the OT cost matrix exploits SWD's advantage in preserving the geometric structure of distributions.
- The parallel decoder design avoids the error accumulation inherent in autoregressive decoders.

## Limitations & Future Work

- The number of action categories $K$ must be specified in advance, which may be difficult to determine in practice.
- As with ASOT, computing OT incurs non-trivial overhead, potentially limiting scalability to very long videos.
- Video-level and activity-level results are not always consistent, suggesting room for improvement in cross-video generalization.
- The strategy for selecting the decoder query count $K'$ is not sufficiently discussed in the codebase.

## Related Work & Insights

- Extends the unbalanced GW-OT framework of ASOT by adding segment-level OT and refinement OT.
- The parallel decoder is inspired by DETR-style designs from action prediction literature (e.g., FUTR3D).
- SWD has been validated in point cloud processing and color transfer; this work is the first to apply it to OT cost matrices for action segmentation.
- Unlike TOT and UFSA, CLOT requires no prior on action order, making it more general.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-level cyclic closed-loop OT design is novel, and the orchestration of three OT problems is well-motivated
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks, two evaluation protocols, and detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure; Fig. 2 conveys rich architectural information
- Value: ⭐⭐⭐⭐ Meaningful advancement in unsupervised action segmentation, achieving state-of-the-art on most datasets

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[ICCV 2025\] Learn2Synth: Learning Optimal Data Synthesis Using Hypergradients for Brain Image Segmentation](learn2synth_learning_optimal_data_synthesis_using_hypergradients_for_brain_image.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](ensemble_foreground_management_for_unsupervised_object_discovery.md)

</div>

<!-- RELATED:END -->
