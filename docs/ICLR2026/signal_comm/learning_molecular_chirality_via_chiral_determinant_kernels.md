---
title: >-
  [Paper Note] Learning Molecular Chirality via Chiral Determinant Kernels
description: >-
  [ICLR2026][Signal & Communication][molecular chirality] This paper proposes Chiral Determinant Kernels (ChiDeK) to encode SE(3)-invariant chiral matrices, achieving for the first time a unified treatment of both central and axial chirality within a GNN framework. Combined with cross-attention for propagating stereochemical information, the method achieves >7% accuracy improvement on a newly constructed axial chirality benchmark.
tags:
  - ICLR2026
  - "Signal & Communication"
  - molecular chirality
  - chiral determinant kernel
  - equivariant graph neural network
  - axial chirality
  - SE(3) invariance
date: 2026-05-08
content_hash: d82ede5db88f9356
---

# Learning Molecular Chirality via Chiral Determinant Kernels

**Conference**: ICLR2026
**arXiv**: [2602.07415](https://arxiv.org/abs/2602.07415)
**Code**: To be confirmed
**Area**: Signal Communication
**Keywords**: molecular chirality, chiral determinant kernel, equivariant graph neural network, axial chirality, SE(3) invariance

## TL;DR
This paper proposes Chiral Determinant Kernels (ChiDeK) to encode SE(3)-invariant chiral matrices, achieving for the first time a unified treatment of both central and axial chirality within a GNN framework. Combined with cross-attention for propagating stereochemical information, the method achieves >7% accuracy improvement on a newly constructed axial chirality benchmark.

## Background & Motivation
- **State of the Field**: Chirality is a central concept in medicinal chemistry: enantiomers share the same chemical formula but have mirror-image 3D structures, and their biological activities can differ drastically (classic example: the R-enantiomer of thalidomide is a sedative, while the S-enantiomer is teratogenic).
- **Limitations of Prior Work**: Existing GNN-based molecular representation methods focus primarily on **central chirality** (tetrahedral carbons), while **axial chirality** (arising from restricted rotation around biaryl bonds, etc.) is equally important in medicinal chemistry yet considerably harder to model. Traditional approaches encode chirality via CIP rules (R/S labels) or chiral volumes, and suffer from two drawbacks: (1) only central chirality is handled, ignoring axial chirality; (2) the chiral volume is an SE(3)-invariant scalar with limited information content.
- **Paper Goals**: A unified framework is needed to capture both types of chirality while preserving SE(3) invariance.

## Method

### Overall Architecture
Given a molecule $\bm{z} = (\bm{X}, \bm{H})$ (3D coordinates + atom features), a molecular graph $\mathcal{G}$ is constructed, and atoms are categorized into chiral atoms $\mathcal{I}_c$, chirality-related atoms $\mathcal{I}_r$, and non-chiral atoms $\mathcal{I}_n$. The pipeline proceeds as follows: (1) a **Chiral Encoder** computes stereochemical embeddings for chiral atoms via determinant kernels and projects features for each of the three atom categories; (2) a **Chiral Transformer** propagates chiral information via cross-attention, using chiral atoms as queries and other atoms as keys/values; (3) a **Predictor** head performs downstream prediction.

### Key Designs
1. **Chiral Determinant Kernel (ChiDeK)**:

   - A chiral matrix $\bm{M}_C(i)$ is constructed from the 3D coordinates of the four substituents of chiral atom $i$, forming a $3 \times 3$ matrix.
   - After transformation by a learnable projection $\bm{W} \in \mathbb{R}^{k \times d_p \times 3}$, QR decomposition is applied and $\det(\bm{R})$ is taken as the determinant feature.
   - Key property: $\det(\bm{R}) = \alpha(\bm{W}) \cdot P_C(i)$, proportional to the original chiral product, preserving SE(3) invariance and changing sign under reflection.
   - Outputs a $k$-dimensional embedding ($k$ determinant kernels), carrying $k$ times more information than a scalar chiral volume.

2. **Unified Treatment of Central and Axial Chirality**:

   - Central chirality: the four neighbors of a tetrahedral center atom form $\bm{M}_C$.
   - Axial chirality: the nearest substituent atoms on each side of the rotationally restricted bond serve as inputs to $\bm{M}_C$.
   - Both chirality types are handled by an **identical mathematical framework**.

3. **Chiral Cross-Attention**:

   - Chiral atom embeddings serve as queries; chirality-related and non-chiral atoms are projected as keys/values respectively.
   - GKPT (Gaussian Kernel with Pair Type) distance biases are incorporated, distinguishing between chiral–chirality-related and chiral–non-chiral atom pairs.
   - $L$ layers are stacked, with pairwise biases updated at each layer.

### Loss & Training
- Standard classification/regression loss plus weight regularization $\mathcal{L}_{reg} = \|W^\top W - I_3\|^2$ to ensure full-rank projection matrices.
- An auxiliary QR decomposition is applied directly to the weight matrices prior to projection, guaranteeing column independence.
- Multi-task evaluation: R/S classification (accuracy), enantiomer ranking (accuracy), ECD spectrum prediction (RMSE), and optical rotation prediction (RMSE).

## Key Experimental Results

### Main Results

| Task | Metric | ChiDeK | ChiGNN | SphereNet | Gain |
|------|--------|--------|--------|-----------|------|
| Central chirality R/S classification | Acc↑ | 98.2% | 97.8% | 94.5% | +0.4% |
| Enantiomer ranking | Acc↑ | 77.8% | 75.6% | 65.7% | +2.2% |
| Central chirality ECD (Position) | RMSE↓ | 2.75 | 3.21 | 3.85 | −14.3% |
| **Axial chirality ECD** | RMSE↓ | **Strong baseline** | **N/A** | Weaker | **>7%↑** |
| **Axial chirality OR** | RMSE↓ | **Strong baseline** | **N/A** | Weaker | **>7%↑** |

### Ablation Study

| Configuration | R/S Acc | ECD RMSE | Notes |
|---------------|---------|----------|-------|
| ChiDeK (full) | 98.2% | 2.75 | Determinant kernel + cross-attention |
| w/o determinant kernel (scalar chiral volume) | 95.1% | 3.45 | Insufficient information in scalar |
| w/o cross-attention | 96.0% | 3.12 | Chiral information cannot propagate to full molecule |
| w/o GKPT distance bias | 97.3% | 2.95 | Distance information plays a clear auxiliary role |

### Key Findings
- ChiDeK outperforms the strongest baseline by >7% on axial chirality tasks—the first systematic evaluation of axial chirality.
- Determinant kernels provide richer stereochemical information than scalar chiral volumes (validated by ablation).
- Cross-attention allows chiral information to propagate beyond chiral centers and influence the entire molecular representation.
- Weight regularization is critical for training stability—rank-deficient projection matrices cause complete loss of chiral information.

## Highlights & Insights
- This is the first work to unify central and axial chirality within a GNN framework—a clear conceptual advance over all prior methods, which address only central chirality.
- The mathematical design of the chiral determinant kernel is elegant: SE(3) invariance is rigorously proven (Proposition 3.1, Lemma 3.1, Lemma 4.1) without relying on empirical approximations.
- The ACMP (Axial Chiral Molecular Properties) benchmark fills a gap in axial chirality evaluation and constitutes a lasting contribution to the community.
- The cross-attention mechanism, augmented by GKPT distance biases, enables chiral information to diffuse across the entire molecular graph rather than remaining localized at chiral centers.
- QR decomposition ensures differentiability while preserving the chirality-discriminating property of the determinant, handling the rank-deficiency edge case.

## Limitations & Future Work
- Detection of axial chirality presupposes knowledge of which bonds are rotationally restricted; the current approach relies on chemical knowledge and heuristic rules—automatic detection is an important future direction.
- Validation is limited to small molecules; chirality modeling for large molecules such as proteins and peptides is not addressed, and computational cost grows with the number of chiral centers.
- Integration with 3D coordinate prediction tasks (e.g., conformer generation) is unexplored—ChiDeK could potentially serve as a chirality constraint in molecular generation.
- More complex forms of chirality (e.g., planar chirality, helical chirality) are not covered, though the theoretical framework is in principle extensible.

## Related Work & Insights
- **vs. ChIRo**: ChIRo learns 3D representations invariant to bond rotation yet sensitive to stereoisomers, but handles only central chirality; ChiDeK unifies both central and axial chirality.
- **vs. ChiGNN**: ChiGNN encodes chirality via an atom-ordering resolution strategy but lacks explicit localized chiral descriptors; ChiDeK's determinant kernels provide richer features.
- **vs. SphereNet/DimeNet**: E(3)-invariant models cannot distinguish enantiomers, since distances and angles are invariant under reflection; ChiDeK's determinant changes sign under reflection, naturally discriminating enantiomers.
- **vs. Tetra-DMPNN**: Chirality encoding on 2D graphs relies solely on symbolic information such as R/S labels and lacks 3D geometric context.
- **ACMP Benchmark**: The first benchmark for axial chiral molecular property prediction, encompassing ECD and OR prediction tasks; it fills a community gap and is expected to become a standard evaluation protocol for future chirality modeling research.
- **Broader Inspiration**: The determinant as a mathematical representation of chirality may also be applicable to other geometric deep learning tasks requiring discrimination of mirror-symmetric structures, such as crystal structure classification and chiral nanomaterial design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Chiral determinant kernels are an entirely new concept; unification of both chirality types)
- Experimental Thoroughness: ⭐⭐⭐⭐ (New benchmark + multi-task validation, though large-molecule experiments are absent)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous mathematical derivations; clear chemical motivation)
- Value: ⭐⭐⭐⭐ (Practical significance for drug design; benchmark contribution is enduring)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](../../ACL2026/signal_comm/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[CVPR 2026\] Dual-Imbalance Continual Learning for Real-World Food Recognition](../../CVPR2026/signal_comm/dual-imbalance_continual_learning_for_real-world_food_recognition.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)

<!-- RELATED:END -->
