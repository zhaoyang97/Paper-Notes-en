---
title: >-
  [Paper Note] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport
description: >-
  [NeurIPS 2025][Signal & Communication][optimal transport] This paper proposes Bispectral Optimal Transport (BOT), which replaces the cost matrix in discrete optimal transport from raw pixel distances to bispectrum (group Fourier invariant) distances. This enables the transport plan to precisely eliminate group-action-induced variation (e.g., rotation) while preserving signal structure, improving class-preservation accuracy from 33% to 84% on rotation-augmented datasets such as MNIST.
tags:
  - NeurIPS 2025
  - "Signal & Communication"
  - optimal transport
  - bispectrum
  - symmetry invariance
  - dataset comparison
  - Fourier transform
date: 2026-05-08
content_hash: e0a8b29691332155
---

# Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport

**Conference**: NeurIPS 2025
**arXiv**: [2509.20678](https://arxiv.org/abs/2509.20678)
**Code**: [https://github.com/annabel-ma/bispectral-ot](https://github.com/annabel-ma/bispectral-ot)
**Area**: Machine Learning Theory
**Keywords**: optimal transport, bispectrum, symmetry invariance, dataset comparison, Fourier transform

## TL;DR
This paper proposes Bispectral Optimal Transport (BOT), which replaces the cost matrix in discrete optimal transport from raw pixel distances to bispectrum (group Fourier invariant) distances. This enables the transport plan to precisely eliminate group-action-induced variation (e.g., rotation) while preserving signal structure, improving class-preservation accuracy from 33% to 84% on rotation-augmented datasets such as MNIST.

## Background & Motivation

**State of the Field**: Optimal transport (OT) is widely used in machine learning for distribution alignment, with applications in transfer learning, domain adaptation, and generative modeling. Standard discrete OT computes transport plans based on pairwise geometric distances (e.g., L2) between samples.

**Limitations of Prior Work**: In settings with symmetry (e.g., image rotation), OT based on raw features may match images according to orientation rather than object shape—i.e., the transport plan is sensitive to nuisance variation and disregards the intrinsic semantic structure of the data.

**Root Cause**: A transport plan is needed that simultaneously eliminates nuisance introduced by symmetric transformations (e.g., differing rotation angles) while retaining discriminative information (e.g., object shape). Simple invariant features such as the power spectrum can remove group actions but lose structural information.

**Paper Goals**: How can symmetry awareness be encoded into OT so that the transport plan is invariant to group actions yet selective with respect to signal structure?

**Starting Point**: The bispectrum is the lowest-order complete invariant in group Fourier analysis—it jointly encodes signal structure and group invariance, removing variation due to group actions and nothing else.

**Core Idea**: Replace raw features in the OT cost matrix with bispectral representations, yielding a transport plan that is invariant to symmetric transformations while retaining discriminative information.

## Method

### Overall Architecture
Input: two datasets (potentially differing by symmetric transformations). Construct bispectral feature representations → compute pairwise distances in bispectral space as the OT cost matrix → solve Sinkhorn OT → obtain a symmetry-aware transport plan.

### Key Designs

1. **Bispectrum Feature Extraction**:

    - Function: Convert images into rotation-invariant bispectral representations.
    - Mechanism: (1) Convert an $M \times N$ image to a polar coordinate representation of size $R \times K$ ($R$ radial bins, $K$ angular bins); (2) apply 1D DFT to the angular slice at each radius $r$; (3) compute the bispectrum $B_{i,j} = \hat{f}_i \hat{f}_j \hat{f}_{i+j}^\dagger$ from the DFT coefficients; (4) concatenate across radii to obtain a global SO(2)-invariant representation.
    - Design Motivation: The bispectrum is the lowest-order complete spectral invariant—the power spectrum is also invariant but discards phase information (incomplete), whereas the bispectrum preserves relative phase structure.

2. **Bispectral OT Formulation**:

    - Function: Solve OT in the bispectral feature space.
    - Mechanism: The cost matrix is $C_{ij} = d(B(\mathbf{x}^{(i)}), B(\mathbf{y}^{(j)}))$, where $B(\cdot)$ denotes the bispectral feature and $d$ can be L1/L2/cosine distance. The standard Sinkhorn algorithm is then applied.
    - Design Motivation: Computing distances in an invariant feature space naturally yields a symmetry-aware transport plan.

### Loss & Training
No training is required. BOT is a purely computational framework that solves a standard OT problem in bispectral feature space.

## Key Experimental Results

### Main Results

| Dataset | OT (L1, raw pixels) | BOT (L1) | BOT (L2) | BOT (cosine) |
|--------|-------------------|----------|----------|-------------|
| MNIST | 0.330 | **0.841** | 0.802 | 0.816 |
| KMNIST | 0.242 | **0.782** | 0.723 | 0.735 |
| FMNIST | 0.300 | **0.762** | **0.766** | 0.732 |
| EMNIST | 0.197 | **0.598** | 0.569 | 0.572 |

Note: Values are class-preservation accuracy (proportion of matches assigned to the same class when transporting from a rotated set to a non-rotated set).

### Ablation Study

| Analysis | Finding |
|------|------|
| MDS visualization | Rotated variants cluster by class in bispectral space; scattered in pixel space |
| Rotationally symmetric digit (0) | Tightest cluster in bispectral space |
| Digits 6 and 9 | Close in bispectral space (equivalent under 180° rotation) |
| Baseline (no rotation) | Raw-pixel OT already high (~97%); BOT also high (~86%) |

### Key Findings
- **BOT improves class-preservation accuracy by 2–3×**: from 20–33% to 60–84%, consistently across all datasets.
- **L1 distance is most effective in bispectral space**: likely because the sparse structure of the bispectrum is better suited to L1.
- **Rotational symmetry is elegantly captured**: digit 0 yields the tightest cluster (high rotational symmetry), digit 2 the most dispersed (low rotational symmetry), and digits 6/9 are adjacent.
- **The gap on the no-rotation baseline reveals information loss in bispectral representations**: without rotation, raw-pixel OT (97%) outperforms BOT (86%), indicating a partial loss of image reconstruction information.

## Highlights & Insights
- **Using the bispectrum as OT cost features is a first**, introducing tools from group theory and harmonic analysis into OT—a novel contribution with theoretical depth.
- **The notion of a complete invariant is central**: the power spectrum is invariant but incomplete (loses information), whereas the bispectrum is the lowest-order complete invariant—achieving an optimal balance between nuisance removal and information preservation.
- **MDS visualizations** intuitively demonstrate the elegant geometric structure of bispectral space and carry pedagogical value.

## Limitations & Future Work
- **Currently limited to SO(2) rotational symmetry**: extending to SO(3) or non-commutative groups significantly increases the computational complexity of bispectrum computation.
- **Validated only on MNIST-style simple datasets**: the effectiveness of bispectral representations on natural images (e.g., ImageNet), where symmetry structures are more complex, remains unknown.
- **Performance degrades without rotation**: bispectral representations incur information loss (97% → 86%), and should not be used when symmetry invariance is unnecessary.
- **Choice of distance metric**: standard norms (L1/L2/cosine) are used currently; dedicated distances tailored to bispectral space may yield better performance.

## Related Work & Insights
- **vs. standard OT**: Standard OT is symmetry-unaware and matching is dominated by nuisance variation.
- **vs. Gromov-Wasserstein**: GW is invariant to isometric transformations but does not handle general group actions.
- **vs. data augmentation + OT**: Invariance can be achieved indirectly via data augmentation, but at high computational cost and without guarantees of completeness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First introduction of the bispectrum into OT; theoretically elegant
- Experimental Thoroughness: ⭐⭐⭐ Limited to MNIST-style datasets; large-scale and natural image validation is lacking
- Writing Quality: ⭐⭐⭐⭐ Mathematical background is clear, though the experimental section is relatively brief
- Value: ⭐⭐⭐⭐ Proposes a promising new direction, though practical impact remains to be demonstrated

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals](masked_symbol_modeling_for_demodulation_of_oversampled_baseband_communication_si.md)
- [\[NeurIPS 2025\] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning](the_surprising_effectiveness_of_negative_reinforcement_in_llm_reasoning.md)

<!-- RELATED:END -->
