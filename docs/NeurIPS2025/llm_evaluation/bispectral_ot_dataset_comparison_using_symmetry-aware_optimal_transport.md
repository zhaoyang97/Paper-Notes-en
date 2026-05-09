---
title: >-
  [Paper Note] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport
description: >-
  [NeurIPS 2025][LLM Evaluation][optimal transport] This paper proposes Bispectral Optimal Transport (BOT), which replaces the cost matrix in discrete optimal transport from raw pixel distances to bispectrum (group Fourier invariant) distances, enabling transport plans to eliminate group-action-induced variation (e.g., rotation) while preserving signal structure. On rotation-augmented MNIST and related datasets, the class-preservation accuracy improves from 33% to 84%.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - optimal transport
  - bispectrum
  - symmetry invariance
  - dataset comparison
  - Fourier transform
date: 2026-05-08
content_hash: 905ec2d2879ee486
---

# Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport

**Conference**: NeurIPS 2025
**arXiv**: [2509.20678](https://arxiv.org/abs/2509.20678)
**Code**: [https://github.com/annabel-ma/bispectral-ot](https://github.com/annabel-ma/bispectral-ot)
**Area**: Machine Learning Theory
**Keywords**: optimal transport, bispectrum, symmetry invariance, dataset comparison, Fourier transform

## TL;DR
This paper proposes Bispectral Optimal Transport (BOT), which replaces the cost matrix in discrete optimal transport from raw pixel distances to bispectrum (group Fourier invariant) distances, enabling transport plans to eliminate group-action-induced variation (e.g., rotation) while preserving signal structure. On rotation-augmented MNIST and related datasets, the class-preservation accuracy improves from 33% to 84%.

## Background & Motivation

**Background**: Optimal transport (OT) is widely used in machine learning for distribution alignment, with applications in transfer learning, domain adaptation, and generative modeling. Standard discrete OT computes transport plans based on pairwise geometric distances (e.g., L2) between samples.

**Limitations of Prior Work**: In settings with inherent symmetries (e.g., image rotation), OT based on raw features may match images according to orientation rather than object shape — i.e., the transport plan is sensitive to nuisance variation and ignores the intrinsic semantic structure of the data.

**Key Challenge**: A transport plan is needed that simultaneously eliminates nuisance introduced by symmetry transformations (e.g., differing rotation angles) and retains discriminative information (e.g., object shape). Simple invariant features such as the power spectrum can remove group actions but sacrifice structural information.

**Goal**: How can symmetry awareness be encoded into OT so that transport plans are invariant to group actions yet selectively sensitive to signal structure?

**Key Insight**: The bispectrum is the lowest-order complete invariant in group Fourier analysis — it jointly encodes signal structure and group invariance, removing variation due to group actions and nothing else.

**Core Idea**: Replace raw features in the OT cost matrix with bispectral representations, yielding transport plans that are invariant to symmetry transformations while retaining discriminative information.

## Method

### Overall Architecture
Input: two datasets (potentially differing by symmetry transformations). Construct bispectral feature representations → compute pairwise distances in bispectral space as the OT cost matrix → solve Sinkhorn OT → obtain a symmetry-aware transport plan.

### Key Designs

1. **Bispectrum Feature Extraction**:

    - **Function**: Transform images into rotation-invariant bispectral representations.
    - **Mechanism**: (1) Convert an $M \times N$ image to polar coordinates of size $R \times K$ ($R$ radial bins, $K$ angular bins); (2) apply 1D DFT to the angular slice at each radius $r$; (3) compute the bispectrum $B_{i,j} = \hat{f}_i \hat{f}_j \hat{f}_{i+j}^\dagger$ from the DFT coefficients; (4) concatenate across radii to obtain a global SO(2)-invariant representation.
    - **Design Motivation**: The bispectrum is the lowest-order complete spectral invariant — the power spectrum is also invariant but discards phase information (incomplete), whereas the bispectrum preserves relative phase structure.

2. **Bispectral OT Problem**:

    - **Function**: Solve OT in the bispectral feature space.
    - **Mechanism**: The cost matrix is defined as $C_{ij} = d(B(\mathbf{x}^{(i)}), B(\mathbf{y}^{(j)}))$, where $B(\cdot)$ denotes bispectral features and $d$ can be L1, L2, or cosine distance. The standard Sinkhorn algorithm is then applied.
    - **Design Motivation**: Computing distances in an invariant feature space allows the transport plan to be naturally symmetry-aware.

### Loss & Training
No training is required. BOT is a purely computational framework that solves a standard OT problem in bispectral feature space.

## Key Experimental Results

### Main Results

| Dataset | OT (L1, raw pixels) | BOT (L1) | BOT (L2) | BOT (cosine) |
|--------|---------------------|----------|----------|--------------|
| MNIST | 0.330 | **0.841** | 0.802 | 0.816 |
| KMNIST | 0.242 | **0.782** | 0.723 | 0.735 |
| FMNIST | 0.300 | **0.762** | **0.766** | 0.732 |
| EMNIST | 0.197 | **0.598** | 0.569 | 0.572 |

Note: Values are class-preservation accuracy (proportion of OT matches between rotated and non-rotated sets that correspond to the same class).

### Ablation Study

| Analysis | Finding |
|----------|---------|
| MDS visualization | Rotated variants cluster by class in bispectral space; scattered in pixel space |
| Rotationally symmetric digit (0) | Tightest cluster in bispectral space |
| Digits 6 and 9 | Close in bispectral space (equivalent under 180° rotation) |
| Baseline (no rotation) | Raw-pixel OT already high (~97%); BOT also high (~86%) |

### Key Findings
- **BOT improves class-preservation accuracy by 2–3×**: from 20–33% to 60–84%, consistently across all datasets.
- **L1 distance is most effective in bispectral space**: likely due to the sparse structure of the bispectrum being better suited to L1.
- **Rotational symmetry is elegantly captured**: digit 0 yields the tightest cluster (high rotational symmetry), digit 2 the most dispersed (low symmetry), and 6/9 are proximal.
- **The gap on the no-rotation baseline reveals partial information loss in bispectral representations**: without rotation, raw-pixel OT (97%) outperforms BOT (86%), indicating a trade-off with image reconstruction fidelity.

## Highlights & Insights
- **Using the bispectrum as OT cost features is a first**: this work introduces tools from group theory and harmonic analysis into OT, offering both novelty and theoretical depth.
- **The concept of a complete invariant is central**: the power spectrum is invariant but incomplete (loses information), whereas the bispectrum is the lowest-order complete invariant — striking an optimal balance between nuisance removal and information preservation.
- **MDS visualizations** intuitively demonstrate the elegant geometric structure of the bispectral space and carry strong pedagogical value.

## Limitations & Future Work
- **Currently limited to SO(2) rotational symmetry**: extending to SO(3) or non-abelian groups substantially increases bispectrum computation complexity.
- **Validated only on MNIST-type simple datasets**: whether bispectral representations remain effective for natural images (e.g., ImageNet) with more complex symmetries is unknown.
- **Performance degrades without rotation**: bispectral representations incur information loss (97% → 86%), and should not be used when symmetry invariance is unnecessary.
- **Choice of distance metric**: standard norms (L1/L2/cosine) are used, but dedicated distances tailored to bispectral space may yield further improvements.

## Related Work & Insights
- **vs. Standard OT**: Standard OT is symmetry-unaware and matching is dominated by nuisance variation.
- **vs. Gromov-Wasserstein**: GW is invariant to isometric transformations but does not handle general group actions.
- **vs. Data Augmentation + OT**: Invariance can be obtained indirectly through data augmentation, but at high computational cost and without completeness guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First introduction of the bispectrum into OT; theoretically elegant
- Experimental Thoroughness: ⭐⭐⭐ Limited to MNIST-type datasets; lacks large-scale or natural image validation
- Writing Quality: ⭐⭐⭐⭐ Mathematical background is clearly presented; experimental sections are relatively brief
- Value: ⭐⭐⭐⭐ Proposes a promising new direction; practical impact remains to be verified

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](../../ICLR2026/llm_evaluation/optimal_transport-induced_samples_against_out-of-distribution_overconfidence.md)
- [\[NeurIPS 2025\] ComPO: Preference Alignment via Comparison Oracles](compo_preference_alignment_via_comparison_oracles.md)
- [\[NeurIPS 2025\] PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations](pfδ_a_benchmark_dataset_for_power_flow_under_load_generation_and_topology_variat.md)
- [\[NeurIPS 2025\] CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)
- [\[NeurIPS 2025\] SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
