---
title: >-
  [Paper Note] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning
description: >-
  [ICLR2026][Physics & Scientific Computing][neural operator] DRIFT-Net is a dual-branch neural operator that addresses autoregressive drift caused by insufficient global spectral coupling in window attention…
tags:
  - "ICLR2026"
  - "Physics & Scientific Computing"
  - "neural operator"
  - "PDE"
  - "spectral coupling"
  - "dual-branch"
  - "Navier-Stokes"
date: 2026-05-08
content_hash: 8916edf5c6546767
---

# DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning

**Conference**: ICLR2026  
**arXiv**: [2509.24868](https://arxiv.org/abs/2509.24868)  
**Code**: None  
**Area**: Scientific Computing  
**Keywords**: neural operator, PDE, spectral coupling, dual-branch, Navier-Stokes

## TL;DR
DRIFT-Net is a dual-branch neural operator that addresses autoregressive drift caused by insufficient global spectral coupling in window attention, via controlled low-frequency mixing (spectral branch), local detail fidelity (image branch), and bandwidth fusion through radial gating. It reduces error by 7%–54% on Navier-Stokes benchmarks.

## Background & Motivation

### State of the Field

**Background**: Most PDE foundation models employ multi-scale window self-attention, which is computationally efficient but propagates global dependencies only gradually through deep stacking and window shifting.

**Limitations of Prior Work**: The locality of window attention weakens global spectral coupling, leading to drift in closed-loop rollouts; purely spectral operators (e.g., FNO) over-emphasize low frequencies.

**Key Challenge**: The inherent trade-off between global coupling and local detail fidelity.

**Goal**: Enhance global spectral coupling while preserving high-frequency details.

**Core Idea**: Learnable low-frequency linear mixing + radial gating bandwidth fusion + frequency-weighted loss.

## Method

### Overall Architecture
A U-Net encoder–decoder in which each scale contains a spectral branch (rFFT2 → low-frequency mixing → bandwidth fusion → iFFT2) and an image branch (ConvNeXt), combined via additive fusion.

### Key Designs

1. **Controlled Low-Frequency Mixing**: After rFFT2, a learnable complex linear transform is applied exclusively to low-frequency components while high frequencies remain unchanged, preventing interference with fine details.
2. **Bandwidth Fusion + Radial Gating**: $\hat{Y}(k) = \alpha(k)\hat{V}_{low}(k) + (1-\alpha(k))\hat{X}_{high}(k)$; the convex combination guarantees no energy overshoot.
3. **Frequency-Weighted Loss**: $w(r) \propto r^\alpha$ up-weights high-frequency errors to counteract spectral bias.

### Loss & Training
Single-step teacher-forcing training; autoregressive closed-loop rollout at test time.

## Key Experimental Results

### Main Results: 7 PDE Benchmarks

| Task | scOT | FNO | **DRIFT-Net** |
|------|------|-----|------|
| NS-SL | 3.96% | 3.69% | **3.40%** |
| NS-PwC | 2.35% | 4.57% | **Best** |
| Poisson-Gauss | — | — | **Best** |
| Allen-Cahn | — | — | **Best** |
| Wave-Gauss | — | — | **Best** |

### Efficiency Comparison
Approximately 15% fewer parameters than scOT with higher throughput; NS errors reduced by 7%–54%.

### Ablation Study

| Configuration | Effect |
|------|------|
| w/o low-frequency mixing | Significant error increase |
| Hard mask instead of radial gating | Instability |
| w/o frequency-weighted loss | Insufficient high-frequency fitting |
| Full DRIFT-Net | Best |

### Key Findings
- Controlled low-frequency mixing is critical — removing it causes a substantial error increase.
- Low drift is maintained over 100-step long-horizon rollouts.
- Effective across elliptic, parabolic, and hyperbolic PDEs.

## Highlights & Insights
- The spectral–spatial dual-branch elegantly decouples global structure from local details, with strong physical intuition.
- The convex combination in non-expansive fusion ensures training stability.
- Modular design — the DRIFT block can replace existing attention blocks.

## Limitations & Future Work
- The low-frequency mask size requires manual tuning.
- Validation is primarily on 2D PDEs; extension to 3D remains untested.
- Comparison with other PDE foundation models such as DPOT is insufficient.

## Related Work & Insights
- **vs. scOT/POSEIDON**: Achieves global coupling via the spectral branch without requiring deep stacking.
- **vs. FNO**: FNO operates over all frequencies but lacks local capacity; DRIFT-Net's dual branches are complementary.
- **vs. PDE-Refiner**: PDE-Refiner relies on iterative refinement, whereas DRIFT-Net achieves fidelity through architectural design.

## Rating
- Novelty: ⭐⭐⭐⭐ An elegant combination of controlled low-frequency mixing, bandwidth fusion, and frequency-weighted loss.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven PDE benchmarks, ablations, and long-horizon rollout evaluation.
- Writing Quality: ⭐⭐⭐⭐ Physical intuition is well articulated.
- Value: ⭐⭐⭐⭐ Provides a stronger backbone for PDE foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](../../ICML2026/physics/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[AAAI 2026\] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](../../AAAI2026/physics/saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)
- [\[ICML 2026\] MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties](../../ICML2026/physics/mōle-λ_learning_the_coupled-cluster_response_state_for_energies_gradients_and_pr.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/physics/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

</div>

<!-- RELATED:END -->
