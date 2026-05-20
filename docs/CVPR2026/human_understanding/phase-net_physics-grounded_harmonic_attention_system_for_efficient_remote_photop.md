---
title: >-
  [Paper Note] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement
description: >-
  [CVPR 2026][Human Understanding][rPPG] Starting from the Navier-Stokes equations, this work derives through rigorous mathematical analysis that rPPG pulse signals obey a second-order damped harmonic oscillator model whos…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "rPPG"
  - "physics-informed network"
  - "temporal convolutional network"
  - "hemodynamics"
  - "Navier-Stokes"
  - "lightweight model"
date: 2026-05-08
content_hash: 761f5dab7d437755
---

# PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement

**Conference**: CVPR 2026
**arXiv**: [2509.24850](https://arxiv.org/abs/2509.24850)  
**Code**: [GitHub](https://github.com/Alex036225/PhaseNet)  
**Area**: Human Understanding
**Keywords**: rPPG, physics-informed network, temporal convolutional network, hemodynamics, Navier-Stokes, lightweight model

## TL;DR

Starting from the Navier-Stokes equations, this work derives through rigorous mathematical analysis that rPPG pulse signals obey a second-order damped harmonic oscillator model whose discrete solution is equivalent to a causal convolution operator, thereby providing a first-principles justification for the TCN architecture. The resulting PHASE-Net, with only 0.29M parameters, achieves state-of-the-art performance across multiple datasets.

## Background & Motivation

**Background**: Remote photoplethysmography (rPPG) extracts physiological signals such as heart rate by capturing subtle changes in subcutaneous blood volume via ordinary cameras, and is a key technology for contactless physiological monitoring. Deep learning methods (PhysNet, PhysFormer, RhythmMamba, etc.) have become the dominant paradigm.

**Limitations of Prior Work**:

1. Most existing deep learning models are designed heuristically—treating rPPG as a generic spatiotemporal signal processing task, with architecture choices relying on empirical trial and error.
2. The lack of physical theoretical grounding means models may overfit dataset-specific noise patterns, resulting in poor cross-domain generalization.
3. Artifacts from head motion and illumination changes are far stronger than genuine pulse signals, and "black-box" models cannot provide reliability guarantees.

**Key Challenge**: High-performance deep learning models vs. lack of physical interpretability and theoretical guarantees.

**Goal**: Can one design an rPPG model whose architecture directly embodies the physical laws governing the signal, starting from first principles?

**Key Insight**: Derive blood flow pulse dynamics from the Navier-Stokes equations and rigorously prove that TCN is the physically correct architectural choice.

**Core Idea**: The physical dynamics of rPPG signals are equivalent to causal convolution; therefore, TCN is not a heuristic choice but a physical inevitability.

## Method

### Overall Architecture

Visual encoder (3 EST Blocks, each containing a ZAS module) extracts spatiotemporal features → Adaptive Spatial Filter (ASF) generates spatial attention masks, aggregates features, and computes temporal differences → Gated Temporal Convolutional Network (GTCN) models long-range temporal dynamics → rPPG waveform output.

### Key Designs

1. **Physics Derivation Chain: From Navier-Stokes to TCN**

    - Starting point: The Beer-Lambert law establishes a linear relationship between pixel variation $\Delta I(t)$ and subcutaneous blood volume $\Delta V(t)$; vascular compliance further links $\Delta V(t)$ to local blood pressure pulsation $z(t)$.
    - Linearizing the Navier-Stokes equations → 1D momentum + continuity equations → eliminating velocity variables yields a damped wave equation: $\frac{\partial^2 p'}{\partial t^2} + \alpha \frac{\partial p'}{\partial t} = c^2 \frac{\partial^2 p'}{\partial x^2}$
    - At a fixed observation point $x_0$, this degenerates to a second-order ODE (damped harmonic oscillator): $\ddot{z} + \alpha \dot{z} + \omega^2 z = u(t)$
    - Semi-implicit Euler discretization → LTI state-space model → **Proposition 1** proves its solution is a causal convolution $z_t = \sum_{m=0}^{\infty} g[m] \cdot a_{t-m}$ → **Proposition 2** proves that an FIR filter can approximate the IIR to arbitrary precision $\varepsilon$ → TCN is the exact computational realization of this physical process.
    - Significance: This establishes, for the first time, a complete logical chain from hemodynamic first principles to a specific network architecture.

2. **Zero-FLOPs Axial Swapper (ZAS)**

    - Applies intra-block spatial transposition to the last $k = \lfloor pC \rfloor$ channels of the feature map (dividing $H \times W$ into $b \times b$ blocks and performing matrix transposition), leaving the remaining channels unchanged.
    - Key properties: self-inverse (ZAS(ZAS($X$)) = $X$, guaranteeing invertibility and gradient stability); energy-preserving ($\|$ZAS($X$)$\|_2 = \|X\|_2$, 1-Lipschitz to prevent signal amplification).
    - Design Motivation: Injects cross-region spatial interaction with zero FLOPs and zero parameters, enhancing feature mixing across distant facial regions.

3. **Adaptive Spatial Filter (ASF)**

    - For each frame, a lightweight convolution generates a spatial logit map → spatial softmax normalization produces an attention mask $M_t$ → weighted aggregation over the spatial dimensions yields a 1D feature vector $\mathbf{z}_t$.
    - Simultaneously computes a first-order temporal difference $\mathbf{v}_t = \mathbf{z}_t - \mathbf{z}_{t-1}$ to encode pulse "velocity."
    - Output = $[\mathbf{z}_t, \mathbf{v}_t]$ channel concatenation, retaining both spatially purified intensity information and short-term temporal variation.
    - Design Motivation: The forehead and cheeks have high SNR, while other regions are dominated by noise—making global average pooling (GAP) suboptimal.

4. **Gated Temporal Convolutional Network (GTCN)**

    - Dual-path causal dilated TCN: one path with tanh activation, one path with sigmoid gating → element-wise multiplication for fusion.
    - Physical significance: Implements the causal convolution operation derived in Propositions 1 & 2, modeling long-range temporal dynamics.

### Loss & Training

Negative Pearson correlation loss: $\mathcal{L}_{\text{pred}} = -\frac{\sum_t (\hat{y}_t - \bar{\hat{y}})(y_t - \bar{y})}{\sqrt{\sum_t (\hat{y}_t - \bar{\hat{y}})^2 \sum_t (y_t - \bar{y})^2}}$, directly optimizing morphological similarity between the predicted waveform and the ground truth.

## Key Experimental Results

### Main Results (In-Domain Evaluation)

| Method | UBFC MAE↓ | UBFC RMSE↓ | PURE MAE↓ | PURE RMSE↓ | BUAA MAE↓ | MMPD MAE↓ | Params |
|--------|-----------|------------|-----------|------------|-----------|-----------|--------|
| PhysNet | 2.95 | 3.67 | 2.10 | 2.60 | 10.89 | 4.80 | Large |
| PhysFormer | 0.92 | 2.46 | 1.10 | 1.75 | 8.45 | 11.99 | Large |
| RhythmFormer | 0.50 | 0.78 | 0.27 | 0.47 | 9.19 | 4.69 | Medium |
| Contrast-Phys+ | 0.21 | 0.80 | 0.48 | 0.98 | - | - | Medium |
| Style-rPPG | 0.17 | 0.41 | 0.39 | 0.62 | - | - | Medium |
| LST-rPPG | 0.16 | 0.57 | 0.32 | 0.62 | - | - | Medium |
| **PHASE-Net** | **0.15** | **0.53** | **0.14** | **0.35** | **5.89** | **4.78** | **0.29M** |

### Ablation Study (Cross-Domain Generalization, Leave-One-Out)

| Method | Others→U MAE↓ | Others→P MAE↓ | Others→B MAE↓ | Others→M MAE↓ |
|--------|---------------|---------------|---------------|---------------|
| PhysFormer | 10.29 | 19.75 | 22.09 | 13.90 |
| RhythmFormer | 14.71 | 21.11 | 6.04 | 16.14 |
| EfficientPhys | 12.87 | 7.15 | 32.30 | 12.87 |
| **PHASE-Net** | **10.04** | **2.86** | - | - |

### Key Findings

- MAE of 0.14 bpm on PURE, halving that of RhythmFormer (0.27)—physical priors as inductive bias significantly improve accuracy.
- SOTA performance achieved with only 0.29M parameters—theoretical rigor and extreme efficiency are unified.
- Cross-domain Others→PURE MAE of 2.86 bpm, substantially outperforming PhysFormer (19.75) and RhythmFormer (21.11)—physical priors enhance generalization.
- On challenging datasets such as BUAA and MMPD, methods like PhysFormer exhibit negative correlation ($R < 0$), while PHASE-Net maintains positive correlation.

## Highlights & Insights

- **First derivation of an rPPG network architecture from first principles**: The complete mathematical proof chain from Navier-Stokes → ODE → SSM → causal convolution → TCN elevates architectural selection from empiricism to physical inevitability.
- **ZAS zero-FLOPs module**: Pure permutation operations enhance cross-region feature interaction; the mathematical proofs of self-inverse and energy-preserving properties elegantly guarantee training stability.
- **Temporal difference design in ASF**: Unifies spatial aggregation and temporal differentiation in a single module, providing the downstream physical model with complete state information in the form of "position + velocity."
- **Paradigm of theoretical rigor combined with engineering minimalism**: 0.29M parameters demonstrate that well-chosen inductive biases can dramatically reduce model complexity.

## Limitations & Future Work

- The physical derivation relies on several simplifying assumptions (laminar flow, linearization, single-point observation, elastic restoring force approximation), which may not hold under extreme motion or atypical vascular conditions.
- The block size $b$ and channel ratio $p$ of ZAS require manual specification, lacking an adaptive mechanism.
- Validation on large-scale in-the-wild datasets such as VIPL-HR has not been performed.
- The cross-domain generalization table has missing entries (Others→B, Others→M), leaving the evaluation incomplete.

## Related Work & Insights

- **vs. PhysFormer/RhythmMamba**: These methods model temporal sequences with Transformers/SSMs, representing general-purpose sequence models; PHASE-Net demonstrates from a physical perspective that causal convolution (TCN) is the correct computational primitive for rPPG.
- **vs. conventional PINN paradigm**: Classical PINNs embed physical equations into the loss function; PHASE-Net's innovation lies in using physical laws to constrain the network architecture itself—"physics determines structure" rather than "physics constrains training."
- Implication: This methodology of "deriving network structure from PDEs" is generalizable to other signal processing tasks with well-defined physical models (e.g., seismic waves, acoustic signals).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First derivation of an rPPG network architecture from first principles; paradigm-level significance.
- Experimental Thoroughness: ⭐⭐⭐⭐ In-domain and cross-domain evaluation on 4 datasets with comprehensive ablations, though some cross-domain entries are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous derivations and a clear, coherent logical chain from physics to architecture.
- Value: ⭐⭐⭐⭐⭐ The physics-driven architecture design paradigm has broad applicability; the extreme efficiency of 0.29M parameters is well-suited for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)
- [\[CVPR 2026\] TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)
- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[AAAI 2026\] MVGD-Net: A Novel Motion-aware Video Glass Surface Detection Network](../../AAAI2026/human_understanding/mvgd-net_a_novel_motion-aware_video_glass_surface_detection_network.md)

</div>

<!-- RELATED:END -->
