---
title: >-
  [Paper Note] Towards a Foundation Model for Partial Differential Equations Across Physics Domains
description: >-
  [AAAI 2026][Scientific Computing][PDE] This paper proposes PDE-FM, a modular PDE foundation model combining spatial-spectral dual-modal tokenization, FiLM-based physics modulation, and a Mamba state-space backbone. It achieves an average 46% reduction in VRMSE across 12 heterogeneous physics-domain datasets from The Well benchmark.
tags:
  - AAAI 2026
  - Scientific Computing
  - PDE
  - foundation model
  - neural operator
  - Mamba
  - FNO
  - multi-physics
  - The Well benchmark
date: 2026-05-08
content_hash: 79987a92f9f8e3c1
---

# Towards a Foundation Model for Partial Differential Equations Across Physics Domains

**Conference**: AAAI 2026
**arXiv**: [2511.21861](https://arxiv.org/abs/2511.21861)
**Code**: None
**Area**: Scientific Computing
**Keywords**: PDE, foundation model, neural operator, Mamba, FNO, multi-physics, The Well benchmark

## TL;DR
This paper proposes PDE-FM, a modular PDE foundation model combining spatial-spectral dual-modal tokenization, FiLM-based physics modulation, and a Mamba state-space backbone. It achieves an average 46% reduction in VRMSE across 12 heterogeneous physics-domain datasets from The Well benchmark.

## Background & Motivation

### State of the Field

Existing neural operators (FNO, GNOT, Transformer-based methods, etc.) achieve strong performance on specific PDE types but are domain-specific—trained on individual datasets and applicable only to narrow classes of PDEs. NLP and vision have widely adopted the "pretrain once, transfer broadly" paradigm of foundation models, yet scientific computing has yet to achieve a comparable breakthrough.

### Limitations of Prior Work

(1) Existing neural operators experience severe performance degradation when boundary conditions or physical laws change, precluding cross-physics-domain transfer. (2) The unique challenges of physical systems—multi-resolution multi-scale dynamics, conservation law constraints, continuous spatiotemporal evolution, and nonlinear operator coupling—make unified modeling extremely difficult. (3) The $O(N^2)$ complexity of Transformer-based methods on high-resolution grids limits the tractable problem scale.

### Root Cause

PDEs from different physics domains (fluid dynamics, radiation, elasticity, astrophysics) exhibit fundamentally different equation forms, boundary conditions, and conservation laws, making unified modeling with a single model seemingly intractable. However, physical systems share an underlying duality of "local structure + global constraints," which provides a basis for a unified architecture.

### Paper Goals

The paper aims to design a unified foundation model architecture that, after pretraining on heterogeneous PDE systems, can transfer to new physics domains without architectural modification. **Key Insight**: spatial tokens capture local structure, spectral tokens encode global constraints, and FiLM modulation injects physical conditions. **Core Idea**: a modular design—comprising replaceable tokenizers, backbones, decoders, and condition injection mechanisms—where systematic ablation identifies the optimal combination.

## Method

### Overall Architecture

Given input PDE state $u \in \mathbb{R}^{C \times H \times W}$, the model applies spatial and spectral dual-modal tokenization, injects physical metadata (e.g., boundary conditions) via FiLM modulation, fuses both token types through cross-attention, models spatiotemporal evolution with a Mamba backbone, and produces predictions via an FNO decoder. The framework supports joint pretraining across multiple datasets.

### Key Designs

1. **Dual-Modal Tokenization (Spatial + Spectral)**

   - **Function**: Simultaneously encodes local spatial structure and global spectral characteristics.
   - **Mechanism**: Spatial tokens $T_{spatial} = \text{PatchConv}(u) \in \mathbb{R}^{N_p \times d}$ are extracted via patch convolution to capture local features; spectral tokens $T_{spectral} = \text{Linear}(\text{FFT}_m(u)) \in \mathbb{R}^{1 \times d}$ retain global structural information from low-frequency modes (keeping only the first $m$ frequency components). Cross-attention enables bidirectional information fusion.
   - **Design Motivation**: PDE solutions simultaneously exhibit local spatial gradient structure and global spectral properties (e.g., periodic boundaries, conserved quantities) that a single tokenization scheme cannot capture. A single spectral token serves as a "global summary" to govern context allocation.

2. **FiLM Physics Condition Modulation**

   - **Function**: Injects physical metadata (boundary conditions, constitutive parameters, temporal grid) into the model.
   - **Mechanism**: Physical conditions $c$ modulate tokens via an affine transformation: $\tilde{T}_{spatial} = T_{spatial} \odot (1 + \gamma(c)) + \beta(c)$, where $\gamma$ and $\beta$ are learnable mappings.
   - **Design Motivation**: PDEs from different physics domains involve distinct parameters (Reynolds number, Mach number, etc.). FiLM injects this conditioning information in an extremely lightweight manner (two vectors), avoiding the need for domain-specific branches for each physical configuration.

3. **Mamba State-Space Backbone + FNO Decoder**

   - **Function**: Efficiently models long-sequence spatiotemporal evolution while preserving spectral smoothness.
   - **Mechanism**: Mamba layers $T^{(l+1)} = T^{(l)} + \text{MambaLayer}(T^{(l)})$ replace the $O(N_p^2)$ complexity of Transformers with $O(N_p d)$ linear complexity, supporting large grids and long contexts. The FNO spectral decoder $\hat{u}(x) = \sum_{|k| \leq m} W_k \cdot \mathcal{F}[z](k) e^{2\pi i k \cdot x}$ preserves spectral smoothness priors.
   - **Design Motivation**: Mamba's selective state-space structure is naturally suited to sequential evolution modeling, as PDE solving is fundamentally a time-marching process. The FNO decoder leverages spectral priors to prevent spatial aliasing.

### Loss & Training

A dual-objective loss is used: $\mathcal{L} = \text{VRMSE} + \lambda \sum_k w(k) \|\hat{U}(k) - U(k)\|^2$ (high-frequency weighted), with optional conservation law constraints. Multi-dataset sampling is governed by $p(i) \propto (\epsilon + \bar{\mathcal{L}}_i)^\alpha \cdot |\mathcal{D}_i|^\tau$, combining difficulty-aware weighting and temperature scaling. Dataset-specific $1\times1$ convolutional adapters unify channel dimensions across datasets.

## Key Experimental Results

### Main Results

Evaluated on 12 cross-physics-domain datasets from The Well benchmark.

| Dataset | FNO VRMSE | CNextU-net | **PDE-FM** | Reduction |
|--------|:---:|:---:|:---:|:---:|
| rayleigh_benard | 0.8395 | 0.6699 | **0.0415** | 95.1% |
| shear_flow | 1.189 | 0.808 | **0.0345** | 97.1% |
| gray_scott_RD | 0.1365 | 0.1761 | **0.0183** | 86.6% |
| post_neutron_star | 0.3866 | — | **0.2995** | 22.5% |
| turbulence_gravity | 0.2429 | 0.2096 | **0.0796** | 67.2% |

PDE-FM achieves state-of-the-art on 6 of 12 datasets and second-best on 5. Average VRMSE is reduced by 46%.

### Ablation Study

| Configuration | Mean VRMSE | Note |
|------|:---:|------|
| Full (Mamba+FNO+SpecTok+XAttn+FiLM) | **0.2581** | Optimal configuration |
| w/o Spectral Token | 0.3012 | −16.7%, global structure lost |
| w/o FiLM Modulation | 0.2891 | −12.0%, physical conditions not injected |
| Transformer replacing Mamba | 0.2743 | −6.3%, with higher complexity |
| w/o FNO Decoder | 0.2956 | −14.5%, spectral smoothness prior absent |

### Key Findings
- Rayleigh-Bénard and shear_flow show the most significant improvements (>95% VRMSE reduction), as these are strongly turbulent scenarios where global spectral modeling is most advantageous.
- Spectral tokens contribute most to performance (16.7% gain), validating the necessity of the global-local dual-modal design.
- Difficulty-aware sampling effectively mitigates negative transfer, with the greatest improvements observed on challenging datasets such as active_matter.

## Highlights & Insights
- **A genuine cross-physics-domain foundation model**: The same model handles fluid turbulence, neutron star mergers, and supernovae, demonstrating the feasibility of unified modeling across physical systems.
- **Complementary combination of Mamba and FNO**: Mamba provides linear-complexity temporal modeling while FNO enforces physics constraints in the spectral domain; the two are synergistic.
- **Spatial-spectral dual-modal tokenization**: A single spectral token serves as a global summary to govern context allocation—a simple yet highly effective design.

## Limitations & Future Work
- Ablations are conducted under short training schedules (8 epochs, 600 steps), which may not fully reflect the contribution of each component.
- Performance on active_matter and helmholtz_staircase is inferior to U-Net variants.
- The model architecture is complex (Tokenizer + CrossAttn + Mamba + FNO), and training costs are not reported.
- Results on 3D datasets are less thoroughly documented than those on 2D datasets.

## Related Work & Insights
- **vs. FNO**: Domain-specific, no pretraining, $O(N\log N)$ complexity; PDE-FM enables cross-domain pretraining with $O(Nd)$ linear complexity.
- **vs. PhysiX**: Partial cross-domain capability but lacks a unified pretraining strategy; PDE-FM's difficulty-aware sampling and FiLM modulation are more systematic.
- **vs. OmniArch**: Shares similar goals but differs in architecture; PDE-FM's Mamba backbone is more computationally efficient.
- FiLM modulation is a lightweight and effective means of incorporating physical metadata, generalizable to other scientific computing tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic attempt at a cross-physics-domain PDE foundation model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across 12 heterogeneous datasets, though ablation training is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, and the modular design is well-documented.
- **Value**: ⭐⭐⭐⭐⭐ — Pioneering significance for the direction of foundation models in scientific computing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Trajectory Physics-Informed Neural Networks for HJB Equations with Hard-Zero Terminal Inventory: Optimal Execution on Synthetic & SPY Data](../../NeurIPS2025/scientific_computing/multi-trajectory_physics-informed_neural_networks_for_hjb_equations_with_hard-ze.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[AAAI 2026\] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids](phys-liquid_a_physics-informed_dataset_for_estimating_3d_geometry_and_volume_of_.md)
- [\[CVPR 2026\] PhysSkin: Real-Time and Generalizable Physics-Based Skin Simulation](../../CVPR2026/scientific_computing/physskin_real-time_and_generalizable_physics-based_animation_via_self-supervised.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)

</div>

<!-- RELATED:END -->
