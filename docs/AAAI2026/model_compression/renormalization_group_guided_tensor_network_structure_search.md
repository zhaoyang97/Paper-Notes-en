---
title: >-
  [Paper Note] Renormalization Group Guided Tensor Network Structure Search
description: >-
  [AAAI 2026][Model Compression][Tensor Network] This paper proposes RGTN, a framework that introduces Renormalization Group (RG) theory from statistical physics into tensor network structure search. Through a multi-scale coarse-graining–expansion–compression pipeline and learnable edge gating, RGTN enables continuous topological evolution, achieving state-of-the-art compression ratios on light field compression, high-order tensor decomposition, and video completion tasks, while running 4–600× faster than existing methods.
tags:
  - AAAI 2026
  - Model Compression
  - Tensor Network
  - Structure Search
  - Renormalization Group
  - Multi-Scale Optimization
  - Tensor Decomposition
date: 2026-05-08
content_hash: 62e0fa1d8bd1bc31
---

# Renormalization Group Guided Tensor Network Structure Search

**Conference**: AAAI 2026
**arXiv**: [2512.24663](https://arxiv.org/abs/2512.24663)
**Code**: [Available](https://github.com/Applied-Machine-Learning-Lab/RGTN)
**Area**: Model Compression
**Keywords**: Tensor Network, Structure Search, Renormalization Group, Multi-Scale Optimization, Tensor Decomposition

## TL;DR

This paper proposes RGTN, a framework that introduces Renormalization Group (RG) theory from statistical physics into tensor network structure search. Through a multi-scale coarse-graining–expansion–compression pipeline and learnable edge gating, RGTN enables continuous topological evolution, achieving state-of-the-art compression ratios on light field compression, high-order tensor decomposition, and video completion tasks, while running 4–600× faster than existing methods.

## Background & Motivation

**Background**: Tensor Network Structure Search (TN-SS) aims to automatically discover optimal tensor network topologies and rank configurations for efficient decomposition of high-dimensional data. Existing approaches include genetic algorithms (TNGA), greedy construction (TNGreedy), local search (TNLS), and the more recent unified optimization method SVDinsTN.

**Limitations of Prior Work**: (1) **Single-scale optimization**—existing methods search at a fixed resolution, ignoring the inherent multi-scale correlation structure of tensor data; (2) **Discrete search spaces**—genetic, greedy, and local search methods all operate in combinatorially explosive discrete spaces, making smooth evolution difficult; (3) **Decoupled structure–parameter optimization**—the sample-and-evaluate paradigm requires full optimization of each candidate structure, incurring prohibitive computational cost.

**Key Challenge**: The population size of genetic methods grows exponentially with tensor order; greedy algorithms make irreversible local decisions leading to suboptimal solutions; local search is prone to local minima. Although SVDinsTN achieves 100–1000× acceleration via unified optimization, single-scale search remains susceptible to local optima.

**Key Insight**: The work draws inspiration from RG theory in condensed matter physics—RG reveals the behavioral regularities of physical systems across different scales. Tensor networks naturally possess three properties that correspond to RG: scale-invariant correlations, hierarchical entanglement, and information flow, making it possible to directly adopt the multi-scale analysis framework of RG.

**Core Idea**: TN-SS is reformulated as a continuous optimization problem over a multi-scale RG flow, progressively refining network structure from coarse to fine, while using node tension and edge information flow as physical indicators to guide structural modifications, thereby unifying structure search and parameter optimization.

## Method

### Overall Architecture

RGTN begins at the coarsest scale $s=S$ and progressively advances toward the finest scale $s=0$. At each scale, three phases are executed sequentially: (1) **Coarse-graining**—downsampling the data $\mathcal{F}_s = \mathcal{D}_s[\mathcal{F}]$ to reduce dimensionality; (2) **Expansion**—splitting high-tension nodes to increase expressive capacity; (3) **Compression**—merging low-information-flow edges to eliminate redundancy. The RG transformation is decomposed into an expansion operator $R_{\text{expand}}: \mathcal{G}_v \mapsto \sum_r \mathcal{G}_u^{(r)} \otimes \mathcal{G}_v^{(r)}$ and a compression operator $R_{\text{compress}}: (\mathcal{G}_u, \mathcal{G}_v) \mapsto \mathcal{G}_u \times_{e_{uv}} \mathcal{G}_v$, with structure and parameter optimization unified through a scale-dependent effective action.

### Key Designs

1. **Learnable Edge Gating Mechanism**:

    - **Function**: Introduces a continuous soft switch $g_{uv} = \sigma(w_{uv})$ for each edge in the network graph, enabling dynamic topological modification during optimization.
    - **Mechanism**: Gated tensor contraction $\mathcal{C}_{uv} = g_{uv} \cdot (\mathcal{G}_u \times \mathcal{G}_v) + (1 - g_{uv}) \cdot \mathcal{I}$ removes an edge as $g_{uv} \to 0$. Temperature annealing $\tau(t) = \tau_0 \exp(-t/t_0)$ gradually transitions from soft to hard decisions.
    - **Design Motivation**: Addresses the issue in conventional methods where structural modifications are discrete operations that disrupt optimization. Continuous gating allows topological evolution and parameter optimization to proceed within the same gradient-based framework, maintaining stable gradient flow.

2. **Physics-Driven Informed Proposal Generation**:

    - **Function**: Guides expansion and compression based on physical quantity indicators derived from the current network, rather than random search.
    - **Mechanism**: **Node tension** $T_v = \|\partial \mathcal{L}_{\text{data}} / \partial \mathcal{G}_v\|_F \cdot \text{degree}(v)$ measures a node's contribution to reconstruction error, with high-tension nodes prioritized for splitting; **edge information flow** $I_{uv} = g_{uv} \cdot \text{MI}(\mathcal{G}_u, \mathcal{G}_v)$ quantifies connection importance, with low-information-flow edges prioritized for merging.
    - **Design Motivation**: Avoids blind combinatorial search by leveraging the "stress distribution" of the current optimization state to guide targeted structural modifications, analogous to material design guided by force analysis in physics.

3. **Multi-Scale Progressive Refinement**:

    - **Function**: After solving at a coarse scale, an upsampling operator $\mathcal{U}_s$ lifts the coarse-scale solution to the finer scale as initialization: $\mathcal{M}_{s-1}^{(0)} = \mathcal{U}_s[\mathcal{M}_s^*]$.
    - **Mechanism**: At coarse scales, the Lipschitz constant $L_s \leq L_0 \cdot 2^{-s\alpha}$ decreases, yielding a smoother loss landscape that facilitates finding globally favorable structures. Scale transitions introduce perturbations that help escape local minima with probability $\mathbb{P}(\text{escape}) \geq 1 - \prod_{s=1}^S \Phi(r/(2^s \sigma_0))$.
    - **Design Motivation**: A divide-and-conquer strategy of solving small problems before large ones reduces search complexity from $\Omega(\exp(N^2))$ to $\mathcal{O}(S \cdot T \cdot N^2 \cdot I^N \cdot R_{\max}^N)$, where $S = \mathcal{O}(\log I)$.

### Loss & Training

The multi-scale loss function comprises six terms: data fidelity (Frobenius error) + temporal consistency + spatial smoothness + diagonal sparsity ($\ell_1$ regularization for rank discovery) + edge entropy regularization $H(g) = -g\log g - (1-g)\log(1-g)$ encouraging decisive gating + tensor nuclear norm low-rank regularization. Coefficients for each term vary across scales following the RG flow equations $d\lambda_k/ds = \beta_k(\{\lambda_j\})$. The core learning rate is larger at finer scales $\eta_{\text{cores}}(s) = \eta_0 \cdot \exp(-s/s_0)$, while the structural parameter learning rate is larger at coarser scales $\eta_{\text{struct}}(s) = \eta_0 \cdot (1 + s/s_1)$.

## Key Experimental Results

### Main Results: Light Field Data Compression

| Method | Bunny CR@RE0.01 | Time (×1000s) | Bunny CR@RE0.1 | Knights CR@RE0.01 | Time (×1000s) |
|--------|----------------|---------------|----------------|-------------------|---------------|
| TRALS | 61.2% | 13.41 | 5.38% | 74.2% | 10.42 |
| TNGA | 28.2% | 1004 | 2.27% | 39.1% | 904.5 |
| TNLS | 24.5% | 1388 | 2.18% | 27.5% | 1273 |
| SVDinsTN | 22.6% | 0.752 | 2.69% | 31.7% | 1.563 |
| **RGTN** | **22.3%** | **0.180** | **0.91%** | **29.9%** | **0.178** |

At RE=0.1, RGTN achieves 2.96× (Bunny) and 1.60× (Knights) better compression ratios than SVDinsTN, while running more than 4500× faster than TNGA/TNLS.

### High-Order Tensor & Video Completion

| Method | 6th-order CR@RE0.01 | 6th-order Time (×1000s) | 8th-order CR@RE0.01 | 8th-order Time (×1000s) |
|--------|--------------------|--------------------------|--------------------|--------------------------|
| TNGreedy | 0.88% | 0.167 | 0.016% | 2.625 |
| TNGA | 0.94% | 3.825 | 0.024% | 51.40 |
| SVDinsTN | 1.13% | 0.002 | 0.016% | 0.017 |
| **RGTN** | **0.76%** | 0.006 | **0.009%** | 0.123 |

| Video Data | RGTN MPSNR | SVDinsTN MPSNR | RGTN Time (s) | SVDinsTN Time (s) |
|-----------|-----------|---------------|--------------|-----------------|
| News | **32.040** | 31.643 | **135.95** | 932.42 |
| Salesman | **31.900** | 31.684 | **144.00** | 769.54 |
| Silent | 30.620 | **32.706** | **142.80** | 532.31 |

### Key Findings
- **High structure discovery success rate**: Over 100 independent trials, structure discovery success rates for 4th- and 5th-order tensors reach 95%–100%.
- **Advantage amplified at higher orders**: At 8th order, RGTN achieves 1.8× better compression than SVDinsTN without encountering OOM (FCTNALS runs out of memory at 8th order).
- **Effect of multi-scale initialization**: 40% faster than random initialization at the RE=0.01 target, while finding superior solutions.

## Highlights & Insights
- **A paradigmatic cross-disciplinary fusion**: The multi-scale nature of RG theory naturally aligns with hierarchical structure discovery in TN-SS. Rather than a superficial analogy, the paper establishes a complete mathematical correspondence (fixed points, criticality, universality classes), providing a fundamentally new theoretical framework for TN-SS.
- **Continuization of discrete search**: Edge gating combined with temperature annealing transforms discrete topological search into continuous gradient optimization, conceptually analogous to DARTS in NAS but arising more naturally from physical motivation.
- **Exponential theoretical speedup**: Search complexity is reduced from $\Omega(\exp(N^2))$ to $\mathcal{O}(\log I \cdot \log(1/\epsilon))$, with rigorous guarantees on convergence and escape from local minima.

## Limitations & Future Work
- Hyperparameters in the multi-scale strategy (number of scales, steps per scale) require manual tuning and have not been made adaptive.
- The downsampling–upsampling operators use fixed powers of 2, which may be suboptimal for non-uniformly structured data.
- Performance on the Silent video dataset falls below SVDinsTN, indicating residual sensitivity to certain data characteristics.
- Experiments focus on compression and completion tasks; extension to recommendation systems, quantum chemistry, and other application domains warrants future investigation.

## Related Work & Insights
- **vs. SVDinsTN (CVPR 2024)**: SVDinsTN achieves unified optimization via diagonal factor sparsification but is limited to single-scale search and susceptible to local optima. RGTN augments this with multi-scale RG flow and edge gating, yielding superior compression and the ability to escape local minima.
- **vs. TNGA/TNLS**: Conventional sample-and-evaluate methods require exponential structural enumeration ($\Omega(\exp(N^2))$); RGTN reduces search cost to polynomial order through physics-driven proposals.
- **vs. MERA/TRG**: In quantum physics, methods such as MERA apply RG principles to state solving under fixed topologies. RGTN is the first to apply RG to topology search itself.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First complete introduction of RG theory into TN-SS; the cross-disciplinary integration is substantive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four task categories—structure discovery, light field, high-order tensor, and video—though ablation studies are not sufficiently systematic.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are complete and rigorous with detailed appendices, though the physics terminology presents a high entry barrier.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for TN-SS with significant speedups, though the target domain remains relatively niche.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[AAAI 2026\] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers](stratified_knowledge-density_super-network_for_scalable_vision_transformers.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[NeurIPS 2025\] Learning to Better Search with Language Models via Guided Reinforced Self-Training](../../NeurIPS2025/model_compression/learning_to_better_search_with_language_models_via_guided_reinforced_self-traini.md)

<!-- RELATED:END -->
