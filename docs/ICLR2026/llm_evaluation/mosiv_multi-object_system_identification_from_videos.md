---
title: >-
  [Paper Note] MOSIV: Multi-Object System Identification from Videos
description: >-
  [ICLR 2026][LLM Evaluation][multi-object physics] This paper proposes MOSIV—the first complete framework for multi-object system identification from multi-view videos—comprising three stages: (1) object-aware 4D dynamic Gaussian reconstruction of per-object geometry and motion; (2) Gaussian-to-continuum lifting to construct MPM simulation particles; and (3) differentiable MPM forward rollout with geometry-alignment objectives (3D Chamfer + 2D silhouette) to back-propagate and optimize per-object continuous material parameters ($E, \nu, \mu$). On a contact-rich synthetic benchmark spanning four material types (elastic, elastoplastic, fluid, and granular), MOSIV achieves PSNR 30.51 vs. OmniPhysGS 25.93 and reduces Chamfer distance by 9.4×, establishing a new baseline for multi-object long-horizon physical simulation.
tags:
  - ICLR 2026
  - LLM Evaluation
  - multi-object physics
  - video system identification
  - differentiable MPM
  - 4D Gaussians
  - continuous parameter optimization
  - contact and friction modeling
date: 2026-05-08
content_hash: 968eeed52c5f9d47
---

# MOSIV: Multi-Object System Identification from Videos

**Conference**: ICLR 2026
**arXiv**: [2603.06022](https://arxiv.org/abs/2603.06022)
**Area**: Physical Modeling / System Identification
**Keywords**: multi-object physics, video system identification, differentiable MPM, 4D Gaussians, continuous parameter optimization, contact and friction modeling

## TL;DR

This paper proposes MOSIV—the first complete framework for multi-object system identification from multi-view videos—comprising three stages: (1) object-aware 4D dynamic Gaussian reconstruction of per-object geometry and motion; (2) Gaussian-to-continuum lifting to construct MPM simulation particles; and (3) differentiable MPM forward rollout with geometry-alignment objectives (3D Chamfer + 2D silhouette) to back-propagate and optimize per-object continuous material parameters ($E, \nu, \mu$). On a contact-rich synthetic benchmark spanning four material types (elastic, elastoplastic, fluid, and granular), MOSIV achieves PSNR 30.51 vs. OmniPhysGS 25.93 and reduces Chamfer distance by 9.4×, establishing a new baseline for multi-object long-horizon physical simulation.

## Background & Motivation

**Background**: Learning object physical properties from video is central to constructing "digital twins." Existing methods (GIC, PAC-NeRF, etc.) are largely restricted to single-object, isolated-motion scenarios, whereas the real world is filled with multi-object collisions, sliding contacts, and occlusions.

**Limitations of Prior Work**:
- (1) Single-object methods cannot handle multi-object interactions—coupled motion during collisions and occlusion-induced tracking difficulties.
- (2) OmniPhysGS performs discrete material classification (selecting from a fixed library), which precludes continuous physical parameter representation and limits accuracy.
- (3) CoupNeRF employs a NeRF+MPM hybrid, resulting in heavy computation, poor temporal consistency, and unsuitability for contact-intensive scenarios.
- (4) The absence of a standardized multi-object system identification benchmark prevents fair evaluation.

**Key Challenge**: Multi-object contact and collision present a double-edged sword: they provide rich signals (revealing hidden physical quantities such as friction and stiffness) while also introducing association ambiguity (scene-level losses can produce misleading gradients through cross-object matching).

**Key Insight**: Continuous parameter identification (rather than category selection) + differentiable physical simulator + object-level geometry-alignment supervision, jointly addressing multi-object system identification.

**Application Prospects**: Accurate multi-object physical parameters enable robot manipulation in cluttered scenes, physically plausible scene editing, and long-horizon behavior prediction.

**Goal**: Formalize the multi-object system identification task, propose the MOSIV framework, and release a synthetic benchmark dataset of 45 multi-view video sequences.

## Method

### Problem Formulation

Given multi-view RGB videos of $K$ deformable objects ($T$ frames, $n$ viewpoints), the goal is to recover:
- (i) the 4D continuum of all objects (3D shape evolving over time);
- (ii) the material parameter set $\boldsymbol{\Theta} = \{\boldsymbol{\theta}_k\}_{k=1}^{K}$ for each object,

such that a physics simulator can reproduce the observed motion and predict future interactions. The only inputs required are video, camera calibration, and instance masks.

### Key Design 1: Object-Aware Dynamic Gaussian Reconstruction

**Function**: Reconstruct an independent 4D Gaussian representation for each object from multi-view video.

**Mechanism**: A low-rank motion decomposition is applied to 3D Gaussian Splatting, where each Gaussian kernel deforms via temporal basis functions $\boldsymbol{\psi}_b^\mu(t)$ and spatial gating $\alpha_b(\boldsymbol{\mu})$:

$$\boldsymbol{\mu}_t = \boldsymbol{\mu} + \sum_{b=1}^{B} \alpha_b(\boldsymbol{\mu}) \boldsymbol{\psi}_b^\mu(t), \quad r_t = r + \sum_{b=1}^{B} \alpha_b(\boldsymbol{\mu}) \psi_b^r(t)$$

The training objective is a photometric consistency loss:

$$\min_{\mathcal{G}_0, \text{net}} \mathcal{L}_1(\hat{\mathbf{I}}_t, \mathbf{I}_t) + \lambda_\text{SSIM} \mathcal{L}_\text{SSIM}(\hat{\mathbf{I}}_t, \mathbf{I}_t) + \lambda_r \|r_t\|_1$$

**Design Motivation**: Instance masks partition Gaussian kernels by object, giving each object an independent motion field for separate tracking and parameter optimization during simulation. Compared to implicit NeRF representations, explicit Gaussians provide more stable geometry and support real-time rendering.

### Key Design 2: Multi-Object Gaussian-to-Continuum Lifting

**Function**: Convert rendering-optimized Gaussian particles into continuum particles suitable for MPM simulation.

**Mechanism**: For each object $k$, particles are randomly sampled within the bounding box of Gaussian points; those consistent with multi-camera depth are retained; a progressively refined density field with mean-filter smoothing is constructed; and surface extraction is performed via thresholding.

**Additional Multi-Object Constraints**:
- **Disjoint support**: Overlapping voxels are assigned to the nearest object surface, eliminating initial interpenetration.
- **Compatible resolution**: Per-object mesh resolutions are aligned so that contact interfaces match.

**Design Motivation**: Dynamic Gaussians are optimized for rendering (with non-uniform spatial distribution) and cannot be used directly in continuum simulation. Explicit density field construction and interface handling are required to ensure a physically valid initial simulation state.

### Key Design 3: Differentiable MPM Simulation and Geometry-Alignment Optimization

**Function**: Perform differentiable MPM forward rollouts and back-propagate through geometry-alignment losses to optimize per-object continuous material parameters.

**Mechanism**: The MPM time-stepping map $\mathbf{z}_{n+1} = \mathcal{T}(\mathbf{z}_n; \boldsymbol{\Theta})$ is fully differentiable. The geometry-alignment objective combines 3D surface Chamfer distance and 2D silhouette L1 loss:

$$\mathcal{L}_\text{ID} = \frac{1}{m}\sum_{i=1}^{m}\left[\sum_{k=1}^{K}\mathcal{L}_\text{CD}(S_k(t_i), \tilde{S}_k(t_i)) + \frac{1}{n}\sum_{j=1}^{n}\sum_{k=1}^{K}\mathcal{L}_1(A_{j,k}(t_i), \tilde{A}_{j,k}(t_i))\right]$$

**Object-Level vs. Scene-Level Supervision**: The core innovation lies in object-wise rather than scene-wise losses. Scene-level Chamfer distance causes cross-object matching during contact, masking parameter errors; object-level losses strictly enforce geometric consistency per object.

**Design Motivation**: Association ambiguity in contact regions is the central challenge of multi-object identification. Scene-level losses allow the optimizer to sacrifice accuracy on one object to satisfy a global objective, producing misleading gradients. Object-level losses block this cross-object borrowing, providing cleaner gradient signals.

### Key Design 4: Material Parameterization and Contact Modeling

**Function**: Independently parameterize material properties for each object instance; model Coulomb friction at material interfaces via a symmetric combination.

**Mechanism**: Different objects do not share parameters even when composed of the same material type—identifiability emerges from each object's individual geometric and silhouette constraints. Interface friction is modeled as:

$$\mu_{m,m'} = g(\mu_m, \mu_{m'}) = \frac{1}{2}(\mu_m + \mu_{m'})$$

**Design Motivation**: This reduces degrees of freedom while maintaining flexibility. Avoiding forced parameter sharing prevents incorrect priors; parameter consistency is validated data-drivenly rather than by manual assumption.

### Loss & Training

- **Three-stage training**: Stage I (4DGS reconstruction) → Stage II (Gaussian-to-continuum lifting) → Stage III (parameter optimization).
- **Horizon curriculum**: The forward rollout length is progressively increased as alignment improves.
- **Alternating updates**: Parameter optimization alternates with particle state re-synchronization to reduce drift.
- **Implementation details**: MPM time step $\tau=1/4800$ (200 sub-steps per frame), grid resolution $4096^3$, Adam optimizer, 80 iterations for velocity estimation + 200 iterations for physical parameter refinement.

## Key Experimental Results

### Dataset

Synthetic benchmark: 45 multi-view video sequences, 10 geometry shapes × 5 material types (elastic / elastoplastic / fluid / granular / snow), 11 camera viewpoints, 30 frames per sequence, with ground-truth physical parameters.

### Table 1: Observable State Simulation

| Method | PSNR↑ | SSIM↑ | CD↓ | EMD↓ |
|--------|-------|-------|-----|------|
| OmniPhysGS-RGB | 25.93 | 0.945 | 11.79 | 0.095 |
| OmniPhysGS-RGB w/ Oracle | 24.39 | 0.930 | 43.50 | 0.168 |
| **MOSIV (Ours)** | **30.51** | **0.977** | **1.256** | **0.049** |

MOSIV outperforms all baselines by a large margin: +4.58 dB PSNR, 9.4× lower CD, 48% lower EMD. Notably, OmniPhysGS with Oracle (ground-truth material model) performs even worse than its standard variant (CD 43.50 vs. 11.79), demonstrating that the discrete selection architecture is itself the bottleneck.

### Table 2: Future State Simulation

| Method | PSNR↑ | SSIM↑ | CD↓ | EMD↓ |
|--------|-------|-------|-----|------|
| OmniPhysGS-RGB | 19.00 | 0.888 | 51.92 | 0.199 |
| OmniPhysGS-RGB w/ Oracle | 17.97 | 0.869 | 215.83 | 0.408 |
| **MOSIV (Ours)** | **28.26** | **0.963** | **3.710** | **0.071** |

The performance gap widens substantially in long-horizon prediction: +9.26 dB PSNR, 14× lower CD. Baseline methods diverge rapidly during long-horizon rollouts, while MOSIV remains stable.

### Table 3: Ablation Study on Supervision Granularity

| Supervision | $\mathcal{L}_\text{CD}$ | $\mathcal{L}_\alpha$ | PSNR↑ | CD↓ |
|-------------|:-:|:-:|-------|-----|
| Scene-level | ✓ | ✓ | 27.89 | 22.13 |
| **Object-level (Ours)** | **✓** | **✓** | **30.24** | **0.696** |

Object-level supervision reduces CD from 22.13 to 0.696 (31.8× improvement), confirming the critical importance of fine-grained object-level supervision.

## Key Findings

1. **Continuous parameters substantially outperform category selection**: MOSIV consistently surpasses the discrete selection paradigm across all material combinations; even OmniPhysGS with Oracle ground-truth material categories cannot match MOSIV.

2. **Object-level supervision is essential for multi-object identification**: Scene-level losses produce cross-object matching errors during contact, causing CD to spike (22.13 vs. 0.696). Object-level losses eliminate this cross-borrowing and provide correct gradients.

3. **Dual-source supervision is indispensable**: Using Chamfer distance or silhouette loss in isolation is insufficient to stabilize training; their combination is necessary for robust physical parameter optimization.

4. **Long-horizon simulation fidelity**: MOSIV maintains PSNR of 28.26 in future-state prediction, while baselines drop sharply from 25.93/24.39 to 19.00/17.97, confirming that accurate parameter identification yields long-term stability.

5. **Generalization to novel interactions**: Holding geometry and initial conditions fixed while swapping material parameters produces physically plausible dynamics, validating that the identified parameters genuinely capture real physics.

## Highlights & Insights

- **"Multi-object = richer signals"**: Object collisions and contacts are not merely a challenge—they are the primary means of revealing hidden physical quantities such as friction and stiffness. A single freely falling object cannot disambiguate different friction coefficients; multi-object interactions provide uniquely identifiable conditions.

- **The fundamental gap between continuous and discrete representations**: Materials occupy points on a continuous spectrum, not a finite set of categories. The Oracle variant of OmniPhysGS performs worse than its standard counterpart, demonstrating that the discrete material library introduces an insurmountable expressiveness bottleneck.

- **Geometry alignment over pixel alignment**: Driving physical parameter optimization with 3D surface and 2D silhouette objectives rather than pixel-level photometric losses yields greater robustness to rendering noise and more directly reflects physical consistency.

- **A complete closed loop for "digital twins"**: Accurate physical parameters not only reproduce observations but also enable prediction of novel scenarios (altered initial conditions, force fields, or material assignments)—a capability critical for downstream applications.

## Limitations & Future Work

1. **Reliance on predefined constitutive models**: The constitutive model type (elastic, plastic, fluid, etc.) must be specified in advance, precluding handling of unknown material types. Neural-network-based direct learning of physical models may address this.

2. **High computational cost**: Differentiable MPM simulation with high-resolution grids ($4096^3$) and many optimization iterations results in substantial training time per scene.

3. **Sensitivity to initial geometry**: The method is sensitive to the quality of initial 3D reconstruction and may degrade in heavily occluded, cluttered scenes.

4. **Validation on synthetic data only**: The current benchmark is entirely synthetic; real-world video introduces additional challenges including complex illumination, sensor noise, and the sim-to-real gap.

5. **Material type must be known**: The material family (elastic/plastic/fluid/granular) for each object must be predefined via masks; fully automatic material type inference remains an open problem.

## Related Work & Insights

### vs. OmniPhysGS (Lin et al., 2025)

| Dimension | OmniPhysGS | MOSIV |
|-----------|-----------|-------|
| Material representation | Classification from fixed expert library | Direct optimization of continuous parameters |
| Physical simulation | Category matching → errors in some scenes | Differentiable MPM → accurate contact/friction |
| Supervision signal | SDS / photometric | Geometry alignment (3D + 2D) |
| Multi-object support | Implicit scene-level | Explicit object-level |
| Observable state PSNR | 25.93 | **30.51** |
| Future state CD | 51.92 | **3.71** |

### vs. CoupNeRF (Li et al., 2024a)

| Dimension | CoupNeRF | MOSIV |
|-----------|---------|-------|
| 3D representation | Implicit NeRF | Explicit 3D Gaussians |
| Computational efficiency | Heavy (temporally optimized NeRF fields) | Lighter (explicit Gaussians) |
| Temporal consistency | Weak (contact-intensive scenes) | Strong (object-level tracking) |
| Material-specific dynamics | Weak inter-material discrimination | Well-preserved material-specific dynamics |
| Applicable scenarios | Free fall / simple interactions | Contact-rich / mixed-material scenes |

### vs. GIC (Cai et al., 2024)

GIC is the single-object predecessor of MOSIV. MOSIV inherits its Gaussian-to-continuum lifting paradigm and extends it to multiple objects by adding disjoint support constraints, object-level supervision, and cross-material contact modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First formal treatment of multi-object video system identification + continuous parameter optimization + object-level geometry-alignment supervision + new synthetic benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 45 multi-view video sequences, 10 geometries × 5 materials, multiple baselines (including Oracle), supervision granularity ablation, and novel-interaction generalization tests; the absence of real-data validation is a minor weakness.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, the method pipeline is logically coherent, and ablation design is insightful (particularly the analysis of scene-level vs. object-level supervision).
- **Value**: ⭐⭐⭐⭐ — Significant contribution to multi-object physical scene understanding; the combination of continuous parameter identification and object-level supervision establishes a strong baseline for future work.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_evaluation/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[ICCV 2025\] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](../../ICCV2025/llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](which_llm_multi-agent_protocol_to_choose.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](../../ACL2026/llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)

<!-- RELATED:END -->
