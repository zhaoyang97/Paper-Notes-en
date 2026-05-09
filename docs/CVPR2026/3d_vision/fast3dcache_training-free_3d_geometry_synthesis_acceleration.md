---
title: >-
  [Paper Note] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration
description: >-
  [CVPR 2026][3D Vision][3D geometry generation acceleration] This paper proposes Fast3Dcache, a training-free geometry-aware caching framework for 3D diffusion models. It dynamically allocates cache budgets via Predictive Cache Scheduling Constraint (PCSC) based on voxel stabilization patterns, and selects stable tokens for reuse via Spatiotemporal Stability Criterion (SSC) using velocity and acceleration signals. The method achieves up to 27.12% throughput improvement and 54.83% FLOPs reduction with only ~2% degradation in geometric quality.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D geometry generation acceleration
  - caching mechanism
  - voxel stabilization
  - training-free
  - diffusion models
date: 2026-05-08
content_hash: 0e1fb12b0b3333c6
---

# Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration

**Conference**: CVPR 2026
**arXiv**: [2511.22533](https://arxiv.org/abs/2511.22533)
**Code**: [https://fast3dcache-agi.github.io](https://fast3dcache-agi.github.io)
**Area**: 3D Vision
**Keywords**: 3D geometry generation acceleration, caching mechanism, voxel stabilization, training-free, diffusion models

## TL;DR

This paper proposes Fast3Dcache, a training-free geometry-aware caching framework for 3D diffusion models. It dynamically allocates cache budgets via Predictive Cache Scheduling Constraint (PCSC) based on voxel stabilization patterns, and selects stable tokens for reuse via Spatiotemporal Stability Criterion (SSC) using velocity and acceleration signals. The method achieves up to 27.12% throughput improvement and 54.83% FLOPs reduction with only ~2% degradation in geometric quality.

## Background & Motivation

1. **Background**: Cache-based acceleration has achieved notable success in 2D image and video diffusion models by reusing intermediate computations from preceding timesteps to reduce redundant inference. Representative methods include various feature caching techniques.
2. **Limitations of Prior Work**:
   - Directly transferring 2D caching strategies to 3D diffusion models severely disrupts geometric consistency;
   - Minor texture errors in 2D/video are perceptually negligible, whereas numerical errors in 3D voxel/point predictions directly affect topology and spatial integrity, leading to surface holes, geometric distortion, or non-manifold meshes;
   - Existing 3D acceleration methods (e.g., Hash3D) are not applicable to diffusion frameworks.
3. **Key Challenge**: 2D caching exploits perceptual redundancy, whereas 3D geometry demands strict numerical correctness—small accumulated errors can lead to topological catastrophe.
4. **Goal**: How to safely cache and reuse computations during 3D diffusion inference while maintaining geometric fidelity alongside acceleration?
5. **Key Insight**: The paper analyzes the evolution of voxel occupancy fields during the sparse structure generation stage in the TRELLIS framework, revealing a three-phase stabilization pattern (unstable → log-linear decay → fine-tuning), and designs adaptive caching strategies accordingly.
6. **Core Idea**: By exploiting the predictable decay pattern of voxel state changes during 3D generation, the method dynamically determines *how many tokens to cache* (PCSC) and *which tokens to cache* (SSC), enabling geometry-aware acceleration.

## Method

### Overall Architecture

Fast3Dcache divides inference into three phases: Phase 1 (full sampling) establishes initial stability and calibrates PCSC; Phase 2 (dynamic caching) uses PCSC to determine the per-step cache budget and SSC to select tokens for reuse, with a full refresh every $\tau$ steps; Phase 3 (CFG-Free Refinement) applies a fixed high cache ratio.

### Key Designs

1. **Predictive Cache Scheduling Constraint (PCSC)**:
   - **Function**: Dynamically determines how many tokens to cache at each timestep based on the decay trend of voxel stabilization.
   - **Mechanism**: The voxel occupancy change $\Delta s_t = \sum_{i,j,k} (\mathcal{O}_{t+1}(i,j,k) \oplus \mathcal{O}_t(i,j,k))$ during denoising exhibits a three-phase pattern: high volatility in Phase 1, log-linear decay in Phase 2, and rapid stabilization in Phase 3. At the anchor step marking the end of Phase 1, the initial change magnitude $\sigma$ is calibrated; subsequent changes are predicted using a fixed slope $\mu$: $\Delta\hat{s} = \sigma \cdot e^{\mu \cdot (t - \lceil T \cdot \rho_a \rceil)}$. The cache budget is then: $c_t = D^3 - \frac{\Delta\hat{s}_t}{\gamma_{\text{up}}}$
   - **Design Motivation**: Unlike the fixed caching ratios used in 2D, the stability of 3D geometry generation varies dramatically across phases. PCSC adaptively allocates budget—caching less in early stages (to protect coarse structure formation) and more in later stages (to exploit geometric convergence). Experiments show that fixed ratios yield CD of 0.0956, whereas PCSC achieves 0.0697.

2. **Spatiotemporal Stability Criterion (SSC)**:
   - **Function**: Precisely selects which tokens can be safely cached given a cache budget.
   - **Mechanism**: A cacheability score is computed for each token: $C_i(t) = \omega \cdot \text{norm}(A_i(t)) + (1-\omega) \cdot \text{norm}(V_i(t))$, where velocity magnitude $V_i(t) = \|v_i(t)\|_2$ reflects feature update intensity, and acceleration $A_i(t) = \|v_i(t) - v_i(t-1)\|_2$ reflects velocity stability (i.e., instantaneous cache error, ICE). Tokens with lower scores are more stable and thus more suitable for caching. Self-attention is computed only for the unstable subset.
   - **Design Motivation**: Velocity magnitude alone is insufficient—tokens with high but directionally stable velocity can still be cached safely (low error). Acceleration alone is also insufficient—tokens with low acceleration but high velocity are still undergoing large updates. The two metrics are complementary, providing finer-grained stability judgment. Ablations confirm that their joint use ($\omega=0.7$) significantly outperforms either individual metric.

3. **Three-Phase Pipeline Integration**:
   - **Function**: Integrates PCSC and SSC into an end-to-end acceleration workflow.
   - **Mechanism**: Phase 1 performs full sampling to establish base geometry and calibrates PCSC at its conclusion. Phase 2 applies dynamic caching with PCSC+SSC, with full refreshes every $\tau$ steps to eliminate error accumulation. Phase 3 uses a fixed high cache ratio $\xi$ with full refreshes every $f_{\text{corr}}$ steps.
   - **Design Motivation**: The three phases correspond to the natural evolution of 3D generation. Error-accumulation elimination steps (full refresh every $\tau$ steps) are essential—completely disabling them degrades CD to 0.0724 and F-Score to 51.8157.

### Loss & Training

Fast3Dcache requires no training whatsoever and is a purely inference-time acceleration method. Hyperparameters include: anchor ratio $\rho_a$, decay slope $\mu$ (default -0.07), refresh interval $\tau$, Phase 3 fixed cache ratio $\xi$, acceleration weight $\omega$ (default 0.7), and correction frequency $f_{\text{corr}}$.

## Key Experimental Results

### Main Results (TRELLIS Framework, Toys4K Dataset)

| Method | Throughput↑ | FLOPs(T)↓ | CD↓ | F-Score↑ |
|--------|-------------|-----------|-----|----------|
| TRELLIS vanilla | 0.5055 | 244.2 | 0.0686 | 54.8244 |
| RAS (25%) | 0.6337 (+25.36%) | 125.1 (-48.77%) | 0.0867 (+26.38%) | 40.2769 (-26.53%) |
| RAS (12.5%) | 0.6177 (+22.20%) | 125.8 (-48.48%) | 0.0846 (+23.32%) | 43.9622 (-19.81%) |
| **Fast3Dcache (τ=3)** | 0.5850 (+15.73%) | 142.4 (-41.69%) | **0.0697 (+1.60%)** | **54.0900 (-1.34%)** |
| **Fast3Dcache (τ=5)** | **0.6344 (+25.50%)** | **121.3 (-50.33%)** | 0.0712 (+3.79%) | 53.5003 (-2.42%) |
| **Fast3Dcache (τ=8)** | **0.6426 (+27.12%)** | **110.3 (-54.83%)** | 0.0703 (+2.48%) | 53.7528 (-1.95%) |

### Ablation Study (SSC Components)

| Configuration | CD↓ | F-Score↑ |
|---------------|-----|----------|
| No SSC (standard deviation filtering) | 0.0743 | 50.9974 |
| Velocity only $V_i$ | 0.0836 | 44.9630 |
| Acceleration only $A_i$ | 0.0709 | 53.5394 |
| Joint $\omega=0.7$ | **0.0697** | **54.0900** |

### Key Findings

- RAS (a 2D method directly transferred to 3D) causes severe geometric degradation (F-Score drops 27%), validating the central claim that 3D generation requires geometry-aware caching.
- At $\tau=8$, throughput improves by 27.12% and FLOPs decrease by 54.83%, while CD increases by only 2.48% and F-Score drops by only 1.95%.
- Combining with TeaCache achieves 3.41× acceleration, with geometric quality superior to TeaCache alone (CD 0.0701 vs. 0.0705), demonstrating that Fast3Dcache is complementary to general-purpose accelerators.
- Acceleration $A_i$ is a more effective standalone metric than velocity $V_i$ (CD 0.0709 vs. 0.0836), as acceleration directly measures instantaneous cache error.
- The PCSC slope $\mu$ exhibits relative robustness over a ±10× range (CD 0.0697–0.0707).

## Highlights & Insights

- **Three-phase stabilization pattern of 3D geometry**: This empirical finding (unstable → log-linear decay → fine-tuning) is likely not unique to TRELLIS but may represent a universal characteristic of 3D diffusion generation, providing a theoretical foundation for future 3D diffusion acceleration work.
- **Decoupled design of "cache budget + token selection"**: PCSC handles macro-level scheduling while SSC handles micro-level selection, with clearly separated responsibilities. This hierarchical design is generalizable to other inference acceleration scenarios requiring adaptive computation allocation.
- **Joint stability measure via velocity and acceleration**: Neither velocity magnitude nor its rate of change alone is sufficient; the two are complementary. This insight is analogous to simultaneously considering velocity and acceleration in physics to characterize motion state.

## Limitations & Future Work

- Only the sparse structure generation stage of TRELLIS is accelerated; the SLat generation stage remains unoptimized, potentially limiting overall end-to-end speedup.
- While training-free, the three-phase boundaries ($\rho_a$) and parameters ($\mu$, $\omega$, $\tau$, $\xi$, $f_{\text{corr}}$) require tuning and may vary across tasks and datasets.
- The caching strategy assumes a consistent voxel decay rate $\mu$ across samples, which may not hold for geometrically extreme cases (e.g., highly complex or fine-grained structures).
- Validation is limited to the TRELLIS and DSO frameworks; applicability to implicit representations (e.g., Hunyuan3D's set-based latent) remains unknown.

## Related Work & Insights

- **vs. RAS**: Directly transferring this 2D DiT caching method to 3D causes F-Score to collapse by 27%. Fast3Dcache limits quality loss to 2% through geometry-aware design.
- **vs. TeaCache**: A general-purpose accelerator that is complementary to Fast3Dcache. Their combination yields a super-additive effect (1+1>2), indicating that modality-aware and modality-agnostic acceleration can be stacked.
- **vs. Hash3D**: Explores 3D acceleration but is inapplicable to diffusion frameworks. Fast3Dcache is specifically designed for diffusion/Flow Matching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-phase stabilization observation and the PCSC/SSC designs are original, though the core approach is an adaptation of caching methods to 3D.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations (PCSC/SSC/τ), multi-framework validation (TRELLIS+DSO), and complementarity experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation–observation–design logical chain is clear, and visualizations are excellent.
- **Value**: ⭐⭐⭐⭐ Practically valuable for inference acceleration in 3D generation; open-source and easy to use.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)
- [\[CVPR 2026\] FE2E: From Editor to Dense Geometry Estimator](from_editor_to_dense_geometry_estimator.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
