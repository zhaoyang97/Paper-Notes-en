---
title: >-
  [Paper Note] Fast-SAM3D: 3Dfy Anything in Images but Faster
description: >-
  [ICML 2026][3D Vision][SAM3D Acceleration] Addressing the slow inference of the SAM3D single-view 3D reconstruction model, this paper provides the first module-level latency profiling. Identifying three types of heteroge…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "SAM3D Acceleration"
  - "Single-view 3D Reconstruction"
  - "Training-free Inference Optimization"
  - "Diffusion Step Caching"
  - "Token Pruning"
date: 2026-05-08
content_hash: 393c35afde12c7b0
---

# Fast-SAM3D: 3Dfy Anything in Images but Faster

**Conference**: ICML 2026  
**arXiv**: [2602.05293](https://arxiv.org/abs/2602.05293)  
**Code**: https://github.com/wlfeng0509/Fast-SAM3D  
**Area**: 3D Vision  
**Keywords**: SAM3D Acceleration, Single-view 3D Reconstruction, Training-free Inference Optimization, Diffusion Step Caching, Token Pruning

## TL;DR
Addressing the slow inference of the SAM3D single-view 3D reconstruction model, this paper provides the first module-level latency profiling. Identifying three types of heterogeneity (differences in shape/layout dynamics, texture sparsity, and geometric spectral variation) as the bottleneck, the authors propose Fast-SAM3D. This training-free framework utilizes modality-aware step caching, spatiotemporal token carving, and spectral-aware token aggregation to achieve a 2.67× object-level speedup with almost no loss in quality, even slightly improving the reconstruction F-Score from 92.34 to 92.59.

## Background & Motivation

**Background**: Models like SAM3D for single-view, open-world, mask-conditioned 3D asset generation have become essential for 3D perception and content creation. The standard pipeline follows a two-stage "coarse-to-fine" diffusion architecture: "Sparse Structure (SS) generator → Sparse Latent (SLaT) generator → Mesh Decoder," which reconstructs multi-object 3D models and decouples layout information from a single image.

**Limitations of Prior Work**: SAM3D suffers from extreme inference costs. Module-level profiling shows a total end-to-end latency of ~462 s per scene, dominated by the SLaT generator (9.7 s/object, 219.8 T FLOPs) and Mesh Decoder (13.8 s/object, 324 T FLOPs), with the SS generator contributing 4.1 s. Such latency makes SAM3D impractical for interactive deployment.

**Key Challenge**: General diffusion acceleration techniques (uniform step skipping, random token pruning, and multi-view caches like Fast3DCache) fail when applied to SAM3D. Random Drop causes 3D-IoU to plummet from 0.403 to 0.094, and Fast3DCache provides only a 1.03× speedup. The failure stems from "multi-level heterogeneity" within SAM3D: (i) Shape tokens in the SS stage evolve smoothly, whereas layout tokens (controlling R/t/s) exhibit high-frequency jitter; applying the same caching strategy to both causes systematic pose drift. (ii) Refinement updates in SLaT are spatially sparse, with most tokens stabilizing early while only edges/seams/thin structures continue to update. (iii) In the Mesh decoder, tolerance for token downsampling varies greatly across objects with different geometric complexities; instance-agnostic uniform downsampling erases high-frequency details of complex objects.

**Goal**: To solve three sub-problems: (1) enabling step skipping in the SS generator without pose drift; (2) allowing the SLaT generator to reuse tokens across time and prune tokens across space; and (3) enabling the Mesh decoder to adaptively aggregate tokens based on object complexity.

**Key Insight**: The authors elevate "heterogeneity" to a unified design principle—**computational power should be allocated non-uniformly**, matching stage difficulty and instance complexity. This implies using different computational budgets for different semantic roles (shape vs. layout), different spatial positions at the same timestep, and different input instances for the same model.

**Core Idea**: Three targeted, plug-and-play, training-free modules are inserted into the three stages to eliminate redundancy, forming a unified "heterogeneity-aware" acceleration framework that reduces object-level latency to 37% of the original.

## Method

### Overall Architecture
Input consists of a scene image $I$ and a target object mask $M$. Outputs include the 3D shape $S$, texture $T$, and layout parameters $(R, t, s)$. Fast-SAM3D does not modify SAM3D weights but inserts an acceleration module into each of the original three stages:

1.  **SS Generator**: Decouples shape and layout tokens during the 25-step denoising process, applying different caching strategies.
2.  **SLaT Generator**: Recomputes only the top-K highly salient tokens spatially and adaptively decides when to skip steps temporally based on curvature.
3.  **Mesh Decoder**: Selects a downsampling factor $\mathcal{S} \in \{1.25, 1.5, 2.0\}$ based on the spectral energy of the mask and coarse voxels, then performs coordinate quantization + max-pool aggregation on sparse 3D tokens.

These synergistically reduce object-level time from 31.04 s to 11.60 s (2.67×) and scene-level time from 462.3 s to 229.7 s (2.01×).

### Key Designs

1.  **Modality-Aware Step Caching for SS Generator**:
    *   **Function**: Decouples update rules for shape and layout tokens in the SS stage, allowing shape tokens to extrapolate aggressively while keeping layout updates conservative to mitigate pose drift.
    *   **Mechanism**: For smoothly evolving shape tokens, first-order finite difference is calculated: $\nabla \mathbf{v}^{\text{shape}}_t = (\mathbf{v}^{\text{shape}}_t - \mathbf{v}^{\text{shape}}_{t+k})/k$. During skipped steps, Taylor extrapolation is used: $\hat{\mathbf{v}}^{\text{shape}}_{t-i} = \mathbf{v}^{\text{shape}}_t + (-i)\nabla \mathbf{v}^{\text{shape}}_t$. For high-frequency layout tokens, the same linear extrapolation yields $\mathbf{v}^{\text{layout}}_{\text{lin}}(t-i)$, which is then smoothed using an anchor from the last full evaluation: $\hat{\mathbf{v}}^{\text{layout}}_{t-i} = \beta \cdot \mathbf{v}^{\text{layout}}_{\text{lin}}(t-i) + (1-\beta) \cdot \mathbf{v}^{\text{layout}}_{\text{anchor}}$, where $\beta \in [0,1)$. Ablations suggest cache stride $k=3$ and momentum $\beta=0.5$ to $0.7$.
    *   **Design Motivation**: The update trajectory reveals that shape tokens are near-linear over short ranges, while layout tokens oscillate. Pure extrapolation for layout leads to accumulated errors manifested as pose drift. The anchor term provides a restorative force to prevent divergence.

2.  **Joint Spatiotemporal Token Carving + Adaptive Step Caching for SLaT Generator**:
    *   **Function**: Simultaneously reduces spatial redundancy (which tokens to compute) and temporal redundancy (which steps to evaluate fully) during SLaT refinement.
    *   **Mechanism**: A unified saliency metric is constructed: $\mathcal{J}_i(t) = \tfrac{1}{2}(\mathcal{M}_i(t) + \gamma \mathcal{A}_i(t)) + \tfrac{1}{2}\mathcal{S}_{\text{freq}}(i)$, where $\mathcal{M}_i(t) = \|\mathbf{v}_{t,i}\|_2$ measures update magnitude, $\mathcal{A}_i(t) = \|\mathbf{v}_{t,i}-\mathbf{v}_{t+1,i}\|_2$ measures abrupt change, and $\mathcal{S}_{\text{freq}}(i)$ represents high-frequency structural intensity via FFT. Only top-K (e.g., top-10%) tokens enter the backbone. Temporally, trajectory nonlinearity is estimated via curvature $\kappa_t = \|\mathbf{v}_t-\mathbf{v}_{t-1}\|_2 / \|\mathbf{x}_t-\mathbf{x}_{t-1}\|_2$. Tangent increments $\Delta_i := \mathbf{v}_i - \mathbf{x}_i$ are cached, and during skips, $\hat{\mathbf{v}}_t = \mathbf{x}_t + \Delta_i$. Full evaluation is triggered if accumulated relative change $E_t = \sum \varepsilon_n$ exceeds threshold $\mathcal{E}$.
    *   **Design Motivation**: Heatmaps of token-wise updates show sparsity—many low-entropy regions converge early. Combining spatial carving and temporal caching cleans up "invalid computation" while error-bounded switching prevents error explosion.

3.  **Spectral-Aware Dynamic Token Aggregation for Mesh Decoder**:
    *   **Function**: Adaptively determines the downsampling intensity for mesh decoder input tokens based on instance complexity—aggressive compression for simple objects, detail preservation for complex ones.
    *   **Mechanism**: FFT is applied to the 2D mask $\mathbf{M}_{2D}$ and coarse 3D voxels $\mathbf{V}_{3D}$. High-frequency energy ratio is defined as $\mathcal{H}(\mathbf{X}) = \sum_{\omega \in \Omega_{\text{high}}} \|\mathcal{F}(\mathbf{X})[\omega]\|_2^2 / \sum_{\omega \in \Omega_{\text{total}}} \|\mathcal{F}(\mathbf{X})[\omega]\|_2^2$, combined into $\mathcal{H}_{\text{joint}} = w\mathcal{H}(\mathbf{M}_{2D}) + (1-w)\mathcal{H}(\mathbf{V}_{3D})$. Downsampling factor $\mathcal{S} \in \{1.25, 1.5, 2.0\}$ is selected via thresholds $\tau_{\text{low}}, \tau_{\text{high}}$. Aggregation involves coordinate quantization $\hat{\mathbf{p}}_i = \lfloor \mathbf{p}_i / \mathcal{S} \rfloor$ and max-pooling within bins, reducing token counts by factor $\approx 1/\mathcal{S}^3$.
    *   **Design Motivation**: Simple objects concentrate spectral energy in low-frequency edges, while complex objects scatter high-frequency energy across the surface. HFER (High-Frequency Energy Ratio) serves as a lightweight, closed-form proxy for complexity to guide instance-level routing.

### Loss & Training
The entire method is **training-free**, requiring no modification to SAM3D weights and no distillation or quantization. All modules are inserted at inference time. Hyperparameters are selected via grid search on a small validation set: $k=3$ and $\beta \approx 0.7$ for SS; top-10% carving and threshold $\mathcal{E}$ for SLaT; $w$ and $\tau$ calibrated by dataset for Mesh.

## Key Experimental Results

### Main Results
Comparison with SOTA acceleration schemes on Toys4K, Aria Digital Twin (ADT), and ISO3D using SAM3D as the base:

| Method | Uni3D↑ | CD↓ | $F_1$@0.05↑ | vIoU↑ | 3D-IoU↑ | Scene Time (s)↓ | Object Time (s)↓ | Object Accel. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SAM-3D (base) | 0.369 | 0.022 | 92.34 | 0.543 | 0.403 | 462.3 | 31.04 | 1.00× |
| Random Drop | 0.264 | 0.030 | 83.52 | 0.327 | 0.094 | 402.2 | 15.93 | 1.95× |
| Uniform Merge | 0.329 | 0.023 | 91.48 | 0.540 | 0.367 | 366.8 | 15.43 | 2.01× |
| Fast3DCache | 0.348 | 0.022 | 91.31 | 0.505 | 0.051 | 443.3 | 30.14 | 1.03× |
| TaylorSeer | 0.344 | 0.028 | 90.95 | 0.504 | 0.374 | 265.6 | 22.93 | 1.35× |
| EasyCache | 0.342 | 0.028 | 87.06 | 0.432 | 0.186 | 244.9 | 23.11 | 1.34× |
| **Fast-SAM3D** | **0.350** | **0.022** | **92.59** | **0.552** | 0.375 | **229.7** | **11.60** | **2.67×** |

Fast-SAM3D's 2.67× object-level acceleration significantly outperforms TaylorSeer/EasyCache (1.35×/1.34×), with $F_1$ and vIoU parity or slight improvement over the base. Fast3DCache effectively fails (1.03×) in the single-view setting, as it relies on multi-view redundancy.

### Ablation Study
Pairwise and full combination of the three modules (Scene-level time, Toys4K):

| SS | SLaT | Mesh | CD↓ | $F_1$@0.05↑ | vIoU↑ | Scene Time (s)↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ✗ | ✗ | ✗ | 0.022 | 92.34 | 0.543 | 462.3 |
| ✓ | ✗ | ✗ | 0.022 | 92.34 | 0.543 | 408.6 |
| ✗ | ✓ | ✗ | 0.022 | 92.50 | 0.540 | 365.9 |
| ✗ | ✗ | ✓ | 0.022 | 92.43 | 0.557 | 320.4 |
| ✓ | ✓ | ✗ | 0.021 | 92.88 | 0.534 | 310.5 |
| ✓ | ✗ | ✓ | 0.022 | 92.58 | 0.553 | 289.9 |
| ✗ | ✓ | ✓ | 0.022 | 92.43 | 0.554 | 301.3 |
| ✓ | ✓ | ✓ | 0.022 | 92.59 | 0.552 | **229.7** |

Key thresholds for cache stride and carving: $k=3$ balances vIoU and speed; $k \ge 4$ causes 3D-IoU to drop from 0.375 to 0.241 (pose drift). Top-10% carving is more stable than top-20%; top-5% shows marginal speed gains.

### Key Findings
-   **Mesh module contributes most**: Enabling only the Mesh module reduces time from 462 s to 320 s, suggesting the mesh decoder is the primary bottleneck. Spectral-aware instance-level aggregation is essential.
-   **Acceleration improves quality**: The SLaT module alone increases $F_1$ from 92.34 to 92.50. This is attributed to saliency-based carving acting as a "spatial filter" to remove low-confidence noise tokens.
-   **SS module is vital for layout**: Random Drop and TaylorSeer suffer from pose drift or semantic drift because they fail to protect the high-frequency nature of layout tokens. The momentum anchor in Fast-SAM3D is key to stabilizing the global coordinate system.
-   **Hyperparameter sensitivity**: $\beta$ is stable within 0.5–0.9. Once the cache stride exceeds the local linear region ($k \ge 4$), layout accuracy drops precipitously, supporting the use of "dynamic-aware step allocation" over uniform skipping.

## Highlights & Insights
-   **"Heterogeneity as an acceleration cue" is a transferable principle**. The authors decompose this into modality (shape vs. layout), spatiotemporal (which tokens/steps), and spectral (which instances) layers. This "profile first, cut second" methodology is more robust than stacking arbitrary tricks and is applicable to various multi-stage diffusion models.
-   **Error-bounded switching is the soul of training-free caching**. Using $E_t = \sum \varepsilon_n$ to trigger anchor refreshes provides a "safety rail," avoiding the need for manual step schedules. This can be extended to video diffusion or 3D Gaussian Splatting generation.
-   **Spectral proxies are cheap and effective routing signals**. The near-zero cost of HFER via FFT allows for stable differentiation between simple and complex objects to guide discrete downsampling. This instance-level adaptive routing is highly valuable for edge inference.
-   **"Speedup $\neq$ degradation" counter-example**. Fast-SAM3D achieves 2.67× acceleration while preserving geometry, layout, and texture, suggesting significant "quality-orthogonal" redundancy in large model inference.

## Limitations & Future Work
-   The method is a training-free inference layer and does not replace backbone improvements. Peak memory remains largely unchanged (Appendix B).
-   Evaluations are centered on SAM3D and a single TRELLIS transfer experiment; generalizability across other single-view 3D diffusion bases (e.g., Hunyuan3D, TripoSR) requires further verification.
-   Several thresholds/coefficients are introduced ($k, \beta, K, \mathcal{E}, w, \tau_{\text{low}}, \tau_{\text{high}}$); it is unclear if these require retraining per dataset.
-   The high-frequency cutoff for HFER is manually set; stability on extremely complex geometries (e.g., hair, fluids) outside the training distribution needs more evidence.
-   Future direction: Making carving ratio $K$ and cache stride $k$ instance-adaptive via a lightweight controller could potentially push acceleration beyond 2.67×.

## Related Work & Insights
-   **vs. TaylorSeer / EasyCache**: These are uniform temporal step caching schemes. This work proves that in decoupled shape-layout diffusion like SAM3D, a single caching strategy for all tokens inevitably causes pose drift. Modality-aware decoupling shows that "role-aware > step-aware."
-   **vs. Fast3DCache**: Designed for multi-view reconstruction, it relies on cross-view redundancy. It degrades in single-view scenarios (1.03×). This paper shifts the focus from "inter-view" to "inter-modality + intra-temporal."
-   **vs. Bolya & Hoffman ToMe (Token Merging)**: ToMe merges tokens uniformly based on similarity in 2D ViTs. This paper uses FFT spectral signals for instance-level adaptive aggregation tailored to 3D geometric spectral variance.
-   **vs. Distillation / Quantization**: Those require retraining, which is expensive for 1.7B multi-stage models. This training-free solution is industry-friendly and can be stacked with distillation/quantization.

## Rating
-   Novelty: ⭐⭐⭐⭐ The "heterogeneity-aware" principle leads the three modules. The three observations (kinematics/sparsity/spectral) are well-supported visually.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, 6 metrics, 6 strong baselines, comprehensive ablation/hyperparameter sweeps.
-   Writing Quality: ⭐⭐⭐⭐ Clear logic for the three modules (Observation → Metric → Mechanism), well-integrated formulas and figures.
-   Value: ⭐⭐⭐⭐ Substantially reduces SAM3D object-level latency towards real-time without retraining, offering clear value for industrial 3D generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](../../ICCV2025/3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](../../CVPR2026/3d_vision/fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](../../ICCV2025/3d_vision/faster_and_better_3d_splatting_via_group_training.md)

</div>

<!-- RELATED:END -->
