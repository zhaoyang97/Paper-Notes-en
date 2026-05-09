---
title: >-
  [Paper Note] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control
description: >-
  [ICLR 2026][Image Generation][Shape Editing] This paper proposes Follow-Your-Shape, a training-free and mask-free shape-aware editing framework. It constructs a Trajectory Divergence Map (TDM) by computing token-level velocity discrepancies between inversion and editing trajectories to precisely localize editing regions, and employs a staged KV injection strategy to achieve large-scale shape transformations while strictly preserving the background.
tags:
  - ICLR 2026
  - Image Generation
  - Shape Editing
  - Trajectory Divergence Map
  - training-free
  - Flow Matching
  - KV Injection
date: 2026-05-08
content_hash: 46ba763f5a93aa12
---

# Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control

**Conference**: ICLR 2026
**arXiv**: [2508.08134](https://arxiv.org/abs/2508.08134)
**Code**: [Project Page](https://follow-your-shape.github.io/)
**Area**: Diffusion Models / Image Editing
**Keywords**: Shape Editing, Trajectory Divergence Map, training-free, Flow Matching, KV Injection

## TL;DR
This paper proposes Follow-Your-Shape, a training-free and mask-free shape-aware editing framework. It constructs a Trajectory Divergence Map (TDM) by computing token-level velocity discrepancies between inversion and editing trajectories to precisely localize editing regions, and employs a staged KV injection strategy to achieve large-scale shape transformations while strictly preserving the background.

## Background & Motivation
**Background**: Diffusion/Flow model-based image editing performs well on general tasks but frequently fails at structural editing involving large-scale shape transformations—either failing to achieve the target shape change or corrupting non-edited regions.

**Limitations of Prior Work**: Existing region control strategies suffer from fundamental drawbacks:
   - External binary masks: overly rigid and ill-suited for fine-grained boundaries
   - Cross-attention map inference: noisy and unstable, unreliable under large deformations
   - Unconditional KV injection: globally preserves structure but suppresses the intended edit

**Key Challenge**: A fundamental tension between edit controllability and content preservation—enabling a Flow model to precisely modify the shape of a target region without affecting surrounding areas.

**Goal**: How to achieve precise, large-scale shape editing without training or external masks?

**Key Insight**: From a dynamical systems perspective—editing regions can be localized by measuring the degree of divergence between denoising trajectories under source and target conditions.

**Core Idea**: Automatically localize editing regions by comparing velocity field discrepancies under source and target prompts, and leverage staged KV injection to achieve stable, shape-aware editing.

## Method

### Overall Architecture
An inference-time editing framework built on FLUX.1-dev. Given a source image and prompt, the method first inverts the image to a noisy latent, then performs denoising in three stages: Stage 1 applies global KV injection to stabilize the trajectory → Stage 2 performs editing while collecting TDMs → Stage 3 applies selective KV injection guided by the TDM mask along with optional ControlNet structural guidance.

### Key Designs
1. **Trajectory Divergence Map (TDM)**: The core contribution. During denoising, the token-level $L_2$ discrepancy between velocity fields conditioned on the source and target prompts is computed:
$$\delta_t^{(i)} = \| v_\theta(\mathbf{z}_t^{(i)}, t, \mathbf{c}_{\text{tgt}}) - v_\theta(\mathbf{x}_t^{(i)}, t, \mathbf{c}_{\text{src}}) \|_2$$
After min-max normalization to obtain $\tilde{\delta}_t^{(i)} \in [0,1]$, regions with high divergence are identified as the target editing regions.

2. **Staged Editing Strategy**:

    - **Stage 1 (Trajectory Stabilization)**: For the first $k_{\text{front}}=2$ steps, unconditional KV injection ($M_S = \mathbf{0}$) is applied to anchor the trajectory onto the faithful reconstruction manifold.
    - **Stage 2 (Editing + TDM Aggregation)**: Within the editing window $N$, $M_S = 1$ is set to allow editing while per-timestep normalized TDMs are collected. Softmax-weighted temporal fusion is used for aggregation:
    $\hat{\delta}^{(i)} = \sum_{t \in N} \alpha_t^{(i)} \cdot \tilde{\delta}_t^{(i)}, \quad \alpha_t^{(i)} = \frac{\exp(\tilde{\delta}_t^{(i)})}{\sum_{t'} \exp(\tilde{\delta}_{t'}^{(i)})}$
    Gaussian blurring $\tilde{M}_S = \mathcal{G}_\sigma * \hat{\delta}$ followed by Otsu thresholding yields the binary mask $M_S$.
    - **Stage 3 (Structural and Semantic Consistency)**: KV features are blended according to $M_S$ (target KV for editing regions, source KV for non-editing regions):
    $\{K^*, V^*\} \leftarrow M_S \odot \{K^{\text{tgt}}, V^{\text{tgt}}\} + (1 - M_S) \odot \{K^{\text{inv}}, V^{\text{inv}}\}$
    Optionally, ControlNet (depth + Canny) provides auxiliary structural constraints.

3. **ReShapeBench**: A new benchmark comprising 120 images spanning single-object (70) and multi-object (50) scenes plus a general set (50), totaling 290 shape editing cases. Source–target prompt pairs differ only in the foreground object description and are verified by human annotators.

### Loss & Training
- **Training-free method**: A purely inference-time framework requiring no training or fine-tuning.
- 14 denoising steps, guidance scale 2.0, $k_{\text{front}} = 2$.
- ControlNet conditioning applied over the normalized denoising interval $[0.1, 0.3]$, depth strength 2.5, Canny strength 3.5.
- RF-Solver second-order inversion is employed.

## Key Experimental Results

### Main Results (ReShapeBench + PIE-Bench)

| Method | AS↑ | PSNR↑ | LPIPS↓(×10³) | CLIP Sim↑ |
|------|-----|-------|-------------|-----------|
| MasaCtrl | 5.83 | 23.54 | 125.36 | 20.84 |
| RF-Edit | 6.52 | 33.28 | 17.53 | 30.41 |
| KV-Edit | 6.51 | 34.73 | 16.42 | 26.97 |
| FLUX.1 Kontext | 6.53 | 32.91 | 18.35 | 28.53 |
| Ours (w/o ControlNet) | 6.52 | 34.85 | **9.04** | 32.97 |
| **Ours (Full)** | **6.57** | **35.79** | **8.23** | **33.71** |

*Follow-Your-Shape achieves top performance across all metrics on ReShapeBench, with LPIPS of only 8.23 (vs. 16.42 for the next best) and CLIP Sim of 33.71 (vs. 30.41).*

### Ablation Study (Effect of $k_{\text{front}}$)

| $k_{\text{front}}$ | AS↑ | PSNR↑ | LPIPS↓(×10³) | CLIP Sim↑ |
|------|-----|-------|-------------|-----------|
| 0 | 6.51 | 32.79 | 10.04 | 31.05 |
| 1 | 6.55 | 34.38 | 9.88 | 32.56 |
| **2** | **6.57** | **35.79** | **8.23** | **33.71** |
| 3 | 6.52 | 31.25 | 10.52 | 29.41 |
| 4 | 6.48 | 30.41 | 12.37 | 27.66 |

*An optimal $k_{\text{front}}$ exists: too small yields an unstable trajectory, too large suppresses editing—$k_{\text{front}}=2$ achieves the best trade-off.*

### Key Findings
- Even without ControlNet, TDM-guided KV injection substantially outperforms all baselines.
- Diffusion-based methods (MasaCtrl, PnPInversion, Dit4Edit) degrade severely under large deformations.
- Flow model methods produce higher image quality overall but still exhibit ghosting or incomplete transformations under drastic shape changes.
- Otsu thresholding is adaptive and requires no manual tuning; the TDM distribution is naturally suited to binary segmentation.
- The timing ($[0.1, 0.3]$) and strength (depth 2.5, Canny 3.5) of ControlNet conditioning are critical hyperparameters.

## Highlights & Insights
- **Dynamical systems perspective**: Reformulating editing region localization as a trajectory divergence measure yields an elegant theoretical motivation.
- **No external masks or attention maps**: TDM dynamically extracts editing regions from the model's own behavior.
- **Three-stage scheduling is carefully designed**: The stabilize→explore→constrain pipeline aligns naturally with denoising dynamics.
- **Softmax-weighted temporal fusion**: Captures temporal dynamics more faithfully than simple averaging, as a token may remain unchanged at certain timesteps but shift significantly at later ones.
- **ReShapeBench fills an evaluation gap**: Existing benchmarks are not designed for large-scale shape transformation tasks.

## Limitations & Future Work
- Only supports prompt-based shape editing; complex geometric transformations that cannot be precisely described in text are not handled.
- Requires second-order inversion via RF-Solver, doubling computational cost.
- TDM is unreliable at high-noise timesteps, necessitating the staged strategy—which in turn introduces additional hyperparameters ($k_{\text{front}}$, window size $N$).
- ControlNet dependency is optional but still required in certain scenarios, deviating from the ideal of a fully tool-free approach.
- Extension to video editing remains unexplored.

## Related Work & Insights
- **RF-Edit / RF-Solver**: This work builds upon RF-Solver inversion and replaces its global KV injection with TDM-guided selective injection.
- **DiffEdit**: Similarly computes source–target prediction differences to generate masks, but operates on diffusion models without considering temporal trajectory dynamics.
- **Stable Flow**: Proposes selective injection at vital layers; Follow-Your-Shape performs selective injection along the spatial dimension.
- **Insight**: The deterministic ODE trajectories of Flow Matching are naturally well-suited for computing divergence measures. This trajectory-based paradigm may generalize to video editing or 3D content editing.

## Rating
- Novelty: ⭐⭐⭐⭐ — The trajectory divergence perspective of TDM is original, though staged injection represents an incremental contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual benchmarks and detailed ablations, though a user study is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation figures are clear and the methodological derivation is well-structured.
- Value: ⭐⭐⭐⭐ — Provides a systematic solution for shape editing, a previously underexplored subtask.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](../../CVPR2026/image_generation/physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)

</div>

<!-- RELATED:END -->
