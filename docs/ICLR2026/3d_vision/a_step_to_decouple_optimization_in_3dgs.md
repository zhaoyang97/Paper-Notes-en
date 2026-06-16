---
title: >-
  [Paper Note] A Step to Decouple Optimization in 3DGS
description: >-
  [ICLR 2026][3D Vision][3DGS] This paper provides an in-depth analysis of two overlooked coupling issues in 3DGS optimization — update-step coupling (implicit updates and momentum rescaling for invisible viewpoints) and g…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3DGS"
  - "Optimizer"
  - "Adam"
  - "Weight Decay"
  - "Sparse Optimization"
date: 2026-05-08
content_hash: 4778e96774700e94
---

# A Step to Decouple Optimization in 3DGS

**Conference**: ICLR 2026
**arXiv**: [2601.16736](https://arxiv.org/abs/2601.16736)  
**Code**: [https://eliottdjay.github.io/adamwgs/](https://eliottdjay.github.io/adamwgs/)  
**Area**: 3D Vision
**Keywords**: 3DGS, Optimizer, Adam, Weight Decay, Sparse Optimization

## TL;DR
This paper provides an in-depth analysis of two overlooked coupling issues in 3DGS optimization — update-step coupling (implicit updates and momentum rescaling for invisible viewpoints) and gradient coupling (entanglement of regularization and photometric loss in Adam momentum) — and proposes AdamW-GS by decoupling and recombining these components, simultaneously improving reconstruction quality and reducing redundant primitives without additional pruning operations.

## Background & Motivation

**Background**: 3DGS directly inherits the Adam optimizer and synchronous update strategy from deep learning, updating all primitives (including those invisible from the current viewpoint) simultaneously.

**Limitations of Prior Work**: (a) Update-step coupling: zero gradients for invisible primitives still trigger momentum rescaling and implicit updates, degrading efficiency and performance; (b) Gradient coupling: regularization loss and photometric loss are entangled in Adam's adaptive gradient, making regularization strength uncontrollable — either too strong or too weak. Sparse Adam improves efficiency but at the cost of performance.

**Key Challenge**: Primitive optimization in 3DGS fundamentally differs from weight optimization in DNNs — each attribute carries physical meaning and each primitive has a different importance, yet existing optimizers do not account for these distinctions.

**Goal**: To understand and decouple the optimization coupling in 3DGS, and to design a more principled optimization strategy.

**Key Insight**: Decompose Adam's behavior in 3DGS into three components (Sparse Adam, Re-State Regularization, and Decoupled Attribute Regularization), study each independently, and then recombine them.

**Core Idea**: Import the decoupling philosophy of AdamW into 3DGS, replacing coupled regularization with an adaptive form $\nabla\mathcal{R}/\sqrt{\hat{v}}$ so that regularization strength automatically adjusts according to each primitive's optimization state.

## Method

### Overall Architecture
3DGS optimization is decoupled in three steps: (1) Sparse Adam restricts updates to visible primitives only; (2) Re-State Regularization (RSR) actively decays momentum to replace the role of momentum rescaling in implicit updates; (3) Decoupled Attribute Regularization (DAR) disentangles regularization gradients from Adam momentum and adaptively scales them using the second moment. These components are recombined into the final AdamW-GS optimizer.

### Key Designs

1. **Sparse Adam + Re-State Regularization (RSR)**:

    - **Function**: Replaces synchronous updates with viewpoint-aware asynchronous updates, and substitutes implicit-update momentum rescaling with active momentum decay.
    - **Mechanism**: Sparse Adam updates only visible primitives via $\beta' = \beta \cdot \mathcal{V} + (1-\mathcal{V})$. RSR periodically samples primitives and applies $m^{new} = \alpha_1 m^{old}$, $v^{new} = \alpha_2 v^{old}$ to actively decay momentum.
    - **Design Motivation**: Observations show that Sparse Adam is more stable but lacks exploration, while Adam's implicit updates, though noisy, are beneficial for regularization activation and redundancy removal. RSR retains the benefit of momentum decay without the side effects of implicit updates.

2. **Decoupled Attribute Regularization (DAR)**:

    - **Function**: Decouples regularization from Adam momentum and adaptively scales regularization strength using the second moment of the photometric loss.
    - **Mechanism**: $\theta_{t+1} = \theta_t - \eta[\frac{\hat{m}'_t}{\sqrt{\hat{v}'_t}+\epsilon} + \min(\lambda_\theta \frac{\nabla\mathcal{R}/N_I}{\sqrt{\hat{v}'_t}+\epsilon}, \mathcal{C}_t)]$, where $\hat{v}'_t$ is computed solely from photometric loss gradients.
    - **Design Motivation**: (a) In under-optimized regions, large $\nabla\ell$ leads to large $\hat{v}$, yielding weak regularization that does not interfere with reconstruction; (b) Near saddle points, small $\nabla\ell$ leads to small $\hat{v}$, yielding strong regularization that aids escape; (c) Under coupled regularization, directly scaling $\lambda$ by 10× causes optimization collapse, whereas the decoupled form allows safe adjustment.

3. **AdamW-GS (Recombination)**:

    - **Function**: Integrates Sparse Adam, RSR, and DAR into a unified optimizer.
    - **Mechanism**: Asynchronous updates + periodic momentum decay + adaptive decoupled regularization.
    - **Design Motivation**: After individually validating each component's contribution, only the beneficial elements are retained and recombined.

### Loss & Training
The photometric loss (L1 + DSSIM) remains unchanged. Regularization losses (opacity L1 + scale L1) are decoupled via DAR. In the vanilla 3DGS setting, noise regularization is additionally applied to encourage exploration.

## Key Experimental Results

### Main Results

**MipNeRF360 Dataset (vanilla 3DGS vs. AdamW-GS):**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | #Primitives (M)↓ | Redundant Primitives↓ |
|--------|-------|-------|--------|-----------------|----------------------|
| 3DGS (Adam) | 27.507 | 0.815 | 0.216 | 3.33 | 0.23M dead |
| 3DGS (Sparse Adam) | 27.285 | 0.809 | 0.228 | 2.53 | 0.04M dead |
| **AdamW-GS** | **27.75+** | **0.82+** | **0.20−** | **~2.5** | **Minimal** |

### Ablation Study

| Component | PSNR | $\Delta N_a$ | Notes |
|-----------|------|------------|-------|
| MCMC Baseline | 27.948 | −3.75% | Standard Adam |
| + Sparse Adam | 27.998 | +4.28% | Efficiency gains but reduced exploration |
| + AIU | 28.050 | +3.62% | Artificial implicit updates restore exploration |
| + RSR | 28.017 | +0.51% | Momentum decay activates regularization |
| + DAR (opacity+scale) | **28.27+** | — | Decoupled regularization yields significant gains |

### Key Findings
- Sparse Adam is more stable but lacks exploration; Adam's implicit updates carry side effects yet are beneficial for regularization activation.
- Momentum rescaling (decreasing $v$) amplifies the effective strength of regularization, explaining the behavioral difference between Adam and Sparse Adam.
- The decoupled regularization automatically removes redundant primitives without additional pruning — AdamW-GS substantially reduces dead primitives in vanilla 3DGS.
- Under the 3DGS-MCMC framework, DAR promotes more primitive redistribution through stronger regularization, improving reconstruction quality.

## Highlights & Insights
- **Deep transfer from DNN optimization to 3DGS optimization**: The philosophy of AdamW (decoupled weight decay) is creatively adapted to 3DGS — not as a direct transplant, but with careful consideration of the physical meaning of primitives, realizing adaptivity via $1/\sqrt{\hat{v}}$ rather than a constant penalty.
- **Discovery and analysis of "implicit updates"**: The momentum rescaling and attribute updates triggered by zero gradients — previously overlooked — are systematically analyzed for the first time, revealing a unique behavioral aspect of Adam in 3DGS.
- **Redundancy elimination without pruning**: Addressing redundancy through optimizer design rather than post-processing represents a methodological advancement.

## Limitations & Future Work
- The sampling schedule in RSR (StSS) requires manual specification; optimal schedules may vary across scenes.
- The clipping constant $\mathcal{C}_t$ in decoupled regularization is empirically robust but lacks theoretical grounding.
- Validation is limited to standard benchmarks (MipNeRF360, Tanks & Temples); generalization to larger-scale scenes remains untested.
- Applicability to the broader ecosystem of 3DGS downstream extensions has not been comprehensively verified.

## Related Work & Insights
- **vs. AdamW (Loshchilov & Hutter)**: AdamW decouples L2 regularization as a constant decay; AdamW-GS further achieves adaptive decoupling via $1/\sqrt{\hat{v}}$, better suited to the varying importance of individual primitives in 3DGS.
- **vs. Rota Bulò et al. 2025**: Their approach replaces opacity reset with constant opacity decay, a form of AdamW-style regularization; this paper argues that constant penalties are insufficient — per-primitive adaptive adjustment is necessary.
- **vs. Sparse Adam (Mallick et al.)**: Sparse Adam addresses efficiency but sacrifices performance; this paper recovers performance through RSR and DAR while maintaining efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces an optimizer-design perspective into 3DGS analysis, discovering and explaining previously overlooked coupling phenomena.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated under dual frameworks (3DGS + 3DGS-MCMC) with extensive ablations and step-by-step component addition experiments.
- Writing Quality: ⭐⭐⭐⭐ Analysis is thorough, though the large number of notations and experimental variants creates a non-trivial reading barrier.
- Value: ⭐⭐⭐⭐ Provides a new understanding of 3DGS optimization and a practically useful optimizer improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](../../ICCV2025/3d_vision/3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [\[CVPR 2026\] Coverage Optimization for Camera View Selection](../../CVPR2026/3d_vision/coverage_optimization_for_camera_view_selection.md)
- [\[CVPR 2026\] GenSplat: Bridging the Generalization Gap in 3DGS Language Comprehension](../../CVPR2026/3d_vision/gensplat_bridging_the_generalization_gap_in_3dgs_language_comprehension.md)
- [\[ICCV 2025\] BokehDiff: Neural Lens Blur with One-Step Diffusion](../../ICCV2025/3d_vision/bokehdiff_neural_lens_blur_with_one-step_diffusion.md)
- [\[CVPR 2026\] Faster-GS: Analyzing and Improving Gaussian Splatting Optimization](../../CVPR2026/3d_vision/faster-gs_analyzing_and_improving_gaussian_splatting_optimization.md)

</div>

<!-- RELATED:END -->
