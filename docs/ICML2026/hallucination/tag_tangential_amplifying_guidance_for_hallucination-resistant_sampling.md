---
title: >-
  [Paper Note] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling
description: >-
  [ICML 2026][Hallucination Detection][Tangential Amplifying] TAG decomposes each diffusion update step along the current latent variable direction into "radial + tangential" components. It applies an additional amplificat…
tags:
  - "ICML 2026"
  - "Hallucination Detection"
  - "Tangential Amplifying"
  - "Inference-time Guidance"
  - "Geometry-aware Sampling"
  - "Hallucination Suppression"
  - "CFG Enhancement"
date: 2026-05-08
content_hash: 512c60b1619960dd
---

# TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling

**Conference**: ICML 2026  
**arXiv**: [2510.04533](https://arxiv.org/abs/2510.04533)  
**Code**: Yes (provided as a Project Page in the paper)  
**Area**: Hallucination Detection  
**Keywords**: Tangential Amplifying, Inference-time Guidance, Geometry-aware Sampling, Hallucination Suppression, CFG Enhancement

## TL;DR
TAG decomposes each diffusion update step along the current latent variable direction into "radial + tangential" components. It applies an additional amplification factor $\eta \ge 1$ only to the tangential component. Using first-order Taylor expansion, it is proven that this is equivalent to monotonically increasing the log-likelihood gain, thereby pulling samples toward high-density regions of the data manifold and mitigating semantic hallucinations in diffusion models with almost zero extra computational cost.

## Background & Motivation

**Background**: Diffusion models have reached SOTA in image generation but frequently suffer from "hallucinations"—extra fingers, incorrectly mixed objects, or structures that violate the prompt. The mainstream remedy is inference-time guidance (CFG and its variants like PAG, SEG, CFG++, etc.), which pushes samples away from low-density regions by applying scalar scaling to conditional/unconditional residuals or making minor network modifications.

**Limitations of Prior Work**: Existing guidance methods are essentially "geometry-unaware"—they simply perform scalar multiplication on the cond–uncond residual without considering the local directional structure of the data manifold at the current noise level. Excessive scaling can disturb ODE/SDE solvers, compress diversity, or even introduce new artifacts.

**Key Challenge**: Diffusion sampling trajectories must follow the noise schedule "radially" (shrinking the radius according to the schedule) while simultaneously moving "tangentially" along the data manifold (refining sample details toward the true distribution). Scaling both directions with a single scalar disrupts the radial schedule, leading to over-smoothing; conversely, failing to distinguish between them misses the opportunity to climb the manifold.

**Goal**: Design a guidance method that requires no architecture changes, no retraining, and is plug-and-play with any off-the-shelf diffusion backbone. It should utilize only the geometric signals available from the trajectory itself to directionally amplify "directions truly useful for sample quality."

**Key Insight**: The authors start from the Tweedie identity and the Gaussian annulus theorem to argue that at noise level $t_k$, data concentrates near a thin spherical shell $\mathcal{M}_k$. After performing orthogonal decomposition of each update step $\Delta_{k+1}$ on $\mathcal{M}_{k+1}$, it is found that the radial component represents the "radius reduction" prescribed by the noise schedule, while the tangential component carries the semantic structure (Figure 2 visualization shows the tangential part corresponds to clear semantic maps, while the radial part appears as noise).

**Core Idea**: Keep the radial component fixed and amplify only the tangential component by a factor $\eta \ge 1$. This allows the single-step update to move further along the data manifold toward higher density regions without altering the noise schedule.

## Method

### Overall Architecture
TAG is a "residual reweighting" operator wrapped around standard base solvers like DDIM, EDM, or Flow Matching. Given the state at step $k+1$, $\bm{x}_{k+1}$:

1. Calculate the "original update" $\Delta_{k+1} = \tilde{\bm{x}}_k - \bm{x}_{k+1}$ using the backbone $\epsilon_\theta$ and the base solver (e.g., the difference in a single-step DDIM prediction).
2. Define spherical projections using the current state: $\bm{P}_{\mathcal{M}_{k+1}} = \hat{\bm{x}}_{k+1}\hat{\bm{x}}_{k+1}^\top$ (radial projection onto $\bm{x}_{k+1}/\|\bm{x}_{k+1}\|_2$) and $\bm{P}^\perp_{\mathcal{M}_{k+1}} = I - \bm{P}_{\mathcal{M}_{k+1}}$ (tangential).
3. Replace the original $\bm{x}_k = \bm{x}_{k+1} + \Delta_{k+1}$ with $\bm{x}_k \leftarrow \bm{x}_{k+1} + \bm{P}_{\mathcal{M}_{k+1}}\Delta_{k+1} + \eta\,\bm{P}^\perp_{\mathcal{M}_{k+1}}\Delta_{k+1}$.

The entire process requires no extra denoiser forward passes, no parameter changes, and is stackable with existing guidance methods like CFG, PAG, SEG, or SSG.

### Key Designs

1. **Tangential Amplifying Update Rule**:
    - **Function**: Decomposes each original update into radial and tangential parts, multiplying only the tangential component by $\eta$ while keeping the radial part unchanged.
    - **Mechanism**: Uses the sample itself as the projection basis $\hat{\bm{x}}_{k+1}$ to define $\bm{P}_{\mathcal{M}_{k+1}}$ and $\bm{P}^\perp_{\mathcal{M}_{k+1}}$. The new increment is $\Delta^{\text{TAG}}_{k+1} = (\bm{P}_{\mathcal{M}_{k+1}} + \eta\bm{P}^\perp_{\mathcal{M}_{k+1}})\Delta_{k+1}$. When substituted into DDIM coefficients, this is equivalent to amplifying the tangential component of $\epsilon_\theta$, thus acting as a "geometry-aware rescaling of predicted noise."
    - **Design Motivation**: Visualizations in Figure 2 show the radial part as structureless noise and the tangential part as clear semantic maps, indicating that semantic information is naturally concentrated in the tangential direction. Amplifying it selectively enhances semantics without amplifying noise.

2. **First-order Taylor Gain Monotonicity Proof**:
    - **Function**: Theoretically explains "why amplifying the tangential component improves image quality."
    - **Mechanism**: Formulates inference-time guidance as a local log-likelihood maximization problem $\max (\bm{x}_k - \bm{x}_{k+1})^\top \nabla_{\bm{x}}\log p(\bm{x}\mid t_{k+1})| _{\bm{x}_{k+1}}$. Under the constraint $\|\bm{x}_k - \bm{x}_{k+1}\|_2 \le \delta_k$, the first-order Taylor gain is defined as $G(\eta) := (\Delta^{\text{TAG}}_{k+1})^\top \nabla_{\bm{x}}\log p$. The paper proves that $G(\eta)$ is monotonically increasing with respect to $\eta$ (Theorem 4.1), meaning that any $\eta > 1$ strictly improves local likelihood.
    - **Design Motivation**: Unlike "empirically larger is better," this interprets tangential amplification as an MLE projection with step-size constraints. It explains why it does not destroy trajectories like blind CFG scaling—because the radial component remains unchanged, maintaining the "single-step calibration" provided by the Tweedie identity ($\langle \hat{\bm{x}}_{k+1}, \Delta^{\text{TAG}}_{k+1}\rangle = \langle \hat{\bm{x}}_{k+1}, \Delta_{k+1}\rangle$, Eq. 22).

3. **Tangential Alignment on CFG ($\text{TAG}_{\text{cfg}}$)**:
    - **Function**: Applies the same geometric perspective to CFG to mitigate artifacts caused by "tangential inconsistency" between conditional and unconditional branches.
    - **Mechanism**: Let the CFG residual be $\bm{g}_k = \bm{\varepsilon}_c - \bm{\varepsilon}_u$. Take its tangential projection $\bm{g}_k^\perp = \bm{P}^\perp_{\mathcal{M}}(\bm{x}_k)\bm{g}_k$. Then project the conditional prediction $\bm{\varepsilon}_c$ onto $\mathrm{span}(\bm{g}_k^\perp)$ to get the alignment vector $\bm{g}_k^{\text{align}} = \langle\bm{\varepsilon}_c, \bm{g}_k^\perp\rangle / \|\bm{g}_k^\perp\|_2^2 \cdot \bm{g}_k^\perp$. Finally, $\tilde{\bm{\varepsilon}}_k = \bm{\varepsilon}_u + \omega\bm{g}_k + \eta\bm{g}_k^{\text{align}}$.
    - **Design Motivation**: The authors argue that CFG's cond/uncond inconsistency mainly stems from mismatches in the tangential direction (as the radial part is jointly constrained by the noise schedule). Aligning them within the tangential subspace allows both predictions to "move in the same direction," avoiding the amplification of unwanted inconsistencies when the scalar $\omega$ is increased.

### Loss & Training
Completely training-free and fine-tuning-free. The algorithm only inserts projection and vector reweighting into the sampling loop (Algorithm 1, ~11 lines of code). The only hyperparameter is $\eta \ge 1$ (typical setting $\eta \approx 1.x$ for unconditional sampling; the CFG version involves two scalars $\omega$ and $\eta$).

## Key Experimental Results

Main backbones: SD v1.5 / v2.1 / XL / SD3, evaluated on COCO 2014; compositionality evaluated on T2I-CompBench.

### Main Results

| Setting | Backbone | FID ↓ | IS ↑ | AES ↑ | CMMD ↓ |
|------|----------|-------|------|-------|--------|
| Uncond. | SD v1.5 | 58.41 | 15.59 | 5.003 | 1.069 |
| Uncond. | SD v1.5 + **TAG** | **46.20** | **16.77** | **5.064** | **0.778** |
| Uncond. | SDXL | 119.14 | 9.08 | 5.645 | 2.474 |
| Uncond. | SDXL + **TAG** | **90.71** | 8.91 | 5.577 | **2.201** |
| Uncond. | SD3 | 84.26 | 11.53 | 5.261 | 1.671 |
| Uncond. | SD3 + **TAG** | **79.11** | **11.73** | **5.365** | **1.564** |

| Setting | Backbone | FID ↓ | ImageReward ↑ | CLIP ↑ |
|------|----------|-------|---------------|--------|
| Cond. (T2I) | SD v1.5 | 33.49 | −0.342 | 25.00 |
| Cond. (T2I) | SD v1.5 + **TAG** | **26.61** | **−0.339** | **25.09** |
| Cond. (T2I) | SD v2.1 | 26.12 | 0.143 | 25.35 |
| Cond. (T2I) | SD v2.1 + **TAG** | **21.59** | **0.424** | **26.16** |
| Cond. (T2I) | SD3 | 29.02 | 1.030 | 26.39 |
| Cond. (T2I) | SD3 + **TAG** | **27.54** | **1.043** | **26.56** |

### Ablation Study

| Configuration (SD v1.5, Uncond.) | FID ↓ | IS ↑ | CMMD ↓ | Note |
|--------------------------|-------|------|--------|------|
| No guidance | 58.41 | 15.59 | 1.069 | No guidance added |
| TAG | 46.20 | 16.77 | 0.778 | TAG only |
| PAG | 53.72 | 21.13 | 0.723 | Existing SOTA guidance |
| **TAG + PAG** | **52.61** | **21.20** | **0.701** | Further gain after stacking |
| SEG | 47.69 | 18.50 | 0.835 | Another existing guidance |
| **TAG + SEG** | **42.71** | **19.45** | **0.746** | Further gain after stacking |

On T2I-CompBench's Spatial/Complex subsets, TAG improved SDXL's 2DSpatial from 0.1857 → 0.1980, BLIP-VQA from 0.4443 → 0.4650, and ImageReward from 0.2596 → 0.3978, indicating significant improvement in structural hallucinations (e.g., three legs).

### Key Findings
- **Radial amplification ruins samples**: Figure 5 shows that amplifying the normal (radial) component together ("+TAG + Normal") leads to severe over-smoothing—explainable by the $\kappa$-fold radial contraction in Eq. 21, verifying the necessity of "tangential only" amplification.
- **TAG at 50 NFE outperforms baseline at 250 NFE**: Using the same backbone, TAG brings 50-step sample quality close to 5× step original sampling, showing it accurately refines the "sampling trajectory itself" rather than relying on more computation.
- **Truly Plug-and-Play**: Can be stacked with PAG / SEG / custom CFG and is monotonically superior; no mutual exclusion found. The single new hyperparameter $\eta$ is very stable.

## Highlights & Insights
- **Elevating "Geometry Awareness" from Empirical to Theoretical**: While PAG/SEG/CFG++ often empirically choose "directions not to amplify," TAG uses Tweedie + Gaussian annulus to prove why tangential amplification monotonically improves first-order likelihood, shifting diffusion guidance toward a geometric optimization perspective.
- **Clever Use of "Sample as Projection Basis"**: No external classifier or additional network forward pass is needed; using only $\hat{\bm{x}}_{k+1}$ splits the high-dimensional space into "directions prescribed by the schedule" and "directions for model creativity." This trick of using one's own normalized vector as a geometric frame can be transferred to any iterative algorithm with spherical shell priors (LM logits, flow matching, Schrödinger bridges, etc.).
- **"Tangential Inconsistency" Hypothesis for CFG**: Attributing the mismatch between conditional and unconditional predictions to the tangential subspace is insightful. If this holds, future work could focus on training a "tangential calibration network" rather than performing full-vector residual scaling.

## Limitations & Future Work
- **Limitations acknowledged by authors**: Theory is based on the local $C^2$ smoothness assumption and first-order Taylor expansion; when $\eta$ is too large, higher-order terms become non-negligible, leading to artifacts. The CFG variant introduces an extra hyperparameter $\eta$ that requires tuning.
- **Self-observed Limitations**: Using $\hat{\bm{x}}_{k+1}$ as a projection basis assumes strictly isotropic noise distribution and that the data manifold consists of concentric spherical shells. Non-Euclidean latent spaces (e.g., spherical flow matching, diffusion on SE(3)) would require redefining the projection. A slight decrease in IS on SDXL in Table 1 (9.08→8.91) suggests improvements in FID/CMMD may come at the cost of some diversity in details.
- **Improvement Ideas**: Replace single-point radial projection with estimated local tangent bundles (using PCA / Jacobian); or make $\eta$ a function of $t_k$ (small early, large late), aligning with observations that hallucinations primarily occur in middle timesteps.

## Related Work & Insights
- **vs CFG / PAG / SEG**: Traditional guidance performs scalar scaling on noise $\bm{\varepsilon}$ or perturbs self-attention; TAG performs geometric decomposition at the vector level, theoretically proving that "amplifying tangential but not radial" is locally optimal and orthogonal to previous methods.
- **vs Mode Interpolation (Aithal 2024)**: That work noted hallucinations arise from trajectories crossing "valleys" between low-density modes; TAG is designed directly based on this: tangential amplification moves further along the manifold to avoid falling into these valleys.
- **vs Tweedie-based score correction**: Traditionally, Tweedie estimates $\mathbb{E}[\bm{x}_0\mid\bm{x}_k]$ for posterior correction; TAG does not estimate $\bm{x}_0$, but uses the identity to argue that the radial component is already "accounted for" by the noise schedule—a novel application of the same identity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Tangential/radial decomposition + first-order monotonicity proof is a clear and novel angle in diffusion guidance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers SD1.5/2.1/XL/SD3 + uncond/cond + stacking with PAG/SEG + T2I-CompBench; reasonable scale, though lacks human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Smooth reasoning, clear transition from geometric intuition to theorems; math-heavy but supported by excellent figures.
- **Value**: ⭐⭐⭐⭐ No architecture change, no retraining, ~10 lines of code; very industry-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization](../../AAAI2026/hallucination/ground_what_you_see_hallucination-resistant_mllms_via_caption_feedback_diversity.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/hallucination/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ICML 2026\] Automatic Layer Selection for Hallucination Detection](automatic_layer_selection_for_hallucination_detection.md)

</div>

<!-- RELATED:END -->
