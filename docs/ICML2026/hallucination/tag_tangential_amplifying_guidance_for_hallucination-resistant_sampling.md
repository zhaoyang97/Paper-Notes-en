---
title: >-
  [Paper Note] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling
description: >-
  [ICML 2026][Hallucination Detection][Paper Note] TAG decomposes each diffusion update step into "radial + tangential" components along the direction of the current latent variable. By applying an amplification factor $\eta \ge 1$ solely to the tangential component, it is proved via first-order Taylor expansion that this is equivalent to monotonically increasing the l
tags:
  - ICML 2026
  - Hallucination Detection
date: 2026-05-08
content_hash: 8b3bf82de6a81f20
---
# TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling

**Conference**: ICML 2026  
**arXiv**: [2510.04533](https://arxiv.org/abs/2510.04533)  
**Code**: Available (provided as a Project Page in the paper)  
**Area**: Hallucination Detection  
**Keywords**: Tangential Amplification, Inference-time Guidance, Geometry-aware Sampling, Hallucination Suppression, CFG Enhancement

## TL;DR
TAG decomposes each diffusion update step into "radial + tangential" components along the direction of the current latent variable. By applying an amplification factor $\eta \ge 1$ solely to the tangential component, it is proved via first-order Taylor expansion that this is equivalent to monotonically increasing the log-likelihood gain. This pulls samples toward high-density regions of the data manifold, mitigating semantic hallucinations in diffusion models with almost zero extra computational cost.

## Background & Motivation

**Background**: Diffusion models have achieved SOTA in image generation but often suffer from "hallucinations"—extra fingers, mismatched objects, and structural violations of the prompt. Mainstream remedies involve inference-time guidance (CFG and its variants like PAG, SEG, CFG++, etc.), which push samples away from low-density regions by scalar scaling of conditional/unconditional residuals or minor network modifications.

**Limitations of Prior Work**: Existing guidance methods are inherently "geometry-agnostic"—they apply a simple scalar multiplication to the cond–uncond residual without considering the local directional structure of the data manifold at the current noise level. Excessive scaling can disturb ODE/SDE solvers, compress diversity, or introduce new artifacts.

**Key Challenge**: Diffusion sampling trajectories must simultaneously follow the noise schedule in the "radial" direction (radius must contract per the schedule) and move along the data manifold in the "tangential" direction (refining sample details toward the true distribution). A single scalar amplifying both directions disrupts the radial schedule, leading to over-smoothing, while failing to distinguish between them misses the opportunity to ascend the manifold.

**Goal**: Design a guidance method that requires no architecture changes, no retraining, and is plug-and-play with any off-the-shelf diffusion backbone. It should utilize only the geometric signals available from the trajectory itself to directionally amplify "directions truly useful for sample quality."

**Key Insight**: Starting from the Tweedie identity and the Gaussian annulus theorem, the authors argue that at noise level $t_k$, data is concentrated near a thin spherical shell $\mathcal{M}_k$. Orthogonal decomposition of each step update $\Delta_{k+1}$ on $\mathcal{M}_{k+1}$ reveals that the radial component is the "radius reduction" prescribed by the noise schedule, while the tangential component carries the semantic structure (Figure 2 visualization shows the tangential part corresponds to clear semantic maps, while the radial part appears as noise).

**Core Idea**: Keep the radial component fixed and amplify only the tangential component by a factor $\eta \ge 1$. This allows the single-step update to move further along the data manifold toward higher-density regions without altering the noise schedule.

## Method

### Overall Architecture
TAG aims to resolve "semantic hallucinations"—extra fingers, mismatched objects, and prompt-violating structures—during diffusion sampling. Instead of modifying the network or retraining, it treats the update calculated by the solver at each step as a vector and performs orthogonal decomposition along the spherical shell where the current sample resides, amplifying only the semantic-bearing component. Specifically, it wraps around standard base solvers such as DDIM/EDM/Flow Matching. Given the state $\bm{x}_{k+1}$ at step $k+1$, it first calculates the original update $\Delta_{k+1} = \tilde{\bm{x}}_k - \bm{x}_{k+1}$ using the backbone $\epsilon_\theta$, then splits this update into "radial + tangential" using the sample's own normalized direction. After amplifying the tangential component and retaining the radial one, the step is executed. The entire process requires no extra denoiser forward passes, changes no parameters, and can be directly overlaid with existing guidance like CFG/PAG/SEG/SSG.

### Key Designs

**1. Tangential Amplification Update Rule: Move further in semantic directions, keep noise schedule fixed**

Diffusion trajectories essentially perform two tasks of different natures: the radial direction is the "time-dependent radius contraction" defined by the noise schedule, while the tangential direction is where the model exercises its freedom to refine details onto the data manifold. Existing guidance applies scalar scaling to the entire residual, disturbing both directions and the noise schedule simultaneously. The key to TAG is treating them separately: using the current sample itself as the projection basis $\hat{\bm{x}}_{k+1} = \bm{x}_{k+1}/\|\bm{x}_{k+1}\|_2$, it defines the radial projection $\bm{P}_{\mathcal{M}_{k+1}} = \hat{\bm{x}}_{k+1}\hat{\bm{x}}_{k+1}^\top$ and the tangential projection $\bm{P}^\perp_{\mathcal{M}_{k+1}} = I - \bm{P}_{\mathcal{M}_{k+1}}$. It then multiplies only the tangential component by a factor $\eta \ge 1$:

$$\Delta^{\text{TAG}}_{k+1} = (\bm{P}_{\mathcal{M}_{k+1}} + \eta\,\bm{P}^\perp_{\mathcal{M}_{k+1}})\Delta_{k+1},\qquad \bm{x}_k = \bm{x}_{k+1} + \Delta^{\text{TAG}}_{k+1}.$$

Substituting this increment back into DDIM coefficients is equivalent to applying the same amplification only to the tangential component of the predicted noise $\epsilon_\theta$, thus it can be understood as "geometry-aware rescaling of predicted noise." Tangential amplification is chosen because visualization in Figure 2 shows the radial part resembles unstructured noise, whereas the tangential part is a clear semantic map—semantic information is naturally concentrated tangentially.

**2. First-order Taylor Gain Monotonicity Proof: Explaining why tangential amplification preserves the trajectory**

Beyond the "tangential looks like semantics" intuition, the authors address why amplifying it does not destroy the sample like blindly increasing CFG. The approach formulates inference-time guidance as a local log-likelihood maximization problem under step length constraints: $\max (\bm{x}_k - \bm{x}_{k+1})^\top \nabla_{\bm{x}}\log p(\bm{x}\mid t_{k+1})|_{\bm{x}_{k+1}}$ subject to $\|\bm{x}_k - \bm{x}_{k+1}\|_2 \le \delta_k$. It defines the first-order Taylor gain $G(\eta) := (\Delta^{\text{TAG}}_{k+1})^\top \nabla_{\bm{x}}\log p$. The paper proves that $G(\eta)$ is monotonically increasing with respect to $\eta$ (Theorem 4.1), meaning anytime $\eta > 1$, local likelihood is strictly improved, pulling the sample toward high-density regions. Crucially, the radial component remains unchanged, meaning the "one-step calibration" provided by the Tweedie identity is preserved—$\langle \hat{\bm{x}}_{k+1}, \Delta^{\text{TAG}}_{k+1}\rangle = \langle \hat{\bm{x}}_{k+1}, \Delta_{k+1}\rangle$ (Eq. 22). The noise schedule's timeline is untouched; thus, tangential amplification ascends the likelihood monotonically without "bursting" the trajectory.

**3. Tangential Alignment on CFG ($\text{TAG}_{\text{cfg}}$): Pulling conditional/unconditional branches back to the same tangent**

The authors apply the same geometric perspective to CFG, arguing that the mismatch between conditional and unconditional predictions primarily occurs in the tangential subspace (as the radial component is jointly constrained by the noise schedule). Instead of using a scalar $\omega$ to scale the entire residual—which also amplifies inconsistent parts that shouldn't be—it is better to realign within the tangential space. Specifically, let the CFG residual be $\bm{g}_k = \bm{\varepsilon}_c - \bm{\varepsilon}_u$. Taking its tangential projection $\bm{g}_k^\perp = \bm{P}^\perp_{\mathcal{M}}(\bm{x}_k)\bm{g}_k$, the conditional prediction $\bm{\varepsilon}_c$ is projected onto this tangential subspace $\mathrm{span}(\bm{g}_k^\perp)$ to obtain an alignment vector:

$$\bm{g}_k^{\text{align}} = \frac{\langle\bm{\varepsilon}_c, \bm{g}_k^\perp\rangle}{\|\bm{g}_k^\perp\|_2^2}\,\bm{g}_k^\perp,\qquad \tilde{\bm{\varepsilon}}_k = \bm{\varepsilon}_u + \omega\,\bm{g}_k + \eta\,\bm{g}_k^{\text{align}}.$$

This way, both predictions "move forward together" in the tangential direction. This preserves the original CFG guidance strength $\omega$ while adding an alignment term $\eta$ acting only tangentially, avoiding artifacts introduced by indiscriminate scalar scaling.

### Loss & Training
Completely training-free and fine-tuning-free. The algorithm only inserts projection and vector re-weighting into the sampling loop (Algorithm 1, approximately 11 lines of code). The only hyperparameter is $\eta \ge 1$ (typical setting $\eta \approx 1.x$ for unconditional sampling; CFG version uses two scalars $\omega$ and $\eta$).

## Key Experimental Results

Primary backbones: SD v1.5 / v2.1 / XL / SD3, evaluated on COCO 2014; compositional evaluation via T2I-CompBench.

### Main Results

| Configuration | Backbone | FID ↓ | IS ↑ | AES ↑ | CMMD ↓ |
|---------------|----------|-------|------|-------|--------|
| Uncond. | SD v1.5 | 58.41 | 15.59 | 5.003 | 1.069 |
| Uncond. | SD v1.5 + **TAG** | **46.20** | **16.77** | **5.064** | **0.778** |
| Uncond. | SDXL | 119.14 | 9.08 | 5.645 | 2.474 |
| Uncond. | SDXL + **TAG** | **90.71** | 8.91 | 5.577 | **2.201** |
| Uncond. | SD3 | 84.26 | 11.53 | 5.261 | 1.671 |
| Uncond. | SD3 + **TAG** | **79.11** | **11.73** | **5.365** | **1.564** |

| Configuration | Backbone | FID ↓ | ImageReward ↑ | CLIP ↑ |
|---------------|----------|-------|---------------|--------|
| Cond. (T2I) | SD v1.5 | 33.49 | −0.342 | 25.00 |
| Cond. (T2I) | SD v1.5 + **TAG** | **26.61** | **−0.339** | **25.09** |
| Cond. (T2I) | SD v2.1 | 26.12 | 0.143 | 25.35 |
| Cond. (T2I) | SD v2.1 + **TAG** | **21.59** | **0.424** | **26.16** |
| Cond. (T2I) | SD3 | 29.02 | 1.030 | 26.39 |
| Cond. (T2I) | SD3 + **TAG** | **27.54** | **1.043** | **26.56** |

### Ablation Study

| Config (SD v1.5, Uncond.) | FID ↓ | IS ↑ | CMMD ↓ | Description |
|---------------------------|-------|------|--------|-------------|
| No guidance | 58.41 | 15.59 | 1.069 | Base sampling |
| TAG | 46.20 | 16.77 | 0.778 | TAG only |
| PAG | 53.72 | 21.13 | 0.723 | Existing SOTA guidance |
| **TAG + PAG** | **52.61** | **21.20** | **0.701** | Combined improvement |
| SEG | 47.69 | 18.50 | 0.835 | Other existing guidance |
| **TAG + SEG** | **42.71** | **19.45** | **0.746** | Combined improvement |

On T2I-CompBench Spatial / Complex subsets, TAG improved SDXL's 2DSpatial from 0.1857 → 0.1980, BLIP-VQA from 0.4443 → 0.4650, and ImageReward from 0.2596 → 0.3978, indicating significant improvements in structural hallucinations (e.g., three legs).

### Key Findings
- **Radial amplification destroys samples**: Figure 5 shows that amplifying the normal (radial) component as well ("+TAG + Normal") leads to severe over-smoothing—explainable by the $\kappa$-fold radial contraction in Eq. 21, validating the necessity of "tangential only" amplification.
- **50-NFE TAG outperforms 250-NFE baseline**: Using the same backbone, TAG brings sample quality at 50 steps close to that of the original 250-step sampling, suggesting it corrects the sampling trajectory itself rather than relying on more computation.
- **Truly Plug-and-Play**: Consistently improves when stacked with PAG / SEG / custom CFG with no observed conflicts; the single added hyperparameter $\eta$ is stable.

## Highlights & Insights
- **Elevating "Geometry-Awareness" from Heuristic to Theory**: While PAG/SEG/CFG++ mostly select "which directions not to amplify" empirically, TAG provides proof via Tweedie + Gaussian annulus for "why tangential amplification monotonically increases first-order likelihood," shifting diffusion guidance toward geometric optimization.
- **Clever use of "sample as its own projection basis"**: Requires no external classifiers or extra network passes; it partitions high-dimensional space into "schedule-prescribed directions" and "model-freedom directions" using only $\hat{\bm{x}}_{k+1}$. This trick of using the normalized vector as a geometric frame could transfer to other iterative algorithms with spherical shell priors (LLM logits, Flow Matching, Schrödinger Bridges).
- **Hypothesis of "Tangential Inconsistency" in CFG**: Attributing the mismatch between conditional and unconditional branches to the tangential subspace is insightful. If this holds, future work could train specialized "tangential calibration networks" rather than scaling whole-vector residuals.

## Limitations & Future Work
- Authors' acknowledged limitations: The theory relies on the local $C^2$ smoothness of $\log p$ and first-order Taylor expansion; when $\eta$ is too large, high-order terms become non-negligible, leading to visible artifacts. The CFG variant introduces $\eta$ as an extra hyperparameter to tune.
- Observed limitations: Taking $\hat{\bm{x}}_{k+1}$ as the projection basis assumes the noise distribution is strictly isotropic and the data manifold forms concentric shells. For non-Euclidean latent spaces (e.g., spherical flow matching, SE(3) diffusion), the projection requires redefinition. The slight drop in IS for SDXL (9.08→8.91) in Table 1 suggests FID/CMMD gains may come at a slight cost to diversity of detail.
- Improvements: Replace single-point radial projection with estimated local tangent bundles (using PCA / Jacobian); or make $\eta$ a function of $t_k$ (small early, large late), aligning with Aithal's observation that hallucinations mostly occur in middle timesteps.

## Related Work & Insights
- **vs CFG / PAG / SEG**: Traditional guidance applies scalar scaling directly to predicted noise $\bm{\varepsilon}$ or perturbs self-attention. TAG performs geometric decomposition at the vector level, proving "tangential amplification without radial" is locally optimal and orthogonal to these methods.
- **vs Mode Interpolation (Aithal 2024)**: That work noted hallucinations stem from trajectories passing through "valleys" between low-density modes. TAG is designed directly on this observation: tangential amplification moves further along the data manifold to avoid falling into valleys.
- **vs Tweedie-based score correction**: Typically, Tweedie is used to estimate $\mathbb{E}[\bm{x}_0\mid\bm{x}_k]$ for posterior correction. TAG does not estimate $\bm{x}_0$ but uses the Tweedie identity to argue that the radial component is already "accounted for" by the noise schedule—a novel application of the identity.

## Rating
- Novelty: ⭐⭐⭐⭐ Tangential/radial decomposition combined with first-order monotonicity proof is a clear, novel angle in diffusion guidance.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers SD1.5/2.1/XL/SD3 + uncond/cond + stacking with PAG/SEG + T2I-CompBench; reasonable scale, though lacks human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Smooth reasoning from geometric intuition to theorems; math-heavy but supported by excellent visualizations.
- Value: ⭐⭐⭐⭐ No architecture changes, no retraining, ~10 lines of code; very friendly for industrial deployment.

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
