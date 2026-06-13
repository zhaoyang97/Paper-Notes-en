---
title: >-
  [Paper Note] Free Lunch for Stabilizing Rectified Flow Inversion
description: >-
  [ICLR 2026][Image Generation][Rectified Flow] This paper proposes PMI (Proximal-Mean Inversion) and mimic-CFG, two training-free methods that stabilize Rectified Flow inversion by applying proximal gradient correction to…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Rectified Flow"
  - "Inversion Stability"
  - "Proximal-Mean Inversion"
  - "Image Editing"
  - "Velocity Field Correction"
date: 2026-05-08
content_hash: 94c8bff1ab6afb29
---

# Free Lunch for Stabilizing Rectified Flow Inversion

**Conference**: ICLR 2026
**arXiv**: [2602.11850](https://arxiv.org/abs/2602.11850)  
**Code**: None  
**Area**: Diffusion Models / Image Editing
**Keywords**: Rectified Flow, Inversion Stability, Proximal-Mean Inversion, Image Editing, Velocity Field Correction

## TL;DR
This paper proposes PMI (Proximal-Mean Inversion) and mimic-CFG, two training-free methods that stabilize Rectified Flow inversion by applying proximal gradient correction toward the historical mean of the velocity field. On PIE-Bench, both methods achieve state-of-the-art reconstruction and editing quality with fewer NFEs.

## Background & Motivation

**Background**: Rectified Flow (RF) models (e.g., FLUX, Wan) have emerged as strong alternatives to conventional diffusion models. Their approximately constant velocity fields enable faster and more stable sampling, and their training-free inversion capability supports downstream tasks such as reconstruction and editing.

**Limitations of Prior Work**: Approximation errors during inversion inevitably accumulate across timesteps. Theory shows that ODE mappings in high-dimensional spaces are intrinsically unstable—the probability that the geometric mean instability coefficient exceeds 1 approaches 1 as dimensionality grows—so small perturbations in latent space can cause large reconstruction errors. Existing methods such as RF-Solver and FireFlow mitigate this by increasing the number of steps/NFEs, at high computational cost.

**Key Challenge**: Higher inversion accuracy requires more steps (and thus more computation), yet practical applications demand efficiency with fewer steps. Perturbations in the velocity field are amplified by instability during inversion, yet eliminating perturbations entirely is infeasible.

**Goal**: To stabilize the velocity field during inversion without increasing NFEs.

**Key Insight**: The RF training objective yields an approximately constant velocity field; therefore, a sliding mean of historical velocities can serve as a stable reference direction. Proximal optimization pulls the current velocity toward this mean while constraining the correction step to lie within a spherically Gaussian region derived from theoretical analysis.

**Core Idea**: Stabilize RF inversion via proximal gradient correction toward the historical mean of the velocity field; during editing, apply mimic-CFG—a velocity projection–interpolation scheme—to balance editing strength and structural preservation.

## Method

### Overall Architecture
Two complementary methods: (1) PMI applied during the inversion phase ($t_0 \to t_N$), stabilizing the velocity field via proximal gradient correction; (2) mimic-CFG applied during the editing/reconstruction phase ($t_N \to t_0$), balancing editing fidelity and structural consistency via velocity projection interpolation. Both are zero-cost (no additional NFEs) and can be plugged into any RF model.

### Key Designs

1. **Proximal-Mean Inversion (PMI)**:

    - **Function**: At each inversion step, apply constrained gradient correction to pull the predicted velocity toward the historical weighted mean.
    - **Mechanism**: Define the weighted mean $\bar{\mathbf{v}}_{t_k} = \frac{1}{t_{k+1}-t_0}\sum_{i=0}^{k}(t_{i+1}-t_i)\mathbf{v}_{t_i}$, construct the proximal objective $F(\mathbf{v}) = \|\mathbf{v} - \mathbf{v}_{t_{k-1}}\|_1 + \frac{1}{2\lambda}\|\mathbf{v} - \bar{\mathbf{v}}_{t_k}\|_2^2$, and obtain a closed-form update via first-order Taylor approximation: $\hat{\mathbf{v}}_{t_k} = \mathbf{v}_{t_k} - r_{t_k}\frac{\nabla F(\mathbf{v}_{t_k})}{\|\nabla F(\mathbf{v}_{t_k})\|_2}$.
    - **Design Motivation**: The proximal objective simultaneously enforces local consistency (the $L_1$ term keeps the update close to the previous step) and global consistency (the $L_2$ term pulls the update toward the mean). The correction radius $r_i$ is derived from instability theory in Proposition 1 to ensure the corrected velocity lies within a high-density region.

2. **Theoretical Derivation of the Correction Radius (Stability Condition)**:

    - **Function**: Derive the maximum safe correction radius at each step.
    - **Mechanism**: Based on concentration inequalities for high-dimensional Gaussian distributions: $r_i = \sqrt{2n + 3\sqrt{2n}} \cdot \frac{\Delta t_i}{T} + \epsilon$, where $n$ is the latent space dimensionality.
    - **Design Motivation**: An overly large correction deviates from the data manifold (entering low-density regions), while an overly small correction fails to stabilize effectively. The theoretical derivation defines the safe operating range.

3. **mimic-CFG Editing**:

    - **Function**: During the editing/reconstruction phase, interpolate the current velocity via projection onto the historical mean direction.
    - **Mechanism**: $\bar{\mathbf{v}}_{t_k}^{\text{proj}} = \frac{\mathbf{v}_{t_k}^\top \bar{\mathbf{v}}_{t_k}^{\text{edit}}}{\|\bar{\mathbf{v}}_{t_k}^{\text{edit}}\|_2^2}\bar{\mathbf{v}}_{t_k}^{\text{edit}}$, then $\hat{\mathbf{v}}_{t_k} = (1-w)\bar{\mathbf{v}}_{t_k}^{\text{proj}} + w \cdot \mathbf{v}_{t_k}$, where $w$ controls editing strength.
    - **Design Motivation**: The name "mimic-CFG" reflects its structural analogy to classifier-free guidance—the projected component resembles the "unconditional" direction (structural preservation) and the original velocity resembles the "conditional" direction (editing effect), with interpolation controlling the trade-off.

### Loss & Training
- Training-free; velocity fields are modified only at inference time.
- PMI: proximal gradient update with no additional NFEs.
- mimic-CFG: velocity projection + linear interpolation with no additional NFEs.

## Key Experimental Results

### Main Results
PIE-Bench (700 editing tasks) on the FLUX model:

| Method | PSNR↑ | LPIPS↓ | Structure Distance↓ | CLIP Similarity↑ | NFE |
|--------|-------|--------|---------------------|------------------|-----|
| DDIM Inversion | Baseline | Baseline | Baseline | Baseline | N |
| RF-Solver | Good | Good | Good | Good | 2N |
| FireFlow | Good | Good | Good | Good | 2N |
| Baseline + PMI | **Better** | **Better** | **Better** | **Better** | N |
| Baseline + PMI + mimic-CFG | **SOTA** | **SOTA** | **SOTA** | **SOTA** | N |

### Ablation Study

| Configuration | PSNR↑ | Notes |
|---------------|-------|-------|
| No correction | Baseline | Original inversion |
| PMI ($L_1$ norm) | **Best** | Full method |
| PMI ($L_2$ norm) | Second best | $L_2$ inferior to $L_1$ |
| PMI ($L_\infty$ norm) | Moderate | Sparse correction |
| Prompt-free reconstruction | Validates inversion quality | Eliminates prompt confounds |

### Key Findings
- PMI adds no NFEs yet significantly improves PSNR (+2–3 dB), justifying the "free lunch" claim.
- The $L_1$ norm performs best in the proximal objective, likely due to its moderately sparse correction behavior.
- Evaluating inversion quality under prompt-free conditions is a methodological contribution that eliminates confounding effects of prompt alignment on reconstruction metrics.
- The mimic-CFG weight $w$ intuitively controls editing strength: larger $w$ favors editing, smaller $w$ favors structural preservation.

## Highlights & Insights
- **The "free lunch" is genuinely free**: PMI requires only a cumulative velocity mean accumulator and a single closed-form update, with zero additional network calls—a truly zero-cost quality improvement.
- **Theory-driven correction radius**: Rather than arbitrarily selecting hyperparameters, the safe correction range is derived from instability theory, grounding practice in principled analysis.
- **Elegant analogy of mimic-CFG**: The velocity projection + interpolation scheme is cast as an analogue of CFG's unconditional/conditional control, yielding an intuitive and easily tunable mechanism. Projection-based interpolation outperforms direct interpolation (experimentally confirmed).
- **Prompt-free evaluation**: The paper introduces a methodology for assessing inversion quality without conditioning prompts, enabling a purer measure of inversion stability.

## Limitations & Future Work
- Hyperparameters such as $\lambda$ and $\epsilon$ require tuning, despite reasonable defaults being provided.
- The theoretical assumptions rely on the velocity field being approximately constant, which may not hold for insufficiently trained or architecturally atypical RF models.
- The mimic-CFG weight $w$ may need to be adjusted for different editing types.
- Validation is limited to image editing; video editing, 3D generation, and other modalities remain unexplored.

## Related Work & Insights
- **vs. RF-Solver / FireFlow**: These methods improve accuracy via higher-order Taylor expansions or two-step iterations at the cost of increased NFEs. PMI adds no NFEs and can be combined with these methods for further gains.
- **vs. Direct Inversion**: Direct Inversion separates source and target diffusion to preserve content; mimic-CFG achieves a similar effect through velocity projection interpolation with lower overhead.
- **vs. FlowEdit**: FlowEdit constructs a direct source-to-target ODE, bypassing inversion entirely. PMI + mimic-CFG retains the flexibility of the inversion paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ — Proximal optimization for velocity field stabilization is a novel idea; the mimic-CFG analogy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive PIE-Bench evaluation, multi-baseline stacking, and prompt-free assessment.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations and intuitive explanations are well integrated; algorithmic pseudocode is clear.
- Value: ⭐⭐⭐⭐⭐ — A genuinely free lunch method with direct practical value for RF inversion and editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)

</div>

<!-- RELATED:END -->
