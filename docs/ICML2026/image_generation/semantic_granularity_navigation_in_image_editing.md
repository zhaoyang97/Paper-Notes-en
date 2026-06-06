---
title: >-
  [Paper Note] Semantic Granularity Navigation in Image Editing
description: >-
  [ICML 2026][Image Generation][Real image editing] NaviEdit decouples the implicit coupling of "model scale coordinate = editing progress clock" in diffusion/flow editors. Under a fixed step budget…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Real image editing"
  - "flow matching"
  - "training-free inference controller"
  - "scale-progress decoupling"
  - "semantic granularity"
date: 2026-05-08
content_hash: ffad910422e618fe
---

# Semantic Granularity Navigation in Image Editing

**Conference**: ICML 2026  
**arXiv**: [2605.21190](https://arxiv.org/abs/2605.21190)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Real image editing, flow matching, training-free inference controller, scale-progress decoupling, semantic granularity

## TL;DR
NaviEdit decouples the implicit coupling of "model scale coordinate = editing progress clock" in diffusion/flow editors. Under a fixed step budget, it utilizes a training-free inference controller to concentrate computational power on the density of an effective scale window rather than expanding the range into high-noise regions, thereby improving background fidelity and semantic consistency across PIE-Bench, ImgEdit-Bench, and various flow backbones simultaneously.

## Background & Motivation

**Background**: T2I models based on diffusion or flow matching (SD3, FLUX, Stable Diffusion series) are widely used as general visual priors. Combined with training-free editing pipelines such as SDEdit, Prompt-to-Prompt, and FlowEdit, they can perform real image editing by modulating the sampling process during inference. This allows the model to modify the source image according to a target prompt without requiring additional training.

**Limitations of Prior Work**: The tension between editability and fidelity remains unresolved. To achieve thorough semantic changes, trajectories are often pushed into higher noise regions (e.g., FlowEdit pulling the anchor to larger noise levels), resulting in drift in non-edited areas, hallucinated objects, or color explosions. Conversely, attempts to preserve structure often fail to modify geometric shapes (e.g., transforming a round cake into a square one). Most existing works (TiNO-Edit, Schedule Your Edit, Dual-Schedule Inversion) focus on the shape of the schedule but remain trapped within the framework where "scale equals progress."

**Key Challenge**: These methods express **two fundamentally different concepts** using the same coordinate: *scale* (which determines the current "editable information domain," from coarse structure to fine texture) and *progress* (which determines the accumulated semantic modification). Through probing experiments, the authors identify a three-stage regime on the scale axis: at high scales, the prompt-conditioned differential field $\Delta V(u)$ diverges spatially and leakage pressure $\rho(u)$ spikes; at low scales, high-frequency reconstruction dominates, making geometry difficult to modify. A "sweet spot" at the valley of $\rho(u)$ exists between these stages, which is both semantically sensitive and spatially anchored. Using scale as progress implies that increasing editing intensity forces the quality integral into high-risk tails, incurring an unavoidable risk floor.

**Goal**: Under the hard constraints of a training-free approach, no modifications to the backbone, and a fixed model-call budget $K$, the objective is to liberate "scale expansion" from being the sole means of strengthening editing and instead redistribute computational power toward the **density** within a fixed effective window.

**Key Insight**: Editing is viewed as a controlled integration of latents on an explicit progress axis $s \in [0, 1]$, where the scale coordinate $u(s)$ is relegated to a controllable "measurement and actuation" input. The evaluation target is the entire rollout (not individual timesteps), characterized by a rollout-level functional termed *semantic granularity*.

**Core Idea**: Decouple progress and scale at the rollout level, enforce a self-consistency contract where mixing, querying, and updating use the same $u_k$ at each step, and utilize the fixed budget to increase density within the effective scale window rather than expanding the range toward the high-noise tail.

## Method

### Overall Architecture
The input consists of the source image latent $x_{\text{src}}$, source prompt $c_{\text{src}}$, target prompt $c_{\text{tar}}$, a fixed step budget $K$, and a frozen flow model (compatible with any base editor like FlowEdit, InfEdit, or FlowAlign). NaviEdit acts as a rollout-level controller that entirely replaces the "budget $\rightarrow$ range" scheduling rules of the original editor. It selects a fixed tail window $\mathcal{U}_{\text{eff}}$ on the scheduler path (anchored by a reference depth $t_{\text{ref}}$, excluding extreme high-noise tails). $K$ sampling points $\{u_k\}$ and corresponding increments $\{\Delta u_k\}$ are determined within this window via a monotonic coordinate $p \in [0, 1]$. At each step: co-located anchor pairs $(z^{\text{src}}, z^{\text{tar}})$ are constructed using the same $u_k$; the model is queried at $t = \tau(u_k)$ to obtain the differential velocity $\Delta V$ (optionally passed through an internal feasible-region gate $M(u_k)$ to yield $\Delta V_{\text{eff}}$); and a first-order Euler update $x_{k+1} = x_k + \Delta u_k \Delta V_{\text{eff}}$ is performed. The entire process requires no training, no inversion, and no external masks.

### Key Designs

1.  **Progress-Granularity Decoupled Controlled Integration**:
    - **Function**: Explicitly models editing as $\frac{dx}{ds} = \frac{du}{ds} \Delta V_{\text{eff}}(x(s); u(s), \epsilon(s))$ and defines a rollout-level semantic granularity functional $\mathcal{G}[x(\cdot), u(\cdot)] = \int_0^1 \phi(x(s), u(s)) \, ds$ as the quality evaluation target.
    - **Mechanism**: Treats scale $u$ as a controlled input rather than progress itself. $\phi(x, u)$ is a non-negative local risk density, assumed to increase with leakage pressure $\rho(u)$ and directional oscillation $\omega(u)$. These probes are obtained by adding fresh noise to the source latent to get $z^{\text{src}} = (1-u)x_{\text{src}} + u\epsilon$, then constructing the target anchor $z^{\text{tar}} = x + (z^{\text{src}} - x_{\text{src}})$. The differential field is given by $\Delta V(u) = v_\theta(z^{\text{tar}}, \tau(u), c_{\text{tar}}) - v_\theta(z^{\text{src}}, \tau(u), c_{\text{src}})$, and $\rho(u) = \|(1-M(u)) \odot \Delta V(u)\|_2 / \|\Delta V(u)\|_2$ measures the energy leaked into non-edited regions.
    - **Design Motivation**: Existing methods only evaluate quality at the final state, failing to see how budget is spent across scales. The rollout functional turns "which $u$ to prioritize" into an optimizable compute allocation problem and leads to Theorem 4.2 (coupled scheduling inevitably results in outside-window progress quality $m_{\text{bad}}$, leading to an irreducible lower bound for $\mathcal{G}$).

2.  **Density-over-range Budget Reallocation**:
    - **Function**: Fixes a tail window $\mathcal{U}_{\text{eff}}$ on the scheduler path and uses the entire step budget to increase density within that window instead of expanding into high-noise regions.
    - **Mechanism**: Parametrizes a monotonic traversal over $\mathcal{U}_{\text{eff}}$ with $p \in [0, 1]$, where $\{p_k\}$ determines $\{u_k\}$ and $\{\Delta u_k\}$. Online density adjustment can optionally be performed using discrete proxies of $\rho$ and $\omega$ calculated during editing without additional model calls. Theorem 4.3 demonstrates that when $K > L_\phi C_E / \gamma$, increasing density within the window is superior to expanding the range into $\mathcal{U}_{\text{bad}}$: the former only incurs a first-order Euler discretization error $C_E/K$, while the latter incurs a constant risk floor $c_{\text{bad}}\delta_K - c_{\text{good}} \geq \gamma$.
    - **Design Motivation**: Experiments show that increasing the number of steps in coupled scheduling can actually degrade background quality (increasing CLIP but decreasing PSNR/SSIM) because increased budget is automatically translated into range expansion. Freezing the "range" and making "density" the sole adjustable dimension corrects the budget-to-quality direction.

3.  **Self-consistency Contract (First-order Consistent Discretization)**:
    - **Function**: Forces the three operations of each step—mixing (anchor construction), querying ($\tau(u_k)$ input to the model), and update (step size $\Delta u_k$)—to use the **same** $u_k$.
    - **Mechanism**: Theorem 4.4 states that if inconsistent scales are used, the differential velocity measures a system different from the one being actuated, accumulating systematic bias that manifests as drift and artifacts. Conversely, $x_{k+1} = x_k + \Delta u_k \Delta V_{\text{eff}}(x_k; u_k, \epsilon_k)$ represents a first-order consistent discretization of Def. 4.1 within the effective window.
    - **Design Motivation**: Many rescheduling works (e.g., SYE, Dual-Schedule Inversion) adjust "shape" without ensuring axis consistency. Axis-mismatch ablations (independently perturbing query/step/mix scales) show that once mismatched, drift and compliance metrics degrade monotonically with $|\delta|$, proving that decoupling requires this contract for theoretical and empirical validity.

### Loss & Training
Completely training-free with no parameter updates. Inference requires only a few hyperparameters, such as $K=50$ (PIE-Bench) or $K=28$ (cross-backbone ablation) and $t_{\text{ref}}=42$. The optional feasible-region gate $M(u)$ is generated from internal signals already exposed by the base editor, introducing no extra model evaluation. It runs on a single RTX 3090.

## Key Experimental Results

### Main Results

Comparison on PIE-Bench (700 real images with GT masks) across various paradigms (fixed schedule / rescheduling / Navi controller):

| Category | Method | Struct.Dist↓ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-Whole↑ | CLIP-Edited↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fixed | FlowEdit (SD3) | 14.64 | 22.46 | 84.08 | 103.00 | 25.91 | 22.50 |
| Fixed | FlowAlign (SD3) | 6.21 | 27.78 | 92.41 | 34.47 | 25.44 | 21.80 |
| Reschedule | SYE (DDIM+PnP) | 27.17 | 21.73 | 87.45 | 110.64 | 24.44 | 21.26 |
| Reschedule | TurboEdit (SDXL-Turbo) | 13.80 | 21.44 | 80.08 | 108.60 | 24.66 | 21.70 |
| Navi | Navi-FlowEdit ($M\equiv 1$) | 14.25 | 22.54 | 89.36 | 92.47 | **26.01** | **22.59** |
| Navi | **Navi-FlowEdit + gate** | 10.67 | 27.94 | **93.85** | 48.74 | **26.18** | **22.72** |
| Navi | **Navi-FlowAlign ($M\equiv 1$)** | **5.40** | **28.33** | 93.40 | **34.49** | 26.15 | 22.44 |

On ImgEdit-Bench (Basic + UGE protocols), the Navi-InfEdit and Navi-FlowAlign ungated variants outperform their respective baselines in average scores. The most significant Gains for FlowAlign are in background, action, and replace categories—scenarios where trajectory drift is most problematic.

### Ablation Study

Controlled comparison of coupling vs. decoupling across backbones (fixed $K=28$, same differential editing mechanism):

| Backbone | Schedule | SSIM↑ | PSNR↑ | CLIP-Whole↑ | CLIP-Edited↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SD3 | couple | 88.22 | 22.18 | 26.01 | 22.55 |
| SD3 | **decouple** | **93.22** | **27.81** | **26.15** | **22.67** |
| SD3.5 | couple | 85.68 | 22.01 | 26.57 | 22.91 |
| SD3.5 | **decouple** | **92.32** | **27.45** | **26.77** | **23.32** |
| FLUX.1 [dev] | couple | 82.14 | 21.81 | 27.02 | 23.35 |
| FLUX.1 [dev] | **decouple** | **91.75** | **26.83** | **27.06** | **23.42** |

The two rows of Navi-FlowEdit (with/without gate) on PIE-Bench isolate the gate's effect: the gate primarily improves background preservation (SSIM 89.36 $\rightarrow$ 93.85), while the core controller alone already outperforms FlowEdit in CLIP and Struct.Dist.

### Key Findings
- **Density Wins over Range**: Figure 6 shows the rollout proxy $\widehat{\mathcal{G}}$ is nearly linearly correlated with $m_{\text{bad}}$, and PSNR-bg decreases monotonically as $m_{\text{bad}}$ increases—confirming that the risk-floor argument in Theorem 4.3 is not merely theoretical.
- **CFG Cannot Save Coupling**: Figure 11 shows that increasing classifier-free guidance cannot consistently replicate the gains of decoupling. CFG modifies only the magnitude of the velocity field, not the budget allocation along the scale axis, leaving it powerless against the cost floor of coupled scheduling.
- **Portability Across Base Editors**: Applying the same controller to FlowEdit, InfEdit, and FlowAlign yields positive results for each, suggesting that progress-scale decoupling is a universal principle rather than a pipeline-specific trick.

## Highlights & Insights
- **Diagnosis-Driven Methodology**: Using $\rho(u)$ and $\omega(u)$ probes to map the scale axis into regimes before designing the controller around the "valley" avoids arbitrary schedule tuning. This paradigm of "spatial diagnosis followed by compute allocation" can be transferred to any inference-time control problem that uses a single coordinate for multiple meanings.
- **Rollout-Level Perspective**: Elevating editing quality from "instantaneous direction at a timestep" to a "compute allocation functional along the scale" transforms schedule tuning from empirical trial-and-error into a measure allocation optimization problem, backed by the existence and inequality guarantees of Theorems 4.2 and 4.3.
- **Transfer Potential of the Consistency Contract**: Any inference-time method using differential velocity (video editing, 3D editing, controllable generation) may suffer from axis mismatch. Using "shared coordinates for mix/query/update" as a condition for valid discretization serves as a universal sanity check.

## Limitations & Future Work
- The controller only adjusts how compute is allocated along the scale; it does not improve support estimation or scene reasoning. If the editable support is too conservative or the effective window is too restrictive for drastic replacements, results may remain in a "partial" state, particularly with gated variants.
- Lack of explicit geometric or relational consistency constraints means that while local drift is suppressed, global consistency in scenes with mirrors, repeated objects, or strong inter-object relationships may still fail.
- The base editor must expose a conditional differential field and a monotone scale path. For strong editors pre-trained end-to-end (which learn some trade-offs during training), the optimization space is reduced.
- Limitations in experimental dimensions: image resolution, prompt complexity (multi-object, long prompts), and user study scale require further stress testing.

## Related Work & Insights
- **vs FlowEdit (Kulikov et al., 2025)**: Both are inversion-free, training-free, and run on flow, but FlowEdit follows the typical coupled route of "expanding scale range to strengthen editing." NaviEdit serves as a direct superior alternative across all metrics.
- **vs Schedule Your Edit / Tino-Edit / Dual-Schedule Inversion**: These rescheduling works recognize the importance of scale allocation but keep progress implicitly tied to scale without an axis consistency contract, leading to inconsistent gains. NaviEdit provides the theoretical and empirical answer to why decoupling and contracts are necessary.
- **vs Prompt-to-Prompt / MasaCtrl / PnP**: These work at the update rule level via attention/feature intervention and are orthogonal to NaviEdit. NaviEdit can be layered on top of these baselines as it only rearranges steps along the scale axis.

## Rating
- Novelty: ⭐⭐⭐⭐ Decoupling progress from scale is a clear and original perspective, supported by risk-floor existence theorems rather than just empirical tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage includes 2 benchmarks, 3 base editors, and 3 flow backbones, with comprehensive ablations on axis mismatch and density-vs-range.
- Writing Quality: ⭐⭐⭐⭐ The logical chain from diagnosis to formalization to theorems and algorithms is very smooth; theorems may be slightly dense for engineering-focused readers.
- Value: ⭐⭐⭐⭐ As a plug-and-play inference-time controller, it provides immediate benefits to all training-free editing pipelines based on differential fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Next Visual Granularity Generation](../../ICLR2026/image_generation/next_visual_granularity_generation.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICCV 2025\] Describe, Don't Dictate: Semantic Image Editing with Natural Language Intent](../../ICCV2025/image_generation/describe_dont_dictate_semantic_image_editing_with_natural_language_intent.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Next Visual Granularity Generation](../../ICLR2026/image_generation/next_visual_granularity_generation.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[ICML 2026\] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation](wise_a_world_knowledge-informed_semantic_evaluation_for_text-to-image_generation.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] DirectEdit: Step-Level Accurate Inversion for Flow-Based Image Editing](directedit_step-level_accurate_inversion_for_flow-based_image_editing.md)

</div>

<!-- RELATED:END -->
