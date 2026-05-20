---
title: >-
  [Paper Note] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models
description: >-
  [ICCV 2025][Image Generation][Image Editing] FlowEdit proposes an inversion-free, optimization-free, model-agnostic text-based image editing method that constructs an ODE path directly between the source and target distr…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Editing"
  - "Flow Models"
  - "Rectified Flow"
  - "Inversion-Free Editing"
  - "FLUX"
date: 2026-05-08
content_hash: dd3462116b63fefe
---

# FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models

**Conference**: ICCV 2025
**arXiv**: [2412.08629](https://arxiv.org/abs/2412.08629)  
**Code**: [https://github.com/fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit)  
**Area**: Image Editing / Flow Models
**Keywords**: Image Editing, Flow Models, Rectified Flow, Inversion-Free Editing, FLUX

## TL;DR

FlowEdit proposes an inversion-free, optimization-free, model-agnostic text-based image editing method that constructs an ODE path directly between the source and target distributions of a pre-trained flow model, achieving structure-preserving editing with lower transport cost than inversion-based approaches.

## Background & Motivation

Editing real images with pre-trained text-to-image (T2I) diffusion/flow models typically involves inverting an image into noise and then re-sampling. However, this "editing-by-inversion" paradigm suffers from three fundamental issues:

**Inversion is inherently inaccurate**: ODE discretization errors cause the inverted noise to be imprecise, leading to reconstructions that deviate from the original image.

**Exact inversion is still insufficient**: Even when editing images with known exact noise, editing-by-inversion fails to preserve structure.

**Attention injection methods lack generalizability**: Many methods compensate for inversion errors by injecting internal representations (e.g., attention maps) during sampling, but these approaches are highly architecture-specific and difficult to transfer to flow models such as FLUX and SD3.

The authors' core insight is that editing-by-inversion can be reinterpreted as a direct path from the source distribution to the target distribution, yet this path is not transport-optimal. Constructing a shorter, more direct path yields better structure preservation.

## Method

### Overall Architecture

The core idea of FlowEdit is to construct an ODE path directly between the source and target distributions, bypassing the Gaussian noise space. The process proceeds from $t=1$ (source image) to $t=0$ (edited image) by solving a specially designed ODE.

**Key formulation**: At each timestep $t$, FlowEdit performs the following steps:

1. Sample a noisy source: $Z_t^{src} = (1-t) \cdot X^{src} + t \cdot N_t$, where $N_t \sim \mathcal{N}(0,1)$
2. Construct the target sample: $Z_t^{tar} = Z_t^{FE} + Z_t^{src} - X^{src}$ (translation relationship)
3. Compute the velocity difference: $V_t^{\Delta} = V^{tar}(Z_t^{tar}, t) - V^{src}(Z_t^{src}, t)$
4. Update the editing path: $Z_{t-1}^{FE} = Z_t^{FE} + (t_{i-1} - t_i) \cdot V_t^{\Delta}$

### Key Designs

**1. Reinterpretation of Editing-by-Inversion**

The authors first demonstrate that editing-by-inversion is equivalent to a direct ODE path $Z_t^{inv} = Z_0^{src} + Z_t^{tar} - Z_t^{src}$. The intermediate states along this path are noise-free, as the noise components in $V^{tar}$ and $V^{src}$ cancel. However, the transport cost of this path is not optimal — experiments on synthetic Gaussian mixture distributions show that inversion conflates different modes, resulting in high transport cost.

**2. Velocity Averaging over Multiple Stochastic Pairings**

A key improvement in FlowEdit is to avoid fixing a single inversion path. Instead, multiple noise samples $N_t$ are drawn, their respective velocity field differences $V_t^{\Delta}$ are computed, and the expectation is taken. In practice, $n_{avg}=1$ (a single sample) works well, since the natural averaging across different timesteps provides sufficient smoothing.

**3. Editing Strength Control**

The parameter $n_{max}$ controls editing strength: setting $n_{max}=T$ traverses the full path for maximum editing; setting $n_{max} < T$ skips early timesteps for weaker edits. This is analogous to controlling inversion depth in inversion-based methods, but without actually performing inversion.

**4. Model Agnosticism**

FlowEdit does not depend on any model-internal structures (e.g., attention maps) and relies solely on forward-pass velocity field predictions. This enables seamless application to different architectures including FLUX and SD3.

### Loss & Training

FlowEdit is a purely inference-time method requiring **no training or optimization**. The only hyperparameters are:

- **T**: Number of ODE discretization steps (50 for SD3, 28 for FLUX)
- **n_max**: Starting timestep (33 for SD3, 24 for FLUX)
- **n_avg**: Number of velocity averaging samples per step (default: 1)
- **CFG scale**: Separate classifier-free guidance scales for source and target

## Key Experimental Results

### Main Results

**Quantitative comparison with other methods on SD3** (CLIP↑ measures text alignment; LPIPS↓ measures structure preservation):

| Method | CLIP↑ | LPIPS↓ | Notes |
|--------|:-:|:-:|--------|
| SDEdit (strength=0.6) | ~0.29 | ~0.20 | Good structure but weak editing |
| ODE Inversion | ~0.30 | ~0.25 | Severe structure degradation |
| iRFDS | ~0.29 | ~0.18 | Weak editing |
| **FlowEdit** | **~0.31** | **~0.12** | Strong editing with good structure preservation |

**Transport cost comparison (synthetic cat→dog editing, 1,000 images)**:

| Method | MSE↓ | LPIPS↓ | FID↓ | KID↓ |
|--------|:-:|:-:|:-:|:-:|
| Editing-by-Inversion | 2239 | 0.25 | 55.88 | 0.023 |
| **FlowEdit** | **1376** | **0.15** | **51.14** | **0.017** |

FlowEdit achieves less than half the transport cost of inversion while also yielding better target distribution fidelity (FID/KID).

### Ablation Study

**Effect of editing strength $n_{max}$ on FLUX**:

| n_max | CLIP↑ | LPIPS↓ | Effect |
|-------|:-:|:-:|--------|
| 16 | ~0.295 | ~0.05 | Editing too weak |
| 20 | ~0.305 | ~0.08 | Moderate editing |
| 24 | ~0.31 | ~0.12 | Best trade-off |
| 28 (full) | ~0.315 | ~0.18 | Strongest editing but increased structure loss |

### Key Findings

1. **FlowEdit consistently dominates the CLIP–LPIPS Pareto frontier**, being the only method that achieves strong editing while preserving structure.
2. **Low sensitivity to the source prompt**: FlowEdit is robust to the source prompt and can even omit it entirely.
3. **$n_{avg}=1$ is sufficient**: Natural averaging across timesteps makes single-sample estimation effective.
4. **Not an optimization process**: Despite superficial similarities, FlowEdit cannot be interpreted as minimizing a loss function — the DDS loss in fact increases across iterations.

## Highlights & Insights

1. **Theoretically grounded**: Reinterpreting editing-by-inversion as a direct ODE path reveals the fundamental reason for its high transport cost.
2. **Remarkably simple**: The entire algorithm can be implemented in a few lines of code, requiring no fine-tuning, optimization, or access to model internals.
3. **Model agnostic**: Validated on both SD3 and FLUX with distinct architectures, demonstrating strong transferability.
4. **Synthetic cat→dog experiment** elegantly illustrates the difference in transport cost and mode preservation ability.

## Limitations & Future Work

1. **Limited support for large-scale edits**: Strong structure preservation may become a constraint for edits requiring substantial modifications (e.g., pose changes, background replacement).
2. **Two model forward passes per step**: Computing both $V^{src}$ and $V^{tar}$ roughly doubles inference cost relative to standard generation.
3. **Restricted to flow models**: The method relies on the linear interpolation property of rectified flow and cannot be directly applied to conventional diffusion models (DDPM).
4. **Hyperparameter tuning required**: $n_{max}$ and CFG scales need to be adapted for different editing types.

## Related Work & Insights

- **SDEdit**: The simplest editing baseline (add noise + re-sample); serves as a primary comparison for FlowEdit.
- **DDS / PDS**: Optimization-based editing methods; FlowEdit provides a detailed analysis of the distinctions in the appendix.
- **Prompt-to-Prompt / MasaCtrl**: Attention injection-based editing methods that are not transferable across architectures.
- **RF-Inversion / RF-Edit**: Inversion-based editing methods for flow models, still constrained by the inversion paradigm.
- **Insight**: Directly modeling the transport path between distributions is more efficient than routing through noise space — a principle that generalizes to video and 3D editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[NeurIPS 2025\] Shortcutting Pre-trained Flow Matching Diffusion Models is Almost Free Lunch](../../NeurIPS2025/image_generation/shortcutting_pre-trained_flow_matching_diffusion_models_is_almost_free_lunch.md)
- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)

</div>

<!-- RELATED:END -->
