---
title: >-
  [Paper Note] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes the TD-Attn framework, which addresses multi-view inconsistency (the Janus problem) caused by prior-view bias in T2I diffusion models for 3D generation an…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Diffusion Models"
  - "Multi-view Consistency"
  - "Janus Problem"
  - "Attention Modulation"
date: 2026-05-08
content_hash: ddb56e793a502493
---

# Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting

**Conference**: AAAI 2026
**arXiv**: [2512.07345](https://arxiv.org/abs/2512.07345)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Diffusion Models, Multi-view Consistency, Janus Problem, Attention Modulation

## TL;DR

This paper proposes the TD-Attn framework, which addresses multi-view inconsistency (the Janus problem) caused by prior-view bias in T2I diffusion models for 3D generation and editing. The framework comprises two modules—3D-Aware Attention Guidance (3D-AAG) and Hierarchical Attention Modulation (HAM)—and can be integrated as a general-purpose plugin into existing 3DGS pipelines.

## Background & Motivation

3D generation and editing tasks that distill T2I diffusion models face a fundamental challenge: **multi-view inconsistency** (the Janus problem), which manifests as conflicting facial features, limbs, or textures when a 3D object is rendered from different viewpoints.

The authors provide a mathematical analysis to identify the root causes:

**Training data distribution bias**: T2I models are trained on datasets heavily skewed toward prior views (e.g., frontal views): $p_{\mathcal{D}}(v_{prior}|y_{obj}) \gg p_{\mathcal{D}}(v_{other}|y_{obj})$

**Subject-token attention bias**: When the probability ratio $\mathcal{R} = \frac{p(v_{prior}|Y)}{p(v^*|Y)} \gg 1$, subject-word tokens preferentially activate prior-view features, overriding the target-view conditioning.

**Gradient interference**: At viewpoints far from the prior view, $\nabla_{z_\phi}\log C \ll 0$ produces strong negative gradient effects that disrupt the 3D optimization process.

**Layer-wise heterogeneity**: Different layers of the UNet respond to prior-view bias to varying degrees.

## Method

### Overall Architecture

TD-Attn consists of two core modules:

1. **3D-Aware Attention Guidance Module (3D-AAG)**: Constructs a view-consistent 3D attention Gaussian field to constrain 2D attention maps.
2. **Hierarchical Attention Modulation Module (HAM)**: Uses a semantically guided tree to locate and modulate high-response cross-attention (CA) layers.

The framework operates as a plugin for different 3D tasks. For generation, training proceeds in three stages (HAM only → HAM + 3D-AAG → 3D-AAG only); for editing, two stages are used.

### Key Designs

#### 3D-AAG: 3D-Aware Attention Guidance

The core idea is to exploit the explicit representation of 3DGS by back-projecting multi-view 2D attention maps into 3D space, constructing a view-consistent 3D attention Gaussian field.

1. **Attention accumulation**: For each Gaussian $i$, multi-view 2D attention weights are accumulated:
   $w_i = \sum_{v \in \Lambda}\sum_{p \in \mathcal{I}(\mathcal{S}_{2D}^v)}[o_i(p)T_i^v(p)\mathcal{I}(\mathcal{S}(p)_{2D}^v)]$
   where $o_i$ is opacity, $T_i^v$ is transmittance, and $\mathcal{S}_{2D}^v$ is the CA map of the subject token.

2. **2D CA map computation**:
   $\mathcal{S}_{2D}^v = \text{Softmax}\left(\frac{Q_v K_{sbj}^T}{\sqrt{d}}\right)$

3. **Attention guidance loss**: KL divergence is used to enforce consistency between the 2D CA map and the rendered 3D attention Gaussian field:
   $\mathcal{L}_{attn} = KL(\text{Softmax}(\widetilde{\mathcal{S}}_{2D}^v) \| \mathcal{I}(\mathcal{S}_{2D}^v))$

4. **Synchronization with 3DGS densification**: The 3D attention Gaussian field is updated synchronously with the adaptive splitting and cloning operations of 3DGS.

#### HAM: Hierarchical Attention Modulation

HAM performs fine-grained modulation to account for the heterogeneous response of different UNet layers to view bias:

1. **Semantic Guided Tree (SGT) construction**: An LLM is used to construct a three-level hierarchy:
    - Root: $M$ semantic categories (Object, Attribute, etc.)
    - Intermediate: $F$ subcategories
    - Leaf nodes: $F$ instance words

2. **Semantic Response Profiling (SRP)**:
    - **Head level**: Computes a response score $W_h^f$ of each CA head to each subcategory.
    - **Layer level**: Computes a response score $W_l^m$ of each UNet layer to each semantic category.

3. **Attention modulation**:
   $\hat{\mathcal{A}}_h = \lambda W_l^{m^*} W_h^{f^*} \mathcal{A}_h$
   This selectively amplifies responses to target semantics (e.g., viewpoint) while suppressing prior-view bias.

4. **Semantic editing capability**: Beyond viewpoint semantics, HAM can also locate and control color, material, and other semantic attributes, enabling fine-grained 3D editing.

### Loss & Training

Generation task: $\mathcal{L} = \mathcal{L}_{Gen} + \lambda_1 \mathcal{L}_{attn}$, where $\lambda_1 = 10$

Editing task: $\mathcal{L} = \mathcal{L}_{Edit} + \lambda_2 \mathcal{L}_{attn}$, where $\lambda_2 = 10$

Generation proceeds in three stages: Stage 1 (0–200 iter, HAM only) → Stage 2 (200–2000, HAM + 3D-AAG) → Stage 3 (2000–4000, 3D-AAG only, for stabilizing fine details).

## Key Experimental Results

### Main Results (3D Generation)

| Method | ImageReward↑ | CLIPsim↑ | Quality↑ | Consistency↑ | f_mf(%)↓ | f_inc(%)↓ |
|------|-------------|----------|----------|-------------|----------|-----------|
| GCS-BEG | 0.158 | 0.312 | 6.13 | 4.18 | 33.3 | 60.0 |
| **GCS-BEG + Ours** | **0.397** | **0.317** | **7.81** | **7.68** | **6.7** | **26.7** |
| LucidDreamer | -0.386 | 0.309 | 5.34 | 4.02 | 26.7 | 60.0 |
| **LucidDreamer + Ours** | **0.124** | **0.320** | **7.27** | **6.67** | **13.3** | **33.3** |

**3D Editing:**

| Method | CLIPsim↑ | CLIPdir↑ | User Study↑ |
|------|----------|----------|-------------|
| Baseline (EditSplat) | 0.253 | 0.101 | 4.18 |
| **TD-Attn** | **0.277** | **0.114** | **6.34** |

### Ablation Study

**Generation task:**

| Method | CLIPsim↑ | f_mf(%)↓ | f_inc(%)↓ |
|------|----------|----------|-----------|
| Janus issue | 0.318 | 100.0 | 100.0 |
| Baseline | 0.307 | 35.6 | 62.2 |
| + HAM | 0.311 | 20.0 | 57.8 |
| + 3D-AAG | 0.313 | 24.4 | 44.4 |
| **TD-Attn** | **0.314** | **17.8** | **37.8** |

**HAM view generation success rate**: Under back-view conditioning, Stable Diffusion achieves a success rate of only 32.4%; with HAM, this increases to 75.2% (+42.8 pp).

### Key Findings

- TD-Attn reduces the Janus problem frequency by approximately 50% on average.
- Anomalously high CLIPsim scores reflect the Janus problem rather than genuine quality—viewpoint distribution analysis is a more reliable evaluation criterion than aggregate scores.
- HAM and 3D-AAG are complementary: HAM produces view-enhanced CA maps, which 3D-AAG leverages to construct a more consistent 3D attention Gaussian field.

## Highlights & Insights

1. **Theory-driven design**: The method is derived from a probabilistic analysis of the mathematical root causes of prior-view bias, enabling targeted solutions.
2. **Universal plugin architecture**: The framework integrates into diverse pipelines—DreamScene, LucidDreamer, GCS-BEG, EditSplat—without retraining the diffusion model.
3. **Semantic Guided Tree** is an elegant design that leverages LLM knowledge to construct a structured semantic space for attention analysis.
4. **Discovery of the CLIPsim evaluation pitfall**: The authors demonstrate that the Janus problem can inflate CLIPsim scores, and propose viewpoint distribution analysis as a more reliable alternative.
5. HAM's **semantic-level control** enables fine-grained 3D editing (e.g., disambiguating the color sense vs. botanical sense of "apricot").

## Limitations & Future Work

1. Validation is limited to Stable Diffusion v2.1/v1.4; applicability to newer models such as SDXL and FLUX remains unclear.
2. The three-stage training pipeline increases hyperparameter tuning complexity.
3. The Semantic Guided Tree relies on LLM generation, whose quality is not fully controllable.
4. The experimental scale is relatively small (100 user study participants), and quantitative geometric quality metrics (e.g., 3D IoU, LPIPS) are absent.
5. Performance on extreme viewpoints (directly above or below) has not been evaluated.

## Related Work & Insights

- **MVDream/Zero-1-to-3**: Address consistency by fine-tuning diffusion models on multi-view data, but incur additional training costs.
- **GaussianEditor**: A predecessor for back-projecting CA maps into 3DGS; TD-Attn builds upon this to construct 3D attention Gaussians.
- **HRV (park2024cross)**: Inspiration for the SRP module, though TD-Attn additionally accounts for lexical ambiguity in natural language.
- **Key insight**: Attention maps serve not only as diagnostic tools but can be inversely applied to guide 3D geometry optimization—establishing a new research paradigm for diffusion-guided 3D tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Deep theoretical analysis; novel designs of the 3D attention Gaussian field and Semantic Guided Tree)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on both generation and editing tasks with multiple baselines and ablations, but lacks geometric quantitative metrics)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clear, though the paper is lengthy)
- Value: ⭐⭐⭐⭐⭐ (The universal plugin approach offers significant practical value to the 3DGS community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[ICCV 2025\] ri3d few-shot gaussian splatting with repair and inpainting diffusion priors](../../ICCV2025/3d_vision/ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
