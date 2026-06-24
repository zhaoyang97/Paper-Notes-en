---
title: >-
  [Paper Note] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This work proposes the TD-Attn framework, which integrates two modules—3D-Aware Attention Guidance (3D-AAG) and Hierarchical Attention Modulation (HAM)—to resolve the multi-view inconsistency (Janus problem) in 3D generation/editing caused by prior viewpoint bias in T2I diffusion models. It can be integrated into existing 3DGS frameworks as a plug-and-play module.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Diffusion Models"
  - "Multi-view Consistency"
  - "Janus Problem"
  - "Attention Modulation"
date: 2026-05-08
content_hash: 7dc585252298513f
---

# Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting

**Conference**: AAAI 2026  
**arXiv**: [2512.07345](https://arxiv.org/abs/2512.07345)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Diffusion Models, Multi-view Consistency, Janus Problem, Attention Modulation

## TL;DR

This work proposes the TD-Attn framework, which integrates two modules—3D-Aware Attention Guidance (3D-AAG) and Hierarchical Attention Modulation (HAM)—to resolve the multi-view inconsistency (Janus problem) in 3D generation/editing caused by prior viewpoint bias in T2I diffusion models. It can be integrated into existing 3DGS frameworks as a plug-and-play module.

## Background & Motivation

3D generation and editing tasks based on T2I diffusion model distillation face a fundamental challenge: **multi-view inconsistency** (the Janus problem). This manifests as conflicting faces, limbs, or textures when rendering 3D objects from different perspectives.

The authors reveal the root causes through mathematical analysis:

**Training Data Distribution Bias**: The training data of T2I models contains far more samples of prior viewpoints (such as the front view) than other perspectives: $p_{\mathcal{D}}(v_{prior}|y_{obj}) \gg p_{\mathcal{D}}(v_{other}|y_{obj})$

**Subject Word Attention Bias**: When the probability ratio is $\mathcal{R} = \frac{p(v_{prior}|Y)}{p(v^*|Y)} \gg 1$, the subject token preferentially activates prior viewpoint features, overriding the target viewpoint conditions.

**Gradient Interference**: When far from the prior viewpoint, $\nabla_{z_\phi}\log C \ll 0$ produces a strong negative gradient effect, disrupting the 3D optimization process.

**Inter-layer Heterogeneity**: Different layers of the UNet respond differently to the prior viewpoint preference.

## Method

### Overall Architecture

TD-Attn comprises two core modules:

1. **3D-Aware Attention Guidance Module (3D-AAG)**: Constructs view-consistent 3D attention Gaussians to constrain 2D attention maps.
2. **Hierarchical Attention Modulation Module (HAM)**: Pinpoints and modulates highly responsive cross-attention (CA) layers via a semantic guidance tree.

The framework is utilized as a plug-in for various 3D tasks. The generation task is split into three stages (HAM only $\to$ HAM+3D-AAG $\to$ 3D-AAG only), and the editing task is divided into two stages.

### Key Designs

#### 3D-AAG: 3D-Aware Attention Guidance

The core idea is to leverage the explicit nature of 3DGS to back-project multi-view 2D attention maps into 3D space, establishing view-consistent 3D attention Gaussians.

1. **Attention Accumulation**: For each Gaussian $i$, accumulate 2D attention weights from multiple views:
    $w_i = \sum_{v \in \Lambda}\sum_{p \in \mathcal{I}(\mathcal{S}_{2D}^v)}[o_i(p)T_i^v(p)\mathcal{I}(\mathcal{S}(p)_{2D}^v)]$
   where $o_i$ is the opacity, $T_i^v$ is the transmittance, and $\mathcal{S}_{2D}^v$ is the CA map of the subject word token.

2. **2D CA Map Computation**:
    $\mathcal{S}_{2D}^v = \text{Softmax}\left(\frac{Q_v K_{sbj}^T}{\sqrt{d}}\right)$

3. **Attention Guidance Loss**: Use KL divergence to constrain consistency between the 2D CA maps and the rendered results of the 3D attention Gaussians:
    $\mathcal{L}_{attn} = KL(\text{Softmax}(\widetilde{\mathcal{S}}_{2D}^v) \| \mathcal{I}(\mathcal{S}_{2D}^v))$

4. **Synchronization with 3DGS Densification**: The 3D attention Gaussians are updated synchronously with the adaptive splitting/cloning operations of 3DGS.

#### HAM: Hierarchical Attention Modulation

HAM performs fine-grained modulation tailored to the heterogeneity in viewpoint preferences across different UNet layers:

1. **Semantic Guidance Tree (SGT) Construction**: Build a three-level hierarchical structure utilizing LLMs.

    - Root: $M$ semantic classes (e.g., Object, Attribute)
    - Middle layer: $F$ subclasses
    - Leaves: $F$ instance words

2. **Semantic Response Analysis (SRP)**:

    - **Head level**: Calculate the response score $W_h^f$ of CA Heads to subclasses.
    - **Layer level**: Calculate the response score $W_l^m$ of UNet layers to semantic classes.

3. **Attention Modulation**:
    $\hat{\mathcal{A}}_h = \lambda W_l^{m^*} W_h^{f^*} \mathcal{A}_h$
   Selectively enhance responses to the target semantics (such as viewpoint) and suppress prior bias.

4. **Semantic Editing Capability**: HAM not only tracks viewpoint semantics but also localizes and controls semantics like color and material to achieve fine-grained 3D editing.

### Loss & Training

Generation task: $\mathcal{L} = \mathcal{L}_{Gen} + \lambda_1 \mathcal{L}_{attn}$, where $\lambda_1 = 10$

Editing task: $\mathcal{L} = \mathcal{L}_{Edit} + \lambda_2 \mathcal{L}_{attn}$, where $\lambda_2 = 10$

Generation is conducted in three stages: Stage 1 (0-200 iter, HAM only) $\to$ Stage 2 (200-2000, HAM+3D-AAG) $\to$ Stage 3 (2000-4000, 3D-AAG only for stabilizing details).

## Key Experimental Results

### Main Results

**3D Generation:**

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

**Generation Task:**

| Method | CLIPsim↑ | f_mf(%)↓ | f_inc(%)↓ |
|------|----------|----------|-----------|
| Janus issue | 0.318 | 100.0 | 100.0 |
| Baseline | 0.307 | 35.6 | 62.2 |
| + HAM | 0.311 | 20.0 | 57.8 |
| + 3D-AAG | 0.313 | 24.4 | 44.4 |
| **TD-Attn** | **0.314** | **17.8** | **37.8** |

**HAM Viewpoint Generation Success Rate**: Under back-view conditions, the success rate of Stable Diffusion is only 32.4%. After incorporating HAM, it increases to 75.2% (+42.8pp).

### Key Findings

- TD-Attn reduces the frequency of the Janus problem by approximately 50% on average.
- The abnormally high CLIPsim scores reflect the Janus problem rather than true quality—viewpoint distribution analysis is more reliable than average scores.
- HAM and 3D-AAG are complementary: HAM provides viewpoint-enhanced CA maps, and 3D-AAG utilizes this information to construct more consistent 3D attention Gaussians.

## Highlights & Insights

1. **Theory-Driven Method Design**: Derives the mathematical root causes of prior viewpoint bias from probabilistic analysis, and subsequently designs targeted solutions.
2. **General Plug-and-Play Architecture**: Can be integrated into multiple frameworks like DreamScene, LucidDreamer, GCS-BEG, and EditSplat without retraining the diffusion model.
3. **The Semantic Guidance Tree** is a clever design, employing LLM knowledge to construct a structured semantic space to guide attention analysis.
4. **Discovery of the CLIPsim Evaluation Pitfall**: Pointing out that the Janus problem counterintuitively leads to high CLIPsim scores, the authors propose viewpoint distribution analysis as a more reliable evaluation method.
5. HAM's **semantic-level control capability** enables fine-grained 3D editing (e.g., distinguishing between the color and botanical meanings of "apricot").

## Limitations & Future Work

1. Only validated on Stable Diffusion v2.1/v1.4; whether it generalizes to newer models such as SDXL or FLUX remains unverified.
2. The three-stage training pipeline increases the complexity of hyperparameter tuning.
3. The semantic guidance tree relies on LLM generation, making its quality less controllable.
4. The experiment scale is relatively small (100 users for human evaluation), and there is a lack of quantitative evaluations of geometry quality (e.g., 3D IoU, LPIPS).
5. The performance under extreme viewpoints (directly below/directly above) has not been validated.

## Related Work & Insights

- **MVDream/Zero-1-to-3**: Address consistency issues by fine-tuning diffusion models on multi-view datasets, but require additional training overhead.
- **GaussianEditor**: A pioneering work that projects CA maps back to 3DGS. TD-Attn builds upon this to construct 3D attention Gaussians.
- **HRV (park2024cross)**: Represents the source of inspiration for the SRP module, but TD-Attn additionally accounts for the polysemy of natural language.
- Insight: Attention maps are not merely diagnostic tools but can also be inversely utilized to guide 3D geometric optimization—forming a new research paradigm for diffusion-guided 3D tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (In-depth theoretical analysis; the designs of 3D attention Gaussians and semantic guidance tree are novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Validated on both generation and editing tasks with multiple baselines compared and abated, but lacks geometric quantitative metrics)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical derivations are clear, but the manuscript is quite long)
- Value: ⭐⭐⭐⭐⭐ (The universal plug-and-play approach has substantial practical value for the 3DGS community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](../../ECCV2024/3d_vision/gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](../../CVPR2026/3d_vision/orbital_video_3d_foundation_priors.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->
