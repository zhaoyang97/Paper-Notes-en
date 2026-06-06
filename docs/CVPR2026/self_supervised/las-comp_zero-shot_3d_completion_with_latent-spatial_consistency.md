---
title: >-
  [Paper Note] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency
description: >-
  [CVPR 2026][Self-Supervised Learning][3D shape completion] This paper proposes LaS-Comp, a zero-shot, category-agnostic 3D shape completion framework. It injects known geometry in the spatial domain via an Explicit Repla…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "3D shape completion"
  - "zero-shot"
  - "3D foundation models"
  - "latent-spatial consistency"
  - "point cloud completion"
date: 2026-05-08
content_hash: 861dfb755d90bfe7
---

# LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency

**Conference**: CVPR 2026
**arXiv**: [2602.18735](https://arxiv.org/abs/2602.18735)  
**Code**: [https://github.com/wylyan/LaS-Comp](https://github.com/wylyan/LaS-Comp)  
**Area**: Self-Supervised Learning
**Keywords**: 3D shape completion, zero-shot, 3D foundation models, latent-spatial consistency, point cloud completion

## TL;DR
This paper proposes LaS-Comp, a zero-shot, category-agnostic 3D shape completion framework. It injects known geometry in the spatial domain via an Explicit Replacement Stage (ERS) and optimizes boundary consistency in the latent space via gradient-based updates in an Implicit Alignment Stage (IAS). The framework bridges the gap between the latent space and spatial domain of pretrained 3D foundation models, achieving state-of-the-art performance across diverse partial observation patterns.

## Background & Motivation

3D shape completion is a fundamental problem in computer vision and graphics, aiming to reconstruct complete 3D shapes from partial observations. It has broad applications in robotics, autonomous driving, and AR/VR. An ideal completion method should: (i) robustly handle diverse partial missing patterns (single-view scans, random crops, semantic part removal); (ii) generalize across categories; (iii) not rely on paired data; and (iv) support both text-guided and automatic completion.

Traditional supervised methods depend on paired data and fail to generalize to unseen categories. Recent methods leveraging generative priors (SDS-Complete, ComPC, GenPC) rely on the assumption that the partial input can render at least one complete image — when missing regions are visible from all viewpoints, incomplete renderings cause severe degradation.

The latest 3D foundation models (TRELLIS, Direct3D-S2) adopt a latent-space generation pipeline: shapes are first encoded into a compact latent space via a VAE, and then diffusion/flow-matching models are trained in that latent space. This introduces a unique challenge: **even when the complete shape and its partial counterpart share identical geometry in overlapping regions, their latent-space encodings differ significantly**. Direct latent-space completion is therefore unreliable.

The core idea of LaS-Comp is to bridge the domain gap between the latent and spatial domains through explicit spatial-domain replacement combined with implicit latent-space alignment, thereby unlocking the completion potential of 3D foundation models.

## Method

### Overall Architecture
Starting from Gaussian noise, the framework iteratively denoises over multiple steps, guided by the partial input $\boldsymbol{S}_p$, to progressively recover the complete geometry. At each step $t \in [0,1]$, two complementary stages are applied:
1. **Explicit Replacement Stage (ERS)**: Explicitly injects geometric information from $\boldsymbol{S}_p$ in the spatial domain, producing updated latent features $\boldsymbol{x}_t^*$.
2. **Implicit Alignment Stage (IAS)**: Optimizes $\boldsymbol{x}_t^*$ via geometric alignment loss gradients, producing spatially aligned features $\boldsymbol{x}_{t-dt}$.

The complete shape is finally obtained by decoding $\boldsymbol{S}_c = \mathcal{D}(\boldsymbol{x}_0)$.

### Key Designs
1. **Explicit Replacement Stage (ERS)**:

    - **Function**: Explicitly injects geometric information from the partial input into the latent features of the generation process, ensuring faithful preservation of known regions.
    - **Mechanism**: Operates via two parallel branches — a Clean Branch and a Noisy Branch.
        - **Clean Branch**: The generator predicts a noise-free latent $\hat{\boldsymbol{x}}_{0|t} = \boldsymbol{x}_t - t \cdot \mathcal{G}(\boldsymbol{x}_t, t)$, which is decoded into a complete shape $\boldsymbol{S}_{0|t}$. A spatial mask replacement is then applied: $\boldsymbol{S}'_{0|t} = \boldsymbol{S}_p \odot \boldsymbol{M} + \boldsymbol{S}_{0|t} \odot (1-\boldsymbol{M})$, and the result is re-encoded into the latent space as $\boldsymbol{x}^*_{0|t} = \mathcal{E}(\boldsymbol{S}'_{0|t})$.
        - **Noisy Branch** + Partial-aware Noise Schedule (PNS): Time-dependent, decreasing perturbations are applied to observed regions ($\boldsymbol{M}=1$) to maintain stability, while pure Gaussian noise is applied to missing regions ($\boldsymbol{M}=0$) to encourage diverse exploration: $\boldsymbol{x}^*_{1|t} = \boldsymbol{M} \odot (\sqrt{1-t} \cdot \hat{\boldsymbol{x}}_{1|t} + \sqrt{t} \cdot \boldsymbol{\epsilon}_1) + (1-\boldsymbol{M}) \odot \boldsymbol{\epsilon}_2$
    - The two branches are combined via flow interpolation: $\boldsymbol{x}^*_t = (1-t) \cdot \boldsymbol{x}^*_{0|t} + t \cdot \boldsymbol{x}^*_{1|t}$
    - **Design Motivation**: Direct latent-space replacement fails due to the domain gap; explicit spatial-domain replacement followed by re-encoding circumvents this issue. PNS assigns different degrees of stochasticity to known and unknown regions.

2. **Implicit Alignment Stage (IAS)**:

    - **Function**: Corrects discontinuities at the boundary between observed and generated regions that ERS may introduce.
    - **Mechanism**: Starting from $\boldsymbol{x}_t^*$, a noise-free latent is predicted and decoded, then a geometric alignment loss is computed: $\mathcal{L}_{\text{align}} = \text{BCE}(\boldsymbol{S}_{0|t} \odot \boldsymbol{M}, \boldsymbol{S}_p \odot \boldsymbol{M})$. A single-step gradient update is applied to the latent features: $\boldsymbol{x}^{\text{aligned}}_{0|t} = \hat{\boldsymbol{x}}_{0|t} - \eta \cdot \nabla_{\hat{\boldsymbol{x}}_{0|t}} \mathcal{L}_{\text{align}}$
    - **Design Motivation**: While ERS guarantees fidelity, it may introduce artifacts at region boundaries. IAS smooths these inconsistencies via gradient optimization in the latent space. Only a single update step is required, keeping computational overhead minimal.
    - Note: This loss updates only the latent features themselves, not the model parameters.

3. **Omni-Comp Benchmark**:

    - **Function**: A new comprehensive evaluation benchmark comprising 30 objects from diverse categories, 3 partial missing patterns (single-view scan, random crop, semantic part removal), and 180 samples in total.
    - **Data Sources**: 10 Redwood real-world scans + 10 YCB everyday objects + 10 synthetic shapes.
    - **Design Motivation**: Existing benchmarks are limited in scale (Redwood: only 10 objects), category diversity (KITTI/ScanNet: ≤2 categories), and partial pattern variety (single pattern only).

### Loss & Training
LaS-Comp is **training-free** — it requires no additional training and directly leverages pretrained 3D foundation models (TRELLIS or Direct3D-S2) at inference time. The gradient update learning rate in IAS is $\eta = 1 \times 10^{-5}$. Completion of each shape takes approximately 20 seconds, more than 3× faster than existing zero-shot methods.

## Key Experimental Results

### Main Results

| Dataset / Metric | Ours (TRELLIS) | Prev. SOTA (ComPC) | Gain |
|---|---|---|---|
| Redwood CD↓/EMD↓ | **1.42/1.84** | 1.95/2.59 | 27.2%/29.0% |
| Synthetic CD↓/EMD↓ | **1.11/1.41** | 1.61/2.09 | 31.1%/32.5% |
| ScanNet-Chair UCD↓/UHD↓ | **0.8/2.0** | 2.0/5.3 | 60%/62% |
| KITTI-Car UCD↓/UHD↓ | **1.4/4.5** | 1.1/5.7 | — |
| Omni-Comp Single Scan CD↓ | **2.21** | 4.24 | 47.9% |
| Omni-Comp Random Crop CD↓ | **2.60** | 5.48 | 52.6% |
| Omni-Comp Semantic Part CD↓ | **3.30** | 6.37 | 48.2% |

### Ablation Study

| Configuration | Redwood CD↓ | Notes |
|---|---|---|
| Full LaS-Comp (TRELLIS) | 1.42 | Best |
| ERS only (w/o IAS) | 1.68 | IAS yields ~15% improvement |
| IAS only (w/o ERS) | 2.31 | ERS is the core component |
| w/o PNS (uniform noise schedule) | 1.89 | PNS is important |
| Direct3D-S2 backbone | 1.64 | TRELLIS performs better |

### Key Findings
- On Omni-Comp, prior methods suffer drastic performance degradation on Random Crop and Semantic Part patterns (due to reliance on the "at least one complete viewpoint" assumption), whereas LaS-Comp remains robust across all patterns.
- The framework is compatible with different 3D foundation models (TRELLIS and Direct3D-S2) and significantly outperforms baselines with both backbones.
- Text-guided completion is supported via the foundation model's CFG mechanism, enabling semantic control over generation results.
- Inference speed is approximately 20 seconds per shape, substantially faster than ComPC (~60s) and SDS-Complete (>5min).

## Highlights & Insights
- **Training-free**: No additional training is required; the method directly leverages pretrained models, making deployment highly practical.
- **Addresses a fundamental domain gap**: The paper identifies and resolves the previously underappreciated discrepancy between latent encodings of partial versus complete shapes.
- **Elegant complementarity of ERS and IAS**: Explicit replacement ensures fidelity; implicit alignment ensures smoothness. Both components are indispensable.
- **Partial-aware Noise Schedule**: Applying distinct noise strategies to known and unknown regions reflects a deep understanding of the asymmetric nature of the completion task.
- **Omni-Comp benchmark** fills a critical gap in multi-pattern completion evaluation.

## Limitations & Future Work
- Performance is bounded by the quality of the pretrained 3D foundation model; weak generative capability for certain categories will limit completion quality.
- Each step requires an encode-decode round trip (ERS), increasing inference overhead.
- IAS performs only a single gradient update step; additional optimization steps could further improve boundary quality at the cost of increased runtime.
- The method has only been validated on rigid objects; articulated and deformable objects remain unexplored.

## Related Work & Insights
- **RepPaint/FlowDPS** inspired the clean-noisy dual-branch design of ERS, transferring concepts from 2D image inpainting to 3D.
- The latent-space generation paradigm of **TRELLIS/Direct3D-S2** and similar 3D foundation models provides the foundation for this work.
- The key distinction from **ComPC/GenPC** is that LaS-Comp does not rely on 2D rendering assumptions and operates directly in the 3D latent space.
- **Insight**: Latent-spatial consistency may be an equally critical challenge in other latent-space generation tasks, such as 3D editing and style transfer.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic solution to the completion problem for latent-space 3D foundation models; ERS+IAS design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 4 existing benchmarks plus the proposed Omni-Comp, with 2 backbones and multiple partial patterns.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method description, and intuitive figures.
- **Value**: ⭐⭐⭐⭐⭐ High practical utility as a training-free solution; Omni-Comp benchmark has lasting value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](../../ICML2026/self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)

</div>

<!-- RELATED:END -->
