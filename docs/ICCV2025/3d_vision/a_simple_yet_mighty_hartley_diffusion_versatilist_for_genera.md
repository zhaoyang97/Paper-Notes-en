---
title: >-
  [Paper Note] A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks
description: >-
  [ICCV 2025][3D Vision][Diffusion Models] This paper proposes HarDiff — a frequency-domain learning strategy based on the Discrete Hartley Transform (DHT) — that enhances the cross-domain generalization capability of diff…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Diffusion Models"
  - "Discrete Hartley Transform"
  - "Domain Generalization"
  - "Semantic Segmentation"
  - "Depth Estimation"
  - "Dehazing"
date: 2026-05-08
content_hash: 08ae39230cf936a0
---

# A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks

**Conference**: ICCV 2025
**arXiv**: N/A (CVF only)
**Code**: No public information
**Area**: Dense Prediction / Diffusion Models / Domain Generalization
**Keywords**: Diffusion Models, Discrete Hartley Transform, Domain Generalization, Semantic Segmentation, Depth Estimation, Dehazing

## TL;DR
This paper proposes HarDiff — a frequency-domain learning strategy based on the Discrete Hartley Transform (DHT) — that enhances the cross-domain generalization capability of diffusion models on dense vision tasks through low-frequency training (extracting structural priors from the source domain) and high-frequency sampling (leveraging target-domain detail guidance). HarDiff achieves state-of-the-art results across 12 benchmarks spanning semantic segmentation, depth estimation, and image dehazing.

## Background & Motivation
Diffusion models have demonstrated strong potential as versatilists for dense vision tasks, yet their ability to generalize to unseen domains remains largely unexplored. A critical open question is how to enable diffusion models to learn task-relevant structural knowledge from the source domain while adapting to the fine-grained characteristics of the target domain at inference time.

## Core Problem
How can the generalization capability of diffusion-based dense prediction methods be improved under domain shift? The core observation is that low-frequency components in the Hartley transform encode broad content information, while high-frequency components preserve fine-grained details — properties that can be exploited separately during training and sampling, respectively.

## Method

### Overall Architecture
HarDiff builds upon diffusion models for dense visual prediction. Its central innovation lies in incorporating the Discrete Hartley Transform (DHT) for frequency-domain analysis, decomposing the diffusion process into two distinct phases: low-frequency training and high-frequency sampling. The overall pipeline proceeds as follows: annotated source-domain image–label pairs serve as input → low-frequency components are extracted in the frequency domain for training, enabling the model to learn domain-invariant structural priors → at inference time, high-frequency components from target-domain images are injected to guide sampling, incorporating target-domain-specific detail information → the final output is a dense prediction result (semantic segmentation map, depth map, or dehazed image).

### Key Designs

1. **Discrete Hartley Transform (DHT) Analysis**:

    - Function: Transforms images/features from the spatial domain to the frequency domain to analyze the roles of low- and high-frequency components.
    - Mechanism: Unlike the Discrete Fourier Transform (DFT), DHT employs the $\text{cas}(\cdot) = \cos(\cdot) + \sin(\cdot)$ function in place of the complex exponential, yielding an entirely real-valued transform with no imaginary components. DHT is defined as $H(u,v) = \frac{1}{N}\sum_{x,y} f(x,y) \text{cas}(\frac{2\pi ux}{N} + \frac{2\pi vy}{N})$. Low-frequency components encode global content and structural information, while high-frequency components retain fine details such as edges and textures.
    - Design Motivation: The purely real-valued computation of DHT is more efficient than DFT and is natively compatible with tensor operations in deep learning frameworks, eliminating the need to handle complex numbers. Analysis reveals that the low-frequency components of images from different domains exhibit high consistency (domain-invariant), whereas high-frequency components differ substantially (domain-specific) — providing a theoretically grounded frequency-domain perspective for domain generalization.

2. **Low-Frequency Training**:

    - Function: Learns domain-invariant structural priors from source-domain data.
    - Mechanism: During diffusion model training, DHT is applied to intermediate features or noise representations, and only the low-frequency components are retained for loss computation or feature propagation. These components capture shared, broad structural content across domains — such as approximate object shapes and scene layouts.
    - Design Motivation: Focusing on low-frequency components naturally filters out domain-specific high-frequency details (e.g., domain-particular texture patterns and noise characteristics), thereby encouraging the model to learn more generalizable structural representations.

3. **High-Frequency Sampling**:

    - Function: Guides generation during inference by incorporating high-frequency information from target-domain images.
    - Mechanism: During the diffusion model's sampling (denoising) process, high-frequency components extracted via DHT decomposition of target-domain images are injected into the sampling steps. This allows the generated dense predictions to retain the structural correctness learned from the source domain while incorporating target-domain-specific fine-grained details.
    - Design Motivation: The high-frequency components of target-domain images encode domain-specific texture and detail information. By introducing these components during sampling, the model can adaptively transition from predictions that are structurally correct but lacking in detail to final outputs that are both structurally accurate and detail-rich.

### Loss & Training

- The training phase employs a standard diffusion model denoising loss, with higher weighting assigned to low-frequency components in the frequency domain.
- The overall strategy is plug-and-play and can be integrated into various diffusion-based dense prediction frameworks.
- No fine-tuning on the target domain is required; domain generalization is achieved at inference time directly via the high-frequency sampling strategy.

## Key Experimental Results

The paper evaluates on 12 public benchmarks across three task categories: semantic segmentation (domain generalization setting), monocular depth estimation, and image dehazing.

### Main Results

| Task | Setting | Summary |
|------|---------|---------|
| Semantic Segmentation | Cross-domain benchmarks (e.g., GTA5→Cityscapes) | Surpasses existing state-of-the-art domain generalization methods |
| Monocular Depth Estimation | Benchmarks including NYU/KITTI | Surpasses existing state-of-the-art methods |
| Image Dehazing | Benchmarks including RESIDE | Surpasses existing state-of-the-art methods |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full HarDiff | Best | Complete pipeline: low-frequency training + high-frequency sampling |
| Low-frequency training only (no high-frequency sampling) | Degraded | Lacks target-domain detail guidance |
| High-frequency sampling only (no low-frequency training) | Degraded | Lacks domain-invariant structural foundation |
| DFT replacing DHT | Slightly lower | DHT offers more efficient purely real-valued computation with comparable or better accuracy |

### Key Findings

- Both the low-frequency training and high-frequency sampling stages are indispensable, demonstrating the complementarity of frequency-domain decomposition.
- DHT achieves comparable accuracy to DFT with greater computational efficiency, as purely real-valued operations avoid the overhead associated with complex arithmetic.
- The same strategy proves consistently effective across segmentation, depth estimation, and dehazing tasks, indicating that frequency-domain decomposition captures domain-invariant and domain-specific properties that are universal across tasks.

## Highlights & Insights

- **Hartley Transform as a Substitute for Fourier Transform**: DHT's purely real-valued computation is better suited to deep learning settings than DFT, while offering equivalent signal analysis capability. This work represents one of the early efforts to introduce DHT into frequency-domain analysis for deep learning.
- **A Frequency-Domain Perspective on Domain Generalization**: Low frequency = structure = domain-invariant; high frequency = detail = domain-specific. This frequency-domain perspective is more theoretically elegant and principled than conventional style transfer or domain adaptation approaches.
- **Plug-and-Play Generality**: The same strategy applies uniformly to different dense prediction tasks including segmentation, depth estimation, and dehazing, demonstrating strong task-agnostic generalizability of the frequency-domain-based domain generalization strategy.

## Limitations & Future Work

- Precise quantitative gains on the 12 benchmarks remain to be verified, as access is currently limited to the CVF PDF.
- Whether high-frequency sampling incurs additional inference-time computational cost has not been explicitly quantified by the authors.
- Extension to other dense prediction tasks — such as optical flow, surface normal estimation, and 3D reconstruction — has yet to be explored.
- Whether the frequency cutoff threshold for separating low and high frequencies is fixed or adaptive is not discussed in detail in the paper.

## Related Work & Insights

- **vs. VPD and other diffusion-based dense prediction methods**: Standard methods such as VPD typically assume that training and testing are conducted within the same domain; HarDiff explicitly addresses domain generalization.
- **vs. FDA and other Fourier domain adaptation methods**: FDA performs frequency swapping in pixel space, whereas HarDiff applies frequency separation within the diffusion process itself — integrating the frequency-domain strategy more deeply into the generative procedure.
- **Generality of Frequency-Domain Priors**: DHT-based decomposition can be transferred to domain generalization designs in other generative models, including GANs and VAEs.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of Hartley transform and diffusion model domain generalization is relatively novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage across 12 benchmarks and 3 task categories is extensive.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clear; the framework is concise and elegant.
- Value: ⭐⭐⭐⭐ The frequency-domain approach offers meaningful inspiration for domain generalization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities](llava-3d_a_simple_yet_effective_pathway_to_empowering_lmms_with_3d_capabilities.md)
- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)

</div>

<!-- RELATED:END -->
