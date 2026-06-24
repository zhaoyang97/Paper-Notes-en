---
title: >-
  [Paper Note] Building Vision Models upon Heat Conduction
description: >-
  [CVPR 2025][LLM (Other)][vision backbone] A vision backbone named vHeat is proposed, which models image patches as heat sources and utilizes physical heat conduction equations via DCT/IDCT transforms to achieve global information propagation with $O(N^{1.5})$ complexity. It achieves 84.0% top-1 accuracy on ImageNet-1K with 3x higher throughput and 80% less GPU memory overhead.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "vision backbone"
  - "heat conduction"
  - "DCT"
  - "efficient attention"
  - "vHeat"
date: 2026-05-08
content_hash: c76a9baf3e38b5e0
---

# Building Vision Models upon Heat Conduction

**Conference**: CVPR 2025  
**arXiv**: [2405.16555](https://arxiv.org/abs/2405.16555)  
**Code**: [https://github.com/MzeroMiko/vHeat](https://github.com/MzeroMiko/vHeat)  
**Area**: LLM/NLP  
**Keywords**: vision backbone, heat conduction, DCT, efficient attention, vHeat

## TL;DR
A vision backbone named vHeat is proposed, which models image patches as heat sources and utilizes physical heat conduction equations via DCT/IDCT transforms to achieve global information propagation with $O(N^{1.5})$ complexity. It achieves 84.0% top-1 accuracy on ImageNet-1K with 3x higher throughput and 80% less GPU memory overhead.

## Background & Motivation

**Background**: Vision backbones have evolved from CNNs to ViTs, and subsequently to various efficient attention variants (Swin, linear attention, etc.). Although self-attention is effective, its $O(N^2)$ complexity limits its application on high-resolution inputs.

**Limitations of Prior Work**: Existing efficient attention methods are mostly approximations of self-attention, which either sacrifice global interaction capabilities (window attention) or sacrifice accuracy (the low-rank issue of linear attention). There is a lack of a global information propagation paradigm fundamentally distinct from the attention mechanism.

**Key Challenge**: Global information interaction requires interactions among all tokens ($O(N^2)$). However, in the physical world, information propagation governed by partial differential equations (such as heat conduction) is inherently global and can be solved efficiently.

**Goal**: Can a new global information propagation operator be designed by leveraging the physical heat conduction equation, maintaining global interaction capabilities while reducing computational complexity?

**Key Insight**: The heat conduction equation can be efficiently solved in the frequency domain (DCT), and the information propagation of image patches can be analogized to heat diffusion among heat sources.

**Core Idea**: Model image patches as heat sources, using learnable frequency-domain thermal diffusivities to achieve $O(N^{1.5})$ global information propagation via DCT/IDCT.

## Method

### Overall Architecture
vHeat adopts a hierarchical architecture (similar to Swin) divided into 4 stages, with progressively decreasing resolutions (H/4→H/8→H/16→H/32). Each stage consists of multiple vHeat Blocks, with each block containing a Heat Conduction Operator (HCO) and an FFN.

### Key Designs

1. **Heat Conduction Operator (HCO)**

    - **Function**: Replaces self-attention for global information propagation.
    - **Mechanism**: Treats the 2D feature map as a temperature field where each patch is a heat source. It utilizes the frequency-domain solution of the heat conduction equation: first transforming to the frequency domain via DCT, multiplying by frequency-dependent thermal diffusion coefficients, and then transforming back to the spatial domain via IDCT.
    - **Complexity**: $O(H \cdot W \cdot \log(H \cdot W))$, which is approximately $O(N^{1.5})$ for square images.
    - **Design Motivation**: The Green's function of the heat conduction equation possesses a global receptive field with distance decay, making it naturally suitable for modeling visual feature interactions that prioritize locality while accounting for global context.

2. **Learnable Frequency Value Embeddings (FVEs)**

    - **Function**: Learns adaptive thermal diffusivities for each frequency component.
    - **Mechanism**: Different frequency components have different diffusion rates; low frequencies (global structure) diffuse quickly, while high frequencies (local details) diffuse slowly. FVE predicts the diffusion coefficient $\alpha(f)$ for each frequency.
    - **Design Motivation**: While diffusivity is a material constant in physical heat conduction, different feature channels and frequencies in visual tasks should have different propagation speeds, hence they are set as learnable parameters.

3. **Hierarchical Architecture**

    - vHeat-Tiny: [2,2,6,2] blocks in each stage.
    - vHeat-Small: [2,2,18,2] blocks in each stage.
    - vHeat-Base: [2,2,18,2] blocks in each stage with wider channels.

### Loss & Training
- Standard 300-epoch training on ImageNet-1K.
- AdamW optimizer with a cosine learning rate schedule.

## Key Experimental Results

### Main Results

**ImageNet-1K Classification:**

| Model | Top-1 | Throughput (img/s) | GPU Memory |
|------|-------|-------------|---------|
| vHeat-T | 82.2% | 1514 | — |
| vHeat-S | 83.6% | 945 | — |
| vHeat-B | 84.0% | 661 | — |
| Swin-B | 83.5% | ~470 | — |

vHeat-B outperforms Swin-B by 0.5% with 40% higher throughput, 80% less GPU memory, and 35% fewer FLOPs.

**COCO Object Detection (1x schedule):**

| Model | mAP(box) | mAP(mask) | FPS |
|------|----------|-----------|-----|
| vHeat-B | 47.7 | 43.0 | 20.2 |
| Swin-B | 46.9 | 42.3 | 13.8 |

**ADE20K Semantic Segmentation**: vHeat-B achieves 49.6 mIoU at 23.6 FPS.

### Ablation Study

| Application | vHeat Variant | Compared Method | Results |
|------|-----------|---------|------|
| Image Denoising | vHeatIR | SwinIR | vHeatIR is superior |
| JPEG Deblocking | vHeatIR | SwinIR | vHeatIR is superior |
| ImageNet-A | vHeat-B | Swin-B | Stronger robustness |
| ObjectNet | vHeat-B | Swin-B | Stronger robustness |

### Key Findings
- The heat conduction operator consistently outperforms Swin across classification, detection, segmentation, and low-level vision tasks.
- The throughput advantage originates from the efficient FFT implementation of DCT/IDCT.
- It exhibits stronger robustness on out-of-distribution data (ImageNet-A, ObjectNet), indicating that physical priors provide a beneficial inductive bias.
- vHeatIR also performs exceptionally well in image restoration tasks, demonstrating the versatility of HCO.

## Highlights & Insights
- **Physics-inspired design paradigm**: Designing an information propagation operator starting from the heat conduction equation is an entirely new perspective, differing from various approximations of attention.
- **Ingenious application of DCT**: The frequency-domain solution of the heat conduction equation is naturally suitable for image processing, and DCT itself is a core tool in image compression (JPEG).
- **Global receptive field + distance decay**: The Green's function of heat conduction inherently possesses this characteristic, eliminating the need to artificially restrict the receptive field like window attention does.
- It can be extended to video understanding (spatiotemporal heat conduction) and 3D point cloud processing.

## Limitations & Future Work
- The efficiency of DCT/IDCT may degrade on non-square or non-power-of-two resolutions.
- Heat conduction is isotropic, whereas image content is typically anisotropic; directional diffusion might be required.
- It has not yet been validated on video or 3D tasks.
- The complementarity with the attention mechanism has not been explored (hybrid architectures might perform better).

## Related Work & Insights
- **vs Swin Transformer**: Window attention restricts the receptive field, whereas vHeat naturally facilitates global interaction and is faster.
- **vs VMamba**: Both are non-attention vision backbones; VMamba employs state space models, while vHeat utilizes the heat conduction equation, offering different physical priors.
- **vs FNet**: FNet replaces attention with FFT but lacks physical meaning, whereas vHeat's heat conduction formulation provides a superior inductive bias.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Designing visual operators from physical equations offers a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of classification, detection, segmentation, and low-level vision.
- Writing Quality: ⭐⭐⭐⭐ The physical motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ Provides a brand-new information propagation paradigm beyond attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/llm_nlp/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICML 2025\] Defending LVLMs Against Vision Attacks through Partial-Perception Supervision](../../ICML2025/llm_nlp/defending_lvlms_against_vision_attacks_through_partial-perception_supervision.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](../../ACL2025/llm_nlp/classifying_unreliable_narrators.md)

</div>

<!-- RELATED:END -->
