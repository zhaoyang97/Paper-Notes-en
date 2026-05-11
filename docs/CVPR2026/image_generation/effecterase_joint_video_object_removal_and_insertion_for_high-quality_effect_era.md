---
title: >-
  [Paper Note] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing
description: >-
  [CVPR 2026][Image Generation][Video Object Removal] This paper proposes EffectErase, a framework that jointly learns video object insertion as an inverse auxiliary task to object removal…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Video Object Removal"
  - "Visual Side Effects"
  - "Diffusion Models"
  - "Dual Learning"
  - "Dataset"
date: 2026-05-08
content_hash: 12fdba2931300957
---

# EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing

**Conference**: CVPR 2026  
**arXiv**: [2603.19224](https://arxiv.org/abs/2603.19224)  
**Code**: [Project Page](https://henghuiding.com/EffectErase/)  
**Area**: Image Generation  
**Keywords**: Video Object Removal, Visual Side Effects, Diffusion Models, Dual Learning, Dataset

## TL;DR

This paper proposes EffectErase, a framework that jointly learns video object insertion as an inverse auxiliary task to object removal, and constructs a large-scale VOR dataset containing 60K video pairs, enabling high-quality erasure of objects along with their associated visual side effects, including occlusion, shadow, reflection, illumination changes, and deformation.

## Background & Motivation

Video object removal requires not only eliminating the target object itself, but also erasing various **visual side effects** introduced by the object (e.g., shadows, reflections, illumination changes, occlusions, deformations). Existing methods face two major challenges:

**Method-level**: Current video inpainting methods rely heavily on input masks for guidance, neglecting side effects outside the mask region. Even approaches like ROSE that predict difference masks still lack explicit modeling of spatiotemporal associations between objects and their effects.

**Data-level**: No large-scale public dataset systematically captures diverse object effects. SVOR does not account for side effects, while ROSE relies only on synthetic data generated via camera motion, with limited scale and diversity.

## Method

### Overall Architecture

The framework is built upon the Wan 2.1 video generation model with LoRA fine-tuning, and consists of three core components: (1) joint removal–insertion learning, (2) Task-Aware Region Guidance (TARG), and (3) Effect Consistency (EC) loss.

### Key Designs

1. **VOR Dataset**: A large-scale dataset combining real footage and 3D synthesis, covering five categories of visual side effects:

    - **Occlusion** (three subtypes: opaque / semi-transparent / transparent)
    - **Shadow** (cast shadows under varying illumination conditions)
    - **Illumination** (brightness and color changes after light source removal)
    - **Reflection** (specular, water, tile, and other reflective surfaces)
    - **Deformation** (physical deformation of curtains, grass, nets, etc.)

   Dataset scale: **60K video pairs**, 366 object categories, 443 scenes, totaling **145+ hours**, far exceeding prior datasets. Real data is captured using fixed cameras with paired recordings; synthetic data is rendered via Blender across 150+ 3D scenes.

2. **Task-Aware Region Guidance (TARG)**: Models spatiotemporal associations between objects and their side effects via cross-attention. The foreground object is encoded using a CLIP image encoder to obtain an embedding $\boldsymbol{e}^f$, which is projected to replace the "object" placeholder in the task prompt:

    $\boldsymbol{e}^{\text{prompt}} = \boldsymbol{e}^{\text{task}}[\text{object}] \leftarrow \mathcal{P}_\psi(\boldsymbol{e}^f)$

   Combined with task tokens, this enables flexible switching between removal and insertion.

3. **Effect Consistency (EC) Loss**: Exploits the fact that removal and insertion, as mutually inverse tasks, share the same affected region. Cross-attention maps from each DiT block in both branches are collected, aggregated via max pooling and a mapper to obtain soft region estimates $f^{\text{rm}}, f^{\text{in}}$, and aligned with a difference-map prior $f^{\text{diff}}$:

    $\mathcal{L}_{\text{EC}} = \mathrm{KL}(f^{\text{diff}} \| f^{\text{rm}}) + \mathrm{KL}(f^{\text{diff}} \| f^{\text{in}})$

   Here $f^{\text{diff}}$ is derived from the normalized distribution of frame differences between videos with and without the object, preserving intensity variation information for illumination and shadows.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{denoise}}^{\text{remove}} + \mathcal{L}_{\text{denoise}}^{\text{insert}} + \lambda \mathcal{L}_{\text{EC}}$$

- Base model: Wan 2.1 + LoRA (rank=256)
- Input resolution $832 \times 480$, with 81 consecutive frames sampled randomly
- Trained for 120K iterations, batch size 8, on 8×H100 GPUs, learning rate $1 \times 10^{-5}$
- Inference with 50 denoising steps

## Key Experimental Results

### Main Results

| Dataset | Metric | EffectErase | ROSE | MinMax-Remover | Gain (vs. ROSE) |
|--------|------|-------------|------|----------------|--------------|
| ROSE-Benchmark | PSNR↑ | **32.161** | 31.122 | 26.770 | +1.04 |
| ROSE-Benchmark | SSIM↑ | **0.948** | 0.917 | 0.905 | +0.031 |
| ROSE-Benchmark | LPIPS↓ | **0.039** | 0.077 | 0.099 | -0.038 |
| ROSE-Benchmark | FVD↓ | **55.578** | 72.177 | 137.840 | -16.6 |
| VOR-Eval | PSNR↑ | **23.750** | 22.966 | 21.963 | +0.784 |
| VOR-Eval | SSIM↑ | **0.806** | 0.792 | 0.802 | +0.014 |
| VOR-Wild | QScore↑ | **9.280** | 9.240 | 8.984 | +0.040 |
| VOR-Wild | User↑ | **7.20** | 6.38 | 5.90 | +0.82 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ | Note |
|------|-------|-------|--------|------|------|
| Real only | 20.409 | 0.720 | 0.243 | 368.664 | Baseline |
| +EC loss | 21.020 | 0.737 | 0.224 | 354.545 | Consistency loss |
| +EC+TARG | 23.101 | 0.780 | 0.193 | 349.094 | Region guidance |
| +EC+TARG+Syn | **23.750** | **0.806** | **0.170** | **342.871** | Full model |

### Key Findings

- TARG contributes most significantly (SSIM improves from 0.737 to 0.780), demonstrating that spatiotemporal association modeling is critical for side effect localization.
- Adding synthetic data improves generalization substantially, reducing LPIPS from 0.193 to 0.170.
- The model can switch to the object insertion task without additional training, generating realistic shadows and reflections.

## Highlights & Insights

- **Novel problem formulation**: Video object removal is reframed as "effect erasing," with a systematic taxonomy of five categories of visual side effects.
- **Elegant dual-learning design**: Removal and insertion are inverse operations sharing the same affected region, naturally forming complementary supervision.
- **High value of the VOR dataset**: 60K paired videos, 145 hours, real-and-synthetic hybrid — the largest dataset in this domain to date.
- Soft difference-map priors retain more intensity information than binary masks.

## Limitations & Future Work

- Recovery quality under extreme occlusion (e.g., large foreground objects covering substantial background regions) remains limited.
- The cost of capturing real paired data is high, making full scene coverage difficult.
- LoRA fine-tuning may constrain generalization to entirely novel effect types.

## Related Work & Insights

- The key differences from ROSE are threefold: (1) joint learning vs. single-task learning; (2) attention-based spatiotemporal association modeling vs. difference-mask prediction; (3) substantially larger data scale and diversity.
- Insight: The dual-learning paradigm can be generalized to other inverse task pairs (e.g., super-resolution/downsampling, colorization/grayscale conversion).

## Rating

- Novelty: ⭐⭐⭐⭐ — Dual-learning with effect consistency is a novel design; the VOR dataset fills a clear gap in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, user study, complete ablation, and insertion task demonstration.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated with rich visual illustrations.
- Value: ⭐⭐⭐⭐⭐ — The VOR dataset and effect-aware removal framework offer significant contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)

</div>

<!-- RELATED:END -->
