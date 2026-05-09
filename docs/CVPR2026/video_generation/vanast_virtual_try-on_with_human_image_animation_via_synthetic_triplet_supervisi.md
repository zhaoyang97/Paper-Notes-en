---
title: >-
  [Paper Note] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision
description: >-
  [CVPR 2026][Video Generation][Virtual Try-On] Vanast proposes a unified framework that simultaneously performs garment transfer and human image animation within a single stage via a Dual Module architecture (HAM + GTM) and a three-stage synthetic data construction pipeline. On the Internet dataset, it achieves a PSNR of 17.95 dB (+5.5 dB over the best two-stage baseline) and an LPIPS of 0.237.
tags:
  - CVPR 2026
  - Video Generation
  - Virtual Try-On
  - Human Image Animation
  - Synthetic Triplet
  - Dual Module
  - Video Diffusion
date: 2026-05-08
content_hash: 016ed2c33d8c88fe
---

# Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision

**Conference**: CVPR 2026
**arXiv**: [2604.04934](https://arxiv.org/abs/2604.04934)
**Code**: [https://hyunsoocha.github.io/vanast/](https://hyunsoocha.github.io/vanast/)
**Area**: Video Generation
**Keywords**: Virtual Try-On, Human Image Animation, Synthetic Triplet, Dual Module, Video Diffusion

## TL;DR

Vanast proposes a unified framework that simultaneously performs garment transfer and human image animation within a single stage via a Dual Module architecture (HAM + GTM) and a three-stage synthetic data construction pipeline. On the Internet dataset, it achieves a PSNR of 17.95 dB (+5.5 dB over the best two-stage baseline) and an LPIPS of 0.237.

## Background & Motivation

1. **Background**: Virtual try-on (VTON) and human image animation are core demands in e-commerce and social media. Existing approaches handle the two tasks in a two-stage pipeline — first generating a static dressed image via CatVTON/OmniTry, then animating it with StableAnimator.
2. **Limitations of Prior Work**: Two-stage methods suffer from severe error accumulation: (1) identity drift — the animation stage loses identity information established during garment transfer; (2) garment distortion — clothing details deform during animation; (3) front-back inconsistency — the appearance of garments breaks across opposite views.
3. **Key Challenge**: A single-stage unified model must simultaneously learn two fundamentally different transformations — garment transfer and animation — yet paired triplet training data (person + garment + motion sequence) is scarce.
4. **Goal**: Construct a large-scale triplet dataset and train a single-stage unified model.
5. **Key Insight**: Compensate for the scarcity of real triplet data through synthetic data, built via three strategies: diffusion-based inpainting, in-the-wild video garment extraction, and studio capture.
6. **Core Idea**: The Dual Module architecture adds a Human Animation Module (HAM) and a Garment Transfer Module (GTM) in parallel on top of a frozen video DiT backbone, achieving unified generation through weighted residual connections.

## Method

### Overall Architecture

Input (person image + garment image + pose-guided video) → VAE encoding to latent space → frozen DiT backbone + HAM for human pose conditioning + GTM for garment conditioning → weighted fusion $h_{l+1} = B^{T2V}_l(h_l) + \alpha \cdot B^{HAM}_l(h_l) + \beta \cdot B^{GTM}_l(h_l)$ ($\alpha=\beta=0.5$) → VAE decoding to generate video.

### Key Designs

1. **Three-Stage Data Construction Pipeline**

   - **Function**: Construct large-scale triplet supervision (dressed image, garment, video) from scratch.
   - **Mechanism**: Stage 1 synthesizes garment-replaced images via FLUX diffusion inpainting; Stage 2 extracts garments from in-the-wild videos and generates corresponding person images via diffusion; Stage 3 captures multi-garment videos in a studio setting. The pipeline yields 9,135 video clips in total.
   - **Design Motivation**: Real triplet data is extremely scarce — video of the same person performing identical motions in different garments is virtually nonexistent in natural settings.

2. **Dual Module Architecture (HAM + GTM)**

   - **Function**: Inject both human animation and garment transfer conditions in parallel onto the frozen backbone.
   - **Mechanism**: HAM and GTM are lightweight adapter modules; only these two branches are trained while the DiT backbone remains frozen. Each module independently processes its respective conditioning signal, then contributes to the main feature stream via weighted residual addition.
   - **Design Motivation**: Decoupling the two highly heterogeneous conditioning signals into independent modules prevents mutual interference. Ablations show that Dual Module outperforms Single Module by 17.8 FID points (91.05 vs. 108.84).

3. **Zero-Shot Garment Interpolation**

   - **Function**: Enable smooth blending between two garments without retraining.
   - **Mechanism**: The outputs of two GTM branches are weighted by $\gamma$: $h_{l+1} = ... + \gamma \cdot B^{GTM}_l(h_l; G_A) + (1-\gamma) \cdot B^{GTM}_l(h_l; G_B)$, where $\gamma \in [0,1]$ controls the mixing ratio.
   - **Design Motivation**: The modular GTM design naturally supports linear interpolation over multiple garment conditions at no additional training cost.

### Loss & Training

Standard diffusion denoising loss (v-prediction), optimizing only the HAM and GTM parameters while the DiT backbone is completely frozen. Training data consists of 9,135 video clips (3–10 seconds); evaluation is conducted on 80 samples (Internet set) and 50 samples (ViViD set).

## Key Experimental Results

### Main Results

| Method | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|--------|-----|-------|-------|--------|------|
| CatVTON+StableAnimator | 0.1242 | 14.56 | 0.765 | 0.327 | 132.09 |
| OmniTry+StableAnimator | 0.1227 | 14.53 | 0.767 | 0.318 | 121.04 |
| VACE (1-stage) | 0.1453 | 13.09 | 0.689 | 0.405 | 115.40 |
| **Vanast** | **0.0719** | **17.95** | **0.755** | **0.237** | **91.05** |

### Ablation Study

| Configuration | L1↓ | PSNR↑ | FID↓ | VFID↓ | Note |
|---------------|-----|-------|------|-------|------|
| Single Module | 0.1162 | 14.28 | 108.84 | 39.64 | Poor single-module performance |
| Backbone-LoRA | 0.1359 | 13.17 | 120.97 | 42.47 | Fine-tuning backbone degrades results |
| w/o SynthHuman | 0.1163 | 14.62 | 110.76 | 38.89 | Synthetic data is critical |
| **Full model** | **0.1069** | **14.74** | **104.59** | **35.60** | Complete model |

### Key Findings

- **Dual Module vs. Single Module**: FID drops from 108.84 to 91.05, validating the necessity of condition decoupling.
- **Frozen backbone vs. LoRA fine-tuning**: Freezing yields better results (FID 91.05 vs. 120.97), likely because LoRA disrupts the pretrained video prior.
- SynthHuman data contributes a 6-point FID improvement, confirming the effectiveness of the synthetic data strategy.
- VFID_ResNeXt is only 0.39 (vs. baselines at 1.69–5.86), demonstrating a substantial lead in temporal consistency.

## Highlights & Insights

- **Engineering elegance of single-stage unification**: Eliminating the error accumulation inherent in two-stage pipelines, the model generates dressed animation videos end-to-end in a single pass.
- **Synthetic data bridging real data gaps**: The three-stage data construction strategy is transferable to other video generation tasks that lack paired training data.
- **Zero-shot interpolation capability**: The modular design naturally confers zero-shot garment mixing ability, offering high commercial value.

## Limitations & Future Work

- Training data is limited to 9,135 video clips, with restricted coverage of garment types.
- Performance may degrade on uncommon garment categories (e.g., jumpsuits, kimonos).
- The quality ceiling of synthetic data is bounded by the capabilities of FLUX inpainting and the VLM used.
- Future work could extend to multi-person scenarios and unified transfer of accessories (hats, bags, etc.).

## Related Work & Insights

- **vs. CatVTON/OmniTry + StableAnimator**: Two-stage baselines achieve FID 121–132; Vanast achieves 91.05. The gap is primarily attributable to error accumulation in the pipeline.
- **vs. VACE**: Although also single-stage, VACE's VFID_ResNeXt of 5.86 is far higher than Vanast's 0.39, indicating a substantial disparity in temporal consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The Dual Module design and synthetic triplet data strategy are genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple baselines, and ablations are provided, though the test set size is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and well-organized.
- **Value**: ⭐⭐⭐⭐ Direct application value for e-commerce virtual try-on.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](../../ICCV2025/video_generation/multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)

<!-- RELATED:END -->
