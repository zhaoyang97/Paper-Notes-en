---
title: >-
  [Paper Note] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography
description: >-
  [ICCV 2025][LLM Evaluation][metalens] This paper proposes DMDiff, a framework that leverages the natural image priors of pretrained diffusion models. Through a positive/neutral/negative tripath multi-prompt diffusion strategy and a Spatially-Varying Degradation-Aware (SVDA) attention module, DMDiff achieves high-fidelity tunable image reconstruction for millimeter-scale metalens cameras, surpassing existing methods across multiple metrics.
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "metalens"
  - "diffusion model"
  - "image restoration"
  - "computational imaging"
  - "LoRA"
date: 2026-05-08
content_hash: 45ece51fa6282663
---

# Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography

**Conference**: ICCV 2025
**arXiv**: [2506.22753](https://arxiv.org/abs/2506.22753)  
**Code**: [https://dmdiff.github.io/](https://dmdiff.github.io/)  
**Area**: LLM Evaluation
**Keywords**: metalens, diffusion model, image restoration, computational imaging, LoRA

## TL;DR
This paper proposes DMDiff, a framework that leverages the natural image priors of pretrained diffusion models. Through a positive/neutral/negative tripath multi-prompt diffusion strategy and a Spatially-Varying Degradation-Aware (SVDA) attention module, DMDiff achieves high-fidelity tunable image reconstruction for millimeter-scale metalens cameras, surpassing existing methods across multiple metrics.

## Background & Motivation
Metalenses hold great promise as ultra-compact imaging systems, yet they face severe challenges from complex optical degradations. Existing approaches either rely on precise optical calibration (difficult to obtain), require large-scale paired datasets (hard to collect), or employ deep learning methods that lack control over the inference process, resulting in hallucination artifacts. The root cause lies in: how to effectively recover spatially-varying metalens degradations using pretrained large-model priors in the absence of large-scale training data, while controlling the generative process to avoid hallucinations. This paper proposes leveraging pretrained diffusion model priors as a substitute for large datasets, employing a multipath prompt strategy to balance detail generation and structural fidelity, and designing a tunable decoder to control reconstruction quality.

## Method

### Overall Architecture
DMDiff is built upon SD-Turbo (a distilled version of Stable Diffusion), comprising a VAE encoder, a latent diffusion UNet, a VAE decoder, and the SVDA module. Metalens-captured input images are encoded into the latent space, where the UNet performs single-step denoising ($k=1$). Guided by text prompts and degradation cues from the SVDA module, the framework produces high-quality reconstructed images. LoRA is applied for efficient fine-tuning of the encoder and UNet.

### Key Designs
1. **Spatially-Varying Degradation-Aware (SVDA) Attention Module**:

    - Function: Quantifies spatially-varying degradations introduced by the metalens and sensor to guide the LoRA fine-tuning process.
    - Mechanism: Combines two degradation metrics—PSF-based FWHM (optical aberration) and MUSIQ-based no-reference image quality assessment (sensor noise). The image is divided into $n \times n$ patches; FWHM and NR-IQA scores are computed per patch, and an attention network generates an $r \times r$ attention matrix $Q$ embedded into the LoRA process: $W^* = W + AQB$
    - Design Motivation: Metalens degradations are spatially varying and cannot be handled by conventional uniform degradation assumptions. Accurate PSF calibration is difficult and susceptible to fabrication errors; both optical and electronic sensor degradation sources must be considered simultaneously.

2. **Multipath Diffusion Training**:

    - Function: Learns distinct objectives via positive, neutral, and negative pathways.
    - Mechanism: The positive path (degraded input → high-quality GT) learns high-frequency detail generation; the neutral path (degraded input → low-pass filtered GT) learns structural fidelity; the negative path (GT input → degraded output) learns metalens degradation patterns and generates pseudo data pairs to augment the training set. The three paths are randomly selected with probabilities $M \sim \text{Cat}(p_1, p_2, p_3)$.
    - Design Motivation: Although diffusion models can generate realistic details, they are prone to hallucinations. The neutral path preserves structural accuracy while the negative path learns degradation characteristics for suppression. The three paths jointly balance perceptual quality and reconstruction fidelity.

3. **On-the-Fly Tunable Decoder**:

    - Function: Dynamically adjusts reconstruction results at inference time between perceptual quality and objective fidelity.
    - Mechanism: Latent codes $z_{pos}$ and $z_{neu}$ are obtained from the positive and neutral paths respectively, then blended via a tunable parameter $\alpha$ before decoding: $I^* = D(\alpha \cdot z_{pos} + (1-\alpha) \cdot z_{neu})$
    - Design Motivation: Different application scenarios demand different reconstruction quality trade-offs. Larger $\alpha$ yields better perceptual quality but may introduce excessive detail, while smaller $\alpha$ provides higher fidelity.

### Loss & Training
The training loss is a weighted combination of L2 loss and LPIPS perceptual loss: $L = L_2 + \lambda \cdot L_{\text{LPIPS}}$, where $\lambda = 2.5$. Training is conducted on 4× A100 80G GPUs for two days with a batch size of 16. The number of patches in SVDA is $n=7$.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | MUSIQ↑ | CLIP-IQA↑ |
|--------|-------|-------|--------|--------|--------|-----------|
| Wiener deconv | 16.06 | 0.5727 | 0.6706 | 0.4393 | 17.41 | 0.2681 |
| Neural nano-optics | 29.25 | 0.8624 | 0.2001 | 0.1765 | 37.26 | 0.2746 |
| SwinIR | 29.46 | 0.8786 | 0.2462 | 0.2111 | 36.86 | 0.3046 |
| SeeSR-s50 | 23.95 | 0.8340 | 0.2315 | 0.1673 | 44.87 | 0.3913 |
| OSEDiff-s1 | 19.69 | 0.8224 | 0.2643 | 0.1868 | 34.52 | 0.3761 |
| **Ours-s1-α0.75** | **30.31** | 0.8731 | 0.1705 | 0.1499 | 44.48 | 0.3869 |
| **Ours-s1-α1.05** | 29.75 | 0.8598 | **0.1485** | **0.1356** | **51.85** | **0.4460** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | MANIQA↑ | MUSIQ↑ | Note |
|---------------|-------|-------|--------|---------|--------|------|
| Base (no modules) | 17.12 | 0.7685 | 0.3455 | 0.2332 | 38.27 | Naive LoRA fine-tuning fails to restore |
| w/o FWHM | 26.62 | 0.8414 | 0.1869 | 0.2966 | 50.55 | Removes optical degradation modeling |
| w/o Neg prompt | 28.21 | 0.8571 | 0.1953 | 0.2587 | 44.15 | Removes negative-path degradation learning |
| **Ours-α1** | **29.89** | **0.8633** | **0.1504** | **0.3078** | **50.63** | Full method |

| α | PSNR↑ | LPIPS↓ | MUSIQ↑ | Note |
|---|-------|--------|--------|------|
| 0 | 30.06 | 0.2715 | 31.24 | Pure neutral path; faithful but lacks detail |
| 0.5 | 30.39 | 0.2039 | 38.96 | Balanced |
| 0.7 | 30.31 | 0.1705 | 44.48 | Good balance point |
| 1.0 | 29.89 | 0.1504 | 50.63 | Stronger perceptual quality |
| 1.05 | 29.75 | 0.1485 | 51.85 | Best perceptual quality |

### Key Findings
- Non-diffusion methods (e.g., SwinIR) produce blurry images lacking high-frequency details but preserve accurate tone and low-frequency structure.
- Diffusion-based methods generate rich details but lack degradation modeling, leading to color tone errors and incorrect details.
- The proposed method maintains high performance at image boundary regions (where degradation is most severe), while other methods degrade noticeably at the borders.
- Pseudo data generated by the negative path effectively augments the training set and improves generalization.

## Highlights & Insights
- The integration of SVDA degradation quantification with LoRA fine-tuning is an elegant design that circumvents the need for precise PSF calibration.
- The tripath design effectively addresses the inherent hallucination problem of diffusion models.
- The tunable decoder enables rapid generation of images at varying diffusion strengths without re-running inference.
- A 1×1×1 mm³ MetaCamera was constructed for real hardware validation, demonstrating strong engineering completeness.

## Limitations & Future Work
- Training data still requires monitor-camera aligned paired data collection, though pseudo data augmentation alleviates this issue.
- Single-step diffusion may still be insufficient in severely degraded regions; multi-step inference could further improve quality.
- PSF in SVDA still relies on simulated parameters; fabrication errors may cause deviations in the actual PSF.
- The chromatic dispersion of metalenses poses challenges for semantics-based text prompts in diffusion models; this paper sidesteps the issue by using only quality-descriptive prompts.
- The current framework is optimized for a single metalens design; transferring to different designs may require re-fine-tuning.
- The sensor resolution is only 400×400 pixels; performance at higher resolutions remains to be verified.

## Related Work & Insights
- Building upon single-step diffusion ideas from OSEDiff/S3Diff, this work introduces degradation modeling to adapt the approach to computational imaging scenarios.
- The degradation-aware attention design in SVDA is generalizable to other image restoration tasks with spatially-varying degradations (e.g., wide-angle lenses, endoscopic imaging).
- The multipath diffusion training strategy has broad applicability for controlling the generative quality of diffusion models.
- Using image quality descriptors rather than scene content as text prompts elegantly avoids interference from metalens chromatic dispersion on semantic prompts.
- The 1 mm³ ultra-compact MetaCamera design provides a hardware reference for biomedical implantable imaging and AR/VR miniaturization.
- The negative-path pseudo data generation method is extensible to other computational imaging tasks that lack paired training data.

## Rating
- Novelty: ⭐⭐⭐⭐ The complete design integrating multipath diffusion, SVDA degradation modeling, and tunable decoding demonstrates high innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation on synthetic and real scenes with comprehensive ablations; however, comparisons with more computational imaging methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, and professionally designed figures and tables.
- Value: ⭐⭐⭐⭐ Significant contribution to ultra-compact computational imaging; the methodology is transferable to other degradation restoration tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[CVPR 2025\] Erase Diffusion: Empowering Object Removal Through Calibrating Diffusion Pathways (EraDiff)](../../CVPR2025/llm_evaluation/erase_diffusion_empowering_object_removal_through_calibrating_diffusion_pathways.md)
- [\[ICLR 2026\] CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density](../../ICLR2026/llm_evaluation/cogniload_a_synthetic_natural_language_reasoning_benchmark_with_tunable_length_i.md)
- [\[ICLR 2026\] ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs](../../ICLR2026/llm_evaluation/parallelbench_understanding_the_trade-offs_of_parallel_decoding_in_diffusion_llm.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](../../ACL2026/llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
