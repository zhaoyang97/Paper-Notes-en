---
title: >-
  [Paper Note] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods
description: >-
  [NeurIPS 2025][3D Vision][novel view synthesis] This paper proposes NerfBaselines, an evaluation framework that addresses unfair comparisons in novel view synthesis (NVS) caused by inconsistent evaluation protocols. By wrapping original method code with a unified API and enforcing environment isolation, the framework ensures that each method's behavior exactly matches its original release. Experiments reveal that seemingly minor protocol differences—such as image resizing strategies and background colors—can significantly alter method rankings.
tags:
  - NeurIPS 2025
  - 3D Vision
  - novel view synthesis
  - evaluation framework
  - NeRF
  - 3D Gaussian Splatting
  - benchmark
date: 2026-05-08
content_hash: 2c390884d3e9f138
---

# NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods

**Conference**: NeurIPS 2025  
**arXiv**: [2406.17345](https://arxiv.org/abs/2406.17345)  
**Code**: [https://nerfbaselines.github.io](https://nerfbaselines.github.io)  
**Area**: 3D Vision  
**Keywords**: novel view synthesis, evaluation framework, NeRF, 3D Gaussian Splatting, benchmark

## TL;DR
This paper proposes NerfBaselines, an evaluation framework that addresses unfair comparisons in novel view synthesis (NVS) caused by inconsistent evaluation protocols. By wrapping original method code with a unified API and enforcing environment isolation, the framework ensures that each method's behavior exactly matches its original release. Experiments reveal that seemingly minor protocol differences—such as image resizing strategies and background colors—can significantly alter method rankings.

## Background & Motivation
Novel view synthesis is a fundamental problem in computer vision. With the rapid development of NeRF and 3DGS, multiple arXiv papers are published daily. However, the field faces a critical yet overlooked issue: **the absence of a unified evaluation protocol**.

Existing methods exhibit numerous inconsistencies in evaluation: different image resizing strategies (manual resizing vs. using pre-scaled JPEGs), varying image resolutions, optional undistortion, choice of VGG vs. AlexNet for LPIPS, background color for images with transparency (white vs. black), whether pixel values are rounded to uint8, and normalization procedures before metric computation. These seemingly minor differences can artificially inflate the reported performance of certain methods, rendering quantitative comparisons in the literature unreliable.

A representative example: Gsplat improved its PSNR ranking by 3 positions on the Mip-NeRF 360 dataset simply by changing the image resizing strategy—without any algorithmic improvement. This means researchers can achieve better numbers by "tuning the protocol" rather than "doing research," which severely undermines the scientific integrity of the field.

Existing development frameworks such as NerfStudio provide unified data loading and evaluation utilities, but their re-implemented methods often exhibit performance gaps relative to original implementations (e.g., TensoRF's PSNR drops from 36.46 to 33.09 under NerfStudio). Moreover, integrating a new method requires re-implementing it within the framework, which is labor-intensive and difficult to guarantee consistency.

The core idea of NerfBaselines is: **wrap original code rather than re-implement methods**. Through a lightweight API and environment isolation, the framework ensures each method behaves identically to its originally published version, while enforcing a unified evaluation protocol.

## Method

### Overall Architecture
The NerfBaselines framework consists of four core components: a unified API, environment isolation, an interactive viewer, and a web platform. The framework decouples method implementations from data loading and evaluation code, enabling any method to be evaluated on any dataset under the same protocol.

### Key Designs
1. **Unified API**:

    - Identifies the shared structure of ray-based and rasterization-based methods and defines a unified method class interface.
    - Core functions include `train_iteration` (execute one training step) and `render` (render a single frame).
    - Each method only requires a lightweight wrapper that calls the original code, rather than a full re-implementation.
    - Design Motivation: Minimize integration effort while ensuring behavioral consistency with the original code. This is the fundamental distinction from NerfStudio.

2. **Environment Isolation**:

    - Creates an isolated runtime environment for each method, freezing source code and dependency versions.
    - Supports three isolation levels: Conda, Docker, and Apptainer, accommodating different system requirements.
    - Interacts with isolated environments via inter-process communication (IPC).
    - Design Motivation: Among 19 integrated methods, 12 could not be installed using official instructions—dependency version conflicts and missing packages are the greatest obstacles to long-term reproducibility. Freezing environments ensures methods can still be installed and produce consistent results in the future.

3. **Standardized Evaluation Protocol**:

    - Images are evaluated within the uint8 range to ensure reproducibility.
    - LPIPS defaults to AlexNet (VGG for older datasets), with the version fixed at 0.10.1.
    - Pre-scaled released images are used to avoid discrepancies introduced by different resizing algorithms across platforms and libraries.
    - SSIM parameters are fixed: kernel size=11, $\sigma=1.5$, $k_1=0.01$, $k_2=0.03$.
    - Dataset-specific details: Blender uses white background with VGG LPIPS; Mip-NeRF 360 uses pre-scaled JPEGs, etc.
    - Design Motivation: Each design choice is carefully considered to align with prevailing practices in the literature while eliminating ambiguity.

### Loss & Training
NerfBaselines does not involve training; each method's original training strategy is used as-is. The framework focuses on standardizing the **evaluation phase**.

## Key Experimental Results

### Main Results (Reproducibility Verification)

| Method (Scene) | Paper PSNR | Official Code PSNR | NerfStudio PSNR | NerfBaselines PSNR |
|----------------|-----------|-------------------|----------------|-------------------|
| TensoRF (lego) | 36.46 | 36.54 | 33.09 | **36.49** |
| 3DGS (garden) | 27.41 | 27.39 | 27.17 | **27.34** |
| Instant-NGP (lego) | 36.39 | 36.09 | 15.24 | **35.65** |
| SeaThru-NeRF (Panama) | 27.89 | 27.85 | 31.28 | **27.82** |
| Zip-NeRF (garden) | 28.20 | 28.22 | - | **28.19** |

### Ablation Study (Impact of Protocol Differences)

| Scene / Protocol Variation | Impact | Notes |
|---------------------------|--------|-------|
| Mip-NeRF 360: manual resizing vs. pre-scaled JPEG | PSNR difference of 0.2–0.3 dB | 3DGS method rankings shift; Gsplat improves by 3 positions |
| Blender: white background vs. black background | PSNR difference of 0.3–0.5 dB | GOF drops from 1st to 2nd; Mip-Splatting rises from 2nd to 1st |
| Photo Tourism: half-image vs. full-image appearance optimization | PSNR difference of 1–2 dB | WildGaussians drops from 1st to 3rd |

### Key Findings
- Results from most published methods can be reproduced, suggesting that "malicious protocol tuning" is not currently the dominant issue.
- Performance differences caused by protocol variations can **exceed** differences between methods evaluated under the same protocol—meaning protocol discrepancies can fabricate illusory progress.
- NerfStudio's re-implemented versions sometimes deviate substantially from original implementations (e.g., Instant-NGP on the lego scene diverges by over 20 dB).
- 3DGS-based methods are generally more sensitive to evaluation protocols than NeRF-based methods, possibly due to properties of explicit representations.

## Highlights & Insights
- This is a systems paper of substantial community value, addressing a problem that every researcher recognizes but no one has systematically resolved. The finding that evaluation protocol differences can reverse method rankings alone warrants community-wide attention.
- "Wrapping original code rather than re-implementing" represents the correct engineering philosophy. NerfStudio's re-implementation approach is inherently unable to guarantee consistency, whereas NerfBaselines' wrapper approach minimizes both integration overhead and the risk of errors.
- The fact that 12 out of 19 methods cannot be installed using official instructions is itself a stark indicator of the reproducibility crisis in the field.
- The interactive viewer and camera trajectory editor go beyond numerical comparison, providing tools for qualitative evaluation outside the training trajectory.
- The web platform allows downloading rendered images and checkpoints, enabling independent verification of results—a reflection of open science principles.

## Limitations & Future Work
- While environment isolation resolves compatibility issues, it incurs additional disk and compute overhead; each method's isolated environment may occupy tens of gigabytes.
- Coverage is currently limited primarily to static-scene NVS methods; dynamic scenes and generative 3D methods are less represented.
- The web platform requires sustained maintenance and updates; long-term sustainability is a challenge as new methods continue to emerge.
- There is limited in-depth discussion of the evaluation metrics themselves (e.g., whether PSNR is the best indicator of perceptual quality).
- Control over training hyperparameters is limited—different methods employ different training strategies, and the framework standardizes only evaluation, not training.
- Emerging NVS tasks such as scene editing and text-driven generation are not covered.

## Related Work & Insights
- **vs. NerfStudio**: NerfStudio re-implements methods, often with performance discrepancies; NerfBaselines wraps original code, guaranteeing consistency.
- **vs. SDFStudio**: Suffers from similar issues as NerfStudio and focuses narrowly on SDF-based representations.
- **vs. Independent Per-Method Evaluation**: Independent evaluations lack a unified protocol and yield incomparable results; NerfBaselines provides a common standard.

## Rating
- Novelty: ⭐⭐⭐ — The framework design is not complex, but the systematic analysis of the problem and the revealed insights are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive comparisons spanning 19 methods, multiple datasets, and various protocol variants.
- Writing Quality: ⭐⭐⭐⭐⭐ — Arguments are clear and well-supported by data; every design choice is accompanied by thorough motivation.
- Value: ⭐⭐⭐⭐⭐ — Provides fundamental infrastructure value for the entire NVS community and has the potential to become the field's standard evaluation tool.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Novel View Synthesis from A Few Glimpses via Test-Time Natural Video Completion](novel_view_synthesis_from_a_few_glimpses_via_test-time_natural_video_completion.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](../../ICCV2025/3d_vision/billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

<!-- RELATED:END -->
