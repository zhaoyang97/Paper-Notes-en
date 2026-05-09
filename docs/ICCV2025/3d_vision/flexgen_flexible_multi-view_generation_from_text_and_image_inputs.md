---
title: >-
  [Paper Note] FlexGen: Flexible Multi-View Generation from Text and Image Inputs
description: >-
  [ICCV 2025][3D Vision][multi-view generation] This paper proposes FlexGen, a flexible multi-view image generation framework that leverages GPT-4V to produce 3D-aware text annotations from tiled orthographic views and introduces an adaptive dual-control module to support single-image, text-only, or joint image-text conditioning for generating consistent multi-view images, enabling capabilities such as unseen-region completion, material editing, and texture control.
tags:
  - ICCV 2025
  - 3D Vision
  - multi-view generation
  - 3D-aware text annotation
  - controllable generation
  - diffusion models
  - material editing
date: 2026-05-08
content_hash: e1940cc9fb24c4ca
---

# FlexGen: Flexible Multi-View Generation from Text and Image Inputs

**Conference**: ICCV 2025
**arXiv**: [2410.10745](https://arxiv.org/abs/2410.10745)
**Code**: [https://xxu068.github.io/flexgen.github.io/](https://xxu068.github.io/flexgen.github.io/) (project page available)
**Area**: 3D Vision
**Keywords**: multi-view generation, 3D-aware text annotation, controllable generation, diffusion models, material editing

## TL;DR

This paper proposes FlexGen, a flexible multi-view image generation framework that leverages GPT-4V to produce 3D-aware text annotations from tiled orthographic views and introduces an adaptive dual-control module to support single-image, text-only, or joint image-text conditioning for generating consistent multi-view images, enabling capabilities such as unseen-region completion, material editing, and texture control.

## Background & Motivation

Multi-view diffusion models (e.g., Zero123++, SyncDreamer, Wonder3D) have demonstrated the potential to generate 3D-consistent multi-view images by leveraging pretrained 2D diffusion models, providing a viable path for rapid 3D content creation. Nevertheless, **controllable generation** remains severely underexplored in this setting.

**Limitations of Prior Work**:
- *Insufficient single-view conditioning*: Most methods condition only on a single image, causing the unseen regions of an object (e.g., the back) to be naively copied from the front view, lacking any 3D-aware guidance signal.
- *Unfriendly 3D guidance*: Coin3D employs primitive shapes as 3D guidance and Clay uses sparse point clouds and 3D bounding boxes, both of which are impractical for general users.
- *Text annotations lacking 3D information*: Cap3D generates per-view captions with BLIP-2 and aggregates them via GPT-4, but the results tend to be high-level summaries that lack local detail and 3D spatial relationships. This stems from two issues: BLIP-2 produces only global descriptions, and single-view information is both redundant and incomplete.
- *Single-modality control*: Instant3D supports text-to-3D but only text conditioning, offering insufficient flexibility.

**Key Challenge**: Text is the most natural control modality for users and can convey rich semantic and spatial relational information. However, how to generate text annotations with sufficient 3D-aware information for 3D objects, and how to effectively fuse image and text control signals within a multi-view diffusion model, remain open problems.

**Key Insight**: (1) Exploit GPT-4V's strong visual reasoning ability to generate global-local 3D-aware text annotations from tiled four-view orthographic images; (2) Design an adaptive dual-control module that enables joint image-text control while supporting three inference modes—image-only, text-only, and joint—via a condition switcher.

## Method

### Overall Architecture

FlexGen is built upon Stable Diffusion 2.1. It accepts a single-view image and/or a text prompt as input and generates four orthographic views (front, left, back, right) arranged in a $2\times2$ layout at $512\times512$ resolution with a fixed elevation of $5°$. The framework comprises three core components: 3D-aware text annotation generation, an adaptive dual-control module, and a flexible training and inference strategy.

### Key Designs

1. **3D-Aware Caption Annotation**

    - *Function*: Generate global-local text descriptions rich in 3D spatial relationship information for 3D objects in the Objaverse dataset.
    - *Mechanism*: Dataset construction proceeds in three steps:
        - **Rendering**: Each 3D object is rendered into four orthographic views (front/left/back/right) at $512\times512$ and assembled into a $2\times2$ tiled image.
        - **Annotation**: The tiled image is fed to GPT-4V, which leverages its cross-view reasoning ability to simultaneously generate a global description (overall attributes and 3D spatial relationships among parts) and local descriptions (color, pose, texture, etc. of individual parts).
        - **Merging**: The global and local descriptions are combined into a "global-local text description." During training, a random subset of local descriptions is sampled to simulate user behavior.
    - Material descriptions (e.g., metallic, roughness) are additionally included, annotated using the actual material parameters from Blender rendering.
    - *Design Motivation*: Unlike Cap3D's per-view annotation and aggregation pipeline, GPT-4V observing all four orthographic views simultaneously can reason about 3D spatial relationships (e.g., "there is a handle on the left side but not on the right"), yielding significantly higher annotation quality.

2. **Adaptive Dual-Control Module**

    - *Function*: Fuse image and text control signals simultaneously during the denoising process of the diffusion model.
    - *Mechanism*:
        - Based on the Reference Attention mechanism—an additional reference image is processed by the denoising UNet and its self-attention key/value matrices are appended to the corresponding attention layers of the target branch.
        - *Novelty*: Text information is injected into the reference attention. The user's text is encoded by a CLIP encoder to obtain per-token embeddings $E \in \mathbb{R}^{L \times D}$, which interact with reference image features via cross-attention.
        - After this interaction, the key/value matrices from the dual-control module are appended to the corresponding layers of the denoising UNet.
    - *Design Motivation*: Single-modality control (image-only or text-only) cannot simultaneously achieve high fidelity and semantic controllability. Reference attention provides image fidelity, while cross-attention injects text for semantic control; the two are fused at the attention level.

3. **Condition Switcher and Flexible Training Strategy**

    - *Function*: Support three inference modes—joint image-text, image-only, and text-only.
    - *Mechanism*: During training, input conditions are randomly dropped according to configurable probabilities:
        - Joint image-text: 0.3
        - Image-only: 0.3
        - Text-only: 0.3
        - Neither: 0.1
        - Missing text is replaced with an empty string; missing images are replaced with black images.
    - *Design Motivation*: By training with dropout-style random condition masking, the model can flexibly adapt to different user input scenarios at inference time (image only, text only, or both).

### Loss & Training

- Standard diffusion denoising loss.
- Built on SD 2.1; trained on 8 × A800 80 GB GPUs for 10 days, 180K iterations, batch size 32.
- Adam optimizer, learning rate $1\times10^{-5}$.
- Inference uses DDIM sampling with 75 steps.
- Training data: 147K high-quality (textured, sufficient polygon count) 3D objects curated from Objaverse.
- For each object, 24 target-view images are rendered (elevation $5°$, uniformly distributed azimuths); the conditioning view is sampled randomly (elevation $-30°$ to $30°$).

## Key Experimental Results

### Main Results

Novel view synthesis and sparse-view 3D reconstruction on the GSO dataset:

| Method | PSNR↑ | LPIPS↓ | CD↓ | FS@0.1↑ |
|--------|-------|--------|-----|---------|
| SyncDreamer | 17.66 | 0.21 | 0.126 | 0.833 |
| Era3D | 18.52 | 0.19 | 0.245 | 0.713 |
| Zero123++ | 18.83 | 0.16 | 0.087 | 0.910 |
| Ours (w/o caption) | 21.12 | 0.14 | 0.078 | 0.921 |
| **FlexGen (Ours)** | **22.31** | **0.12** | **0.076** | **0.928** |

Text-to-multi-view comparison (300 GSO samples):

| Method | FID↓ | IS↑ | CLIP↑ |
|--------|------|-----|-------|
| MVDream | 44.42 | 12.98±1.22 | 0.79 |
| **FlexGen (Ours)** | **35.56** | **13.41±0.87** | **0.83** |
| Ground truth | N/A | 13.81±1.40 | 0.89 |

### Ablation Study

| Configuration | PSNR | LPIPS | Note |
|---------------|------|-------|------|
| Ours (w/o caption) | 21.12 | 0.14 | Image control only, no text annotation |
| Ours (Cap3D caption) | ~20.5 | ~0.15 | Cap3D annotations, lacking 3D-aware information |
| **Ours (full)** | **22.31** | **0.12** | GPT-4V 3D-aware annotations + dual-control module |

### Key Findings

- Incorporating 3D-aware text annotations improves PSNR from 21.12 to 22.31 (+1.19), demonstrating that text control significantly aids unseen-region completion.
- FlexGen's FID and CLIP scores approach ground-truth levels (FID 35.56, CLIP 0.83 vs. 0.89), substantially outperforming MVDream (FID 44.42, CLIP 0.79).
- Multi-view images generated under joint image-text conditioning yield better CD and FS scores in downstream 3D reconstruction compared to image-only methods.
- Material properties (e.g., "high metallic, low roughness") can be directly controlled by modifying the material descriptions in the text prompt.

## Highlights & Insights

- Using GPT-4V to observe a tiled four-view orthographic image for generating 3D-aware annotations is an elegant approach to extracting structured spatial priors from large vision-language models.
- The adaptive dual-control module enables thorough interaction between image and text information at the attention level, which is superior to naive concatenation approaches.
- The condition switcher training strategy allows a single model to support three inference modes simultaneously, improving practical usability.
- Material-controllable generation (metallic/roughness) is a valuable contribution with clear utility for 3D asset creation.

## Limitations & Future Work

- The model's ability to parse complex user instructions is limited, likely due to the relatively modest training data scale (147K objects).
- GPT-4V annotation requires API calls, making dataset construction costly and dependent on a closed-source model.
- Only four orthographic views (2×2 layout) are generated, which is insufficient for applications requiring more views or arbitrary viewpoint control.
- The fixed $5°$ elevation restricts viewpoint diversity and may be unsuitable for certain applications (e.g., top-down or bottom-up perspectives).
- 3D reconstruction quality depends on the downstream method (InstantMesh), leaving room for end-to-end quality improvement.

## Related Work & Insights

- **Zero123++/SyncDreamer/Wonder3D**: Baseline single-image-to-multi-view methods; image-only conditioning is insufficient for controllable generation.
- **MVDream**: Text-to-multi-view method, but lacks fine-grained 3D-aware text control.
- **Cap3D**: A pioneering work on text annotation for 3D objects, but per-view independent annotation followed by aggregation fails to capture 3D spatial relationships.
- **ControlNet**: A representative method for controllable 2D generation; FlexGen extends analogous ideas to multi-view 3D generation.
- **Insight**: Leveraging the visual reasoning capabilities of large models to provide structured priors for 3D tasks (e.g., 3D-aware text) is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ (The GPT-4V annotation strategy and dual-control module are innovative, though the overall framework builds on well-established components.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation across NVS, text-to-multiview, and 3D reconstruction; comparisons with more controllable generation baselines would strengthen the study.)
- Writing Quality: ⭐⭐⭐⭐ (Method descriptions are clear and visualizations are rich; some ablation comparisons could be more detailed.)
- Value: ⭐⭐⭐⭐ (Multi-modal controllable multi-view generation addresses a genuine practical need and advances 3D content creation.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)

</div>

<!-- RELATED:END -->
