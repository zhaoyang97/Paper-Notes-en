---
title: >-
  [Paper Note] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos
description: >-
  [CVPR 2026][Video Generation][Video Manipulation Detection] This work introduces the first activity-level video forgery localization task and the large-scale ActivityForensics benchmark (6K+ forged clips). A grounding-as…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Video Manipulation Detection"
  - "Activity-Level Forgery"
  - "Temporal Localization"
  - "Diffusion Model Feature Regularization"
  - "Video Forensics"
date: 2026-05-08
content_hash: 845dfc94a90a37bc
---

# ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos

**Conference**: CVPR 2026
**arXiv**: [2604.03819](https://arxiv.org/abs/2604.03819)  
**Code**: [https://activityforensics.github.io](https://activityforensics.github.io)  
**Area**: Video Generation
**Keywords**: Video Manipulation Detection, Activity-Level Forgery, Temporal Localization, Diffusion Model Feature Regularization, Video Forensics

## TL;DR
This work introduces the first activity-level video forgery localization task and the large-scale ActivityForensics benchmark (6K+ forged clips). A grounding-assisted automated data construction pipeline is proposed to produce highly realistic activity manipulations, and a baseline method, Temporal Artifact Diffuser (TADiff), is presented to amplify forgery cues via diffusion-based feature regularization.

## Background & Motivation

**Background**: Video manipulation localization aims to identify tampered segments in untrimmed videos. Existing benchmarks (ForgeryNet, Lav-DF, AV-Deepfake1M, TVIL) primarily focus on appearance-level forgeries (face swapping, object removal).

**Limitations of Prior Work**: With the rapid advancement of video generation technologies (Wan, Sora, VACE, etc.), activity-level forgery has emerged as a new threat—modifying human actions to distort event semantics (e.g., replacing a politician's neutral stance with inappropriate behavior). Such forgeries are highly realistic and deceptive, posing serious threats to media authenticity and credibility. Yet no benchmark exists for activity-level forgery localization.

**Key Challenge**: The detection logic for appearance-level and activity-level forgeries is fundamentally different—the former relies on pixel-level texture anomalies, while the latter requires understanding semantic changes in actions and temporal consistency. Directly transferring action localization models to forgery localization leads to over-reliance on semantic information.

**Key Insight**: Construct the first activity-level forgery localization benchmark using video captioning and temporal grounding to automate data construction (avoiding costly manual annotation), while proposing a targeted baseline method.

**Core Idea**: (1) A grounding-assisted automated data pipeline seamlessly embeds manipulated clips into original videos; (2) TADiff suppresses semantic bias via noise perturbation, then amplifies forgery artifact cues through diffusion denoising.

## Method

### Overall Architecture

**Data Construction Pipeline**: Raw video → video captioning + temporal grounding (obtaining activity descriptions and time spans) → LLM-based description modification (semantic manipulation) → video generation/editing models synthesize forged clips → seamless replacement of original clips → precise temporal annotations.

**TADiff Method**: Frame-level feature extraction → ActionFormer multi-scale Transformer encoder → TADiff diffusion-based feature regularization → forgery confidence head + boundary regression head.

### Key Designs

1. **Grounding-Assisted Data Construction**:

    - Function: Automatically generate activity-level forged videos with precise temporal annotations.
    - Mechanism: Video captioning and temporal grounding models automatically localize activity segments; an LLM rewrites descriptions into semantically manipulated versions (e.g., "waving" → "thumbs up"); video generation/editing models (Wan, Scifi, FCVG, Vidu, VACE, LTX) synthesize forged clips and seamlessly fuse them into the original video.
    - Design Motivation: Addresses the high cost of manually constructing activity-level forgery data while ensuring high visual consistency between forged and contextual content.

2. **Temporal Artifact Diffuser (TADiff)**:

    - Function: Inject noise into temporal feature space and denoise to amplify forgery artifact cues.
    - Mechanism:
        - **Forward Process**: Inject Gaussian noise into the feature sequence $x_s = \sqrt{\bar{\alpha}_s} f + \sqrt{1-\bar{\alpha}_s} \epsilon$, perturbing the representation away from the semantic manifold.
        - **Reverse Process**: A lightweight temporal convolutional denoiser (FiLM-conditioned) performs DDIM-style updates $x_{s-1} = \sqrt{\bar{\alpha}_{s-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{s-1}-\sigma_s^2}\hat{\epsilon} + \sigma_s z$.
        - Denoising steps set to 3.
    - Design Motivation: Features in action localization models encode high-level semantics, rendering them insensitive to low-level artifact cues (textural inconsistencies, motion discontinuities) required for forgery detection. The forward diffusion process suppresses semantic bias via noise injection, while the reverse process amplifies forgery-sensitive signals.

3. **Dataset Statistics**:

    - 6 manipulation methods (4 video generation + 2 video editing).
    - 6K+ forged clips with balanced distribution.
    - Over 60% of forged clips occupy less than 30% of total video duration (high localization challenge).
    - Three evaluation settings: in-domain, cross-domain, and open-world.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{reg}$: focal loss (forgery confidence) + smooth L1 loss (boundary regression). End-to-end training with AdamW optimizer, batch_size=16, lr=0.001.

## Key Experimental Results

### Main Results (In-Domain and Open-World)

| Setting | Method | AP@0.75 | AP@0.95 | avg AP | avg AR |
|------|------|---------|---------|--------|--------|
| In-Domain | ActionFormer | 86.29 | 46.79 | 70.67 | 74.31 |
| | UMMAFormer | 87.02 | 48.55 | 71.94 | 75.74 |
| | DiGIT | 78.61 | 44.92 | 64.69 | 70.43 |
| | **TADiff (Ours)** | **87.52** | **56.57** | **75.05** | **77.15** |
| Open-World | ActionFormer | 89.81 | 57.08 | 77.82 | 83.31 |
| | UMMAFormer | 91.13 | 57.57 | 78.79 | 84.15 |
| | **TADiff (Ours)** | **92.35** | **69.06** | **83.64** | **87.92** |

### Ablation Study (Cross-Domain Transfer Between Manipulation Methods)

| Direction | Method | avg AP | avg AR |
|------|------|--------|--------|
| A→B | ActionFormer | 67.18 | 72.14 |
| | **TADiff (Ours)** | **69.63 (+2.45)** | **74.91 (+2.77)** |
| B→A | ActionFormer | 37.14 | 51.03 |
| | **TADiff (Ours)** | **40.89 (+3.75)** | **52.56 (+1.53)** |

### Key Findings
- TADiff yields the most significant improvements at high IoU thresholds (AP@0.95): +9.78 in-domain, +11.98 open-world, indicating that diffusion regularization particularly benefits precise boundary localization.
- The B→A cross-domain direction is substantially harder than A→B (avg AP ~40 vs. ~70), highlighting generalization across manipulation methods as a key challenge.
- The open-world setting (training on mixed manipulation methods) achieves the best performance, demonstrating the benefit of diverse training for generalization.
- DiGIT (an appearance-level forensics method) performs poorly on activity-level forgery, validating the fundamental distinction between activity-level and appearance-level forgery detection.

## Highlights & Insights
- **New Task Definition**: This paper is the first to formally define the activity-level forgery localization task, complementing appearance-level forgery research. As video generation models rapidly advance, the real-world significance of this task continues to grow.
- **Automated Data Pipeline**: The grounding-assisted data construction approach eliminates the high cost of manual annotation while ensuring visual consistency between forged clips and their context, and is readily extensible to additional video generation models.
- **Diffusion-Based Feature Regularization**: The core insight of TADiff—disrupting semantic encoding via noise injection and amplifying artifact signals through denoising—is both elegant and effective, and is transferable to other detection tasks requiring sensitivity to low-level cues.

## Limitations & Future Work
- The current benchmark covers only 6 manipulation methods; models such as Sora that do not support controlled start/end frames are excluded, and ongoing expansion is needed.
- TADiff is a lightweight modification on top of ActionFormer; deeper forensics-oriented architecture designs (e.g., incorporating optical flow or frequency-domain analysis) warrant further exploration.
- Activity-level forgery localization relies on visual artifact cues, which may diminish as video generation quality continues to improve, necessitating a shift toward higher-level temporal-semantic consistency detection.
- Cross-domain generalization remains limited (avg AP of only 40 in the B→A direction), calling for stronger domain-invariant feature learning.

## Related Work & Insights
- **vs. ForgeryNet/Lav-DF**: These benchmarks address face forgery (appearance-level), whereas this work addresses activity forgery (semantic-level); the underlying detection logic is fundamentally different.
- **vs. TVIL**: TVIL targets temporal video inpainting localization (object removal), while this work focuses on activity modification—a more covert forgery type.
- **vs. ActionFormer**: Action localization architectures applied directly to forgery localization suffer from semantic bias; TADiff addresses this limitation through feature regularization.

## Rating
- Novelty: ⭐⭐⭐⭐ First activity-level forgery localization benchmark with a forward-looking task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three evaluation protocols, multiple state-of-the-art baselines, and comprehensive cross-domain transfer analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation; complete descriptions of the data construction pipeline and method design.
- Value: ⭐⭐⭐⭐⭐ Highly timely; the importance of this task will continue to grow alongside advances in AI video generation capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[CVPR 2026\] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)
- [\[AAAI 2026\] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](../../AAAI2026/video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)

</div>

<!-- RELATED:END -->
