---
title: >-
  [Paper Note] WorldScore: A Unified Evaluation Benchmark for World Generation
description: >-
  [ICCV 2025][Video Generation][World generation] This paper proposes WorldScore — the first unified evaluation benchmark for world generation. It decomposes world generation into a series of next-scene generation tasks…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "World generation"
  - "unified evaluation benchmark"
  - "3D/4D scene generation"
  - "multi-dimensional evaluation metrics"
date: 2026-05-08
content_hash: b83ea59343f67da8
---

# WorldScore: A Unified Evaluation Benchmark for World Generation

**Conference**: ICCV 2025  
**arXiv**: [2504.00983](https://arxiv.org/abs/2504.00983)  
**Code**: [GitHub](https://haoyi-duan.github.io/WorldScore/)  
**Area**: Video Generation  
**Keywords**: World generation, unified evaluation benchmark, 3D/4D scene generation, video generation, multi-dimensional evaluation metrics

## TL;DR

This paper proposes WorldScore — the first unified evaluation benchmark for world generation. It decomposes world generation into a series of next-scene generation tasks, enabling unified evaluation of 3D, 4D, I2V, and T2V models across 3,000 test samples and 10 evaluation metrics.

## Background & Motivation

Recent years have witnessed rapid advances in video generation (e.g., Sora, CogVideoX) and 3D/4D scene generation (e.g., LucidDreamer, WonderWorld), giving rise to the concept of "world generation" — the creation of large-scale worlds composed of multiple diverse scenes seamlessly connected. However, existing benchmarks suffer from three critical limitations:

**Single-scene quality evaluation only**: Metrics such as VBench and EvalCrafter assess only the visual quality of individual video clips, without addressing scene transitions or layout control.

**Incompatibility with 3D/4D methods**: Existing benchmarks lack camera trajectory and reference image inputs, making it impossible to evaluate 3D/4D methods that require camera poses or seed images.

**Absence of multi-scene and long-sequence evaluation**: No existing benchmark requires models to perform multi-step scene generation or long-sequence world generation tasks.

WorldScore is motivated by the observation that **world generation is not equivalent to video generation**, and that a unified evaluation protocol covering controllability, quality, and dynamics is needed to enable fair comparisons across different method families.

## Method

### Overall Architecture

WorldScore decomposes the world generation task into a series of **next-scene generation steps**, each described by a triplet $(\mathcal{C}, \mathcal{N}, \mathcal{L})$:

- $\mathcal{C} = \{\mathbf{I}, \mathcal{P}\}$: the current scene (image + text description)
- $\mathcal{N}$: the text prompt for the next scene
- $\mathcal{L} = \{\mathcal{T}, \mathcal{Y}\}$: the layout (camera trajectory + textual camera motion description)

The generation process is formulated as: $\mathbf{V} = g_{\text{world}}(w_{\text{proc}}(\mathcal{C}, \mathcal{N}, \mathcal{L}))$

The benchmark further divides evaluation into **static worlds** (assessing controllability and quality) and **dynamic worlds** (assessing dynamic behavior). Static tasks require models to generate new scene sequences, while dynamic tasks require models to generate motion within a single scene.

### Key Designs

1. **Unified World Specification**: Each test sample simultaneously provides image conditions and text prompts (for I2V and T2V), camera matrices, and textual camera motion descriptions (for 3D/4D), enabling evaluation of all model families. All methods produce outputs in a unified video format, eliminating incomparability across approaches.

2. **Dataset Curation**: A total of 3,000 high-quality test samples are collected. The static world split contains 2,000 samples across 5 indoor and 5 outdoor scene categories (including small worlds with 1 scene and large worlds with 3 scenes); the dynamic world split contains 1,000 samples covering 5 motion types (rigid body motion, fluid motion, etc.). Each sample also has a stylized counterpart (7 candidate styles), supporting both photorealistic and stylized evaluation.

3. **WorldScore Metric System**: A total of 10 sub-metrics organized along three dimensions:

    - **Controllability**: Camera error $e_{\text{camera}} = \sqrt{e_\theta \cdot e_t}$ (rotation + translation error), object controllability (open-vocabulary object detection success rate), content alignment (CLIPScore)
    - **Quality**: 3D consistency (DROID-SLAM reprojection error), photometric consistency (optical flow AEPE for texture flickering detection), style consistency (Gram matrix Frobenius norm difference), perceptual quality (ensemble of CLIP-IQA+ and CLIP Aesthetic)
    - **Dynamics**: Motion accuracy (optical flow contrast between designated vs. non-designated regions), motion magnitude (optical flow magnitude between consecutive frames), motion smoothness (comparison with frame interpolation ground truth)

### Loss & Training

WorldScore is an evaluation benchmark rather than a training method, and therefore involves no loss functions. The scoring procedure applies linear normalization to map each sub-metric to $[0, 100]$, then computes arithmetic means to obtain WorldScore-Static (average of controllability and quality metrics) and WorldScore-Dynamic (further incorporating dynamic metrics). For 3D methods that do not support dynamic generation, dynamic metrics are set to 0.

## Key Experimental Results

### Main Results

| Model | WorldScore-Static | WorldScore-Dynamic | Camera Ctrl. | 3D Consistency | Motion Accuracy |
|------|-----|-----|-----|-----|-----|
| WonderWorld (3D) | 72.69 | 50.88 | 92.98 | 86.87 | 0.00 |
| LucidDreamer (3D) | 70.40 | 49.28 | 88.93 | 90.37 | 0.00 |
| CogVideoX-I2V | 62.15 | 59.12 | 38.27 | 86.21 | 69.56 |
| Gen-3 | 60.71 | 57.58 | 29.47 | 68.31 | 54.53 |
| Hailuo | 57.55 | 56.36 | 22.39 | 67.18 | 63.46 |
| CogVideoX-T2V | 54.18 | 48.79 | 40.22 | 68.81 | 25.00 |
| 4D-fy | 27.98 | 32.10 | 69.92 | 35.47 | 22.22 |

### Ablation Study

| Dimension | Key Findings |
|---------|---------|
| Indoor vs. Outdoor | Video models underperform significantly on outdoor scenes compared to 3D models; the gap is smaller indoors |
| Small vs. Large Worlds | Video model performance drops sharply on long-sequence tasks (large worlds = 4 scenes) |
| T2V vs. I2V | T2V achieves higher controllability and larger motion magnitude; I2V produces higher quality but tends to "adhere" to the input viewpoint |
| Motion Magnitude vs. Smoothness | Larger motion often comes at the cost of smoothness, revealing a motion quality bottleneck in current models |

### Key Findings

- 3D models substantially outperform video models on static world generation (WonderWorld 72.69 vs. CogVideoX-I2V 62.15), owing to their inherently high camera controllability and 3D consistency.
- The primary weakness of video models lies in **camera controllability**: the best-performing video model, CogVideoX-T2V, scores only 40.22, far below 3D methods.
- The strongest open-source video model, CogVideoX-I2V, achieves an overall score surpassing closed-source commercial models Gen-3 and Hailuo.
- High motion magnitude does not imply high motion accuracy; models may produce unintended camera motion or irrelevant dynamic content.

## Highlights & Insights

- The **unified evaluation paradigm** is elegantly designed: by simultaneously providing text, images, camera matrices, and textual camera motion descriptions, a single set of test samples accommodates all method families.
- **Decomposing world generation into next-scene generation** is a core contribution, endowing the benchmark with scalability (scene sequence length can be adjusted).
- The photometric consistency metric design is noteworthy — CLIP/DINO features fail to capture fine-grained texture changes, whereas optical flow AEPE proves more effective.
- The 3,000 carefully curated test samples span indoor/outdoor, photorealistic/stylized, and static/dynamic dimensions, reflecting high data quality.

## Limitations & Future Work

- Evaluation relies primarily on automatic metrics; certain sub-metrics (e.g., perceptual quality, content alignment) may diverge from human judgments.
- Only one 4D model (4D-fy) is included, leaving the 4D generation family underrepresented.
- Camera controllability metrics depend on the accuracy of existing SLAM methods, which may introduce noise when processing low-quality generated videos.
- The fixed-camera assumption in dynamic evaluation may be overly simplistic, as real-world generation frequently involves coupling between camera motion and scene dynamics.

## Related Work & Insights

- VBench, EvalCrafter, and WorldModelBench evaluate only single-scene quality; WorldScore fills the gap in multi-scene world generation assessment.
- The high controllability of 3D scene generation methods (e.g., SceneScape, WonderJourney) suggests that video models would benefit from explicit camera conditioning (e.g., CamI2V).
- Benchmark results reveal the complementarity of 3D and video methods, motivating future exploration of hybrid approaches that combine 3D and video priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First unified world generation benchmark with an elegant and conceptually novel design
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 20 models, 3,000 test samples, and 10 metrics — highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich figures and tables, though some metric details require consulting the supplementary material
- **Value**: ⭐⭐⭐⭐⭐ Provides the world generation community with a much-needed standardized evaluation infrastructure

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](../../CVPR2026/video_generation/vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)

</div>

<!-- RELATED:END -->
