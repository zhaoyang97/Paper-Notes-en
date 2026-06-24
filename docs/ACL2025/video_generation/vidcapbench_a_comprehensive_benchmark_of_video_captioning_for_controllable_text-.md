---
title: >-
  [Paper Note] VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation
description: >-
  [ACL 2025][Video Generation][Video Captioning] This work proposes VidCapBench, the first video captioning evaluation benchmark designed specifically for controllable text-to-video (T2V) generation. It evaluates caption quality across four dimensions: aesthetics, content, motion, and physical laws. Comprising 643 videos and 10,644 QA pairs, experiments demonstrate that VidCapBench scores are highly positively correlated with T2V generation quality.
tags:
  - "ACL 2025"
  - "Video Generation"
  - "Video Captioning"
  - "Benchmark"
  - "Text-to-Video"
  - "Caption Evaluation"
  - "Multi-dimensional Evaluation"
date: 2026-05-08
content_hash: 98f63ba33bfc3c76
---

# VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation

**Conference**: ACL 2025  
**arXiv**: [2502.12782](https://arxiv.org/abs/2502.12782)  
**Code**: [https://github.com/VidCapBench/VidCapBench](https://github.com/VidCapBench/VidCapBench)  
**Area**: Video Understanding / Video Generation  
**Keywords**: Video Captioning, Benchmark, Text-to-Video, Caption Evaluation, Multi-dimensional Evaluation

## TL;DR
This work proposes VidCapBench, the first video captioning evaluation benchmark designed specifically for controllable text-to-video (T2V) generation. It evaluates caption quality across four dimensions: aesthetics, content, motion, and physical laws. Comprising 643 videos and 10,644 QA pairs, experiments demonstrate that VidCapBench scores are highly positively correlated with T2V generation quality.

## Background & Motivation

**Background**: Controllable T2V generation relies on the alignment between high-quality video captions and video content. Existing caption evaluation benchmarks (such as MSR-VTT and VATEX) assess short descriptions using traditional metrics like CIDEr, which are inadequate for evaluating the detailed, multi-dimensional captions required for T2V generation.

**Limitations of Prior Work**: (1) Existing evaluations do not cover key T2V dimensions (aesthetics, motion, physical laws); (2) Automatic evaluation is unstable—only $41\%$ of QAs in the VDC benchmark yield consistent results across multiple evaluations; (3) The correlation between caption evaluation and T2V generation quality remains unverified.

**Key Challenge**: While T2V model training necessitates high-quality captions, a standard to evaluate whether captions meet T2V requirements is lacking.

**Goal**: (1) Define T2V-oriented caption evaluation dimensions; (2) Construct a stable and reliable evaluation benchmark; (3) Verify the positive correlation between caption evaluation and T2V quality.

**Key Insight**: Evaluation dimensions are designed based on the core elements focused on by T2V generation models (aesthetics, content, motion, physical laws), rather than starting from the captions themselves.

**Core Idea**: Divide QA pairs into an "automated evaluation subset" (stable to evaluate) and a "human evaluation subset" (difficult to evaluate), balancing both efficiency and accuracy.

## Method

### Overall Architecture
Data Collection $\rightarrow$ Four-Dimension Annotation (Video Aesthetics / Content / Motion / Physical Laws) $\rightarrow$ QA Pair Generation $\rightarrow$ Stratification by Evaluation Stability $\rightarrow$ Hybrid Automated and Human Evaluation.

### Key Designs

1. **Four-dimensional Evaluation System**:

    - **Video Aesthetics (VA)**: Cinematography, post-processing, visual composition, etc.
    - **Video Content (VC)**: Narrative content, descriptions of subjects, backgrounds, and scenes.
    - **Video Motion (VM)**: Foreground subject movement, background object movement, camera motion.
    - **Physical Laws (PL)**: Plausibility and consistency of physical phenomena.
    - **Design Motivation**: Fully aligned with the core dimensions of T2V model evaluation (e.g., VBench, EvalCrafter).

2. **Data Annotation Pipeline**:

    - **Function**: Create multi-dimensional QA annotations for 643 videos.
    - **Mechanism**: (1) Collect multi-source videos (open-source datasets + YouTube + UGC) to ensure subject diversity (even distribution across 10 categories including humans, animals, plants, food, objects, landscapes, etc.); (2) Automate video attribute annotation using expert models (pose estimation, object detection, optical flow, etc.); (3) Generate QA pairs based on these attributes, followed by expert manual verification and refinement.
    - **Design Motivation**: Combine automated annotation and human refinement to balance data quality and annotation cost.

3. **Stratified Evaluation Strategy**:

    - **Function**: Categorize QA pairs into an automated evaluation subset and a human evaluation subset.
    - **Mechanism**: QA pairs undergo repeated evaluations ($3 \times 5$ models). Only QAs with **consistent evaluations across all runs** are categorized into the "automated evaluation subset" (approximately $41\%$), with the remaining assigned for human evaluation.
    - **Design Motivation**: Relying solely on automated evaluation was found to cause significant bias (e.g., short captions receiving artificially high scores). The stratified strategy simultaneously addresses the need for both "rapid iteration" and "precise verification".

4. **Four-dimensional Evaluation Metrics**:

    - **Accuracy (Acc)**: The proportion of completely correct answers.
    - **Precision (Pre)**: The proportion of correct details among those mentioned.
    - **Coverage (Cov)**: The proportion of QA content covered by the caption.
    - **Conciseness (Con)**: The contribution of each token to Acc (encouraging conciseness).
    - **Design Motivation**: A single metric (e.g., CIDEr) cannot comprehensively reflect caption quality; these four metrics complement each other from different perspectives.

### Correlation Verification with T2V
Using captions from different models as prompts for T2V models (CogVideoX, Hunyuan Video), the correlation between T2V quality metrics (VBench) and VidCapBench scores is calculated.

## Key Experimental Results

### Main Results (VidCapBench-AE Automated Evaluation)

| Model | Overall Acc | Video Aesthetics | Video Content | Video Motion | Physical Laws |
|------|-----------|-----------------|---------------|-------------|--------------|
| GPT-4o | **16.8** | 14.1 | 17.5 | 10.2 | **27.9** |
| Gemini 1.5 Pro | 17.1 | 16.4 | 16.9 | 9.8 | 28.4 |
| Qwen2-VL-72B | 15.2 | 14.3 | 15.0 | 5.0 | 25.9 |
| CogVLM2-Caption | 13.1 | 12.5 | 12.7 | 5.7 | 27.9 |
| Tarsier-34B | 11.1 | 10.7 | 10.2 | 3.2 | 26.2 |
| LLaVA-Next-Video-7B | 10.6 | 11.3 | 9.6 | 4.4 | 24.4 |

### T2V Correlation Verification

| Caption Model | VidCapBench Acc | CogVideoX VBench ↑ |
|-------------|----------------|-------------------|
| GPT-4o | 16.8 | Highest |
| CogVLM2-Caption | 13.1 | Medium |
| LLaVA-Next-Video | 10.6 | Lower |

Pearson correlation coefficient $r > 0.8$, proving that VidCapBench scores are highly positively correlated with T2V quality.

### Key Findings
- **All models perform worst in the motion dimension**: Video Motion Acc is generally $< 10\%$, indicating that current VLMs struggle to accurately describe video motion—A key bottleneck for T2V alignment.
- **Closed-source models > Open-source models**: GPT-4o and Gemini lead in Overall Acc, but their advantage is not as large as expected.
- **Longer captions are not always better**: The Conciseness metric reveals that excessively long captions have low information density (e.g., InternVL2 achieves an Acc of 10.2 but a Con of only 2.5).
- **VDC evaluation is unstable**: Only $41\%$ of QA pairs are consistent across multiple automated evaluations, validating the necessity of stratified evaluation.
- **Caption quality $\rightarrow$ T2V quality**: VidCapBench scores are significantly positively correlated with VBench scores, providing quantitative guidance for caption optimization.

## Highlights & Insights
- **First T2V-oriented caption benchmark**: Reverses the caption evaluation dimensions from T2V evaluation dimensions, establishing a closed-loop evaluation from caption $\rightarrow$ T2V.
- **Stratified automated + human evaluation**: Cleverly leverages evaluation consistency to stratify QA pairs, offering a general strategy to resolve instability in automated evaluation.
- **Diagnostic value of the Video Motion dimension**: All models perform poorly on motion descriptions ($< 10\%$ Acc), directly pointing out directions for caption model improvement.

## Limitations & Future Work
- The dataset size of 643 videos is relatively small, with limited scene coverage.
- Automated evaluation still relies on GPT-4o as the judge; the judge's capability acts as the ceiling of evaluation performance.
- The design of QA pair granularity significantly affects results, but the optimal granularity has not been deeply explored.
- Correlation has only been validated with two T2V models (CogVideoX and Hunyuan Video).
- Annotating the Physical Laws (PL) dimension is highly challenging, which may introduce annotation noise.

## Related Work & Insights
- **vs VDC**: VDC contains ~100K QAs but suffers from unstable evaluation (only $41\%$ consistent). VidCapBench is smaller but stable, covering more dimensions (aesthetics, physical laws).
- **vs DREAM-1K**: DREAM-1K evaluates event-level descriptions but does not assess aesthetic or physical plausibility.
- **vs MSR-VTT/VATEX**: These are traditional short-caption evaluations using metrics like CIDEr, which are disconnected from T2V requirements.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Evaluating captions from a T2V perspective is a strong entry point, and the stratified evaluation strategy provides clear highlights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model evaluations are thorough, and the T2V correlation verification is convincing, though the dataset is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, and the problem definitions are precise.
- **Value**: ⭐⭐⭐⭐ Directly provides guidance for caption quality optimization in the T2V domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VABench: A Comprehensive Benchmark for Audio-Video Generation](../../CVPR2026/video_generation/vabench_a_comprehensive_benchmark_for_audio-video_generation.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](../../ICCV2025/video_generation/vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)
- [\[ICLR 2026\] $PhyWorldBench$: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models](../../ICLR2026/video_generation/phyworldbench_a_comprehensive_evaluation_of_physical_realism_in_text-to-video_mo.md)

</div>

<!-- RELATED:END -->
