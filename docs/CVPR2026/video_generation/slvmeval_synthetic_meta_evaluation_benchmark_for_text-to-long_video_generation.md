---
title: >-
  [Paper Note] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation
description: >-
  [CVPR 2026][Video Generation][Long video generation evaluation] This paper proposes SLVMEval, a meta-evaluation benchmark that synthesizes controlled degradations to construct "high-quality vs. low-quality" video pairs (up to ~3 hours) from densely captioned video datasets, and tests whether existing T2V evaluation systems can distinguish long-video quality differences. Human annotators achieve 84.7%–96.8% accuracy across 10 dimensions, whereas existing automatic evaluation systems fall behind humans on 9 out of 10 dimensions.
tags:
  - CVPR 2026
  - Video Generation
  - Long video generation evaluation
  - meta-evaluation benchmark
  - text-to-video
  - synthetic degradation
  - VLM-as-a-judge
date: 2026-05-08
content_hash: 6f898453663d33e3
---

# SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation

**Conference**: CVPR 2026
**arXiv**: [2603.29186](https://arxiv.org/abs/2603.29186)
**Code**: [https://slvmeval.github.io/](https://slvmeval.github.io/)
**Area**: Video Generation
**Keywords**: Long video generation evaluation, meta-evaluation benchmark, text-to-video, synthetic degradation, VLM-as-a-judge

## TL;DR

This paper proposes SLVMEval, a meta-evaluation benchmark that synthesizes controlled degradations to construct "high-quality vs. low-quality" video pairs (up to ~3 hours) from densely captioned video datasets, and tests whether existing T2V evaluation systems can distinguish long-video quality differences. Human annotators achieve 84.7%–96.8% accuracy across 10 dimensions, whereas existing automatic evaluation systems fall behind humans on 9 out of 10 dimensions.

## Background & Motivation

1. **Background**: Text-to-video (T2V) models are evolving from short clips (a few seconds) to long-form videos (minutes to hours), with systems such as StreamingT2V and Phenaki theoretically capable of generating videos of arbitrary length.
2. **Limitations of Prior Work**: Widely used evaluation metrics such as VideoScore were originally designed for short videos of a few seconds to tens of seconds, and applying them directly to long videos introduces a length mismatch. Meta-evaluation benchmarks such as VBench and UVE also cover only ~10-second clips, making it impossible to verify whether evaluation metrics are reliable for long videos.
3. **Key Challenge**: Long video generation is becoming a frontier research direction, yet no testbed exists to verify whether evaluation systems possess the basic capability to assess long-video quality.
4. **Goal**: To construct a meta-evaluation benchmark specifically targeting long videos, testing whether existing evaluation systems can at least perform quality judgments that humans find straightforward.
5. **Key Insight**: Starting from a densely captioned video dataset, controlled degradations (contrast reduction, resolution downscaling, segment deletion, etc.) are applied to original videos to construct paired controlled experiments, with crowdsourced annotation to verify the perceptibility of each degradation.
6. **Core Idea**: By synthesizing controllably degraded long video pairs, the paper identifies evaluation bottlenecks where humans distinguish quality effortlessly but automatic systems fail.

## Method

### Overall Architecture

Long videos are sampled from the Vript dense video description dataset as high-quality videos $v^+$. Degradation operations are applied to each of 10 evaluation dimensions to generate low-quality videos $v^-$, which are then filtered through crowdsourced annotation to ensure perceptibility of the degradation. During evaluation, automatic systems or human annotators are presented with pairs $(p, \{v^+, v^-\})$ and their accuracy in correctly identifying the high-quality video is measured.

### Key Designs

1. **Degradation Operations across 10 Dimensions**:

    - **Function**: Comprehensively cover two major evaluation capability categories: video quality and video–text consistency.
    - **Mechanism**: *Video quality*—aesthetics (contrast reduction), technical quality (resolution downscaling), appearance style (OpenCV-based style transfer to oil painting, cartoon, etc.), background consistency (background removal via rembg + random landscape replacement). *Video–text consistency*—temporal flow (shuffling 5 consecutive segments), completeness (randomly deleting 5 segments), object completeness (localizing prompt-mentioned objects via GroundingDINO + erasing them via Stable Diffusion Inpainting), spatial relations (horizontal flipping of segments containing left/right descriptions), dynamic degree (replacing motion-described segments with the middle frame to freeze motion), color (modifying specific object colors via Qwen-Image-Edit).
    - **Design Motivation**: Each dimension's degradation affects only that dimension's quality while leaving others unchanged, enabling fine-grained decomposed capability testing.

2. **Controlled Degradation Application Strategy (Algorithm 1)**:

    - **Function**: Selectively degrade partial segments within long videos to preserve overall naturalness.
    - **Mechanism**: Five clips are randomly selected from the video; degradation is applied only to these clips while the remainder stays intact. This local degradation better approximates the quality inconsistencies observed in real T2V generation. Qwen3-8B is used to identify segments containing relevant semantics (e.g., color mentions, spatial-relation mentions).
    - **Design Motivation**: Global degradation is overly simplistic and unrealistic; local degradation challenges evaluation systems to locate and aggregate quality signals within long videos.

3. **Crowdsourced Filtering and Validation**:

    - **Function**: Ensure that the quality difference in each degraded video pair is clearly perceptible to humans.
    - **Mechanism**: Five crowdworkers rate each video pair on a three-tier scale: A (all selected segments successfully degraded) / B (partially successful) / C (complete failure). Retention criteria: (1) no C ratings, and (2) the number of A ratings exceeds the number of B ratings. After filtering, 3,932 video pairs are retained.
    - **Design Motivation**: Ensures benchmark validity—imperceptible degradations cannot serve as effective test cases. Additionally, the finding that evaluation results are highly correlated before and after filtering suggests that the benchmark could be scaled in the future without costly human filtering.

### Loss & Training

SLVMEval is an evaluation benchmark rather than a training method. The main evaluated systems include:
- **Video-based VLM-as-a-judge**: GPT-5, GPT-5-mini, and Qwen3-VL-235B directly judge video pairs.
- **Text-based VLM-as-a-judge**: A VLM first generates descriptions of the video, then a language model compares the descriptions against the prompt.
- **CLIPScore**: Computes the average CLIP similarity between the center frame of each segment and the prompt.
- **VideoScore v1.1**: Quality scoring based on a VLM combined with a regression head.

## Key Experimental Results

### Main Results

Accuracy (%) comparison across evaluation systems (representative dimensions shown):

| System | Aesthetics | Technical Quality | Object Completeness | Temporal Flow | Dynamic Degree |
|--------|------------|-------------------|---------------------|---------------|----------------|
| GPT-5 (video) | **90.1** | **85.8** | 72.0 | 50.3 | 35.3 |
| GPT-5 (text) | 74.8 | 46.2 | 68.0 | 43.5 | 43.1 |
| CLIPScore | 56.4 | 72.3 | **76.0** | 50.5 | 51.7 |
| VideoScore | 52.5 | 33.8 | 66.0 | 46.3 | 48.6 |
| **Human** | **96.5** | **91.8** | **86.6** | **86.6** | **95.9** |

### Ablation Study

Pearson correlation of evaluation system accuracy before and after crowdsourced filtering:

| Dimension | $\rho_P$ |
|-----------|----------|
| Aesthetics | High |
| Technical Quality | High |
| Object Completeness | High |

(All 10 dimensions show strong positive correlation before and after filtering, demonstrating that a reliable benchmark can be produced without filtering.)

Relationship between video duration and accuracy:

| Trend | Description |
|-------|-------------|
| Most dimensions | Longer videos lead to lower accuracy for automatic evaluation systems |
| Dynamic degree | Weak correlation (accuracy is already low even for short videos) |

### Key Findings

- **Semantic and temporal dimensions are the primary bottlenecks**: Dynamic degree (GPT-5 achieves only 35.3%, below 50% chance), temporal flow (50.3% ≈ random), and completeness (51.3% ≈ random) demonstrate that current evaluation systems cannot reason about motion and event ordering across frames.
- **GPT-5 video-based is strongest on visual quality dimensions**: 90.1% on aesthetics, 98.9% on background consistency—yet still below human performance.
- **CLIPScore shows unexpected strength on object completeness and completeness**: CLIP's contrastive pretraining makes it sensitive to the disappearance of prompt-mentioned objects (76.0% on object completeness, ranking second); however, its frame-independent processing yields near-random performance on temporal and dynamic dimensions.
- **Text-based outperforms video-based on certain dimensions**: Qwen3 text-based exceeds video-based by 23.3 points on background consistency and 17.1 points on appearance style, suggesting that projecting video into text space may benefit certain evaluation tasks.
- **VideoScore falls below 50% on multiple dimensions**: Its predefined five evaluation dimensions do not fully align with those in this work, resulting in inconsistent judgments.
- Dataset statistics: 3,932 video pairs; average duration 1,141 seconds (~19 minutes); maximum duration 10,486 seconds (~2 hours 54 minutes); average prompt length 57,884 characters.

## Highlights & Insights

- **Design philosophy of "minimum competency" testing**: Rather than asking what advanced judgments an evaluation system can make, the paper asks whether the system can perform what humans find trivially easy. This "lower-bound test" precisely exposes fundamental deficiencies in existing systems and offers greater diagnostic value than increasingly complex tests.
- **Scalability of synthetic degradation**: The high pre/post-filter correlation validates that the benchmark can be scaled without costly human filtering, lowering the barrier to building large-scale T2LV evaluation benchmarks.
- **First benchmark extended to hour-long videos**: The longest video is approximately 3 hours, far exceeding the seconds-to-tens-of-seconds range of existing benchmarks, filling a critical gap in long-video evaluation validation.

## Limitations & Future Work

- Degradation operations are manually designed and may not fully simulate quality issues arising in real T2V generation (e.g., semantic drift, character inconsistency).
- Source videos come from real footage rather than AI-generated content, so T2V-specific artifacts (e.g., flickering, geometric distortion) are not covered.
- Degradation is applied to only 5 segments; the effect of degradation density on evaluation difficulty is insufficiently explored.
- Large models such as GPT-5 cannot process all frames due to context length constraints, causing loss of inter-frame details.
- The p-values for Spearman correlation coefficients are not significant at the 0.05 level; the video duration effect is a general trend rather than a strong effect.

## Related Work & Insights

- **vs. VBench**: VBench provides fine-grained human annotations across 16 dimensions but is limited to 3.3-second clips. SLVMEval covers 10 dimensions but extends to hour-long videos; the two are complementary—VBench for short videos, SLVMEval for long videos.
- **vs. UVE-Bench**: UVE-Bench focuses on meta-evaluation of LM-based evaluators, but its longest video is only 6.1 seconds. SLVMEval's videos are more than 1,700× longer.
- **vs. VideoScore**: As one of the evaluated systems, VideoScore performs poorly on SLVMEval, confirming the need for evaluation metrics specifically designed for long videos.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First meta-evaluation benchmark for long-form T2V, with a methodologically valuable pipeline of synthetic degradation + crowdsourced filtering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison of 8 evaluation systems × 10 dimensions, with human baselines and duration analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Framework definitions are clear and Algorithm 1 is concise, though occasional encoding issues affect readability.
- **Value**: ⭐⭐⭐⭐ Precisely identifies long-video evaluation bottlenecks (semantic and temporal dimensions) and provides clear research directions for the community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](../../ICCV2025/video_generation/worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)

<!-- RELATED:END -->
