---
title: >-
  [Paper Note] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization
description: >-
  [CVPR 2026][Image Generation][Story Visualization] ViStoryBench constructs a comprehensive benchmark comprising 80 multi-style stories, 344 characters, and 1,317 shots. It proposes 12 automated evaluation metrics (covering character consistency, style similarity, prompt alignment, copy-paste detection, etc.) to systematically evaluate over 25 open-source and commercial story visualization methods, filling the gap of unified evaluation standards in the field.
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Story Visualization"
  - "Benchmarking"
  - "Character Consistency"
  - "Multi-dimensional Evaluation"
  - "Narrative Generation"
date: 2026-05-08
content_hash: ecad47a48cf078c8
---

# ViStoryBench: Comprehensive Benchmark Suite for Story Visualization

**Conference**: CVPR 2026  
**arXiv**: [2505.24862](https://arxiv.org/abs/2505.24862)  
**Code**: [https://github.com/ViStoryBench/ViStoryBench](https://github.com/ViStoryBench/ViStoryBench)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Story Visualization, Benchmarking, Character Consistency, Multi-dimensional Evaluation, Narrative Generation

## TL;DR
ViStoryBench constructs a comprehensive benchmark comprising 80 multi-style stories, 344 characters, and 1,317 shots. It proposes 12 automated evaluation metrics (covering character consistency, style similarity, prompt alignment, copy-paste detection, etc.) to systematically evaluate over 25 open-source and commercial story visualization methods, filling the gap of unified evaluation standards in the field.

## Background & Motivation

**Background**: Story visualization aims to generate a visually consistent sequence of images based on a narrative text and character reference images. Recent advancements in diffusion models and autoregressive models have driven rapid development in this area, leading to training-free methods like StoryDiffusion, UNO, and USO, as well as LLM-based multi-stage pipelines such as MMStoryAgent and MovieAgent.

**Limitations of Prior Work**: Existing benchmarks suffer from three major limitations: (1) simplified test scenarios, often restricted to short text prompts or single-image generation, failing to reflect the complexity of real narratives; (2) lack of character reference images, making it impossible to test character consistency; (3) incomplete evaluation metrics, usually relying on general metrics like FID/CLIP-Score while ignoring dimensions specific to story visualization, such as character matching precision, style consistency, and copy-paste behavior.

**Key Challenge**: Story visualization is inherently a multi-dimensional problem—requiring simultaneous character identity consistency, stylistic unity, narrative alignment, and aesthetic quality—but existing evaluation frameworks cannot systematically measure these dimensions, leading to a lack of credibility in comparisons between different methods.

**Goal**: (1) Construct a dataset with diverse story scripts and character references; (2) design an automated metric system covering multiple key dimensions; (3) evaluate and compare a large number of methods under a unified framework.

**Key Insight**: The authors start from "real narrative scenarios," collecting 80 story segments from literature, movies, and folklore across 10 visual styles. They use LLMs to assist in generating structured scripts (including scene descriptions, character actions, and shot designs), which are then manually verified.

**Core Idea**: Construct the first comprehensive benchmark for story visualization covering multiple styles, characters, and metrics to systematically reveal the pros and cons of existing methods.

## Method

### Overall Architecture
ViStoryBench addresses a question often avoided by existing benchmarks: when multiple methods claim to generate coherent illustration sequences based on stories and character references, which one is truly better, and in what dimension? It implements this in three steps: first, collecting 80 stories across 10 styles from various sources and using an LLM to rewrite each into a structured script with character references; second, designing 12 automated metrics around dimensions unique to story visualization (character, style, prompt alignment, aesthetics, copy-paste); finally, conducting a horizontal evaluation of over 25 open-source and commercial methods (both image and video-based) using the same data and metrics. The difficulty lies not in generation itself, but in translating subjective human judgments—like character resemblance and stylistic consistency—into reproducible numerical values.

### Key Designs

**1. Structured Script Generation: Rewriting natural language stories into divisible scripts for evaluation**

For fine-grained evaluation, the first step is having fine-grained ground truth. If a model is only given "Little Red Riding Hood meets a wolf in the forest," it is impossible to determine whether a failure is due to the scene, the character, or the action. ViStoryBench uses five prompt engineering strategies to guide an LLM to summarize and split stories into "Shots." Each shot is forced to include five standardized components: scene description, plot correspondence, a list of characters present, static shot description, and shot angle design (down to shot size, type, and camera position). All LLM outputs are manually reviewed to ensure narrative coherence and logical consistency. This decomposition provides clear references for subsequent "character action alignment" and "shot design alignment" evaluations.

**2. Character Identity Similarity (CIDS): Cropping characters before comparison**

The core requirement of story visualization is that "the same character must be the same person across different shots." However, calculating CLIP similarity on the whole image is biased by background and composition. CIDS uses a four-stage pipeline: first, it uses Grounding DINO to crop character regions from both reference and generated images; next, it selects feature extractors based on style—CLIP for non-realistic styles and ArcFace/AdaFace/FaceNet for realistic faces—to extract 512-dimensional feature vectors; then, it constructs a similarity matrix and uses bipartite matching to find optimal correspondences; finally, it calculates the average cosine similarity of matched pairs. It is further divided into Cross-CIDS (generated image vs. reference) and Self-CIDS (between generated images). Detecting before comparing isolates "character consistency" from background noise.

**3. Multi-granular Prompt Alignment: Splitting text alignment into four levels**

A coarse CLIP-Score of 0.3 doesn't reveal if a model failed the scene, the camera position, or the character interactions. Multi-granular prompt alignment splits alignment into four sub-dimensions: Scene Score (overall correspondence), Shot Score (perspective consistency), Character Interaction (group alignment), and Individual Actions (accuracy of movement). Each dimension is evaluated by a VLM—Gemini-Pro for primary evaluation and Qwen-VL for reproducible evaluation—using a Likert 5-point scale (0–4), then mapped to a 100-point scale. This allows specific failure modes, like "correct scene but wrong action," to be identified.

> ⚠️ Note: Specific versions of Gemini-Pro / Qwen-VL are as specified in the original paper.

**4. Copy-Paste Detection: Identifying cheats where reference images are directly pasted**

High CIDS is not always positive—some methods (like Story-Adapter) achieve high character similarity simply by pasting the reference image into the output. This "false consistency" pollutes rankings. Copy-Paste detection introduces a second reference image of the same character as a "proxy target." Let $g$ be the generated character feature, $r$ be the input reference feature, and $t$ be the feature of the second reference image. If $g$ is significantly closer to $r$ than to $t$, it indicates the model is replicating the specific input image rather than learning the character's identity.

$$\text{Copy-Paste} \;\propto\; \mathbb{1}\big[\,d(g, r) < d(g, t)\,\big]$$

> ⚠️ Note: The formula above is illustrative; the original paper uses geometric normalization to calculate the Copy-Paste Rate.

Aggregating these results provides the Copy-Paste Rate. The second reference image is key because it represents "the same character in a different presentation"—a model that has truly learned the character should be roughly equidistant from $r$ and $t$.

### Loss & Training
ViStoryBench is an evaluation benchmark and does not involve model training. Its core contributions lie in the design of evaluation protocols and the validation of metric correlation with human scores.

## Key Experimental Results

### Main Results

| Method | CSD-Cross↑ | CIDS-Cross↑ | PA-Avg↑ | OCCM↑ | Inc↑ | Aes↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| OmniGen2 | 0.454 | 0.548 | 2.49 | 70.2 | 11.05 | 5.25 |
| UNO (FLUX1) | 0.391 | 0.485 | 2.30 | 74.2 | 12.40 | 5.23 |
| QwenImageEdit | 0.381 | 0.475 | 2.51 | 59.8 | 13.42 | 5.50 |
| AnimDirector (SD3) | 0.288 | 0.401 | 2.55 | 67.4 | 12.02 | 5.59 |
| Story-Adapter (scale=0) | 0.456 | 0.460 | 1.90 | 69.0 | 12.98 | 4.99 |
| StoryDiffusion (SDXL) | 0.269 | 0.397 | 1.85 | 62.9 | 15.72 | 5.76 |

### Ablation Study (Metric Validation)

| Metric Dimension | Human Correlation | Description |
|:---|:---:|:---|
| CIDS (Cross) | High | Significantly positively correlated with human character consistency scores. |
| PA (Scene) | Medium-High | Low variance in VLM evaluation stability analysis. |
| Copy-Paste Rate | - | Copy-Paste Baseline scores 0.474, normal methods < 0.28. |
| Inception Score | High | Good discrimination for diversity; StoryDiffusion (15.72) vs SEED-Story (6.30). |

### Key Findings
- **OmniGen2 performs best in character consistency** (CIDS-Cross=0.548), but also has the highest copy-paste rate (0.275), suggesting over-reliance on replicating reference images.
- **A trade-off exists between prompt alignment and character consistency**: AnimDirector leads in PA-Avg (2.55) but only reaches 0.401 in CIDS; Story-Adapter has high character similarity but weak prompt alignment.
- **Video methods exhibit better scene consistency**: MovieAgent-SD3 reaches a PA-Avg of 2.54, on par with the best image methods, though it lags in character consistency and aesthetic scores.
- **OCCM metrics reveal severe character count hallucinations**: Even the best (Vlogger) only reaches 76.6%, indicating that controlling the number of characters in multi-character scenes remains a universal challenge.

## Highlights & Insights
- **The Copy-Paste detection metric is the most ingenious design of this work**: By introducing a "second reference image" as a proxy target and using geometric normalization to distinguish between "generated consistency" and "pasted consistency," this approach can be transferred to any evaluation scenario requiring the detection of model "cheating."
- **The multi-granular prompt alignment strategy is exemplary**: Breaking prompt alignment into four sub-dimensions (scene/shot/CI/IA) provides significantly more information than a single CLIP-Score and can be generalized to other conditional generation tasks like video generation.
- **VLM-as-an-evaluator** has been rigorously validated for stability (low variance), providing a reliable paradigm for future large-scale automated evaluations.

## Limitations & Future Work
- **Character detection relies on Grounding DINO**: When detection fails (especially for non-realistic styles), CIDS and OCCM metrics are affected; the error introduced by the detector itself has not been fully quantified.
- **Limited dataset size**: 80 stories and 344 characters are still statistically small, especially for rare styles (e.g., 3D rendering) where samples are insufficient.
- **Lack of temporal evaluation**: For video methods, only keyframes are evaluated, losing information on inter-frame coherence and animation smoothness.
- **VLM evaluation bias**: Gemini-Pro's evaluation criteria might systematically shift from human preferences, and closed-source models are difficult to reproduce.

## Related Work & Insights
- **vs. VinaBench**: While VinaBench target story visualization, it lacks character reference images and supports only 6 styles. ViStoryBench is more comprehensive in character consistency evaluation and style coverage.
- **vs. DreamBench++**: DreamBench++ focuses on single-image generation; this work extends to multi-image sequence scenarios, adding story-level metrics like narrative alignment and character matching.
- **Insight**: The metric system of this benchmark can be directly applied to evaluate character consistency and narrative alignment in long video generation, and can serve as an evaluation standard for interactive story creation systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The metric system is comprehensive and innovative (e.g., copy-paste detection), though as a benchmark paper, the core contribution is "systematicity" rather than "theoretical breakthrough."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Over 25 methods were evaluated, with thorough validation of metric correlation with human scores and rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich visualizations, though tables are slightly crowded due to the number of methods.
- Value: ⭐⭐⭐⭐⭐ Fills the gap of unified evaluation standards in story visualization, highly valuable for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models](i2i-bench_a_comprehensive_benchmark_suite_for_image-to-image_editing_models.md)
- [\[CVPR 2026\] DreamingComics: A Story Visualization Pipeline via Subject and Layout Customized Generation using Video Models](dreamingcomics_a_story_visualization_pipeline_via_subject_and_layout_customized_.md)
- [\[ICLR 2026\] Story-Iter: A Training-free Iterative Paradigm for Long Story Visualization](../../ICLR2026/image_generation/story-iter_a_training-free_iterative_paradigm_for_long_story_visualization.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)

</div>

<!-- RELATED:END -->
