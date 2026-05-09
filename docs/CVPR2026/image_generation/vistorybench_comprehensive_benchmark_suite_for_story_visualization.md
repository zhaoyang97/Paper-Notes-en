---
title: >-
  [Paper Note] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization
description: >-
  [CVPR 2026][Image Generation][Story Visualization] ViStoryBench constructs a comprehensive benchmark comprising 80 multi-style stories, 344 characters, and 1,317 shots, and proposes 12 automated evaluation metrics covering character consistency, style similarity, prompt alignment, and copy-paste detection. The benchmark systematically evaluates over 25 open-source and commercial story visualization methods, addressing the lack of unified evaluation standards in this field.
tags:
  - CVPR 2026
  - Image Generation
  - Story Visualization
  - Benchmark
  - Character Consistency
  - Multi-dimensional Evaluation
  - Narrative Generation
date: 2026-05-08
content_hash: 4cc290ce9eac39d4
---

# ViStoryBench: Comprehensive Benchmark Suite for Story Visualization

**Conference**: CVPR 2026
**arXiv**: [2505.24862](https://arxiv.org/abs/2505.24862)
**Code**: [https://github.com/ViStoryBench/ViStoryBench](https://github.com/ViStoryBench/ViStoryBench)
**Area**: Diffusion Models / Image Generation
**Keywords**: Story Visualization, Benchmark, Character Consistency, Multi-dimensional Evaluation, Narrative Generation

## TL;DR
ViStoryBench constructs a comprehensive benchmark comprising 80 multi-style stories, 344 characters, and 1,317 shots, and proposes 12 automated evaluation metrics covering character consistency, style similarity, prompt alignment, and copy-paste detection. The benchmark systematically evaluates over 25 open-source and commercial story visualization methods, addressing the lack of unified evaluation standards in this field.

## Background & Motivation

**State of the Field**: Story visualization aims to generate a sequence of visually consistent images from narrative text and character reference images. Recent advances in diffusion and autoregressive models have driven rapid progress, yielding training-free methods such as StoryDiffusion, UNO, and USO, as well as LLM-based multi-stage pipelines including MMStoryAgent and MovieAgent.

**Limitations of Prior Work**: Existing benchmarks suffer from three major shortcomings: (1) test scenarios are overly simplistic, typically limited to short text prompts or single-image generation, failing to capture the complexity of real narratives; (2) character reference images are absent, precluding evaluation of character consistency; (3) evaluation metrics are insufficiently comprehensive, relying on general-purpose metrics such as FID and CLIP-Score while ignoring story-visualization-specific dimensions such as character matching accuracy, style consistency, and copy-paste behavior.

**Root Cause**: Story visualization is inherently a multi-dimensional problem requiring simultaneous preservation of character identity, stylistic coherence, narrative alignment, and visual aesthetics. Existing evaluation frameworks cannot systematically measure these dimensions, undermining the credibility of cross-method comparisons.

**Paper Goals**: (1) Construct a diverse dataset of story scripts paired with character reference images; (2) design an automated metric suite covering multiple critical dimensions; (3) conduct comparative evaluation of a large number of methods within a unified framework.

**Starting Point**: The authors take "real narrative scenarios" as their starting point, collecting 80 story excerpts from literature, film, and folklore across 10 visual styles. LLMs are used to assist in generating structured scripts (including scene descriptions, character actions, and shot designs), followed by human review.

**Core Idea**: Construct the first comprehensive story visualization benchmark encompassing multiple styles, characters, and metrics to systematically reveal the strengths and weaknesses of existing methods.

## Method

### Overall Architecture
The ViStoryBench pipeline consists of three stages: (1) dataset construction—extracting scripts from multi-source stories and collecting character reference images; (2) metric design—defining 12 automated metrics spanning character, style, prompt, aesthetics, and copy-paste dimensions; (3) model evaluation—assessing 25+ methods (including both story image and story video approaches) within a unified framework.

### Key Designs

1. **Structured Script Generation**:

    - Function: Converts natural-language stories into structured scripts amenable to quantitative evaluation.
    - Mechanism: Five prompt engineering strategies are employed to guide LLMs in story summarization and script generation. Each shot contains five standardized components: scene description, plot correspondence, list of present characters, static shot description, and shot perspective design (including shot scale, shooting type, and camera angle). All LLM outputs undergo human review to ensure narrative coherence and logical consistency.
    - Design Motivation: Structured scripts provide unambiguous ground truth for multi-dimensional evaluation, enabling fine-grained assessment of character action alignment and shot design alignment as separate dimensions.

2. **Character Identification Similarity (CIDS)**:

    - Function: Quantifies visual consistency between generated characters and reference characters.
    - Mechanism: A four-stage pipeline first crops character regions from reference and generated images using Grounding DINO, then extracts 512-dimensional feature vectors using CLIP (for non-photorealistic styles) or ArcFace/AdaFace/FaceNet (for photorealistic styles), computes a similarity matrix and applies bipartite graph matching to find optimal character correspondences, and finally takes the average cosine similarity of matched pairs. Two variants are defined: Cross-CIDS (generated images vs. reference images) and Self-CIDS (generated images vs. each other).
    - Design Motivation: Using full-image CLIP similarity cannot precisely measure character-level consistency; character regions must first be detected and cropped before feature comparison.

3. **Multi-grained Prompt Alignment**:

    - Function: Measures the degree to which generated images align with textual descriptions at different levels of granularity.
    - Mechanism: Alignment is decomposed into four sub-dimensions—Scene Score (overall correspondence between scene and narrative), Shot Score (shot perspective consistency), Character Interaction (group interaction alignment), and Individual Actions (accuracy of individual character actions). Each dimension is scored on a Likert 5-point scale (0–4) by Gemini-3-Pro (primary evaluator) or Qwen3-VL (reproducible evaluator) and subsequently mapped to a percentage scale.
    - Design Motivation: Coarse-grained CLIP-Score cannot distinguish between cases such as "correct scene but wrong character actions" or "correct composition but wrong interaction relationships."

4. **Copy-Paste Detection Metric**:

    - Function: Detects whether a model lazily copies reference images rather than generating new content.
    - Mechanism: For each generated character feature $g$, the metric compares its distance to the input reference feature $r$ and a second reference feature $t$ of the same character. If $g$ is closer to $r$ than to $t$, the generated result likely reproduces the input reference directly. A Copy-Paste Rate is computed via geometric normalization.
    - Design Motivation: Some methods (e.g., Story-Adapter) achieve high character similarity by directly pasting reference images; such "spurious consistency" must be identified and penalized.

### Loss & Training
ViStoryBench is an evaluation benchmark and does not involve model training. Its core contributions lie in the design of evaluation protocols and metric validation.

## Key Experimental Results

### Main Results

| Method | CSD-Cross↑ | CIDS-Cross↑ | PA-Avg↑ | OCCM↑ | Inc↑ | Aes↑ |
|--------|-----------|------------|---------|-------|------|------|
| OmniGen2 | 0.454 | 0.548 | 2.49 | 70.2 | 11.05 | 5.25 |
| UNO (FLUX1) | 0.391 | 0.485 | 2.30 | 74.2 | 12.40 | 5.23 |
| QwenImageEdit | 0.381 | 0.475 | 2.51 | 59.8 | 13.42 | 5.50 |
| AnimDirector (SD3) | 0.288 | 0.401 | 2.55 | 67.4 | 12.02 | 5.59 |
| Story-Adapter (scale=0) | 0.456 | 0.460 | 1.90 | 69.0 | 12.98 | 4.99 |
| StoryDiffusion (SDXL) | 0.269 | 0.397 | 1.85 | 62.9 | 15.72 | 5.76 |

### Ablation Study (Metric Validation)

| Metric Dimension | Correlation with Human Evaluation | Notes |
|-----------------|----------------------------------|-------|
| CIDS (Cross) | High | Significantly positively correlated with human character consistency ratings |
| PA (Scene) | Medium-High | Low variance in VLM evaluator stability analysis |
| Copy-Paste Rate | — | Copy-Paste Baseline scores 0.474; normal methods score <0.28 |
| Inception Score | High | Diversity metric shows strong discriminability: StoryDiffusion (15.72) vs. SEED-Story (6.30) |

### Key Findings
- **OmniGen2 achieves the best character consistency** (CIDS-Cross = 0.548), but also exhibits the highest copy-paste rate (0.275), suggesting potential over-reliance on reference image copying.
- **A trade-off exists between prompt alignment and character consistency**: AnimDirector leads on PA-Avg (2.55) but achieves only CIDS = 0.401; Story-Adapter shows high character similarity but weak prompt alignment.
- **Video-based methods perform better on scene consistency**: MovieAgent-SD3 achieves a PA-Avg of 2.54, on par with the best image-based methods, but scores lower on character consistency and aesthetics.
- **The OCCM metric reveals severe character count hallucination**: even the best-performing method, Vlogger, reaches only 76.6%, indicating that controlling character count in multi-character scenes is a widespread challenge.

## Highlights & Insights
- **The copy-paste detection metric is the most ingenious design in this work**: by introducing a second reference image of the same character as a proxy target and applying geometric normalization, it distinguishes "generated consistency" from "pasted consistency." This idea is transferable to any evaluation scenario requiring detection of model "cheating behavior."
- **The decomposition strategy for multi-grained prompt alignment** is worth emulating: splitting prompt alignment into four sub-dimensions (scene/shot/CI/IA) yields substantially more information than a single CLIP-Score and is generalizable to evaluation of other conditional generation tasks such as video generation.
- **Using VLMs as evaluators**, validated through rigorous stability analysis (low variance), provides a reliable paradigm for future large-scale automated evaluation.

## Limitations & Future Work
- **Character detection relies on Grounding DINO**: when detection fails—particularly for non-photorealistic styles—CIDS and OCCM metrics are adversely affected, and the error introduced by the detector itself is not fully quantified.
- **Dataset scale is limited**: 80 stories and 344 characters remain statistically small, especially for certain rare styles (e.g., 3D rendering) where sample sizes are insufficient.
- **Temporal evaluation is absent**: video methods are assessed using only keyframes, discarding inter-frame coherence and animation smoothness information.
- **VLM evaluation bias**: Gemini-3-Pro's scoring criteria may exhibit systematic deviation from human preferences, and reproducibility is hindered by the closed-source nature of the model.

## Related Work & Insights
- **vs. VinaBench**: Although VinaBench also targets story visualization, it lacks character reference images and supports only 6 visual styles; ViStoryBench is more comprehensive in character consistency evaluation and style coverage.
- **vs. DreamBench++**: DreamBench++ focuses on single-image generation; this work extends evaluation to multi-image sequential scenarios and introduces story-level metrics such as narrative alignment and character matching.
- **Insights**: The metric suite from this benchmark can be directly applied to evaluating character consistency and narrative alignment in long-video generation, and can also serve as an evaluation standard for interactive story creation systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The metric design system is comprehensive and innovative (e.g., copy-paste detection), though as a benchmark paper the core contribution lies in "systematicity" rather than "theoretical breakthroughs."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Over 25 methods are evaluated; correlation between metrics and human judgments is thoroughly validated; statistical analysis is rigorous.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and figures are abundant, though tables appear somewhat crowded due to the large number of evaluated methods.
- Value: ⭐⭐⭐⭐⭐ Fills the gap left by the absence of unified evaluation standards in story visualization, providing an important reference for future research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](realunify_do_unified_models_truly_benefit_from_unification_a_comprehensive_bench.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)
- [\[CVPR 2026\] ShowTable: Unlocking Creative Table Visualization with Collaborative Reflection and Refinement](showtable_unlocking_creative_table_visualization_with_collaborative_reflection_a.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

<!-- RELATED:END -->
