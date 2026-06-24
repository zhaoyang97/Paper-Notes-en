---
title: >-
  [Paper Note] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation
description: >-
  [CVPR 2026][Image Generation][Multi-reference image generation] This paper proposes MultiBanana—the first large-scale benchmark to systematically evaluate multi-reference image generation capabilities. It comprises 3,769 evaluation samples with up to 8 reference images across 5 difficulty dimensions (cross-domain, scale, rare concepts, and multilingualism), revealing a complementary failure mode where closed-source models "overfit reference details" while open-source models "…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Multi-reference image generation"
  - "benchmarking"
  - "cross-domain mixing"
  - "rare concepts"
  - "multilingual"
date: 2026-05-08
content_hash: 04ee2cec0f600a53
---

# MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation

**Conference**: CVPR 2026  
**arXiv**: [2511.22989](https://arxiv.org/abs/2511.22989)  
**Code**: [GitHub](https://github.com/matsuolab/multibanana)  
**Area**: Image Generation  
**Keywords**: Multi-reference image generation, benchmarking, cross-domain mixing, rare concepts, multilingual

## TL;DR

This paper proposes MultiBanana—the first large-scale benchmark to systematically evaluate multi-reference image generation capabilities. It comprises 3,769 evaluation samples with up to 8 reference images across 5 difficulty dimensions (cross-domain, scale, rare concepts, and multilingualism), revealing a complementary failure mode where closed-source models "overfit reference details" while open-source models "ignore reference subjects."

## Background & Motivation

Multi-reference image generation requires models to inherit the appearance of subjects from multiple reference images and render them within new scenes. Although models such as GPT-Image-1 and Nano Banana have demonstrated this capability, evaluation benchmarks have lagged significantly:

1. Existing benchmarks restrict the number of reference images (typically 1–4), failing to evaluate performance under higher reference counts.
2. Task definitions are vague, merely distinguishing "what to edit" or "the number of references" without capturing the intrinsic challenges of heterogeneous reference combinations.
3. There is a lack of systematic evaluation for difficult conditions such as cross-domain transfers, scale mismatches, rare concepts, and multilingual instructions.

MultiBanana fills this critical gap, enabling fair comparisons and the measurement of progress in the field.

## Method

### Overall Architecture

MultiBanana is not a new model but a benchmark suite designed to measure exactly how difficult multi-reference image generation truly is. The **Core Problem** it addresses is that existing benchmarks are limited to 1–4 references and possess vague task definitions, making it impossible to measure the true capabilities of models under more numerous and heterogeneous references. To this end, the paper establishes a complete pipeline from raw images to final evaluation samples: collecting images from both real and synthetic sources, filtering low-quality and harmful content, categorizing them into hierarchical classes, and using Gemini to generate and verify editing instructions alongside human auditing. Finally, difficult references are specifically annotated. The benchmark is supported by three pillars: a multi-dimensional task definition system that quantifies "the number and combination of references" into difficulty gradients, the aforementioned dual-source (real + synthetic) data construction pipeline, and a VLM-weighted evaluation protocol aligned with human judgment.

### Key Designs

**1. Multi-dimensional Task Definition: Quantifying "Reference Counts and Combinations" into Difficulty Gradients**

Legacy benchmarks only broadly distinguish "what to edit" or "the number of images," masking the true challenges of heterogeneous combinations. MultiBanana expands tasks along two axes: reference quantity and compositional structure. Single-reference tasks represent standard image editing (11 types); dual-reference tasks distinguish between subject and auxiliary references; and multi-reference tasks (3–8 images) are created by crossing 4 compositional structures with 6 reference counts for a total of 24 tasks. Furthermore, 4 difficulty dimensions are overlaid—cross-domain (28.2%), scale mismatch (36.0%), rare concepts (19.7%), and multilingual (2.6%)—specifically targeting scenarios where models are most prone to failure, thereby clearly identifying the capability boundaries of different models.

**2. Data Construction Pipeline: Dual-source Real + Synthetic Data with Hierarchical Classification to Mitigate Imbalance**

To cover such a wide range of task types, relying solely on real data would lead to severe imbalances in categories like people and objects. The pipeline thus adopts a two-pronged approach: real data is filtered from LAION-5B based on aesthetic scores $>6.25$ and resolutions $>512\text{px}$, while gaps are filled using synthetic data from Nano Banana and GPT-Image-1. All images undergo hierarchical classification across 6 major categories (People, Objects, Background, Lighting, Color, Style) and 13 subcategories. Gemini then generates candidate editing instructions and evaluates their visual plausibility, followed by manual verification of each entry. This "machine generation—machine evaluation—human oversight" structure ensures that large-scale samples remain diverse and controlled.

**3. VLM Evaluation Protocol: Weighted Multi-dimensional Scoring Aligned with Human Judgment**

The quality of multi-reference generation cannot be summarized by a single dimension, and simple pixel similarity fails to measure whether the "reference subject has been faithfully inherited." The paper designs a 5-dimension weighted scoring system: instruction alignment (weight 3), reference consistency (weight 3), background-subject matching (1), physical realism (1), and visual quality (1). The first two dimensions are given the highest weights as they represent the most critical requirements of multi-reference tasks. Judgments are performed by Gemini 2.5 and GPT-5 (with Qwen3-VL as an open-source alternative), using a 10-point scale for the weighted total score. This protocol demonstrates strong correlation with human ratings (Pearson $r=0.69$, Cohen’s $\kappa=0.61$ for GPT-5), proving its reliability as an automated proxy.

### Loss & Training

This is a pure benchmarking work and does not involve model training.

## Key Experimental Results

### Main Results (Average Scores Across Task Types)

| Model | Single-Ref | Dual-Ref | X-Object | X-1 + Background |
|------|--------|--------|-------|----------|
| GPT-Image-1 | 7.80 | 6.59 | 5.09 | 5.02 |
| Nano Banana | 7.82 | 4.89 | 4.45 | 3.58 |
| Qwen-Image | 7.50 | 3.70 | 2.26 | 2.03 |
| DreamOmni2 | 6.52 | 4.07 | 2.80 | 2.59 |

### Impact of Reference Quantity

| Ref Count | GPT-Image-1 | Nano Banana | Qwen-Image |
|-----------|-------------|-------------|------------|
| 3 | ~5.5 | ~4.8 | ~3.0 |
| 5 | ~5.0 | ~4.2 | ~2.5 |
| 8 | ~4.5 | ~3.8 | ~2.0 |

### Key Findings

- **Closed-source Models**: Tend to struggle to satisfy all reference constraints, leading to global scene distortion (overfitting reference details $\rightarrow$ compositional collapse).
- **Open-source Models**: Generate visually clean images but frequently ignore multiple reference subjects (sacrificing fidelity for visual quality).
- Background replacement is the most difficult task for all models, regardless of the number of references.
- Performance drops significantly for all models under cross-domain and scale mismatch conditions.
- VLM judging correlates well with human scoring (GPT-5 Pearson $r=0.69$, Cohen’s $\kappa=0.61$).

## Highlights & Insights

- The first systematic benchmark for multi-reference image generation, filling a major vacancy in the field.
- Reveals a complementary failure mode between closed and open-source models: the trade-off between reference fidelity and visual consistency.
- The design of difficult reference dimensions (cross-domain, rare concepts, etc.) is highly targeted and effectively distinguishes model capability boundaries.
- With 3,769 samples, 36 task types, and 5-dimensional evaluation, the scale and coverage far exceed existing benchmarks.

## Limitations & Future Work

- Multilingual samples account for only 2.6% (99 samples), providing limited statistical power.
- The bias introduced by synthetic data is not fully discussed (e.g., evaluating Nano Banana using data it helped generate).
- The 10-point scoring granularity may be insufficient to distinguish subtle quality differences.
- Preliminary explorations into Agent frameworks (such as IPR) showed limited effectiveness; stronger pipeline strategies remain an area for research.

## Related Work & Insights

- **vs. DreamBooth**: Supports only single references and does not involve compositional challenges.
- **vs. OmniContext/DreamOmni2**: Limited to 3–4 references and lacks difficult reference combination dimensions.
- **vs. EditBench/EMU-Edit**: Focuses on evaluation of editing quality without addressing multi-reference combinations.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic multi-reference benchmark; novel design of difficulty dimensions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 closed/open-source models; deep analysis; comprehensive reliability validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich statistical charts, and well-summarized findings.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in multi-reference image generation evaluation; likely to drive field progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GenColorBench: A Color Evaluation Benchmark for Text-to-Image Generation](gencolorbench_a_color_evaluation_benchmark_for_text-to-image_generation.md)
- [\[CVPR 2026\] I2I-Bench: A Comprehensive Benchmark Suite for Image-to-Image Editing Models](i2i-bench_a_comprehensive_benchmark_suite_for_image-to-image_editing_models.md)
- [\[CVPR 2026\] DynFusion: Rethinking Condition Fusion for Adaptive Multi-Conditional Text-to-Image Generation](dynfusion_rethinking_condition_fusion_for_adaptive_multi-conditional_text-to-ima.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] Garments2Look: A Multi-Reference Dataset for High-Fidelity Outfit-Level Virtual Try-On with Clothing and Accessories](garments2look_a_multi-reference_dataset_for_high-fidelity_outfit-level_virtual_t.md)

</div>

<!-- RELATED:END -->
