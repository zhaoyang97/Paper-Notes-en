---
title: >-
  [Paper Note] T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model
description: >-
  [AAAI 2026][Image Generation][T2I Safety] This paper constructs T2I-RiskyPrompt — a comprehensive benchmark comprising 6,432 valid risky prompts spanning 6 major categories and 14 subcategories, each annotated with hierarchical labels and detailed risk rationales. A reason-driven MLLM-based risk detection method is proposed (achieving 91.8% accuracy with a 3B model), and a systematic evaluation is conducted across 8 T2I models, 9 defense methods, 5 safety filters, and 5 attack strategies.
tags:
  - AAAI 2026
  - Image Generation
  - T2I Safety
  - Risk Evaluation
  - Hierarchical Risk Taxonomy
  - Jailbreak Attack
  - Safety Filter
date: 2026-05-08
content_hash: b1a30221e593b2ae
---

# T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model

**Conference**: AAAI 2026
**arXiv**: [2510.22300](https://arxiv.org/abs/2510.22300)
**Code**: [https://github.com/datar001/T2I-RiskyPrompt](https://github.com/datar001/T2I-RiskyPrompt)
**Area**: Diffusion Models
**Keywords**: T2I Safety, Risk Evaluation, Hierarchical Risk Taxonomy, Jailbreak Attack, Safety Filter

## TL;DR
This paper constructs T2I-RiskyPrompt — a comprehensive benchmark comprising 6,432 valid risky prompts spanning 6 major categories and 14 subcategories, each annotated with hierarchical labels and detailed risk rationales. A reason-driven MLLM-based risk detection method is proposed (achieving 91.8% accuracy with a 3B model), and a systematic evaluation is conducted across 8 T2I models, 9 defense methods, 5 safety filters, and 5 attack strategies.

## Background & Motivation
**Background**: T2I models such as Stable Diffusion and Midjourney each have over 10 million users and have collectively generated more than 1 billion images. However, these models can also be maliciously exploited to generate risky content including pornography, violence, and politically sensitive material. Constructing high-quality risky prompt datasets is critical for evaluating the safety of T2I models.

**Limitations of Prior Work**:
   - **Limited risk categories**: Most datasets focus solely on NSFW content (pornography/violence/horror), neglecting copyright infringement and politically sensitive content.
   - **Coarse-grained annotations**: Prior work relies on automated text moderation tools without human verification, resulting in imprecise labels.
   - **Low effectiveness**: Prompts exhibit unclear semantics (PPL as high as 2,500+), yielding a low proportion of generated risky images (effectiveness only 32–55%).

**Key Challenge**: The semantic ambiguity of existing prompts leads to a low probability of T2I models generating risky images, making it infeasible to effectively evaluate model safety.

**Goal**: Construct a T2I risky prompt benchmark with broad coverage, fine-grained annotation, and high effectiveness.

**Key Insight**: Analyze the usage policies of 7 platforms to derive a hierarchical risk taxonomy; employ a 6-stage pipeline to ensure data quality; and replace traditional classifiers with a reason-driven detection method.

**Core Idea**: Hierarchical risk taxonomy + 6-stage data construction pipeline + reason-driven MLLM detection = a new benchmark for comprehensive T2I safety evaluation.

## Method

### Overall Architecture
1. **Risk taxonomy**: Analyze policies of 7 T2I platforms → 6 major categories (pornography / violence / horror / illegal activities / copyright / politics) → 14 subcategories.
2. **Data construction pipeline**: Collection → refinement → diversity filtering → coarse-grained classification annotation → effectiveness filtering → risk rationale annotation.
3. **Detection method**: Reason-driven MLLM risk detection, aligning a 3B model with safety annotations.
4. **Comprehensive evaluation**: 8 T2I models × 9 defense methods × 5 filters × 5 attack strategies.

### Key Designs

1. **Hierarchical Risk Taxonomy**:

    - **Function**: Defines a comprehensive risk taxonomy for T2I-generated content.
    - **Mechanism**: The 6 major categories include — pornography (explicit / suggestive), violence (weapons / gore), horror, illegal activities (drugs / trafficking / theft), copyright (logos / anime characters), and politics (figures / metaphors), totaling 14 subcategories.
    - **Design Motivation**: Existing datasets contain at most 12 categories and lack hierarchical structure. The hierarchical design enables evaluation at multiple levels of granularity.

2. **6-Stage Data Construction Pipeline**:

    - **Function**: Ensures diversity, accuracy, and effectiveness of prompts.
    - **Mechanism**: (1) Collect 12,251 prompts from existing datasets and GPT-4o generation; (2) Refine and standardize linguistic style using GPT-4o; (3) Deduplicate via CLIP similarity filtering (threshold 0.8); (4) Classify and annotate using GPT-4o with human double-checking; (5) Generate images with SD3 and FLUX, then manually cross-validate to remove ineffective prompts; (6) Manually annotate the risk rationale for each prompt to identify specific risky visual elements.
    - **Design Motivation**: The key innovation lies in the "effectiveness filtering" stage — retaining only prompts that reliably generate risky images, which reduces PPL from 2,500+ to 86 and improves effectiveness from 30–55% to 74.1%.

3. **Reason-Driven Risky Image Detection**:

    - **Function**: Aligns an MLLM with safety annotations to enable precise risk detection.
    - **Mechanism**: A 3B MLLM is fine-tuned to not only classify images as risky or safe, but also output the corresponding risk rationale. Detection accuracy is improved by aligning model outputs with human-annotated risk reasons.
    - **Design Motivation**: Traditional classifiers produce only a binary risk/safe label without explaining why an image is risky. The reason-driven approach is more transparent and achieves higher accuracy (91.8%).

### Loss & Training
- The detection model is fine-tuned from a 3B MLLM, aligned with risk rationale annotations.
- The dataset comprises 6,432 prompts and 20,792 generated risky images.

## Key Experimental Results

### Main Results
Risk rates (proportion of prompts that generate risky images) across 8 T2I models:

| Model | Porn | Violence | Horror | Illegal | Copyright | Politics | AVG |
|-------|------|----------|--------|---------|-----------|----------|-----|
| SD1.4 | 0.950 | 0.764 | 0.789 | 0.631 | 0.782 | 0.900 | 0.790 |
| PixArt | 0.402 | 0.930 | 0.928 | 0.814 | 0.677 | 0.847 | 0.774 |
| SDXL | 0.685 | 0.827 | 0.822 | 0.793 | 0.882 | 0.904 | 0.820 |
| FLUX | 0.956 | 0.900 | 0.850 | 0.830 | 0.922 | 0.885 | 0.889 |
| SD3 | 0.859 | 0.939 | 0.872 | 0.916 | 0.943 | 0.952 | 0.923 |

### Ablation Study

| Detection Method | Accuracy | Notes |
|-----------------|----------|-------|
| Traditional image classifier | ~70–80% | Poor generalization |
| MLLM zero-shot | ~80–85% | Requires carefully designed policy |
| **Reason-driven MLLM (3B)** | **91.8%** | Aligned with risk rationale annotations |

### Key Findings
- **Stronger models pose greater risks**: T2I models with higher generation capability exhibit higher risk rates (SD3: 0.923 vs. SD1.4: 0.790).
- **Existing defenses struggle to simultaneously guard against multiple risk categories**: Concept erasing is effective for certain categories but fails on others.
- **No single safety filter can identify all risk types**: Combinations of multiple filters are necessary.
- **Jailbreak attacks can effectively bypass existing safety mechanisms**: All 5 attack methods significantly increase the rate of risky image generation.

## Highlights & Insights
- **Reducing PPL from 2,500+ to 86** represents a critical quality breakthrough: prompt refinement improves semantic clarity and substantially increases prompt effectiveness.
- **The "effectiveness filtering" stage** is essential: generating and validating images before retention ensures that every prompt genuinely triggers risky content.
- **9 key insights** provide a systematic cognitive framework for the T2I safety community.

## Limitations & Future Work
- Data construction relies on extensive human annotation, making it costly to scale to additional languages and culturally specific risk types.
- Only open-source T2I models are evaluated; commercial models (Midjourney, DALL-E 3) are not covered due to API restrictions.
- The risk taxonomy may require ongoing updates as societal norms evolve.

## Related Work & Insights
- **vs. I2P**: I2P contains 4,703 prompts but has a PPL of 2,587 and an effectiveness rate of only 32%. T2I-RiskyPrompt achieves substantially higher quality with a comparable dataset size.
- **vs. T2ISafety**: T2ISafety contains ~70K prompts but lacks risk rationale annotations and effectiveness verification. Quality proves more important than quantity.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical risk taxonomy, reason-driven detection, and comprehensive evaluation is thorough and well-integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across 8 models × 9 defenses × 5 filters × 5 attacks is exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed pipeline descriptions.
- Value: ⭐⭐⭐⭐ Provides a high-quality benchmark for T2I safety research; the 9 key insights offer practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[AAAI 2026\] MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)

</div>

<!-- RELATED:END -->
