---
title: >-
  [Paper Note] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts
description: >-
  [CVPR 2025][VLM Reasoning][Multi-image math reasoning] This paper proposes the MV-MATH benchmark, consisting of 2,009 high-quality multi-image math problems (sourced from real K-12 scenarios) to systematically evaluate the capability of 25 multimodal large models in multi-image math reasoning scenarios. It is found that all models perform well below human levels (the best, Claude, only achieves 33.9%), revealing that multi-image math reasoning remains a significant challenge…
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Multi-image math reasoning"
  - "multimodal benchmark"
  - "K-12 mathematics"
  - "multi-image understanding"
  - "MLLM evaluation"
date: 2026-05-08
content_hash: a647d9e0a0e3df17
---

# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts

**Conference**: CVPR 2025  
**arXiv**: [2502.20808](https://arxiv.org/abs/2502.20808)  
**Code**: [https://eternal8080.github.io/MV-MATH.github.io/](https://eternal8080.github.io/MV-MATH.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Multi-image math reasoning, multimodal benchmark, K-12 mathematics, multi-image understanding, MLLM evaluation

## TL;DR

This paper proposes the MV-MATH benchmark, consisting of 2,009 high-quality multi-image math problems (sourced from real K-12 scenarios) to systematically evaluate the capability of 25 multimodal large models in multi-image math reasoning scenarios. It is found that all models perform well below human levels (the best, Claude, only achieves 33.9%), revealing that multi-image math reasoning remains a significant challenge for MLLMs.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in mathematical reasoning, with top models even outperforming human performance on benchmarks like MathVista. However, existing multimodal math benchmarks (such as MathVista, MathVision, and MathVerse) are almost entirely limited to single-image scenarios—each problem contains only one image.

**Limitations of Prior Work**: The single-image setting is drastically disconnected from real mathematical application scenarios. In actual K-12 education, students frequently need to simultaneously understand relations among multiple diagrams, coordinate systems, and geometric shapes to solve problems. Although attempts like MathVerse-mv (788 problems) and CMM-Math (765 multi-image samples) have tried to fill this gap, they either generate multi-image problems by manually modifying single-image ones (introducing distribution bias) or contain low-quality images, and both lack fine-grained classification and diverse problem types.

**Key Challenge**: Existing multi-image math datasets are severely insufficient in both quantity and diversity, making it impossible to comprehensively evaluate the mathematical reasoning capabilities of MLLMs in multi-image contexts. The coefficient of variation (CV) for MathVerse-mv is only 0.19, whereas the question length distribution in real-world scenarios is far more diverse.

**Goal**: (1) Construct a large-scale, high-quality multi-image math benchmark from real-world scenarios; (2) systematically evaluate the performance of MLLMs on multi-image math reasoning; (3) conduct in-depth analysis of error patterns and performance bottlenecks of these models.

**Key Insight**: The authors directly sample from over 300,000 real K-12 math problems. Through a three-stage filtering and cross-verification process, they guarantee that each problem is a genuine multi-image question rather than manually concatenated.

**Core Idea**: Build a large-scale multi-image math benchmark, MV-MATH, from real K-12 scenarios to systematically reveal the significant shortcomings of MLLMs in multi-image reasoning.

## Method

### Overall Architecture

The construction pipeline of MV-MATH includes: data collection $\to$ three-stage filtering $\to$ data annotation $\to$ benchmark evaluation. The input consists of over 300,000 raw mathematical problem PDFs, and the output is 2,009 rigorously filtered and annotated multi-image math questions, covering 11 subjects and 3 difficulty levels.

### Key Designs

1. **Three-stage data filtering pipeline**:

    - **Function**: Filter out high-quality multi-image questions from 300,000 raw questions.
    - **Mechanism**: The first stage verifies alignment between text and images (Mathpix OCR often makes mistakes), retaining 35,562 out of 49,538 multi-image questions. The second stage checks for missing text fields and semantic errors, classifying questions into multiple-choice and fill-in-the-blank types. The third stage manually filters out low-quality images (blurry, containing text, etc.), ultimately yielding 1,109 multiple-choice and 900 fill-in-the-blank questions. Each step is cross-verified by at least two graduate students.
    - **Design Motivation**: Automated OCR tools suffer from high error rates, necessitating a multi-stage filtering process to ensure data quality.

2. **Image relevance classification (MD/ID)**:

    - **Function**: Classify questions into two subsets: "Mutually Dependent" (MD) and "Independent" (ID).
    - **Mechanism**: In MD-type questions, images are correlated, and understanding one image requires referencing another (e.g., different perspectives of the same geometric shape). In ID-type questions, images are independent. The classification is decided by voting among GPT-4o, Claude-3.5-Sonnet, and Qwen-VL-Max, followed by manual verification.
    - **Design Motivation**: Distinguishing image relevance allows for a deeper analysis of model performance differences when requiring cross-image reasoning versus independent reasoning.

3. **Multi-dimensional difficulty and subject annotation**:

    - **Function**: Provide fine-grained categorization of questions.
    - **Mechanism**: Difficulty is categorized into three levels (Easy/Medium/Hard) using a weighted combination of question length (weight 0.4) and explanation length (weight 0.6). Subjects are classified into 11 categories (analytic geometry, algebra, metric geometry, combinatorics, etc.) via three-model voting.
    - **Design Motivation**: Fine-grained annotations allow researchers to precisely pinpoint the weaknesses of models.

### Loss & Training

This paper introduces an evaluation benchmark rather than a training method, and thus does not involve loss function design. The evaluations are conducted under various configurations: vanilla prompts, CoT prompts, CoT + 2-shot, etc.

## Key Experimental Results

### Main Results

| Model | Overall | Easy | Medium | Hard |
|------|---------|------|--------|------|
| Claude-3.5-sonnet | **33.9%** | 35.7 | 37.5 | 26.6 |
| GPT-4o | 32.1% | 40.3 | 32.7 | 22.9 |
| LLaVA-OV-72B | 26.2% | 34.6 | 26.0 | 19.2 |
| Qwen2VL-7B | 16.5% | 18.8 | 17.1 | 13.9 |
| Human | ~60%+ | - | - | - |

### Ablation Study (CoT Strategy Comparison)

| Model | Original | CoT | CoT+2-shot |
|------|----------|-----|------------|
| Claude-3.5 | 29.2 | 32.6 (+3.4) | **33.9** (+1.3) |
| GPT-4o | **31.8** | 30.9 (-0.9) | 32.1 (+1.2) |
| Gemini-1.5 | **29.8** | 28.3 (-1.5) | 29.1 (+0.8) |
| LLaVA-OV-72B | **27.3** | 26.7 (-0.6) | 26.2 (-0.5) |

### Key Findings

- CoT prompting provides a clear boost for Claude (+3.4), but conversely degrades performance for GPT-4o, Gemini, etc., indicating that CoT is not always effective in multi-image mathematical tasks.
- The performance of all models drops drastically on the Hard difficulty, with the best model (Claude) achieving only 26.6%, demonstrating that multi-step reasoning remains the core challenge.
- The open-source model LLaVA-OneVision-72B (26.2%) delivers a competitive performance, surpassing GPT-4V (24.5%).
- Models perform significantly lower on Mutually Dependent (MD) questions than on Independent (ID) questions, suggesting that cross-image relational reasoning is a major bottleneck.
- Sequential input of multiple images outperforms merged inputs, indicating that leveraging the sequential information of images is crucial for models.

## Highlights & Insights

- **Real Data vs. Artificial Rewriting**: Sourcing from 300,000 real K-12 questions rather than rewriting existing datasets avoids the distribution bias caused by manual concatenation in MathVerse-mv. This "large-pool filtering" strategy is highly practical for benchmark construction.
- **CV Metric for Measuring Distribution Diversity**: Utilizing the coefficient of variation ($CV = \sigma/\mu$) to quantify the richness of the question length distribution (MV-MATH 0.74 vs MathVerse-mv 0.19) is a simple and effective metric choice.
- **MD/ID Categorization Unveiling Cross-Image Reasoning Bottleneck**: By distinguishing between mutually dependent and independent images, this work is the first to quantitatively prove that cross-image relational reasoning is the core weakness of current models.

## Limitations & Future Work

- All data is sourced from the Chinese K-12 education system, which may introduce cultural/educational system biases and might not fully reflect mathematical education scenarios in other countries.
- The difficulty definition relies on the weighted sum of question/explanation lengths, which is relatively coarse and may not accurately reflect cognitive difficulty.
- The evaluation is limited to models released before 2024 (including Claude-3.5) and does not cover subsequent versions of GPT-4o or newer open-source models.
- A lack of analysis regarding the models' internal representations remains—while it is known that the models failed, it is unclear at which step the error occurred (visual understanding or mathematical reasoning).

## Related Work & Insights

- **vs. MathVerse-mv**: MathVerse-mv generates multi-image questions by rewriting single-image questions, yielding only 788 multiple-choice-only questions with a CV = 0.19. MV-MATH filters 2,009 questions from real-world scenarios, including multiple question types, with a CV = 0.74, offering more realistic and diverse data.
- **vs. CMM-Math**: CMM-Math focuses on Chinese scenarios, and some images suffer from poor quality. MV-MATH provides an English version along with image relevance annotations and finer-grained subject classifications.
- **vs. MathVista/MathVision**: These benchmarks are restricted to single-image scenarios. MV-MATH is the first to systematically evaluate multi-image mathematical reasoning at scale.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The gap in multi-image mathematical reasoning benchmarks is substantial, making this filling valuable, though the methodology leans toward data engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluates 25 models, incorporating various configurations and in-depth error analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich charts, and comprehensive statistics.
- **Value**: ⭐⭐⭐⭐ Holds significant reference value for the community to understand the bottlenecks of multi-image reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning](../../ICLR2026/vlm_reasoning/we-math_20_a_versatile_mathbook_system_for_incentivizing_visual_mathematical_rea.md)
- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](../../ACL2025/vlm_reasoning/wemath_knowledge_reasoning.md)
- [\[ICLR 2026\] Math Blind: Failures in Diagram Understanding Undermine Reasoning in MLLMs](../../ICLR2026/vlm_reasoning/math_blind_failures_in_diagram_understanding_undermine_reasoning_in_mllms.md)
- [\[CVPR 2025\] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [\[ICLR 2026\] Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations](../../ICLR2026/vlm_reasoning/unfolding_spatial_cognition_evaluating_multimodal_models_on_visual_simulations.md)

</div>

<!-- RELATED:END -->
