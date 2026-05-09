---
title: >-
  [Paper Note] GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models
description: >-
  [ICLR 2026][Image Generation][Geographical Diversity] This paper proposes GeoDiv, a framework that leverages the world knowledge embedded in LLMs and VLMs to systematically evaluate the geographical diversity of T2I models along two dimensions — the Socioeconomic Visual Index (SEVI) and the Visual Diversity Index (VDI) — revealing systematic impoverishment biases in model outputs for countries such as India and Nigeria.
tags:
  - ICLR 2026
  - Image Generation
  - Geographical Diversity
  - Text-to-Image Models
  - Socioeconomic Bias
  - VLM Evaluation
  - Interpretable Metrics
date: 2026-05-08
content_hash: 0ce0264db4d9bb00
---

# GeoDiv: Framework for Measuring Geographical Diversity in Text-to-Image Models

**Conference**: ICLR 2026
**arXiv**: [2602.22120](https://arxiv.org/abs/2602.22120)
**Code**: [GitHub](https://github.com/moha23/geodiv)
**Area**: Text-to-Image Generation / Fairness Evaluation
**Keywords**: Geographical Diversity, Text-to-Image Models, Socioeconomic Bias, VLM Evaluation, Interpretable Metrics

## TL;DR
This paper proposes GeoDiv, a framework that leverages the world knowledge embedded in LLMs and VLMs to systematically evaluate the geographical diversity of T2I models along two dimensions — the Socioeconomic Visual Index (SEVI) and the Visual Diversity Index (VDI) — revealing systematic impoverishment biases in model outputs for countries such as India and Nigeria.

## Background & Motivation
- **Background**: T2I models (e.g., Stable Diffusion, FLUX.1) are widely deployed in commercial applications, yet their generated outputs frequently lack geographical diversity and perpetuate stereotypical depictions of different regions.
- **Limitations of Prior Work**: Existing diversity metrics either rely on annotated datasets (e.g., GeoDE) or focus solely on low-level visual similarity (e.g., Vendi-Score), failing to capture the multidimensional nature of geographical diversity in an interpretable manner.
- **Key Challenge**: Geographical diversity spans multiple dimensions — economic, environmental, and cultural — which cannot be comprehensively captured by a single metric; moreover, existing approaches have limited capacity for fine-grained bias detection at the country level.
- **Key Insight**: Exploit the implicit world knowledge of LLMs/VLMs to design an interpretable and automated evaluation framework.
- **Core Idea**: Decompose geographical diversity into four interpretable dimensions — Affluence and Maintenance (constituting SEVI) and Entity Appearance and Background Appearance (constituting VDI) — and quantify diversity using Hill Numbers.

## Method

### Overall Architecture
The GeoDiv pipeline operates as follows: given an entity $e$ and a country $c$, an LLM generates entity-specific attribute questions alongside fixed background questions; a VQA model predicts answer distributions over the image set; VDI (normalized Hill Number) is computed from these distributions; simultaneously, the VQA model scores each image to produce SEVI.

### Key Designs

1. **Visual Diversity Index (VDI)**:

    - **Function**: Evaluates visual variation across images along two axes — entity appearance and background appearance.
    - **Mechanism**: An ensemble of LLMs generates attribute question–answer pairs; a VQA model predicts answer distributions; diversity is quantified using normalized Hill Numbers.
    - Diversity score: $\text{Diversity-Score} = \frac{\exp(H(\hat{P_k})) - 1}{|\hat{\mathcal{A}_k}| - 1}$, where $H(\cdot)$ denotes Shannon entropy.
    - **Design Motivation**: Questions vary in the number of possible answers; normalization enables fair comparison across questions.

2. **Socioeconomic Visual Index (SEVI)**:

    - **Function**: Captures Affluence (1–5 scale) and Maintenance (1–5 scale) of generated images.
    - **Mechanism**: A VLM directly scores each image; Hill Numbers are then applied to the resulting score distributions to quantify diversity.
    - **Design Motivation**: Integrates socioeconomic indicators with visual analysis to ensure consistency across subjective concepts.

3. **Reliability Mechanisms**:

    - *Visibility Step*: Filters out images in which the target attribute is not visible, reducing VQA hallucinations.
    - *Multi-Select*: Allows multiple answer selections to avoid distribution distortion caused by forced single-choice.
    - *NOTA Option*: Appends a "None of the Above" option (selected in only 2.6% of cases) to reduce guessing.
    - *Large-Scale Human Validation*: Local annotators from 14 countries verify the alignment between SEVI and human judgments.

### Loss & Training
- Gemini-2.5-flash is used as the VQA/VLM backbone (best accuracy: 86%; human correlation: ρ = 0.76/0.69).
- Evaluation covers 4 T2I models, 10 entities, and 16 countries, comprising a total of 160,000 synthetic images.

## Key Experimental Results

### Main Results

| VQA Model | VDI Entity Acc. | VDI Background Acc. | SEVI-Affluence ρ | SEVI-Maintenance ρ |
|---|---|---|---|---|
| Gemini-2.5-flash | 0.87 | 0.85 | 0.76 | 0.69 |
| gpt-4o | 0.85 | 0.81 | 0.76 | 0.76 |
| Qwen2.5-VL | 0.85 | 0.77 | 0.69 | 0.71 |
| LLaVA-v1.6 | 0.70 | 0.66 | 0.65 | 0.68 |

### Key Findings: Country-Level Bias

| Country Group | Avg. Affluence | Avg. Maintenance | Diversity Score |
|---|---|---|---|
| India / Nigeria / Colombia | 2.31 | 3.34 | Low |
| Japan / UAE / UK | 3.53 | 4.30 | Low |
| FLUX.1 (Global) | 3.82 | 4.73 | Extremely low (0.15) |

### Key Findings
- FLUX.1 produces the most refined images but exhibits the lowest diversity, revealing a trade-off between refinement and diversity.
- Overall geographical diversity has declined in newer model versions.
- Background diversity (0.31) is substantially lower than entity diversity (0.44); mountains appear in only 12% of images.
- Compared to Vendi-Score: only entity diversity shows moderate correlation (ρ = 0.56); correlations across other dimensions are low.

## Highlights & Insights
- GeoDiv is the first systematic and interpretable evaluation framework for geographical diversity in T2I models, supporting arbitrary extension to new entities and countries.
- The discovery of FLUX.1's "high quality, low diversity" trade-off offers direct guidance for model development.
- All data, annotations, and code are publicly released.

## Limitations & Future Work
- Coverage is limited to 16 countries and 10 entities; extending to more regions may reveal additional bias patterns.
- The framework relies on the world knowledge of LLMs/VLMs, which may themselves harbor biases.
- Cultural representation remains a limitation, with annotator–VQA model disagreements observed for certain countries.

## Related Work & Insights
- **vs. Vendi-Score**: Vendi-Score measures only visual variation and cannot capture the socioeconomic dimension.
- **vs. GRADE**: GRADE evaluates diversity only for everyday objects and does not address the complexity of the geographical dimension.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First framework to decompose geographical diversity into four interpretable dimensions via SEVI + VDI.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 160K images, large-scale human validation, multi-model and multi-country comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with insightful findings.
- **Value**: ⭐⭐⭐⭐ Direct applicability to fairness evaluation of T2I models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] The Intricate Dance of Prompt Complexity, Quality, Diversity, and Consistency in T2I Models](the_intricate_dance_of_prompt_complexity_quality_diversity_and_consistency_in_t2.md)
- [\[AAAI 2026\] How Bias Binds: Measuring Hidden Associations for Bias Control in Text-to-Image Compositions](../../AAAI2026/image_generation/how_bias_binds_measuring_hidden_associations_for_bias_control_in_text-to-image_c.md)
- [\[CVPR 2026\] AutoDebias: An Automated Framework for Detecting and Mitigating Backdoor Biases in Text-to-Image Models](../../CVPR2026/image_generation/autodebias_automated_framework_for_debiasing_text-to-image_models.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)

<!-- RELATED:END -->
