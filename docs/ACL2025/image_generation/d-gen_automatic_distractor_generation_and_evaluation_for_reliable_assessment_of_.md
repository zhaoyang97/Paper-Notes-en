---
title: >-
  [Paper Note] D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models
description: >-
  [ACL 2025 (Findings)][Image Generation][distractor generation] Proposes D-GEN—the first open-source distractor generation model (fine-tuned LLaMA, 8B/70B) that automatically converts open-ended evaluation questions into multiple-choice formats, paired with two evaluation methods (ranking alignment and entropy analysis) to verify distractor quality, maintaining model ranking consistency with Spearman's ρ=0.99 on MMLU.
tags:
  - "ACL 2025 (Findings)"
  - "Image Generation"
  - "distractor generation"
  - "multiple-choice evaluation"
  - "LLM benchmark"
  - "ranking alignment"
  - "entropy analysis"
date: 2026-05-08
content_hash: 63c0f7af6861cae9
---

# D-GEN: Automatic Distractor Generation and Evaluation for Reliable Assessment of Generative Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2504.13439](https://arxiv.org/abs/2504.13439)  
**Code**: [GitHub](https://github.com/emorynlp/D-Gen)  
**Area**: Image Generation  
**Keywords**: distractor generation, multiple-choice evaluation, LLM benchmark, ranking alignment, entropy analysis

## TL;DR
Proposes D-GEN—the first open-source distractor generation model (fine-tuned LLaMA, 8B/70B) that automatically converts open-ended evaluation questions into multiple-choice formats, paired with two evaluation methods (ranking alignment and entropy analysis) to verify distractor quality, maintaining model ranking consistency with Spearman's ρ=0.99 on MMLU.

## Background & Motivation

**Background**: Multiple-choice (MC) formats (such as MMLU/HellaSwag) in LLM evaluation are more reliable than open-ended generation, as they directly extract predictions from logits. However, constructing MC datasets requires carefully designed distractors.

**Limitations of Prior Work**: (a) Human annotation is time-consuming and expensive; (b) retrieval-based methods tend to introduce biases; (c) metrics like BLEU/ROUGE do not measure the "distractedness" of distractors.

**Key Challenge**: While open-ended questions are abundant, they lack MC versions—automatic conversion is needed while ensuring evaluation validity.

**Goal**: Build a high-quality distractor generator + design automatic evaluation methods to measure distractor quality.

**Key Insight**: High-quality distractors should keep the model rankings consistent with those obtained using human-written distractors (ranking alignment) and should yield similar model confidence distributions (entropy analysis).

**Core Idea**: Fine-tune LLaMA on an auxiliary MMLU set + automatically filter out low-quality items + implement a double validation of ranking alignment and entropy.

## Method

### Overall Architecture
Input: Question + correct answer. Output: 3 high-quality distractors.

### Key Designs

1. **D-GEN Training**

    - **Function**: Fine-tune LLaMA 8B/70B to generate distractors.
    - **Mechanism**: Trained on the MMLU auxiliary set, taking "Question+Answer" as input and outputting a list of distractors.
    - **Design Motivation**: Fine-tuning is more stable than prompting.

2. **Automatic Filtering and Regeneration**

    - **Filtering conditions**: Duplicate of the correct answer / duplicate among distractors / format errors.
    - Re-generate unqualified items until conditions are met.

3. **Ranking Alignment**

    - **Function**: Verify if D-GEN distractors maintain valid LLM rankings.
    - **Mechanism**: Evaluate 20+ models using original and D-GEN distractors respectively → compute Spearman/Kendall correlation.
    - **Design Motivation**: Maintaining discriminative power equals good distractors.

4. **Entropy Analysis**

    - **Function**: Compare model confidence distributions on the two sets of distractors.
    - **Mechanism**: $H = -\sum p_i \log p_i$, the entropy distribution of original vs. D-GEN should be similar.
    - **Design Motivation**: Measure whether the "level of confusion" is consistent.

## Key Experimental Results

### Ranking Alignment (MMLU, 20+ models)

| Metric | D-GEN 8B | D-GEN 70B |
|------|---------|----------|
| Spearman's ρ | 0.98 | **0.99** |
| Kendall's τ | 0.92 | **0.94** |

### Entropy Distribution Matching (KL Divergence vs. Original)

| Configuration | KL Divergence |
|------|--------|
| Random Distractors | 0.45 |
| GPT-4 Generated | 0.12 |
| **D-GEN 70B** | **0.08** |

### Human Evaluation (1-5 scale)

| Dimension | D-GEN 70B |
|------|----------|
| Fluency | 4.7 |
| Coherence | 4.5 |
| Distractingness | 4.2 |
| Correctness | 4.8 |

### Key Findings
- **Almost Perfect Ranking Preservation**: ρ=0.99 means D-GEN distractors do not alter the LLM rankings.
- **Highly Matched Entropy Distribution**: Model confidence patterns are almost identical to those with human distractors.
- **Cross-Domain Generalization**: Effective on non-MMLU tasks such as math, commonsense, and structured data.
- **70B > 8B**: Larger models generate higher-quality distractors.
- **Crucial Automatic Filtering**: Filtering out 5-10% of low-quality items is critical.

## Highlights & Insights
- **Standard "MC conversion" pipeline**: Automatically converts any open-ended benchmark into an MC format, eliminating the pain points of parsing output formats.
- **Functional evaluation via ranking alignment**: Instead of judging individual distractor quality, it assesses whether overall evaluation effectiveness is preserved.
- **Entropy analysis beyond text similarity**: Observes "how much the model is confused" rather than "how similar the texts are"—aligning closer with the goal of evaluation.
- **Applicable for data contamination detection**: Models performing significantly worse on the D-GEN version compared to the original version implies they have memorized the original distractors.

## Limitations & Future Work
- **Training data is derived from MMLU**: Restricted domain coverage.
- **English only**: Multilingual environments remain unexplored.
- **Fixed number of distractors**: Fixed at 3.
- **Dependency on logit access**: Inapplicable to API-only models.

## Related Work & Insights
- **vs. GPT-4 Direct Generation**: Fine-tuned D-GEN models perform better in entropy matching (KL=0.08 vs 0.12).
- **vs. Raina et al. (2023)**: While they evaluate the impact of individual distractors, D-GEN evaluates the overall entropy distribution.
- **Insights**: D-GEN coupled with ranking alignment can detect benchmark data contamination.

## Rating
- Novelty: ⭐⭐⭐⭐ First open-source distractor generator + ranking alignment / entropy analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ model rankings + entropy analysis + cross-domain + human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear and intuitive methodology.
- Value: ⭐⭐⭐⭐⭐ Standardization pipeline for LLM evaluation MC conversion with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conf-Gen: Conformal Uncertainty Quantification for Generative Models](../../ICML2026/image_generation/conf-gen_conformal_uncertainty_quantification_for_generative_models.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](../../ICCV2025/image_generation/adiee_automatic_dataset_creation_and_scorer_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](../../CVPR2025/image_generation/chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)
- [\[CVPR 2026\] Toward Early Quality Assessment of Text-to-Image Diffusion Models](../../CVPR2026/image_generation/toward_early_quality_assessment_of_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
