---
title: >-
  [Paper Note] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories
description: >-
  [ACL 2026][AIGC Detection][multilingual bias] BiasedTales-ML constructs a corpus of ~350K LLM-generated children's stories across 8 languages, using full-permutation prompt design and a distributional analysis framework to reveal that **social attribute distributions in narratives vary significantly across languages**, and English-centric evaluation fails to capture bias patterns in multilingual settings.
tags:
  - ACL 2026
  - AIGC Detection
  - multilingual bias
  - narrative generation
  - social attribute distribution
  - cross-lingual consistency
  - children stories
date: 2025-05-08
content_hash: 0c54cb1278e57b87
---

# BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories

**Conference**: ACL 2026
**arXiv**: [2604.17008](https://arxiv.org/abs/2604.17008)
**Code**: [https://huggingface.co/spaces/Linyuana/BIASEDTALES-ML](https://huggingface.co/spaces/Linyuana/BIASEDTALES-ML)
**Area**: AIGC Detection
**Keywords**: multilingual bias, narrative generation, social attribute distribution, cross-lingual consistency, children stories

## TL;DR
BiasedTales-ML constructs a corpus of ~350K LLM-generated children's stories across 8 languages, using full-permutation prompt design and a distributional analysis framework to reveal that **social attribute distributions in narratives vary significantly across languages**, and English-centric evaluation fails to capture bias patterns in multilingual settings.

## Background & Motivation

**Background**: LLMs are increasingly used to generate narrative content (especially children's stories), which implicitly conveys notions about social roles, occupations, and environments. Existing social bias research primarily focuses on English short-text tasks (e.g., sentence completion, classification).

**Limitations of Prior Work**: (1) Short-text bias evaluation cannot capture biases expressed indirectly through characters, scenes, and plot structures in long-form narratives; (2) existing bias benchmarks (e.g., StereoSet, BBQ) are static classification tasks disconnected from real generation scenarios; (3) virtually no work has systematically studied cross-lingual consistency of bias in multilingual narrative generation.

**Key Challenge**: RLHF and other safety alignment techniques are primarily developed on English data and Western norms, but model bias behavior in other languages may be entirely different — conclusions of "safe" from English evaluation may not hold in low-resource languages.

**Goal**: (1) Construct a large-scale multilingual parallel narrative corpus; (2) propose a systematic narrative-level social attribute distribution analysis framework; (3) empirically study cross-lingual bias consistency.

**Key Insight**: Children's stories are chosen as a controlled yet expressive narrative domain — encouraging positive and imaginative content while requiring models to make structured choices about characters, settings, and social roles.

**Core Idea**: Generate parallel stories across 8 languages through full-permutation prompt design (systematically varying nationality × religion × social class × parental role × child gender), and analyze bias using distributional metrics rather than instance-level annotation.

## Method

### Overall Architecture
Three-stage pipeline: (1) **Prompt design and localization**: construct standardized prompt templates, localized into 8 target languages by native speakers; (2) **Large-scale parallel generation**: use 3 LLMs to generate stories across all prompt configurations (5 independent samples per configuration); (3) **Narrative feature extraction and distributional analysis**: use LLM extractors to extract character traits, environments, and cultural references from stories, then compare distributions using statistical metrics.

### Key Designs

1. **Full-Permutation Prompt Design**:

    - Function: Construct controlled cross-lingual comparative experiments
    - Mechanism: Systematically combine 27 nationalities × 6 religions × 2 social classes × 3 parental roles × 3 child genders = 2,916 unique prompt configurations, generated across 8 languages × 3 models with 5 samples per configuration, totaling ~350K stories. Language selection covers languages without grammatical gender (EN/ZH/JA/KO), with grammatical gender (ES/RU/AR), and low-resource (Swahili)
    - Design Motivation: Full-permutation design allows separating the effects of language medium from cultural content, avoiding language-specific patterns that translation benchmarks might mask

2. **LLM-based Narrative Feature Extractor**:

    - Function: Extract structured social attribute representations from long-form stories
    - Mechanism: Use Qwen-3-14B to extract three-dimensional representations $E = (A_{\text{adj}}, V_{\text{env}}, C_{\text{cul}})$ from each story $S$: character descriptive adjectives (e.g., brave, obedient), environment keywords (e.g., forest, kitchen), and cultural references (e.g., menorah, dates). Human validation on 800 stories achieves 85.6% precision with Cohen's $\kappa = 0.618$
    - Design Motivation: Narrative bias is expressed indirectly through character descriptions and scene settings, requiring structured extraction beyond surface keywords

3. **Multi-Dimensional Distributional Bias Metrics**:

    - Function: Quantify and compare cross-lingual social attribute distribution differences
    - Mechanism: Four complementary metrics: (1) Directional bias $S_C = \ln(P(C|g_m)/P(C|g_f))$ measures the association direction between specific attribute categories and gender; (2) JSD measures overall distributional divergence; (3) cosine similarity measures cross-lingual bias pattern consistency; (4) valid story rate (VSR) controls generation quality
    - Design Motivation: No single metric can fully characterize bias — direction, magnitude, cross-lingual consistency, and generation quality require multi-dimensional synthesis

### Loss & Training
Pure evaluation/analysis work with no model training involved. Uses the vLLM inference framework with higher sampling temperature to encourage narrative diversity.

## Key Experimental Results

### Main Results

| Analysis Dimension | Key Finding | Model |
|---|---|---|
| Directional bias | Communality descriptions skew toward female stories across all languages; intellect descriptions skew toward males in Arabic/Russian | 8B model |
| Grammatical gender effect | Llama-3.1-8B shows higher JSD (greater bias divergence) in gendered languages; Qwen-3-8B shows no significant difference | - |
| Cross-lingual consistency | Qwen-3 shows high cross-lingual cosine similarity (consistent); Llama-3 shows large bias pattern differences between English and low-resource languages | - |
| Small model effect | 1B model bias directionality approaches zero, not due to better safety but due to **insufficient vocabulary diversity** falling back to generic patterns | Llama-3.2-1B |

### Ablation Study

| Config | Effect | Note |
|---|---|---|
| Gender condition | Male → outdoor/activity words; Female → home/relationship words | Consistent across languages |
| Social class condition | Working class → practical/labor words; Affluent → leisure/aesthetic words | Qwen-3 data |
| Low-resource language | Swahili: low VSR, high JSD | Especially pronounced in 1B models |

### Key Findings
- Bias patterns observed in English **cannot** be simply extrapolated to other languages, especially low-resource languages
- The relationship between model scale and bias is non-monotonic: smaller models are not "safer" but "more mediocre" (vocabulary diversity bottleneck)
- The effect of grammatical gender on bias divergence varies by model, not a universal rule
- Qwen-3 shows higher cross-lingual consistency than Llama-3, possibly reflecting differences in multilingual coverage of training data

## Highlights & Insights
- **Full-permutation experimental design** is the paper's greatest highlight: by systematically varying each social attribute dimension, the influence of each factor can be precisely isolated. This methodology is transferable to any NLP evaluation involving multi-factor analysis
- **The finding that "small model bias appears low but is actually due to capability limitations"** is very important: it warns against using surface distributional uniformity to assert safety, as vocabulary poverty can also produce uniform distributions
- Distribution-level bias analysis (rather than instance-level annotation) is better suited for large-scale generation scenarios, avoiding the unscalability of per-sample annotation

## Limitations & Future Work
- All stories are generated by LLMs and cannot directly reflect bias patterns in human narratives
- Feature extraction relies on LLMs, which may introduce extraction bias themselves
- While 8 languages are representative, many low-resource languages remain uncovered
- Analysis is limited to the distributional level, without delving into individual story quality or actual impact on children

## Related Work & Insights
- **vs Biased Tales (Rooein et al., 2025)**: The latter covers only English + a few languages; BiasedTales-ML extends to 8-language full-permutation design
- **vs StereoSet/BBQ**: Static classification benchmarks; this paper analyzes bias closer to real scenarios through long-form generation
- **vs Yong et al., 2025**: The latter studies cross-lingual transfer of safety interventions; this paper complements with representational safety analysis in non-adversarial scenarios

## Rating
- Novelty: ⭐⭐⭐⭐ Large-scale multilingual narrative bias analysis is a novel research direction; the full-permutation design methodology is valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ 350K stories, 8 languages, 3 models, multi-dimensional analysis, but lacks comparison with human narratives
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich visualizations, but discussion section is somewhat generic

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)

</div>

<!-- RELATED:END -->
