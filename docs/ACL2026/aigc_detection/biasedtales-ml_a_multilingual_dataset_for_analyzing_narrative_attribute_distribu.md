---
title: >-
  [Paper Note] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories
description: >-
  [ACL 2026][AIGC Detection][Multilingual bias] BiasedTales-ML constructs a multilingual corpus of approximately 350,000 LLM-generated children's stories across 8 languages. Through an exhaustive combinatorial prompt desig…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "Multilingual bias"
  - "narrative generation"
  - "social attribute distribution"
  - "cross-lingual consistency"
  - "children's stories"
date: 2026-05-08
content_hash: bffe063a7e684dca
---

# BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17008](https://arxiv.org/abs/2604.17008)  
**Code**: [https://huggingface.co/spaces/Linyuana/BIASEDTALES-ML](https://huggingface.co/spaces/Linyuana/BIASEDTALES-ML)  
**Area**: AIGC Detection  
**Keywords**: Multilingual bias, narrative generation, social attribute distribution, cross-lingual consistency, children's stories

## TL;DR
BiasedTales-ML constructs a multilingual corpus of approximately 350,000 LLM-generated children's stories across 8 languages. Through an exhaustive combinatorial prompt design and a distributional analysis framework, it reveals that **social attribute distributions in narratives vary significantly across languages**, and English-centric evaluations fail to reflect bias patterns in multilingual scenarios.

## Background & Motivation

**Background**: LLMs are increasingly utilized to generate narrative content (especially children's stories), which implicitly convey notions of social roles, occupations, and environments. Existing social bias research primarily focuses on English short-text tasks such as sentence completion and classification.

**Limitations of Prior Work**: (1) Short-text bias assessment cannot capture biases indirectly expressed through characters, settings, and plot structures in long-form narratives; (2) Existing bias benchmarks (e.g., StereoSet, BBQ) are static classification tasks, which are decoupled from real-world generation scenarios; (3) There is almost no systematic research on the cross-lingual consistency of biases in multilingual narrative generation.

**Key Challenge**: Safety alignment techniques like RLHF are predominantly developed based on English data and Western norms. However, a model's bias performance in other languages might be entirely different—conclusions of "safety" derived from English evaluations may not hold in low-resource languages.

**Goal**: (1) Construct a large-scale multilingual parallel narrative corpus; (2) Propose a systematic narrative-level social attribute distribution analysis framework; (3) Empirically investigate cross-lingual bias consistency.

**Key Insight**: Children's stories are selected as a controlled yet expressive narrative domain—they encourage positive and imaginative content while requiring the model to make structured choices regarding characters, settings, and social roles.

**Core Idea**: By employing an exhaustive combinatorial prompt design (systematically varying nationality × religion × social class × parental roles × child gender), parallel stories are generated across 8 languages. Bias is then analyzed using distributional measures rather than instance-level annotations.

## Method

### Overall Architecture
The pipeline consists of three stages: (1) **Prompt Design and Localization**: Constructing a standardized prompt template localized into 8 target languages by native speakers; (2) **Large-scale Parallel Generation**: Generating stories using 3 LLMs across all prompt configurations (with 5 independent samples per configuration); (3) **Narrative Feature Extraction and Distributional Analysis**: Using an LLM-based extractor to pull character traits, settings, and cultural references from stories, followed by statistical measures to compare distributional differences.

### Key Designs

1. **Exhaustive Combinatorial Prompt Design**:

    - **Function**: Constructing controlled cross-lingual comparative experiments.
    - **Mechanism**: Systematically combining 27 nationalities × 6 religions × 2 social classes × 3 parental roles × 3 child genders = 2,916 unique prompt configurations. Generation is performed for 8 languages × 3 models, with 5 samples per configuration, totaling 350,000 stories. Language selection covers non-gendered (EN/ZH/JA/KO), grammatical gender (ES/RU/AR), and low-resource (SW) languages.
    - **Design Motivation**: This design allows for the isolation of the impact of the language medium from cultural content, avoiding language-specific patterns that might be obscured by translation benchmarks.

2. **LLM-based Narrative Feature Extractor**:

    - **Function**: Extracting structured social attribute representations from long-form stories.
    - **Mechanism**: Utilizing Qwen-3-14B to extract a three-dimensional representation $E = (A_{\text{adj}}, V_{\text{env}}, C_{\text{cul}})$ from each story $S$: character description adjectives (e.g., brave, obedient), setting keywords (e.g., forest, kitchen), and cultural references (e.g., menorah, dates). Manual verification of 800 stories showed 85.6% accuracy with a Cohen's $\kappa = 0.618$.
    - **Design Motivation**: Narrative bias is expressed indirectly via character descriptions and scene settings, necessitating structured extraction beyond surface-level keywords.

3. **Multi-dimensional Distributional Bias Measures**:

    - **Function**: Quantifying and comparing cross-lingual social attribute distribution differences.
    - **Mechanism**: Four complementary metrics: (1) Directional Bias $S_C = \ln(P(C|g_m)/P(C|g_f))$ measuring the association direction between specific attribute categories and gender; (2) JSD measuring overall distributional divergence; (3) Cosine similarity measuring cross-lingual consistency of bias patterns; (4) Valid Story Rate (VSR) to monitor generation quality.
    - **Design Motivation**: A single metric cannot fully characterize bias—a multi-dimensional synthesis of direction, magnitude, consistency, and generation quality is required.

### Loss & Training
This is an evaluation and analysis study; no model training was performed. The vLLM inference framework was used with a relatively high sampling temperature to encourage narrative diversity.

## Key Experimental Results

### Main Results

| Analysis Dimension | Key Findings | Model |
|--------|------|------|
| Directional Bias | Communality descriptions bias towards female stories in all languages; intellect descriptions bias towards males in Arabic and Russian. | 8B models |
| Grammatical Gender Influence | Llama-3.1-8B shows higher JSD (greater bias divergence) in languages with grammatical gender; Qwen-3-8B shows no significant difference. | - |
| Cross-lingual Consistency | Qwen-3 shows high cross-lingual cosine similarity (consistent); Llama-3 shows large differences in bias patterns between English and low-resource languages. | - |
| Small Model Effect | 1B models show near-zero directional bias, not due to better safety, but because of **insufficient lexical diversity** reverting to generic patterns. | Llama-3.2-1B |

### Ablation Study

| Configuration | Effect | Description |
|------|---------|------|
| Gender Conditioning | Male → outdoor/action words; Female → family/relationship words | Consistent across languages |
| Social Class Conditioning | Working class → practical/labor words; Wealthy → leisure/aesthetic words | Observed in Qwen-3 data |
| Low-resource Languages | Swahili exhibits low VSR and high JSD | Particularly evident in 1B models |

### Key Findings
- Bias patterns observed in English **cannot** be simply extrapolated to other languages, especially low-resource ones.
- The relationship between model scale and bias is non-monotonic: small models are not "safer" but rather "more mediocre" due to lexical diversity bottlenecks.
- The impact of grammatical gender on bias divergence varies by model and is not a universal rule.
- Qwen-3 exhibits higher cross-lingual consistency than Llama-3, possibly reflecting differences in the multilingual coverage of their training data.

## Highlights & Insights
- The **exhaustive combinatorial experimental design** is the primary highlight: by systematically varying each social attribute dimension, the influence of various factors can be precisely isolated. This methodology is transferable to any NLP evaluation involving multi-factor analysis.
- The insight that **"small model bias appears low but is actually due to lack of capability"** is significant: it warns against using surface-level distributional uniformity to assert safety, as lexical poverty can also produce uniform distributions.
- Distributional bias analysis (as opposed to instance-level annotation) is better suited for large-scale generation scenarios, avoiding the lack of scalability inherent in sample-by-sample labeling.

## Limitations & Future Work
- The stories are exclusively LLM-generated and may not directly reflect bias patterns in human narratives.
- Feature extraction depends on LLMs, which may introduce its own extraction biases.
- Although the 8 selected languages are representative, they do not cover the vast array of low-resource languages.
- The analysis is limited to the distributional level and does not delve into individual story quality or the actual psychological impact on children.

## Related Work & Insights
- **vs Biased Tales (Rooein et al., 2025)**: The latter covers only English and a few other languages; BiasedTales-ML expands this to an 8-language exhaustive design.
- **vs StereoSet/BBQ**: While those are static classification benchmarks, this work analyzes bias performance through long-form generation, which is closer to real-world scenarios.
- **vs Yong et al., 2025**: The latter focuses on the cross-lingual transfer of safety interventions, whereas this work complements that by providing representational safety analysis in non-adversarial contexts.

## Rating
- Novelty: ⭐⭐⭐⭐ Large-scale multilingual narrative bias analysis is a novel direction; the combinatorial design methodology is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ 350k stories, 8 languages, 3 models, and multi-dimensional analysis, though missing comparison with human narratives.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich visualizations, though the discussion section is somewhat general.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection](detectrl-x_towards_reliable_multilingual_and_real-world_llm-generated_text_detec.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization](gigacheck_detecting_llm-generated_content_via_object-centric_span_localization.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)

</div>

<!-- RELATED:END -->
