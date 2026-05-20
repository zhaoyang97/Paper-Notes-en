---
title: >-
  [Paper Note] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual LLM] This paper introduces LocQA, a benchmark comprising 2,156 location-sensitive QA pairs across 12 languages and 49 regions. By employing geographically ambig…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual LLM"
  - "Geographic Bias"
  - "US-centrism"
  - "Cultural Localization"
  - "Implicit Bias"
date: 2026-05-08
content_hash: 62c2850198d21963
---

# Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs

**Conference**: ACL 2026
**arXiv**: [2604.19292](https://arxiv.org/abs/2604.19292)  
**Code**: [https://github.com/google-research-datasets/locqa/](https://github.com/google-research-datasets/locqa/)  
**Area**: Multilingual / Bias Analysis
**Keywords**: Multilingual LLM, Geographic Bias, US-centrism, Cultural Localization, Implicit Bias

## TL;DR

This paper introduces LocQA, a benchmark comprising 2,156 location-sensitive QA pairs across 12 languages and 49 regions. By employing geographically ambiguous queries (e.g., "What is the emergency phone number?"), it exposes implicit biases in LLMs: a persistent US-centric default across languages (50% of model responses contain US answers vs. only 26% in the data), a within-language "demographic probability engine" effect driven by population size, and an exacerbation of global bias following instruction fine-tuning.

## Background & Motivation

**Background**: Multilingual LLMs have substantially narrowed fluency gaps across languages. Existing cultural benchmarks (e.g., INCLUDE, Global-MMLU) evaluate explicit knowledge (e.g., "What is the capital of Peru?"), disambiguating the target region by explicitly specifying it in the prompt.

**Limitations of Prior Work**: (1) Existing multilingual evaluations conflate two distinct capabilities—linguistic fluency (generating fluent text) and localization (generating contextually relevant facts for the population using that language). (2) When regions are explicitly specified in prompts, models' implicit biases are masked, leaving their default behavior in the absence of geographic cues unexamined. (3) A single language often corresponds to multiple regions (French spans 29 countries), meaning language alone cannot uniquely determine region.

**Key Challenge**: A model may "know" the legal drinking age in Indonesia, yet if it defaults to the US standard when queried in Indonesian, that knowledge is effectively erased. A critical gap exists between possessing a fact and choosing to surface it.

**Goal**: (1) Quantify cross-lingual global bias (US-centrism) in LLMs; (2) Quantify within-language regional bias (population size effect); (3) Analyze the effect of instruction fine-tuning on bias.

**Key Insight**: Semantically invariant, geographically ambiguous queries—providing no regional cues in the prompt beyond the language itself—are used to analyze models' spontaneous default behavior, thereby exposing implicit biases.

**Core Idea**: By analyzing how models spontaneously resolve geographic ambiguity, one can map their tendencies, biases, and implicit representational hierarchies—aspects that explicit knowledge benchmarks cannot capture.

## Method

### Overall Architecture

LocQA contains 44 semantically parallel location-sensitive questions (covering domains such as law, dates, and measurements), translated into 12 languages, with answers annotated by bilingual annotators for 49 regions. Evaluation employs two metrics: global bias $B_{US}$ (the difference between the frequency of US answers in model outputs and the collision frequency in the data) and regional bias $B_R$ (the ratio of the frequency of each region's answers in model outputs to its distribution in the data). Gemini-2.5-Flash is used as an automated evaluator, achieving 92% agreement with human judgments upon validation.

### Key Designs

1. **Geographically Ambiguous Question Design**:

    - **Function**: Exposes models' default geographic assumptions by providing no regional cues.
    - **Mechanism**: Questions such as "What is the legal drinking age?" or "What is the emergency phone number?" have different answers across regions. By posing these in 12 languages, the model's responses reveal its default regional assumption. Collision-aware evaluation ensures that US answers are not miscounted when they happen to coincide with the target region's answer.
    - **Design Motivation**: Explicit benchmarks (e.g., "What is the emergency number in France?") test capability; LocQA tests tendency—the model's default behavior under ambiguity.

2. **Global Bias Metric $B_{US}$**:

    - **Function**: Quantifies the degree of US-centrism across languages.
    - **Mechanism**: $B_{US} = P_{\text{obs}}(A_{\text{US}}) - P_{\text{exp}}(A_{\text{US}})$, where $P_{\text{obs}}$ is the frequency of US answers in model outputs and $P_{\text{exp}}$ is the collision-aware expected frequency (the proportion of regions in the data whose answers happen to equal the US answer). Macro-averaged over 11 non-English languages. A positive value indicates that the model favors US norms beyond what random collision can explain.
    - **Design Motivation**: Naive counting conflates genuine US bias with answer collision (e.g., Indonesia and the US sharing the same drinking age). The collision-aware formulation eliminates this confound.

3. **Cultural Alignment Tax Analysis**:

    - **Function**: Quantifies the effect of instruction fine-tuning on bias.
    - **Mechanism**: The base and instruction-tuned versions of the same model are compared on $B_{US}$ and $B_R$. Instruction-tuned versions exhibit lower regional bias $B_R$ (more "equitable") but significantly higher global bias $B_{US}$ (more US-centric), forming a "cultural alignment tax"—current alignment practices pursue a more generic and "safer" homogeneity at the expense of cultural nuance.
    - **Design Motivation**: Does safety alignment come at the cost of geographic fairness? This question has direct policy implications for global AI deployment.

### Loss & Training

This is a purely evaluative work; no training is involved. Thirty-two models are evaluated in a zero-shot format, receiving only the question as input, with no instructions or examples.

## Key Experimental Results

### Main Results

**Distribution of Global Bias $B_{US}$ Across 32 Models**

| Model Group | $B_{US}$ Range | Notes |
|-------------|----------------|-------|
| Lowest bias | ~0 (Falcon 3) | Almost no US bias |
| Average | 0.24 | Mean across all models |
| Highest bias | 0.42 (Grok 4) | Severe US-centrism |
| Avg. share of US answers in outputs | 50% | vs. 26% in the data |

### Ablation Study

| Configuration | Global Bias $B_{US}$ | Regional Bias $B_R$ | Notes |
|---------------|----------------------|----------------------|-------|
| Base model | Lower | Higher | Greater geographic diversity but unequal |
| Instruction-tuned | **Significantly higher** | **Lower** | Cultural alignment tax: "safer" but more US-centric |

### Key Findings

- Nearly all models exhibit US bias—even when queried in non-English languages, models frequently default to US norms.
- Global bias increases after instruction fine-tuning—the alignment process may encode US norms as the "default safe" answer.
- Regional bias correlates strongly with population size—models behave like a "demographic probability engine," prioritizing larger-population regions.
- In Spanish, the US, Spain, Mexico, and Argentina are over-represented, while Honduras, Bolivia, and others are systematically "erased."
- In French, France is over-represented while Haiti, Congo, and Mali are underrepresented.

## Highlights & Insights

- The conceptual distinction between "capability vs. tendency" is a significant theoretical contribution—knowing a fact does not equal defaulting to it.
- The discovery of the "cultural alignment tax" has profound implications for AI alignment research—safety-oriented alignment may sacrifice geographic fairness.
- The LocQA design paradigm (geographic ambiguity + collision-aware metrics) is transferable to other implicit bias research settings.

## Limitations & Future Work

- The 12 languages and 49 regions still fail to cover the majority of the world's languages and regions.
- The scale of 44 questions is limited, and domain coverage could be substantially expanded.
- Translation quality may affect non-English evaluations.
- Future work should promote a shift from multilingual modeling toward multicultural and multi-regional modeling.

## Related Work & Insights

- **vs. INCLUDE/Global-MMLU**: These benchmarks test explicit cultural knowledge, whereas LocQA tests implicit geographic bias.
- **vs. M-RewardBench**: Focuses on multilingual judgment quality, whereas LocQA focuses on default geographic behavior.
- **vs. Han et al. (2025)**: Identifies the "transfer–localization trade-off"; LocQA provides quantitative tools for measuring it.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of implicit geographic bias in LLMs; significant conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 models, collision-aware metrics, base vs. instruction-tuned comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous argumentation, vivid examples, and clear policy implications.
- Value: ⭐⭐⭐⭐⭐ Direct and far-reaching impact on global AI fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[AAAI 2026\] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](../../AAAI2026/multilingual_mt/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)
- [\[ACL 2026\] Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic](vocab_diet_reshaping_the_vocabulary_of_llms_via_vector_arithmetic.md)

</div>

<!-- RELATED:END -->
