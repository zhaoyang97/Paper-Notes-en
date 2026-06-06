---
title: >-
  [Paper Note] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual LLMs] This paper proposes the LocQA benchmark (12 languages, 49 regions, 2156 geoculturally relevant QAs)…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual LLMs"
  - "Geocultural Bias"
  - "US-centrism"
  - "Cultural Localization"
  - "Implicit Bias"
date: 2026-05-08
content_hash: e8690291012686ab
---

# Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.19292](https://arxiv.org/abs/2604.19292)  
**Code**: [https://github.com/google-research-datasets/locqa/](https://github.com/google-research-datasets/locqa/)  
**Area**: Multilingual / Bias Analysis  
**Keywords**: Multilingual LLMs, Geocultural Bias, US-centrism, Cultural Localization, Implicit Bias

## TL;DR

This paper proposes the LocQA benchmark (12 languages, 49 regions, 2156 geoculturally relevant QAs), exposing implicit biases in LLMs through geoculturally ambiguous questions (e.g., "What is the emergency phone number?"): a persistent US-centric default behavior exists across languages (50% of model answers contain US answers vs. only 26% in the data), while intra-language biases exhibit a population-driven "population probability engine" effect, and instruction tuning exacerbates global bias.

## Background & Motivation

**Background**: Multilingual LLMs have significantly narrowed the gap in cross-lingual fluency. Existing cultural benchmarks (e.g., INCLUDE, Global-MMLU) test explicit knowledge ("What is the capital of Peru?"), where ambiguity is resolved by explicitly specifying the target region in the prompt.

**Limitations of Prior Work**: (1) Existing multilingual evaluations conflate two capabilities—linguistic fluency (generating fluent text) and localization (generating based on relevant facts for the population using that language); (2) Implicit biases in models are masked when regions are explicitly specified—the default behavior without regional prompts remains unknown; (3) A single language often corresponds to multiple regions (French covers 29 countries), so language alone does not uniquely identify a region.

**Key Challenge**: A model might "know" the drinking age in Indonesia, but if it defaults to US standards when asked in Indonesian, this knowledge is effectively "erased." There is a critical gap between knowing a fact and choosing to present it.

**Goal**: (1) Quantify cross-lingual global bias (US-centrism) in LLMs; (2) Quantify regional bias within the same language (population size effect); (3) Analyze the impact of instruction tuning on bias.

**Key Insight**: Using semantically invariant, geoculturally ambiguous queries—providing no regional cues in the prompt (except the language itself)—and analyzing the model's spontaneous default behavior to expose implicit bias.

**Core Idea**: By analyzing how models spontaneously resolve geocultural ambiguity, one can map the model's tendencies, biases, and implicit representation hierarchies—attributes that explicit knowledge benchmarks cannot capture.

## Method

### Overall Architecture

LocQA contains 44 semantically parallel geoculturally relevant questions (e.g., law, dates, measurements) translated into 12 languages, with answers provided by bilingual annotators for 49 regions. Evaluation uses two metrics: Global Bias $B_{US}$ (the difference between the frequency of US answers in model output and the collision frequency in data) and Regional Bias $B_R$ (the ratio of the frequency of each region's answer in model output to the distribution in data). Gemini-2.5-Flash is used as an automatic evaluator (with 92% agreement with human validation).

### Key Designs

1.  **Geoculturally Ambiguous Question Design**:

    - **Function**: Exposes the model's default geocultural assumptions by providing no regional cues.
    - **Mechanism**: Questions such as "What is the legal drinking age?" or "What is the emergency phone number?" have different answers in different regions. By asking in 12 languages, the model's responses reveal which region it defaults to. Collision-aware evaluation ensures US answers are not miscounted when the target region's answer happens to match the US answer.
    - **Design Motivation**: Explicit benchmarks ("What is the emergency number in France?") test capability; LocQA tests tendency—the model's default behavior under ambiguity.

2.  **Global Bias Metric $B_{US}$**:

    - **Function**: Quantifies the degree of cross-lingual US-centrism.
    - **Mechanism**: $B_{US} = P_{\text{obs}}(A_{\text{US}}) - P_{\text{exp}}(A_{\text{US}})$, where $P_{\text{obs}}$ is the frequency of US answers in model output and $P_{\text{exp}}$ is the expected collision-aware frequency (the proportion of regional answers in the data that happen to equal the US answer). This is macro-averaged across 11 non-English languages. A positive value indicates a model preference for US norms beyond what random collision can explain.
    - **Design Motivation**: Simple counts conflate true US bias with answer collisions (e.g., Indonesia and the US sharing the same drinking age). Collision-awareness eliminates this confusion.

3.  **Cultural Alignment Tax Analysis**:

    - **Function**: Quantifies the impact of instruction tuning on bias.
    - **Mechanism**: Compares $B_{US}$ and $B_R$ between base and instruction-tuned versions of the same model. The study finds that instruction-tuned versions have lower regional bias $B_R$ (more "equitable") but significantly higher global bias $B_{US}$ (more US-centric), forming a "cultural alignment tax"—current alignment practices pursue more general, "safer" homogeneity at the cost of cultural nuances.
    - **Design Motivation**: Does safety alignment come at the cost of geocultural equity? This question has direct policy implications for global AI deployment.

### Loss & Training

A pure evaluation work involving no training. 32 models were evaluated in a zero-shot format, with only the question as input, without instructions or examples.

## Key Experimental Results

### Main Results

**Global Bias $B_{US}$ Distribution across 32 Models**

| Model Group | $B_{US}$ Range | Description |
|:---|:---|:---|
| Lowest Bias | ~0 (Falcon 3) | Almost no US bias |
| Average | 0.24 | Average of all models |
| Highest Bias | 0.42 (Grok 4) | Severely US-centric |
| Average: Model output contains US answer | 50% | vs. 26% proportion of US answers in data |

### Ablation Study

| Configuration | Global Bias $B_{US}$ | Regional Bias $B_R$ | Description |
|:---|:---|:---|:---|
| Base Model | Lower | Higher | More regional diversity but unequal |
| Instruction Tuning | **Significantly Higher** | **Lower** | Cultural Alignment Tax: More "safe" but more US-centric |

### Key Findings

- Almost all models demonstrate US bias—frequently mentioning US norms even when asked in non-English languages.
- Global bias intensifies after instruction tuning—the alignment process may adopt US norms as "default safe" answers.
- Regional bias correlates strongly with population size—models prioritize high-population regions like a "population probability engine."
- In Spanish, the US, Spain, Mexico, and Argentina are over-represented, while Honduras, Bolivia, etc., are systematically "erased."
- In French, France is over-represented, while Haiti, Congo, and Mali are underestimated.

## Highlights & Insights

- The distinction between "capability vs. tendency" is a significant conceptual contribution—knowing a fact $\neq$ choosing it by default.
- The finding of a "cultural alignment tax" has profound implications for AI alignment research—pursuing safe alignment may sacrifice geocultural equity.
- The LocQA design paradigm (geocultural ambiguity + collision-aware metrics) is transferable to other implicit bias studies.

## Limitations & Future Work

- 12 languages and 49 regions still do not cover the majority of global languages and regions.
- The scale of 44 questions is limited; the covered domains could be broader.
- Translation quality may affect non-English evaluation.
- Future work should drive the shift from multilingual modeling to multicultural/multiregional modeling.

## Related Work & Insights

- **vs. INCLUDE/Global-MMLU**: These test explicit cultural knowledge, whereas LocQA tests implicit geocultural bias.
- **vs. M-RewardBench**: Focuses on multilingual judgment quality, whereas LocQA focuses on geocultural default behavior.
- **vs. Han et al. (2025)**: Identified a "transfer-localization trade-off," while LocQA provides quantification tools.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically quantify implicit geocultural bias in LLMs; significant conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 models, collision-aware metrics, comparison of base/instruction-tuned models.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous argumentation, vivid examples, clear policy implications.
- Value: ⭐⭐⭐⭐⭐ Direct and far-reaching impact on global AI equity research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[AAAI 2026\] GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](../../AAAI2026/multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)

</div>

<!-- RELATED:END -->
