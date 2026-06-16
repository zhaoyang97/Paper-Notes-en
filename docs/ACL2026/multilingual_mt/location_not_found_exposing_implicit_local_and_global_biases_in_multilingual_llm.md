---
title: >-
  [Paper Note] Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs
description: >-
  [ACL 2026][Multilingual & Translation][Paper Note] This paper introduces the LocQA benchmark (12 languages, 49 regions, 2,156 region-relevant Q&As) to reveal implicit biases in LLMs through geographically ambiguous questions (e.g., "What is the emergency phone number?"). It uncovers persistent cross-lingual US-centric defaults (50% of model responses contain US answers
tags:
  - ACL 2026
  - Multilingual & Translation
date: 2026-05-08
content_hash: bebd3e97c649bb12
---
# Location Not Found: Exposing Implicit Local and Global Biases in Multilingual LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.19292](https://arxiv.org/abs/2604.19292)  
**Code**: [https://github.com/google-research-datasets/locqa/](https://github.com/google-research-datasets/locqa/)  
**Area**: Multilingual / Bias Analysis  
**Keywords**: Multilingual LLMs, Regional Bias, US-centrism, Cultural Localization, Implicit Bias

## TL;DR

This paper introduces the LocQA benchmark (12 languages, 49 regions, 2,156 region-relevant Q&As) to reveal implicit biases in LLMs through geographically ambiguous questions (e.g., "What is the emergency phone number?"). It uncovers persistent cross-lingual US-centric defaults (50% of model responses contain US answers vs. 26% in the data) and a "population probability engine" effect driven by population size within languages. Furthermore, instruction tuning is found to exacerbate global bias.

## Background & Motivation

**Background**: Multilingual LLMs have significantly narrowed the gap in cross-lingual fluency. Existing cultural benchmarks (such as INCLUDE, Global-MMLU) test explicit knowledge (e.g., "What is the capital of Peru?"), which disambiguates queries by explicitly specifying the target region in the prompt.

**Limitations of Prior Work**: (1) Existing multilingual evaluations confuse two distinct capabilities—linguistic fluency (generating fluent text) and localization (generating facts relevant to the population using that language); (2) Implicit biases are masked when regions are explicitly specified—the default behavior without regional prompts remains unknown; (3) A single language often corresponds to multiple regions (e.g., French covers 29 countries), meaning language alone does not uniquely determine a region.

**Key Challenge**: A model might "know" the legal drinking age in Indonesia, but if it defaults to US standards when asked in Indonesian, this knowledge is effectively "erased." A critical gap exists between knowing a fact and choosing to present it.

**Goal**: (1) Quantify cross-lingual global bias (US-centrism) in LLMs; (2) Quantify intra-lingual regional bias (population size effect); (3) Analyze the impact of instruction tuning on bias.

**Key Insight**: By using semantically invariant, geographically ambiguous queries—providing no regional cues in the prompt except for the language itself—the model's spontaneous default behavior can be analyzed to expose implicit bias.

**Core Idea**: Assessing how a model spontaneously resolves geographical ambiguity maps its tendencies, biases, and implicit representation hierarchies—factors that explicit knowledge benchmarks fail to capture.

## Method

### Overall Architecture

LocQA consists of 44 semantically parallel region-relevant questions (e.g., law, dates, measurements) translated into 12 languages, with answers provided by bilingual annotators for 49 regions. Evaluation utilizes two metrics: Global Bias $B_{US}$ (the difference between the frequency of US answers in model output and the collision frequency in data) and Regional Bias $B_R$ (the ratio of regional answer frequency in model output to the distribution in data). Gemini-2.5-Flash serves as an automated evaluator (validated with 92% human agreement).

### Key Designs

**1. Geographically Ambiguous Question Design: Forcing models to expose default assumptions**

Existing cultural benchmarks typically specify the target region (e.g., "What is the emergency number in France?"), which resolves ambiguity and only tests knowledge. LocQA uses ambiguous questions (e.g., "What is the legal drinking age?", "What is the emergency phone number?") that have different answers across regions but lack regional cues in the prompt. The model's response directly reveals its assumed default region. To avoid misjudgment, the authors incorporate collision awareness: if a US answer happens to match the target region's answer (e.g., the drinking age in Indonesia is the same as in the US), it is not counted as bias.

**2. Global Bias Metric $B_{US}$: Decoupling "US-centrism" from answer collisions**

Simply counting US-centric answers can be misleading due to coincidental answer overlaps. The authors define global bias as the observed frequency minus the expected frequency of collisions:

$$B_{US} = P_{\text{obs}}(A_{\text{US}}) - P_{\text{exp}}(A_{\text{US}}),$$

where $P_{\text{obs}}$ is the frequency of US answers in model output and $P_{\text{exp}}$ is the expected proportion where regional answers match US standards in the data. This metric is macro-averaged across 11 non-English languages. A positive value indicates a preference for US norms beyond random chance (average $B_{US}=0.24$; 50% of outputs contain US answers vs. 26% in data).

**3. Regional Bias Metric $B_R$: Identifying the "population probability engine" via lift**

To determine which regions are favored within a single language (e.g., Spanish across 20 regions), the authors define regional bias as a lift ratio:

$$B_R(c) = \frac{N_{\text{model}}(c)}{N_{\text{data}}(c)},$$

where $N_{\text{model}}(c)$ is the count of model answers hitting the standard of region $c$, and $N_{\text{data}}(c)$ is the frequency of that answer in the data. This reveals that models act as "population probability engines," over-representing large populations (e.g., US, Spain, Mexico in Spanish) while systematically "erasing" smaller ones (e.g., Honduras, Bolivia).

**4. Cultural Alignment Tax Analysis: Testing if safety alignment costs regional equity**

The authors compare base and instruction-tuned versions of the same models. They identify a trade-off: instruction tuning lowers regional bias $B_R$ (making regions appear more "equal") but significantly increases global bias $B_{US}$ (making models more US-centric). This "cultural alignment tax" suggests that current alignment practices treat US norms as the "default safe" answer, sacrificing cultural nuance for homogenized safety.

### Loss & Training

This is a pure evaluation study and does not involve training. 32 models were evaluated in a zero-shot format, using only the question as input without instructions or exemplars.

## Key Experimental Results

### Main Results

**Distribution of Global Bias $B_{US}$ across 32 models**

| Model Group | $B_{US}$ Range | Description |
| :--- | :--- | :--- |
| Lowest Bias | ~0 (Falcon 3) | Almost no US bias |
| Average | 0.24 | Mean across all models |
| Highest Bias | 0.42 (Grok 4) | Severe US-centrism |
| Mean: US answers in output | 50% | vs. 26% US answers in data |

### Ablation Study

| Configuration | Global Bias $B_{US}$ | Regional Bias $B_R$ | Description |
| :--- | :--- | :--- | :--- |
| Base Model | Lower | Higher | More regional diversity but unequal |
| Instruction Tuned | **Significantly Higher** | **Lower** | Cultural Alignment Tax: "Safer" but more US-centric |

### Key Findings

- Nearly all models exhibit US bias; even when questioned in non-English languages, they frequently reference US norms.
- Global bias intensifies after instruction tuning, as the alignment process may involve adopting US norms as "default safe" responses.
- Regional bias is strongly correlated with population size; models prioritize high-population regions like a "population probability engine."
- In Spanish, the US, Spain, Mexico, and Argentina are over-represented, while countries like Honduras and Bolivia are systematically "erased."
- In French, France is over-represented, while Haiti, Congo, and Mali are undervalued.

## Highlights & Insights

- The distinction between "capability vs. tendency" is a major conceptual contribution—knowing a fact $\neq$ choosing it by default.
- The discovery of the "cultural alignment tax" provides deep insights into AI alignment—pursuing safety may inadvertently sacrifice regional equity.
- The LocQA design paradigm (geographical ambiguity + collision-aware metrics) is transferable to other implicit bias research.

## Limitations & Future Work

- 12 languages and 49 regions do not cover the majority of global diversity.
- The scale of 44 questions is limited; coverage of domains could be expanded.
- Translation quality may impact non-English evaluations.
- Future work should drive the shift from multilingual modeling to multicultural/multiregional modeling.

## Related Work & Insights

- **vs. INCLUDE/Global-MMLU**: Tests explicit cultural knowledge; LocQA tests implicit regional bias.
- **vs. M-RewardBench**: Focuses on multilingual judgment quality; LocQA focuses on regional default behavior.
- **vs. Han et al. (2025)**: Identifies the "transfer-localization trade-off"; LocQA provides quantification tools.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of implicit regional bias in LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 models, collision-aware metrics, base vs. instruction-tuned comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous argumentation, vivid examples, clear policy implications.
- Value: ⭐⭐⭐⭐⭐ Direct and profound impact on global AI fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](../../ACL2025/multilingual_mt/implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](../../AAAI2026/multilingual_mt/bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)

</div>

<!-- RELATED:END -->
