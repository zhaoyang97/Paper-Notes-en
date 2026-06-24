---
title: >-
  [Paper Note] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs
description: >-
  [ACL 2025][LLM (Other)][Arabic NLP] The Palm dataset, constructed over a year through a community-driven effort by 44 Arabic-world researchers, covers all 22 Arabic countries, 20 cultural themes, and 10 dialects, comprising 17,411 human-created instruction-response pairs for evaluating and improving Arabic cultural and dialectal capabilities of LLMs.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Arabic NLP"
  - "Cultural Awareness"
  - "Dialectal Arabic"
  - "Instruction Dataset"
  - "LLM Evaluation"
  - "Community-Driven"
date: 2026-05-08
content_hash: cddd6db7ac31e392
---

# Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.00151](https://arxiv.org/abs/2503.00151)  
**Code**: [github.com/UBC-NLP/palm](https://github.com/UBC-NLP/palm)  
**Area**: LLM NLP / Multilingual & Cultural Alignment  
**Keywords**: Arabic NLP, Cultural Awareness, Dialectal Arabic, Instruction Dataset, LLM Evaluation, Community-Driven

## TL;DR

The Palm dataset, constructed over a year through a community-driven effort by 44 Arabic-world researchers, covers all 22 Arabic countries, 20 cultural themes, and 10 dialects, comprising 17,411 human-created instruction-response pairs for evaluating and improving Arabic cultural and dialectal capabilities of LLMs.

## Background & Motivation

**Core Problem**: LLM training data is dominated by English, and translated data introduces Western/Anglo-centric bias. For example, some Arabic LLMs suggest drinking beer after a user prays—completely defying Arabic cultural values, religious practices, and social norms.

**Complexity of the Arabic World**:
   - Covers 22 countries across Africa and Asia, with a population exceeding 450 million.
   - Diverse local cultures, customs, traditions, political systems, and social practices.
   - Linguistic diversity: Classical Arabic, Modern Standard Arabic (MSA), and various Dialectal Arabic (DA) varieties, where dialectal differences can scale to national levels.

**Limitations of Prior Work**:
   - **AYA**: Only 5K Arabic instructions within 204K multilingual instructions, with limited cultural coverage.
   - **AraDiCE**: Based on translation and data redirection, non-original; covers only 6 dialects.
   - **CIDAR**: Based on localization, not built from scratch.
   - **BLEnD**: Covers only 1 Arabic country.
   - No single dataset simultaneously satisfies: covering all 22 countries + multiple dialects + completely human-created + open-ended instructions.

**Limitations of Prior Arabic LLMs**: Instruction data for models like JAIS and AceGPT are primarily machine-generated/translated, lacking evaluation on country-specific culture.

## Method

### Team Structure

- **44 Researchers**: All co-authors of the paper, who are local native speakers from 15 Arabic countries.
- **Coverage Strategy**: 15 countries have local researchers; the remaining 7 (Bahrain, Comoros, Djibouti, Iraq, Libya, Qatar, Somalia) are covered by researchers from neighboring countries.
- **Diversity Quality Assurance**: At least 2 annotators per country from different regions to ensure cultural and dialectal coverage.

### Annotation Guidelines

- **Iterative Development**: Perfected through $\sim 3$ months of iteration, led by 4 senior members, reaching a final version of 100 pages.
- **Two Main Categories of Instructions**:
  1. **General Class**: General knowledge written in MSA (science, technology, etc.).
  2. **Country-Specific Class**: Content reflecting country-specific culture, written in MSA or local dialect—covering 20 topics such as celebrations, customs, geography, history, proverbs, and food.

### Data Construction Pipeline

1. **Collection**: Gathering content from reliable sources (official information, academic literature, etc.).
2. **Annotation**: Using the Label Studio platform, grouped by country.
3. **Quality Assurance**: Weekly meetings + real-time collaboration via Slack channels.
4. **Cross-Review**: Once completed, team members cross-reviewed, ensuring each sample was checked by at least 2 reviewers.

### Dataset Analysis

- **Total Scale**: 17,411 instruction-response pairs.
- **High-Resource Countries** ($>500$ samples): Egypt, Jordan, Mauritania, Morocco, Palestine, Saudi Arabia, Sudan, Syria, Tunisia, UAE, Yemen + General (1,109 samples) = 16,066 samples (92%).
- **Low-Resource Countries** ($\sim 100$ samples): 11 countries including Algeria and Bahrain = 1,345 samples (8%).
- **Dialectal Data**: Around 380 dialectal examples for each of the 10 high-resource countries, totaling 4,211 dialectal instructions.
- **6 Instruction Types**: Summarization/Explanation, Instruction/Procedure, Fact/Information Search, Creativity/Construction, Analysis/Evaluation, Narration/Description.
- **Data Split**: Train 13,559 + Public Test 1,926 + Private Test 1,926.

### Evaluation Setup

#### Surface-Level Evaluation
- **Repetition Rate**: Measures sequence repetition in the generated responses.
- **Language Consistency**: Whether Arabic instructions receive responses in Arabic.
- **Dialect Consistency**: Whether dialectal instructions receive responses in dialect.

#### LLM-as-Judge Evaluation
- **Judge Models**: GPT-4o, CMDR+, QWEN 2.5-72B.
- **Scoring**: Correctness scoring from 1-10, based on the instruction and ground truth.
- **Evaluator Consistency**: $\text{ICC} = 0.68$ (good agreement).

#### Human Evaluation
- **Scale**: 92 MSA + 20 dialectal samples for each of 4 countries = 172 samples.
- **Evaluators**: The same group of researchers from the data creation team.
- **Consistency with Automatic Evaluation**: Pearson $r = 0.76$ ($p < 0.05$), $\text{ICC} = 0.78$.

## Experiments

### Surface-Level Results (Table 2)

| Model | Repetition Rate↓ | Language Consistency↑ | Dialect Consistency↑ |
|------|---------|------------|------------|
| GPT-4o | 0.00 | 91.07 | 8.33 |
| Claude-3.5-Sonnet | 0.00 | 91.02 | 9.44 |
| Qwen2.5-72B | 0.00 | 91.33 | 8.33 |
| gemma-2-9b | **42.47** | 90.76 | 6.11 |
| Llama-3.1-8B | 8.20 | 91.28 | 10.56 |

**Key Findings**:
- Dialect consistency is extremely low across all models ($<11\%$); even when prompted in a dialect, approximately 90% of responses are in MSA.
- Gemma-2-9b suffers from severe repetition issues (42%).
- Language consistency is generally high ($\sim 91\%$).

### LLM-as-Judge Correctness Scoring

**Overall Ranking** (Figure 4a): GPT-4o and Claude-3.5-Sonnet perform best (median $>6.0$) $>$ CMDR+/Qwen2.5-72B ($\sim 5.8$-$6.0$) $>$ AceGPT/Gemma/Llama-70B ($\sim 4.5$-$5.0$) $>$ Jais-13b/Llama-8B ($\sim 3.0$-$4.0$).

**National Discrepancies** (Figure 4b):
- GPT-4o: Syria 7.5, Djibouti 7.3 (highest); some countries are lower.
- Jais-13b: Saudi Arabia 2.2 (lowest).
- Claude: Yemen 7.0 but Lebanon only 5.8, showing significant disparities across countries.

**Dialect Discrepancies** (Figure 5):
- Egyptian and Tunisian dialects perform better (GPT-4o: 8.1/8.4).
- Moroccan and Palestinian dialects are challenging for most models.
- Smaller models generally score $<4.0$ on dialects.

### Human Evaluation Results (Table 3 - Dialect)

| Country | AceGPT-32B | Llama-8B | Qwen-72B | Claude-3.5 | Jais-13b |
|------|-----------|---------|---------|-----------|---------|
| Egypt | **6.47** | 4.26 | 4.71 | 4.15 | 4.08 |
| Morocco | 4.55 | 2.87 | 4.44 | **6.23** | 3.10 |
| Syria | 3.27 | 3.40 | 4.03 | **4.65** | 2.27 |
| Yemen | 2.13 | 1.85 | 2.58 | **4.28** | 2.90 |

Models perform differently across countries' dialects, and no single model dominates overall—AceGPT excels in Egyptian dialect, while Claude performs best in Moroccan, Syrian, and Yemeni dialects.

### Model Scale and Performance

Model scale positively correlates with performance ($>70B$ models generally outperform smaller ones), yet even the largest models have cultural blind spots—consistently performing poorly on certain countries.

## Highlights & Insights

1. **Community-Driven Data Construction Paradigm**: Fully annotated by 44 local researchers who are all credited as authors, setting a benchmark for inclusivity and ownership in NLP dataset development.
2. **Unparalleled Coverage**: The first dataset covering all 22 Arabic countries $\times$ 20 cultural themes $\times$ 10 dialects, filling a critical gap.
3. **Revealing Dialectal Capabilities**: Dialect consistency is $<11\%$ across all models, exposing severe limitations of current LLMs in non-standard linguistic forms.
4. **Disparity in Cultural Equity**: Certain countries (e.g., Egypt, UAE) are well-represented in existing models, while others (e.g., Iraq, Mauritania, Yemen) are severely underrepresented.
5. **Validation of Evaluation Methods**: The high agreement between human evaluation and LLM-as-Judge ($r=0.76$, $\text{ICC}=0.78$) builds confidence in applying automatic evaluation methods to cultural domains.

## Limitations & Future Work

1. Content for low-resource countries is supplemented by annotators from neighboring countries, potentially lacking local depth.
2. Many Arabic countries have multiple regional dialects internally, and the current coverage is not fully complete.
3. LLM automatic evaluation may exhibit biases when processing dialects and nuanced cultural elements.
4. Some countries have relatively small volumes of data ($\sim 100$ samples), which might not fully represent their entire cultural landscape.
5. Recently released Arabic LLMs, such as Fanar and Allam, have not been evaluated.

## Related Work & Insights

- **Arabic LLMs**: JAIS, AceGPT, Jasmine, NileChat, Fanar, Allam, etc.
- **Arabic Encoders**: AraBERT, CAMeLBERT, MARBERT, etc.
- **Cultural Evaluation**: World Value Survey alignment evaluation (AlKhamissi et al.), CALText bias detection (Naous et al.).
- **Multilingual Instruction Data**: AYA (204K multilingual), BLEnD (cultural blind-test).

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐ The dataset construction method itself is not brand new, but the coverage scale and community-driven paradigm are unique contributions.
- **Data Quality**: ⭐⭐⭐⭐⭐ Fully human-created + 100-page annotation guidelines + cross-review + human evaluation validation.
- **Analysis Depth**: ⭐⭐⭐⭐ Very rich multi-dimensional analysis spanning country, dialect, and thematic levels.
- **Social Impact**: ⭐⭐⭐⭐⭐ Takes a significant step forward for the cultural representation of 450 million Arabic speakers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] A Modular Dataset to Demonstrate LLM Abstraction Capability](a_modular_dataset_to_demonstrate_llm_abstraction_capability.md)
- [\[ACL 2025\] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)

</div>

<!-- RELATED:END -->
