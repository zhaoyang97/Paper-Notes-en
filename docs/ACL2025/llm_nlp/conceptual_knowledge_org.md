---
title: >-
  [Paper Note] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian
description: >-
  [ACL 2025][LLM (Other)][Conceptual Categories] By constructing the first Italian subordinate-level psycholinguistic dataset (covering 187 basic categories), this work systematically compares the category organization structures of humans and LLMs at the subordinate concept level, finding low alignment overall but significant variation across different semantic domains.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Conceptual Categories"
  - "Subordinate Level"
  - "Typicality"
  - "Category Organization"
  - "Cognitive Plausibility"
date: 2026-05-08
content_hash: 028449b5891bff94
---

# How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian

**Conference**: ACL 2025  
**arXiv**: [2505.21301](https://arxiv.org/abs/2505.21301)  
**Code**: None (Dataset publicly available on GitHub and OSF)  
**Area**: LLM Reasoning / Cognitive Science  
**Keywords**: Conceptual Categories, Subordinate Level, Typicality, Category Organization, Cognitive Plausibility  

## TL;DR

By constructing the first Italian subordinate-level psycholinguistic dataset (covering 187 basic categories), this work systematically compares the category organization structures of humans and LLMs at the subordinate concept level, finding low alignment overall but significant variation across different semantic domains.

## Background & Motivation

**Background**:
   - Concepts are the "building blocks" of human cognition, and humans can comprehend the same entity across multiple taxonomic levels (e.g., grizzly bear $\rightarrow$ bear $\rightarrow$ animal).
   - The taxonomic hierarchy in cognitive science is categorized into three levels: superordinate (e.g., "animal"), basic (e.g., "bear"), and subordinate (e.g., "grizzly bear").
   - Prior research has primarily focused on the basic and superordinate levels, leaving exploration of the subordinate level highly limited.
   - While LLMs exhibit human-like performance in language understanding and generation, whether their conceptual organization aligns with that of humans remains highly debated.

**Limitations of Prior Work**:
   - No study has systematically investigated the differences in conceptual organization between humans and LLMs at the **subordinate category** level.
   - Prior LLM evaluations have been mostly conducted in English and at the superordinate level, lacking exploration in other languages and at the subordinate level.
   - LLMs frequently generate hallucinated subordinate category exemplars, which are highly inconsistent with the most typical exemplars produced by humans.

**Key Challenge**:
   - While the semantic knowledge of LLMs is derived purely from text distributions, human conceptual knowledge integrates both linguistic and sensorimotor experiences (e.g., vision, touch).
   - Subordinate categories depend more heavily on fine-grained perceptual details and linguistic compositional abilities than superordinate categories, making them an ideal scenario for testing LLM cognitive alignment.

**Goal**:
   - RQ1: How do humans construct and organize basic categories at the subordinate level?
   - RQ2: Do LLMs exhibit the same category organization structure as humans?

**Key Insight**:
   - Constructing a novel Italian psycholinguistic dataset where 365 human subjects generated subordinate exemplars for 187 basic concepts.
   - Probing multiple LLMs with the same task to systematically compare human and model outputs.

**Core Idea**:
   - Comparing the category organization structures of humans and LLMs at the subordinate concept level for the first time, exposing systematic biases in LLMs regarding fine-grained conceptual knowledge.

## Method

### Overall Architecture

The research is divided into two studies:
- **Study 1**: Human data collection and analysis (constructing the psycholinguistic dataset)
- **Study 2**: LLM probing experiments (exemplar generation + category induction + typicality detection)

### Key Designs

1. **Study 1: Human Psycholinguistic Dataset Construction**:

    - **Function**: Collecting subordinate exemplars produced by 365 native Italian speakers for 187 basic concepts (belonging to 12 superordinate categories).
    - **Mechanism**: Prompting participants to list as many subordinate types as possible for a given concept (e.g., "list a type of dog"), then calculating metrics such as dominance (production frequency), availability, and first occurrence.
    - Following data cleaning, 24,659 exemplars were collected, of which 1,696 main exemplars with dominance $\ge 0.1$ were retained.
    - **Key Findings**: Exemplar richness varied drastically across different categories, with FOOD having the most (270) and PLANTS the fewest (77).

2. **Study 2: LLM Exemplar Generation and Comparison**:

    - **Function**: Prompting multiple LLMs (LLaMA 3.1-8B/70B, LLaMA 3.2-3B, Mistral-7B, Mixtral-8x7B, NeMo-12B, LLaVA-7B, Idefics2-8B) to generate subordinate exemplars for the same 187 concepts.
    - **Key Metrics**:
        - Proportion of valid exemplars (validated via frequency in the Italian corpus ItTenTen)
        - Overlap rate with the top-$n$ most typical human-produced exemplars
    - Hallucination Analysis: LLMs tend to generate non-existent exemplars through structural extrapolation.

3. **Subtask A: Category Induction**:

    - **Function**: Providing the model with 10 of the most typical human-generated subordinate exemplars and tasking it to select their basic or superordinate category.
    - **Mechanism**: Selecting the best-matching category via perplexity.
    - Results: Basic category identification accuracy was high (reaching 98% for Mixtral-8x7B), but superordinate categorization was significantly more challenging (peaking at 64%).

4. **Subtask B: Typicality Detection**:

    - **Function**: Providing the model with one highly typical and one highly atypical exemplar, and asking it to judge which is more typical.
    - Evaluating whether LLMs are sensitive to the typicality gradient perceived by humans.

### Loss & Training

- No model training is involved; this is a pure inference/probing study.
- Few-shot prompting is used for exemplar generation.
- Perplexity is used to evaluate the classification tasks.

## Key Experimental Results

### Main Results

**Proportion of valid exemplars generated by LLMs**:

| Model | Proportion of Valid Exemplars |
|------|-------------|
| LLaMA-3.1-70B | 82% |
| NeMo-12B | ~75% |
| Mistral-7B | 52% |
| LLaVA-7B | 44% |

- The FOOD category yielded the highest validity rate (85%), while PLANTS yielded the lowest (52%).

**Top-$n$ exemplar overlap rate** (Humans vs. LLMs):

| Model | Top-1 | Top-3 | Top-5 |
|------|-------|-------|-------|
| nemo-12B | 0.25 | 0.24 | **0.24** |
| llama-3.1-70B | 0.18 | 0.20 | 0.21 |
| mistral-7B | 0.13 | 0.12 | 0.13 |
| idefics2-8B | 0.08 | 0.10 | 0.10 |

- Even the best-performing model (NeMo-12B) achieved a Top-5 overlap rate of only 24%.

**Category Induction Accuracy**:

| Model | Basic Category | Superordinate Category |
|------|---------|---------|
| mixtral-8x7B | 0.98 | 0.57 |
| llama-3.1-70B | 0.95 | 0.64 |
| llama-3.1-8B | 0.96 | 0.63 |

### Key Findings

1. **Conceptual organization between humans and LLMs is highly misaligned**: The best model achieved a Top-5 overlap rate of only ~24%.
2. **LLMs generate a massive number of hallucinated exemplars**: They produce plausible-sounding but non-existent combinations through structural extrapolation (e.g., 'oak-leaf geranium').
3. **Significant differences exist across semantic domains**: FOOD and ANIMALS show higher overlap rates (~29-37%), while BODY PARTS and FURNISHING exhibit the lowest (~12-16%).
4. **Vision-language models do not necessarily perform better**: LLaVA and Idefics2 performed even worse, suggesting that visual pre-training provides limited assistance for subordinate conceptual organization.
5. **LLMs exhibit a 'flattened' category organization**: They lack the typicality gradients found in humans, showing no clear availability ranking in generated exemplars.
6. **Basic category identification is far superior to superordinate classification**: LLMs know that 'a Labrador is a dog' but are less certain that 'a Labrador belongs to animals'.

## Highlights & Insights

- **First exploration of human-AI comparison at the subordinate level**: Fills an important gap in the intersection of cognitive science and NLP.
- **Cross-lingual perspective**: The study on Italian provides new benchmarking data for non-English LLM evaluation.
- **In-depth analysis of hallucination mechanisms**: Reveals systematic strategies of LLMs generating hallucinations by 'imitating known patterns for combinatoric extrapolation'.
- **Implications for LLM cognitive alignment research**: The way LLMs organize conceptual knowledge is fundamentally different from humans, especially at fine-grained levels.

## Limitations & Future Work

1. Validated only in Italian; cross-lingual generalizability remains unknown.
2. Using corpus frequency to validate exemplar validity might omit low-frequency but legitimate exemplars.
3. Definitions of subordinate categories may vary across cultures and languages.
4. The impact of the varying proportions of Italian in LLM training data on the results was not considered.
5. Evaluation of vision-language models was relatively brief and did not utilize image-based prompts.
6. Advanced prompting strategies, such as chain-of-thought, can be further explored to see if they improve LLM performance.

## Related Work & Insights

- **Rosch (1975, 1978)**: Founder of prototype theory, defining taxonomic hierarchies and typicality effects.
- **Nighojkar et al. (2022)**: Modeled semantic fluency tasks using Transformers, where RoBERTa-Large achieved only 16% accuracy.
- **Heyman and Heyman (2024)**: ChatGPT's typicality ratings shared a similarity of approximately 0.60–0.64 with humans.
- **Misra et al. (2021, 2023)**: LLM correlations with humans on fine-grained property attribution were only 0.24–0.41.
- **Insight**: The limitations of LLMs as 'cognitive models' are particularly evident at fine-grained semantic levels.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic human-AI comparison at the subordinate category level.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive, featuring human data, multi-LLM probing, and various subtasks.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with a thorough introduction to the cognitive science background.
- Value: ⭐⭐⭐⭐ — Provides important references for understanding LLM semantic organization and cognitive alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](human_nlp_cooperation_survey.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] HiCUPID: Exploring the Potential of LLMs as Personalized Assistants](exploring_the_potential_of_llms_as.md)

</div>

<!-- RELATED:END -->
