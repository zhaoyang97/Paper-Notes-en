---
title: >-
  [Paper Note] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs
description: >-
  [ACL 2025][LLM (Other)][discourse particles] Using the English polysemous discourse particle "just" as a case study, this paper systematically evaluates the capabilities of LLMs to understand the fine-grained semantics of discourse particles through two metalinguistic experiments (few-shot semantic labeling and pairwise comparison) using both expert-constructed datasets and movie subtitle annotation data. The findings show that while models can distinguish broad categories (a…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "discourse particles"
  - "pragmatics"
  - "linguistic evaluation"
  - "sense disambiguation"
  - "LLM"
date: 2026-05-08
content_hash: 64ed01fb5666631e
---

# Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.04534](https://arxiv.org/abs/2506.04534)  
**Code**: [GitHub](https://github.com/sheffwb/IsItJUSTSemantics)  
**Area**: LLM/NLP  
**Keywords**: discourse particles, pragmatics, linguistic evaluation, sense disambiguation, LLM

## TL;DR

Using the English polysemous discourse particle "just" as a case study, this paper systematically evaluates the capabilities of LLMs to understand the fine-grained semantics of discourse particles through two metalinguistic experiments (few-shot semantic labeling and pairwise comparison) using both expert-constructed datasets and movie subtitle annotation data. The findings show that while models can distinguish broad categories (adjectival and temporal meanings), they fail to fully capture the subtle semantic differences of discourse particles (exclusive, unelaboratory, unexplanatory, and emphatic meanings).

## Background & Motivation

**Background**: Discourse particles are critical factors in language understanding, where a single word carries multiple semantic functions (e.g., "just" has at least 12 senses, such as exclusive, temporal, emphatic, unelaboratory, and unexplanatory). Existing research has extensively explored LLM capabilities in processing function words and discourse connectives.

**Limitations of Prior Work**: Prior work (Chan et al. 2024; Yung et al. 2024) has found deficiencies in LLMs' understanding of discourse relations, but these studies remain at a relatively coarse level. The performance of LLMs regarding the polysemous, subtle differences within a single word—especially discourse particles that are deeply studied in formal semantics—remains completely unknown.

**Key Challenge**: Discourse particles are extremely high-frequency in daily dialogue and crucial for discourse understanding, but their semantic differentiation highly depends on pragmatic context, and the boundaries between different senses are fuzzy (e.g., the unelaboratory sense of "Fido is just a dog" vs. the unexplanatory sense of "The lights just turn on and off"), which poses a unique challenge for LLMs.

**Goal**: To quantitatively evaluate whether LLMs can distinguish the fine-grained semantics of the discourse particle "just."

**Key Insight**: Grounded in formal semantic theory (the unified analysis of "just" by Deo and Thomas 2025), high-quality datasets are constructed by linguistic experts, and two complementary metalinguistic evaluation paradigms are designed.

**Core Idea**: Using expert-annotated data coupled with two evaluation paradigms (few-shot labeling and pairwise comparison), this work shows that LLMs possess significant deficiencies in understanding the fine-grained semantics of polysemous discourse particles.

## Method

### Overall Architecture

This paper falls under benchmark/evaluation studies. Two sets of experiments are designed around the English discourse particle "just": (1) Few-shot semantic labeling experiment—providing the model with definitions and examples of six senses, and having the model judge the semantic meaning of "just" in a sentence; (2) Pairwise comparison experiment—providing the model with two sentences and having it judge whether the usage of "just" is the same in both. The two sets of experiments utilize hand-crafted data (90 sentences, 15 sentences per sense) and annotated movie subtitle data (149 sentences), respectively.

### Key Designs

1. **Data Construction and Annotation**:

    - Function: Construct two high-quality datasets for evaluation.
    - Mechanism: (a) Hand-crafted corpus—90 unambiguous sentences (15 sentences per sense) are meticulously crafted by linguistics graduate students, ensuring that each sentence has a unique reading of the "just" sense; (b) Annotated corpus—149 movie subtitle sentences containing "just" are extracted from OpenSubtitles and annotated by 2 senior semanticists and 8 graduate students, with disagreements resolved by 2 additional senior annotators.
    - Design Motivation: Hand-crafted data ensures evaluation precision (eliminating ambiguity), while subtitle data tests actual performance in natural settings.

2. **Target Sense Selection**:

    - Function: Select 4 target senses + 2 control senses from the 12 senses in Deo and Thomas 2025.
    - Mechanism: The 4 target senses are exclusive ("Betsy just eats chicken nuggets"), unelaborate/unelaboratory ("Fido is just a dog"), unexplanatory ("The lights just turn on and off"), and emphatic ("This pumpkin bisque is just delicious!"); the 2 control senses are temporal and adjective.
    - Design Motivation: The 4 target senses are relatively distinguishable semantically but still subtly different, whereas the 2 control senses provide a clear gradient of difficulty (adjective is the easiest to distinguish, temporal is medium).

3. **Experiment 1: Few-shot Semantic Labeling**:

    - Function: Evaluate whether the model can correctly predict the semantics of "just" when provided with sense definitions and examples.
    - Mechanism: Use the conditional log probability $\arg\max_{l \in L} P_M(l|S)$ to select the label with the highest probability rather than parsing the model's free-text output. The minicons library is used to calculate the conditional probabilities.
    - Design Motivation: Avoid the difficulty of parsing lengthy generated text, and directly compare the attention probabilities of each label.

4. **Experiment 2: Pairwise Comparison**:

    - Function: Probe whether the model perceives the similarity/difference of "just" usages across two sentences without assuming the model knows the sense labels.
    - Mechanism: Given two sentences $s_i, s_j$, calculate $H^M_{ij} = \log P_M(\text{Yes}|Z_{ij}) - \log P_M(\text{No}|Z_{ij})$ and normalize to $[0,1]$ to generate heatmaps. Ideally, the values of same-sense sentence pairs should be higher than those of different-sense pairs.
    - Design Motivation: Eliminate reliance on sense label names, allow for gradience in semantic distinction, and provide a more flexible evaluation perspective than labeling experiments.

### Evaluation Models

Tested 10 instruction-tuned models: Llama-3-8b, Llama-3.2-1b/3b, Llama-3.3-70b, Mistral-7b-v0.3, OLMo-7b, OLMo2-7b/13b, Gemma2-2b/9b.

## Key Experimental Results

### Main Results (Few-shot Semantic Labeling Accuracy)

| Model | Hand-crafted Data (4 Target Senses) | Subtitle Data (No Context) | Subtitle Data (With Context) |
|------|---------------------|--------------------|--------------------|
| Llama-3.2-1b | ≈ chance (0.167) | ≈ chance (0.403) | - |
| Gemma-2-2b | +0.28 vs 1b | Decreased | - |
| Mistral-7b-v0.3 | One of the best performers | -0.24 avg | - |
| Gemma-2-9b | One of the best performers | Decreased | Further -0.05 |
| Llama-3.3-70b | On par with 7B | Decreased | +0.10 (still near chance) |

### Pairwise Comparison Experiment

| Model | Same/Different Distinction Significance | Cohen's d |
|------|-------------------|-----------|
| Llama-3.2-1b | p=.11 (Not significant) | - |
| Llama-3.3-70b | p<.005 | 2.32 |
| Gemma-2-9b | p<.005 | 1.91 |
| Mistral-7b-v0.3 | p<.005 | 1.66 |

### Key Findings
- A critical model scale threshold exists: models exhibit above-chance performance only above 2B parameters (1B -> 2B accuracy +0.28).
- The largest model (70B) does not perform significantly better than the best 7B/9B models (Mistral, Gemma-9b).
- Moving from hand-crafted data to natural subtitles, all models' accuracy drops by an average of 0.24, indicating poorer understanding in real-world scenarios.
- Contextual information (the preceding 2 dialogue lines) does not help models disambiguate; instead, it reduces accuracy by an average of 0.05 (except for the 70B model).
- In the pairwise comparison: adjectival and temporal meanings are well distinguished, but the distinction among the 4 target discourse particle senses is extremely weak.
- Control experiments (using words like "bat", "bank") prove that this method is effective for explicit polysemous words; hence, the performance bottleneck lies indeed in the subtlety of discourse particle semantics.

## Highlights & Insights
- Elegant research design, where two complementary evaluation paradigms (labeling vs. pairwise comparison) jointly provide robust evidence: the former tests explicit metalinguistic capabilities, while the latter tests implicit semantic perception, leading to consistent conclusions.
- An excellent integration of formal semantic theory and empirical NLP research; the data is professionally constructed by semantics experts, with quality far exceeding automatically generated benchmarks.

## Limitations & Future Work
- The study only examines a single English discourse particle "just"; whether the conclusions generalize to other discourse particles needs validation.
- Metalinguistic prompting methods might underestimate LLMs' linguistic capabilities (Hu and Levy 2023), but standard log-probability-based methods are not applicable to discourse particle semantic differentiation.
- Text-only modality is used, which cannot leverage intonation information (intonation is a key cue for disambiguation in spoken language).

## Related Work & Insights
- **vs Chan et al. 2024**: Examined the performance of ChatGPT on discourse relations (temporal, causal) but at a coarser granularity; this work focuses on the subtle polysemous differences within a word.
- **vs Pandia et al. 2021**: Studied the sensitivity of LMs to discourse connectives; this work delves deeper into the differentiation of specific senses of discourse particles.
- **vs Coppock and Beaver 2014**: A classical analysis of exclusive meanings in formal semantics; this work applies its semantic theory to LLM evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation of LLMs' understanding of the fine-grained semantics of discourse particles, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Wide coverage with two complementary experiments, 10 models, and control experiments; however, lack of comparison with closed-source large models (such as GPT-4).
- Writing Quality: ⭐⭐⭐⭐ Smooth connection between linguistic and NLP perspectives.
- Value: ⭐⭐⭐⭐ Unreveals a major blind spot of LLMs in pragmatic understanding, providing insights for both NLP semantic evaluation and computational linguistic experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility](meaning_beyond_truth_conditions_evaluating_discourse_level_understanding_via_ana.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](../../ACL2026/llm_nlp/understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ACL 2025\] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study](algorithmic_fidelity_german_opinion.md)

</div>

<!-- RELATED:END -->
