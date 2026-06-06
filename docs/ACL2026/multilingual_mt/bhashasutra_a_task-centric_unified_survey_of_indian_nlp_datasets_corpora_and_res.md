---
title: >-
  [Paper Note] BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources
description: >-
  [ACL 2026][Multilingual & Machine Translation][Indian Language NLP] This is the first unified survey dedicated to Indian language NLP resources, covering 200+ datasets, 50+ benchmarks…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Indian Language NLP"
  - "Dataset Survey"
  - "Multilingual Resources"
  - "Low-Resource Languages"
  - "Cultural NLP"
date: 2026-05-08
content_hash: a579fed55adefd80
---

# BhashaSutra: A Task-Centric Unified Survey of Indian NLP Datasets, Corpora, and Resources

**Conference**: ACL 2026
**arXiv**: [2604.18423](https://arxiv.org/abs/2604.18423)  
**Code**: None  
**Area**: Multilingual Translation
**Keywords**: Indian Language NLP, Dataset Survey, Multilingual Resources, Low-Resource Languages, Cultural NLP

## TL;DR

This is the first unified survey dedicated to Indian language NLP resources, covering 200+ datasets, 50+ benchmarks, and 100+ models/tools. Resources are organized under 17 task categories spanning core linguistic processing to sociocultural tasks. The survey systematically analyzes persistent challenges including uneven language coverage, annotation fragmentation, and evaluation inconsistency.

## Background & Motivation

**Background**: India possesses one of the most linguistically diverse ecosystems in the world (22 official languages, hundreds of dialects). Indian language NLP has advanced rapidly in recent years, with datasets, benchmarks, and pretrained models emerging across domains such as healthcare, law, education, and governance.

**Limitations of Prior Work**: Progress remains severely fragmented—most work concentrates on a small number of relatively resource-rich languages, with substantial variation in quality and documentation scattered across disparate publication venues. Existing surveys either cover a narrow scope (targeting specific task families) or subsume Indian languages within broader multilingual settings, leaving a gap for a dedicated, full-task-coverage survey of Indian NLP.

**Key Challenge**: A large gap exists between the rate of resource growth in Indian language NLP and the systematic consolidation of that growth, making it difficult for researchers to obtain a global picture of the field or identify genuine resource gaps.

**Goal**: To provide the first unified, task-centric survey of Indian NLP resources, encompassing text, speech, multimodal, and cultural tasks.

**Key Insight**: Organizing resources by task category (rather than by language) allows researchers to quickly locate available resources for a specific task direction.

**Core Idea**: A taxonomy of six major categories and seventeen fine-grained tasks is constructed to systematically catalog 200+ datasets and analyze resource coverage patterns and gaps.

## Method

### Overall Architecture

The survey organizes Indian NLP into six major categories: (1) core linguistic processing (tokenization, POS tagging, NER); (2) text classification and semantics (sentiment analysis, hate speech detection, topic classification, NLU); (3) generation and translation (summarization, machine translation, question answering); (4) retrieval and interaction (information retrieval, dialogue systems); (5) speech and multimodal; and (6) sociocultural and emerging tasks (misinformation detection, cultural reasoning, etc.).

### Key Designs

1. **Task-Centric Taxonomy**

    - **Function**: Provides a unified organizational framework for fragmented Indian NLP resources.
    - **Mechanism**: Resources are organized not by language (which would introduce extensive redundancy) but across 17 NLP task subdivisions. Under each task, key datasets, benchmarks, and tools are aggregated, including both monolingual resources targeting specific languages and multilingual resources that include English.
    - **Design Motivation**: Researchers typically seek resources starting from a task requirement; a task-centric organization better aligns with practical usage patterns.

2. **Resource Coverage Analysis and Visualization**

    - **Function**: Reveals uneven distribution patterns in Indian NLP resources.
    - **Mechanism**: A language-level resource count visualization (Figure 2) illustrates which languages are well-resourced and which are severely underserved. Hindi, Bengali, and Tamil are the most resource-rich, while Northeastern languages (Assamese, Manipuri, etc.) and endangered dialects are critically underrepresented. Multilingual resources are uniformly labeled as "Indic Languages."
    - **Design Motivation**: To quantify the degree of resource imbalance and direct the community toward the most urgent resource-building needs.

3. **Cross-Task Gap and Cultural Challenge Analysis**

    - **Function**: Identifies systemic challenges in Indian NLP.
    - **Mechanism**: The analysis examines issues that cut across multiple tasks: language imbalance, annotation fragmentation, domain skew, evaluation inconsistency, and cross-lingual brittleness. Special attention is given to sociocultural challenges (bias evaluation, code-mixing, misinformation) and cultural fidelity issues in translation pipelines.
    - **Design Motivation**: To move beyond single-task perspectives and surface structural problems at the ecosystem level.

### Loss & Training

As a survey paper, no technical implementation is involved.

## Key Experimental Results

### Main Results

Resource statistics overview:

| Category | Count |
|----------|-------|
| Datasets | 200+ |
| Benchmarks | 50+ |
| Models / Tools / Systems | 100+ |
| Tasks Covered | 17 |
| Modalities Covered | Text + Speech + Multimodal |

### Ablation Study

Language coverage analysis:

| Resource Level | Representative Languages | Characteristics |
|----------------|--------------------------|-----------------|
| High-resource | Hindi, Bengali, Tamil | Complete multi-task coverage |
| Medium-resource | Telugu, Marathi, Kannada | Coverage on core tasks |
| Low-resource | Assamese, Odia, Manipuri | Data available for only a few tasks |
| Extremely low-resource | Bhojpuri, Maithili, Santhali | Nearly no dedicated resources |

### Key Findings

- Hindi has the richest resources yet still exhibits task-coverage gaps; Northeastern languages and endangered dialects are critically underserved.
- Code-mixing (e.g., Hinglish) constitutes a distinctive challenge for Indian NLP and is pervasive across tasks such as sentiment analysis and hate speech detection.
- A large proportion of Indian language resources are constructed via translation pipelines; while this enables rapid scaling, it may fail to capture native linguistic, pragmatic, and sociocultural nuances.
- Evaluation practices are severely inconsistent: standards for reporting train/dev/test splits, metric definitions, and annotation procedures vary widely across works.

## Highlights & Insights

- This is currently the most comprehensive resource map for Indian language NLP and serves as essential reading for any researcher seeking to conduct NLP work on Indian languages. The task-centric organization makes it highly practical.
- The discussion of the trade-off between translation-pipeline construction and native data collection addresses a central issue in low-resource NLP: scalability and cultural fidelity are often difficult to achieve simultaneously.
- Treating sociocultural tasks (bias, misinformation, cultural reasoning) as a distinct category reflects the broader shift in NLP research toward social impact.

## Limitations & Future Work

- The survey cannot fully keep pace with a rapidly evolving ecosystem, as new datasets and models continue to emerge.
- The work synthesizes published literature rather than replicating experiments, so datasets from industry or with incomplete documentation may be missed.
- No explicit quality ranking of resources is provided, given the large variation in evaluation criteria across tasks and modalities.
- Discussion of efficiency and hardware constraints is insufficient—the accessibility of large models in low-resource environments represents an important challenge.

## Related Work & Insights

- **vs. Kakwani et al. (2020) / IndicNLP**: Prior work has narrower scope, focusing primarily on basic NLP tools; this survey covers 17 tasks and incorporates cultural and social dimensions.
- **vs. General Multilingual NLP Surveys**: Existing surveys treat Indian languages as a small subset of a broader multilingual setting, which is insufficient to capture India-specific challenges such as code-mixing, script diversity, and caste-related bias.
- **vs. AI4Bharat**: AI4Bharat is an important resource-building initiative but is not a survey; this paper systematically consolidates all available resources, including those from AI4Bharat.

## Rating
- Novelty: ⭐⭐⭐⭐ — First dedicated unified survey of Indian language NLP resources
- Experimental Thoroughness: ⭐⭐⭐ — No experiments as a survey, but resource coverage is exceptionally comprehensive
- Writing Quality: ⭐⭐⭐⭐ — Clear organization and practical taxonomy
- Value: ⭐⭐⭐⭐⭐ — Significant reference value for the Indian language NLP community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP](scripts_through_time_a_survey_of_the_evolving_role_of_transliteration_in_nlp.md)
- [\[ACL 2026\] No One Fits All: From Fixed Prompting to Learned Routing in Multilingual LLMs](no_one_fits_all_from_fixed_prompting_to_learned_routing_in_multilingual_llms.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents](indotabvqa_a_benchmark_for_cross-lingual_table_understanding_in_bahasa_indonesia.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)

</div>

<!-- RELATED:END -->
