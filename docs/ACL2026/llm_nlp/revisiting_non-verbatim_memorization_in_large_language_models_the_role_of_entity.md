---
title: >-
  [Paper Note] Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms
description: >-
  [ACL 2026][LLM/NLP][non-verbatim memorization] This paper constructs the RedirectQA dataset—leveraging Wikipedia redirect information to associate the same entity with multiple surface forms—and systematically investigates how non-verbatim memorization in LLMs is affected by entity naming variants. The findings show that factual memorization is neither purely surface-form-specific nor entirely surface-form-agnostic, and that entity-level frequency makes an independent contribution beyond surface-level frequency.
tags:
  - ACL 2026
  - LLM/NLP
  - non-verbatim memorization
  - entity surface forms
  - factual QA
  - frequency analysis
  - RedirectQA
date: 2026-05-08
content_hash: 42d8214eeef45ae0
---

# Revisiting Non-Verbatim Memorization in Large Language Models: The Role of Entity Surface Forms

**Conference**: ACL 2026
**arXiv**: [2604.21882](https://arxiv.org/abs/2604.21882)
**Code**: [https://huggingface.co/datasets/naist-nlp/RedirectQA](https://huggingface.co/datasets/naist-nlp/RedirectQA) (dataset)
**Area**: LLM/NLP
**Keywords**: non-verbatim memorization, entity surface forms, factual QA, frequency analysis, RedirectQA

## TL;DR
This paper constructs the RedirectQA dataset—leveraging Wikipedia redirect information to associate the same entity with multiple surface forms—and systematically investigates how non-verbatim memorization in LLMs is affected by entity naming variants. The findings show that factual memorization is neither purely surface-form-specific nor entirely surface-form-agnostic, and that entity-level frequency makes an independent contribution beyond surface-level frequency.

## Background & Motivation

**State of the Field**: Large language models store substantial factual knowledge in their parameters and can answer knowledge-intensive questions without external retrieval. Entity-based QA is a common framework for analyzing non-verbatim memorization, and prior work has shown that facts about low-frequency or low-prominence entities are less reliably memorized.

**Limitations of Prior Work**: In typical evaluations, each entity is queried using a single canonical surface form (e.g., the Wikipedia title). This makes it difficult to distinguish whether a model has "memorized facts about an entity" or merely "can access those facts via a specific name." For instance, a model may answer correctly for "Pelé" but fail for "Edson Arantes do Nascimento"—the underlying fact is identical, only the name differs.

**Root Cause**: A preliminary diagnosis reveals that 23.7% of canonical–redirect question pairs on Pythia-12B yield inconsistent predictions. This indicates that single-surface-form evaluations substantially underestimate the unreliability of fact access, and canonical-name-based evaluations may miss a large number of surface-conditioned failure cases.

**Paper Goals**: (1) Construct a QA dataset that holds fact triples constant while varying only the entity surface form; (2) systematically evaluate the effect of surface form variation on factual QA; (3) analyze the independent contributions of entity-level and surface-level frequency to memorization accuracy.

**Starting Point**: Wikipedia redirect pages serve as a natural resource for entity surface forms. Redirect pages carry category annotations (aliases, abbreviations, spelling variants, common errors, etc.), enabling controlled analysis.

**Core Idea**: Fix the fact triple and the gold answer, vary only the surface form of the subject entity, and construct the large-scale controlled QA dataset RedirectQA via Wikipedia's redirect structure to quantify the impact of surface form on fact access.

## Method

### Overall Architecture
RedirectQA is constructed in three steps: (1) collect fact triples (subject, relation, object) from Wikidata; (2) use Wikipedia redirect information to associate each subject entity with its canonical and redirect surface forms, grouped by category; (3) render surface instances into questions using relation-specific templates. The final dataset contains 30,560 surface instances, 14,672 fact triples, and 61,120 question realizations.

### Key Designs

1. **Redirect Category Grouping**:

    - Function: Classify surface form variants by type to support fine-grained consistency analysis.
    - Mechanism: Manually curate 33 frequent redirect categories from Wikipedia redirect pages, organized into three broad types: alternative names and abbreviations (e.g., birth name → stage name), spelling variants (e.g., with/without diacritics), and typical errors (e.g., common misspellings). Each type represents a different degree of lexical variation.
    - Design Motivation: Different naming variant types pose different challenges to models; categorization enables quantification of the difference in impact between "minor orthographic changes" and "major lexical changes."

2. **Dual-Template Disambiguation**:

    - Function: Reduce confounding effects of question wording on results.
    - Mechanism: Two question templates are used for each relation type—an original template and a semantically equivalent paraphrase generated by GPT-4o. Each surface instance is rendered into two question realizations, and results are averaged.
    - Design Motivation: Prior evidence shows that LLM predictions are sensitive to surface-level variation in question templates; the dual-template design ensures that observed surface form effects are not artifacts of a single template.

3. **Frequency Decomposition Analysis**:

    - Function: Disentangle the independent contributions of entity-level and surface-level frequency to memorization accuracy.
    - Mechanism: DBpedia Spotlight is applied at scale to the pretraining corpus for entity linking, computing for each entity its entity frequency (total count of all associated mentions) and surface frequency (count of a specific surface form used as a mention of that entity). Partial correlation analyses $\rho(\text{Ent}, \text{Acc} | \text{Surf})$ and $\rho(\text{Surf}, \text{Acc} | \text{Ent})$ are used to isolate the independent effects of the two frequency types.
    - Design Motivation: If accuracy correlates only with the frequency of a specific surface form, this supports a "strongly surface-specific" view; if entity frequency contributes independently beyond surface frequency, it indicates cross-surface-form coupling of knowledge.

### Loss & Training
This is an analysis paper; no model training is involved. Evaluation uses 15-shot prompting, and answer correctness is determined via alias-aware string matching.

## Key Experimental Results

### Main Results
Prediction consistency (canonical vs. redirect surface forms) is evaluated across 13 LLMs.

| Redirect Type | Consistency Trend | Notes |
|---|---|---|
| Spelling variants | Highest | Models are relatively robust to minor orthographic changes (case, punctuation, diacritics) |
| Alternative names / abbreviations | Lowest | Large lexical changes (aliases, acronyms) significantly disrupt fact access |
| Typical errors | Moderate | Partially robust to misspellings but not perfectly so |

### Frequency Analysis (Partial Correlation)

| Model | $\rho(\text{Ent}, \text{Acc} | \text{Surf})$ canonical | $\rho(\text{Surf}, \text{Acc} | \text{Ent})$ canonical |
|---|---|---|
| Pythia 12B | 0.148* | -0.009 |
| OLMo 2 32B | 0.113* | -0.032 |
| OpenSciRef 1.7B | 0.125* | 0.000 |

### Key Findings
- Across all model families, surface form variation causes non-negligible correctness reversals; even strong instruction-tuned and commercial models cannot achieve perfect consistency.
- For the canonical surface form subset, the partial correlation of entity frequency is consistently significantly positive and stronger than that of surface frequency, indicating cross-surface-form coupling in fact access rather than independent memorization per surface form.
- A noteworthy reverse pattern exists: models sometimes fail under the canonical name but succeed under an alternative name, suggesting that human-defined canonicality does not necessarily align with the surface forms through which LLMs most reliably access facts.
- Acronyms (e.g., NYT → The New York Times) represent the most challenging variant type.

## Highlights & Insights
- **The experimental design is particularly elegant**: Wikipedia redirects are exploited as a free, naturally occurring, category-annotated resource for surface form variants, enabling a rigorously controlled experiment in which fact triples are held constant and only surface forms vary. This "minimal change" experimental design is transferable to other robustness evaluation tasks.
- **Frequency decomposition** is the paper's most substantive contribution: it refines the coarse-grained "entity frequency → accuracy" analysis to the surface level, finding that entity frequency retains an independent contribution after controlling for surface frequency. This reveals a cross-surface-form knowledge coupling mechanism within LLMs, rather than independent storage of facts for each name.
- The conclusion that memorization is "neither purely surface-specific nor entirely surface-agnostic" provides a more accurate conceptual framework than a binary characterization, avoiding overly simplistic claims.

## Limitations & Future Work
- The dataset covers only English entities and 16 relation types; behavior across languages and additional relation types remains unknown.
- Using DBpedia Spotlight for entity linking may introduce bias (zero-frequency cases are filtered out), and linking quality may be poor for long-tail entities.
- Causality is not established—frequency correlations cannot directly imply causation, and confounding variables may exist (e.g., entity prominence simultaneously affecting frequency and training data quality).
- The paper does not explore how to leverage these findings to improve models (e.g., surface-form-augmented training, multi-name data augmentation).
- Evaluation uses only the 15-shot prompting format; different prompting strategies (e.g., zero-shot, chain-of-thought) may influence surface form sensitivity.
- The selection of redirect categories is based on manual curation and may omit certain important types of surface form variation.

## Related Work & Insights
- **vs. PopQA/SimpleQA**: These datasets use a single canonical name per entity; RedirectQA reveals inconsistencies they miss by diversifying surface forms.
- **vs. Kandpal et al.**: The prior work studies the relationship between entity frequency and memorization; this paper decomposes frequency into entity-level and surface-level components, finding independent contributions from both, providing a more fine-grained understanding.
- **Implications**: RAG systems should consider surface form diversity when querying (using multiple names to improve recall); knowledge editing methods should verify cross-surface-form consistency (i.e., whether editing one name propagates to other names).

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic study of the surface form dimension is novel, though the core intuition (naming variants affect QA) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 models, multi-category analysis, partial correlation decomposition, dual-template validation—extremely rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with well-justified design decisions throughout.
- Value: ⭐⭐⭐⭐ Provides meaningful insights into the mechanisms of factual knowledge storage in LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Scaling Equitable Reflection Assessment in Education via Large Language Models and Role-Based Feedback Agents](../../AAAI2026/llm_nlp/scaling_equitable_reflection_assessment_in_education_via_large_language_models_a.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)

<!-- RELATED:END -->
