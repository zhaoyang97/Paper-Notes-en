---
title: >-
  [Paper Note] Culture is Not Trivia: Sociocultural Theory for Cultural NLP
description: >-
  [LLM (Other)] Starting from sociocultural linguistic theory, this paper points out the methodological limitations of current cultural NLP (coarse-grained national boundaries, static benchmarks, and the lack of a unified definition of culture), argues that culture is a dynamically constructed process rather than static knowledge, and proposes "localization" as a more viable research framework.
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: 37a62dad710aca32
---

# Culture is Not Trivia: Sociocultural Theory for Cultural NLP

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|--------|
| ACL 2025 | 2502.12057 | - | LLM / NLP / Culture | Cultural NLP, Sociocultural Linguistics, Localization, Indexicality Theory, Stereotypes |

## TL;DR

Starting from sociocultural linguistic theory, this paper points out the methodological limitations of current cultural NLP (coarse-grained national boundaries, static benchmarks, and the lack of a unified definition of culture), argues that culture is a dynamically constructed process rather than static knowledge, and proposes "localization" as a more viable research framework.

## Background & Motivation

**Core Problem:** The field of cultural NLP lacks a unified definition of culture, and various cultural proxies (nationality, religion, food, etc.) lead to recurring methodological limitations. How can this theoretical gap be bridged?

**Limitations of Prior Work:**
- **Under-representation (40% of papers):** Datasets only cover a small number of cultures, typically high-resource ones.
- **Coarse-grained Boundaries (36% of papers):** Using nationality as a proxy for culture, ignoring intra-national cultural heterogeneity.
- **Lack of Dynamism (12% of papers):** Culture evolves dynamically, but benchmarks remain static.
- **Proxy Indicator Limitations (37% of papers):** Food, values, etc., are only small facets of culture and cannot represent the whole.
- **Stereotype Dilemma (14% of papers):** "Cultural knowledge" mined from text consists mostly of stereotypes.

**Core Motivation:** These recurring limitations are not isolated technical issues but symptoms of a theoretical gap. It is necessary to draw upon established cultural theories to guide the construction of cultural capabilities in NLP systems.

## Method

### Overall Architecture

This paper is a **position paper** that does not propose a new model but rather:
1. Outlines four core objectives of cultural NLP (adaptability, discernment, inclusivity, and fine-grainedness).
2. Surveys common limitations across 57 papers.
3. Introduces a sociocultural linguistic theoretical framework.
4. Demonstrates the practical value of the theory through case studies.
5. Proposes "localization" as a more pragmatic research direction.

### Key Designs

**Five Principles of Sociocultural Linguistics (Bucholtz & Hall, 2005):**

1. **Emergence:** Identity and culture emerge through interaction rather than pre-existing. This supports dynamic cultural representation and inductive cultural categorization from data.
2. **Positionality:** Identity encompasses multiple levels, including macro-demographic categories, local social positions, and context-specific stances. Nationality is only one level.
3. **Indexicality:** Identity is constructed through the semiotic association of linguistic forms with social meanings. It distinguishes between first-order indexicality (actual use) and higher-order indexicality (discussions about use, i.e., stereotypes).
4. **Relationality:** Identities acquire meaning in relation to other identities. Contrastive learning may be more suitable for cultural representation than supervised classification.
5. **Partialness:** Any cultural description is inherently partial and incomplete, as it is itself situated within a specific context.

**Case Study — Indexicality Reveals the Essence of Stereotypes:**
- Mining "belief Y is widely accepted in culture X" from web texts actually learns high-order indexes (stereotypes).
- These "cultural facts" reflect the worldview of the text's author rather than the objective reality of the described culture.
- First-order indexes (such as actual linguistic variations observed through geographic location or community metadata) require different computational methods.

### Localization as an Alternative Framework

- **Actionability:** Specifying the application domain limits the required depth of cultural knowledge.
- **Explicit Audience:** Clearly enumerating the target user group makes the choice of cultural boundaries no longer arbitrary.
- **Interactional Positioning:** Forcing the definition of the NLP system's role and expected behavior in human-computer interaction.

## Experiments

This paper is a theoretical/position paper and does not contain traditional experiments. The core empirical contributions are:

### Literature Survey Results

| Category of Limitation | Proportion of Papers | Explanation |
|----------|-------------|------|
| Under-representation | 40% | Datasets only cover a small number of cultures |
| Proxy Indicator Limitations | 37% | Selected cultural proxies cannot represent the whole culture |
| Nationality as Proxy | 36% | The most commonly used but heavily criticized cultural boundary |
| Lack of Intra-group Variation | 28% | Ignores differences within cultural groups |
| Risk of Stereotypes | 14% | Collected cultural knowledge may contain harmful stereotypes |
| Lack of Dynamism | 12% | Static benchmarks fail to reflect cultural evolution |

### Key Findings

- "Cultural knowledge" mined from meta-discourse inherently yields only stereotypes (high-order indexicals); this is not an issue of data bias but a methodological limitation.
- The five principles of sociocultural linguistics provide corresponding theoretical explanations and potential resolution paths for each core challenge in cultural NLP.
- The localization framework decomposes the grand goal of "cultural competence" into actionable, domain-specific tasks, aligning more closely with practical system-building requirements.
- Even for the same linguistic style of the same speaker, its cultural meaning depends on the interactional context (e.g., a "foreign accent" can represent accommodation or mockery).

## Highlights & Insights

- Systematically introduces established cultural theories from social sciences into NLP, filling the theoretical gap in the field.
- Elegantly explains why mining from text inevitably yields stereotypes through indexicality theory.
- The "localization" framework offers a more pragmatic research direction than "cultural competence," helping to guide engineering practices.
- The literature survey methodology is rigorous, making the issues quantifiable through the analysis of limitations across 57 papers.

## Limitations & Future Work

- As a position paper, it lacks concrete algorithms or system implementations to validate the feasibility of the theoretical proposals.
- It primarily relies on the theoretical tradition of sociocultural linguistics, whereas other cultural theories might offer different perspectives.
- Discussions on how computational methods can concretely implement theoretical principles (e.g., learning representation spaces of social meaning) remain at the level of suggestions.
- The 57 surveyed papers may not be fully comprehensive and might not represent the entire field.

## Related Work & Insights

- **Surveys on Cultural NLP:** Comprehensive reviews of cultural LLM research by Adilazuarda et al. (2024) and Liu et al. (2024b).
- **Cultural Benchmarks:** Value surveys based on nationality (Cao et al., 2024a), benchmarks based on local knowledge (Koto et al., 2024).
- **Sociocultural Linguistics:** Bucholtz & Hall's (2005) framework of identity and interaction, Eckert's (2012) three waves of sociolinguistics, and Silverstein's (2003) indexicality theory.
- **Computational Sociolinguistics:** Grieve et al. (2019) studying lexical variation using Twitter geolocation, Lucy & Bamman (2021) studying semantic variation via subreddits.
- **Stereotype Studies:** Stereotype mitigation datasets of Jha et al. (2023) and Ma et al. (2023).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisiting Common Assumptions about Arabic Dialects in NLP](arabic_dialects_assumptions_revisited.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

</div>

<!-- RELATED:END -->
