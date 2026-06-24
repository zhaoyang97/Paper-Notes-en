---
title: >-
  [Paper Note] Expect the Unexpected? Testing the Surprisal of Salient Entities
description: >-
  [ACL 2026][LLM (Other)][Uniform Information Density] This paper investigates the relationship between discourse-level salient entities and surprisal. Using over 70K manually annotated entity mentions and a novel minimal-pair prompting method, the study finds that while global salient entities are themselves more unexpected (higher surprisal), they systematically reduce the surprisal of surrounding content. This effect varies by genre, being strongest in texts with high topic…
tags:
  - "ACL 2026"
  - "LLM (Other)"
  - "Uniform Information Density"
  - "Discourse Salience"
  - "Surprisal"
  - "Entity Prominence"
  - "Discourse Structure"
date: 2026-05-08
content_hash: 3f0326bc9d079450
---

# Expect the Unexpected? Testing the Surprisal of Salient Entities

**Conference**: ACL 2026  
**arXiv**: [2604.10724](https://arxiv.org/abs/2604.10724)  
**Code**: None  
**Area**: Computational Linguistics / Information Theory  
**Keywords**: Uniform Information Density, Discourse Salience, Surprisal, Entity Prominence, Discourse Structure

## TL;DR

This paper investigates the relationship between discourse-level salient entities and surprisal. Using over 70K manually annotated entity mentions and a novel minimal-pair prompting method, the study finds that while global salient entities are themselves more unexpected (higher surprisal), they systematically reduce the surprisal of surrounding content. This effect varies by genre, being strongest in texts with high topic coherence.

## Background & Motivation

**Background**: The Uniform Information Density (UID) hypothesis suggests that speakers tend to distribute information uniformly across an utterance, keeping surprisal roughly constant. However, multiple studies have identified systematic deviations—"competing pressures" such as phonetic constraints (high surprisal at word starts), syntactic constraints, and discourse structure constraints create local non-uniformity.

**Limitations of Prior Work**: (1) Previous UID research largely ignored the relative salience of discourse participants—who the "protagonists" of the text are; (2) existing results are contradictory regarding whether salient entities are more predictable or more unexpected; (3) various factors (grammatical role, recency, referring form, etc.) influence entity predictability, making it difficult to isolate the effect of salience in natural contexts.

**Key Challenge**: On one hand, salient entities may be more predictable due to repeated mentions; on the other hand, as primary information carriers, they may contain higher information content. How do these two effects interact at the discourse level?

**Goal**: To systematically study the relationship between global entity salience and surprisal for the first time, distinguishing between the surprisal of the entity itself and the effect of the entity on the surprisal of surrounding content.

**Key Insight**: Utilize manual annotations from the GUM-SAGE dataset (salience scores based on summary consistency) and the diversity of 16 genres, combined with a minimal-pair prompting method to control for confounding factors.

**Core Idea**: Global salient entities act as "anchors"—they carry more information themselves (high surprisal) but significantly reduce the uncertainty of subsequent content by establishing topic expectations, creating a local surprisal "trough."

## Method

### Overall Architecture

This paper presents an observational linguistic analysis built entirely on surprisal values provided by language models (the negative log-probability of the next word $-\log p(w)$), with no model training involved. The statistical foundation is the GUM v11 corpus (250K+ tokens, 16 genres). The research is organized into three progressive questions: first, after controlling for confounders like position, length, and nesting in natural corpora, determine if the surprisal of salient entities themselves is high or low (RQ1); second, use minimal-pair prompting to isolate "entity identity" and measure the causal impact of salient entities on the predictability of subsequent content (RQ2); finally, apply the same measurements across 16 genres to see how effect intensity varies with topic coherence (RQ3). The input consists of discourse with salience annotations, the intermediate step involves controlled surprisal comparisons, and the output is the dual-layered conclusion that "salient entities have high surprisal themselves but lower surrounding surprisal."

### Key Designs

**1. Global Salience Metric Based on Summary Consistency: Quantifying "Protagonists"**

Previous UID studies lacked a quantitative measure for "protagonists." This paper addresses this using 5 independent summaries provided for each document in the GUM-SAGE dataset: if an entity is mentioned in all 5 summaries, it receives a score of 5 (highly salient); if mentioned in only 1, it receives 1; if never mentioned, it receives 0. The intuition is simple—if an entity is truly important, it is difficult to write a summary without mentioning it. Thus, "summary consistency" becomes a robust and operational definition of salience. Applied to the data, this covers 70K+ entity mentions and 31K unique entities, where approximately 84.5% of entities score 0, indicating that most entities are "supporting characters" and salient entities are a sparse minority.

**2. Minimal-Pair Prompting Paradigm: Isolating Causality from Confounding Factors**

In natural corpora, factors like grammatical role, recency, and referring form are intertwined, making it impossible to determine the independent contribution of salience from correlation alone. The minimal-pair prompting approach fixes the subsequent text and only replaces the entity used as the prefix—once with a salient entity and once with a non-salient entity—then compares the surprisal calculated by the language model for the same subsequent segment. The logic is straightforward: if salient entities indeed establish stronger topic expectations, trailing content should be more predictable (lower surprisal) when preceded by them. This effectively constructs a quasi-controlled experiment on observational data to isolate the causal direction of "salience → surrounding predictability."

**3. Cross-Genre Analysis: Testing Mechanisms via Topic Coherence**

If salient entities lower surrounding surprisal by establishing "topic expectations," the effect should be more pronounced in texts focused on a single topic. The GUM corpus spans 16 genres (academic papers, biographies, vlogs, conversations, court transcripts, essays, fiction, forums, etc.). The paper measures effect intensity by genre, expecting the strongest effects in highly coherent academic papers and the weakest in conversations with frequent topic shifts. This dimension serves as both a robustness check and establishes topic coherence as a key moderator of the salience-surprisal relationship.

## Key Experimental Results

### Main Results

| Research Question | Key Findings |
|----------|----------|
| RQ1: Surprisal of salient entities themselves | Surprisal of global salient entities is significantly higher than non-salient entities, even after controlling for position, length, and nesting. |
| RQ2: Impact on surrounding content | Salient entities systematically reduce the surprisal of subsequent content, creating a local "trough." |
| RQ3: Genre differences | The effect is strongest in topic-coherent texts (academic papers) and weakest in conversational contexts. |

### Ablation Study

| Analysis Dimension | Result |
|----------|------|
| Salience score vs. Surprisal | Positive correlation—the higher the score, the higher the entity's own surprisal. |
| Minimal-pair: Salient vs. Non-salient prompts | Surprisal of subsequent content is significantly lower under salient entity prompts. |
| Topic-coherent vs. Topic-shifting genres | Effect intensity in topic-coherent genres is approximately 2-3 times higher than in topic-shifting genres. |

### Key Findings

- Global salient entities are "more unexpected" but "make the context more predictable"—two seemingly contradictory findings that reflect information organization at different levels.
- This pattern resembles the "word-initial high surprisal" phenomenon in phonetics—information is locally non-uniform but serves overall uniformity on a larger scale.
- Genre effects align with the topic coherence hypothesis, adding referential structure as a new dimension to the UID competing pressures framework.
- Approximately 84.5% of entities score 0 (non-salient), indicating most entities are "supporting characters."

## Highlights & Insights

- The insight that "salient entities are information anchors" elegantly unifies findings in both directions—their own high surprisal is due to carrying key information, while lowering surrounding surprisal is due to establishing strong topic expectations.
- The minimal-pair prompting method cleverly introduces causal inference into observational corpus analysis and can be generalized to study other discourse phenomena.
- The paper extends the "competing pressures" in the UID framework to the dimension of referential structure—previous work primarily considered phonetics, syntax, and discourse structure.

## Limitations & Future Work

- Uses only English data; cross-linguistic generalizability is unknown.
- Salience is based on summary consistency, which may bias toward extractable information rather than deep thematic importance.
- Surprisal calculated by language models is not equivalent to human cognitive surprisal.
- Dynamic salience—how local salience of an entity changes as the discourse progresses—was not explored.

## Related Work & Insights

- **vs. Centering Theory**: The latter focuses on local attentional salience (grammatical role, recency), while Ours focuses on global discourse salience—the two are complementary.
- **vs. Clark et al. (2023)**: Found that syntactic constraints limit the degree of UID achievement; Ours finds that referential structure constraints function similarly.
- **vs. Tsipidi et al. (2024)**: Found that discourse structure predicts non-uniformity in surprisal profiles; Ours extends this to the dimension of entity salience.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of the relationship between global entity salience and surprisal; novel minimal-pair method.
- Experimental Thoroughness: ⭐⭐⭐⭐ 70K annotations across 16 genres provide broad coverage, though limited to English.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are well-layered, analytical logic is rigorous, and conclusions are clear.
- Value: ⭐⭐⭐⭐ Adds an important referential structure dimension to UID theory, providing insights for discourse processing and language model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2025\] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs](../../ACL2025/llm_nlp/gift-sw_gaussian_noise_injected_fine-tuning_of_salient_weights_for_llms.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM vs. Human Writing Diversity using Progressive Conditional Surprisal](../../ICML2026/llm_nlp/ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)

</div>

<!-- RELATED:END -->
