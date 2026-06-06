---
title: >-
  [Paper Note] Expect the Unexpected? Testing the Surprisal of Salient Entities
description: >-
  [ACL 2026][LLM/NLP][Uniform Information Density] This paper investigates the relationship between discourse-level salient entities and surprisal. By analyzing 70K+ manually annotated entity mentions and employing a novel…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Uniform Information Density"
  - "Discourse Salience"
  - "Surprisal"
  - "Entity Prominence"
  - "Discourse Structure"
date: 2026-05-08
content_hash: ea341de5f67d3177
---

# Expect the Unexpected? Testing the Surprisal of Salient Entities

**Conference**: ACL 2026  
**arXiv**: [2604.10724](https://arxiv.org/abs/2604.10724)  
**Code**: None  
**Area**: Computational Linguistics / Information Theory  
**Keywords**: Uniform Information Density, Discourse Salience, Surprisal, Entity Prominence, Discourse Structure

## TL;DR

This paper investigates the relationship between discourse-level salient entities and surprisal. By analyzing 70K+ manually annotated entity mentions and employing a novel minimal-pair prompting method, it finds that while global salient entities are themselves more unexpected (higher surprisal), they systematically reduce the surprisal of surrounding content. This effect varies by genre, being strongest in texts with high topical coherence.

## Background & Motivation

**Background**: The Uniform Information Density (UID) hypothesis suggests that speakers tend to distribute information evenly across an utterance to keep surprisal roughly constant. However, multiple studies have identified systematic deviations—"competing pressures" from phonetic constraints (high surprisal at word beginnings), syntactic constraints, and discourse structure constraints that produce local non-uniformity.

**Limitations of Prior Work**: (1) Previous UID research has largely ignored the relative salience of discourse participants—which entities are the "protagonists" of the text; (2) Existing results are contradictory regarding whether salient entities are more predictable or more unexpected; (3) Multiple factors (grammatical role, recency, referring form, etc.) affect entity predictability, making it difficult to isolate the effect of salience in natural contexts.

**Key Challenge**: On one hand, salient entities may be more predictable due to repeated mentions; on the other hand, as primary information carriers, they may contain higher information content. How do these two effects interact at the discourse level?

**Goal**: To conduct the first systematic study of the relationship between global entity salience and surprisal, distinguishing between the surprisal of the entity itself and the entity's influence on the surprisal of surrounding content.

**Key Insight**: Utilizing manual annotations from the GUM-SAGE dataset (salience scores based on summary consistency) across 16 diverse genres, combined with a minimal-pair prompting method to control for confounding factors.

**Core Idea**: Global salient entities play the role of "anchors"—they carry more information themselves (high surprisal), but significantly reduce the uncertainty of subsequent content by establishing topical expectations, creating local surprisal "troughs."

## Method

### Overall Architecture

The research is conducted across three levels: (1) RQ1—Analyzing the surprisal characteristics of salient entities in natural corpora (controlling for confounders such as position, length, and nesting); (2) RQ2—Using a minimal-pair prompting paradigm (replacing salient vs. non-salient entities) to measure the causal impact of entities on document content predictability; (3) RQ3—Comparing the strength of these effects across 16 genres.

### Key Designs

1.  **Global Salience Metric based on Summary Consistency**:
    - **Function**: Provides a quantitative discourse-level entity importance score.
    - **Mechanism**: Utilizes the GUM-SAGE dataset, where each document has 5 independent summaries. If an entity is mentioned in all 5 summaries, it receives a score of 5 (most salient); if it appears in only 1, the score is 1; if it never appears, the score is 0 (approx. 84.5% of entities). The data includes over 70K entity mentions covering 31K unique entities.
    - **Design Motivation**: Based on the intuition that "if an entity is salient, it is difficult to write a summary without mentioning it"—summary consistency provides a robust, operational definition of salience.

2.  **Minimal-Pair Prompting**:
    - **Function**: Controls for confounders to measure the causal effect of entities on subsequent content surprisal.
    - **Mechanism**: For the same document content, salient and non-salient entities are used separately as prompt prefixes to compare the language model's surprisal regarding the subsequent text. If salient entities indeed enhance the predictability of the document content, the surprisal of the subsequent content should be lower when prompted with a salient entity.
    - **Design Motivation**: In natural corpora, various factors act in coordination, making it impossible to isolate the independent contribution of salience. The minimal-pair design fixes other factors and only varies the entity identity, enabling quasi-causal inference.

3.  **Cross-genre Analysis**:
    - **Function**: Reveals moderators of the salience-surprisal relationship.
    - **Mechanism**: The GUM corpus covers 16 genres (academic papers, biographies, vlogs, conversations, court records, essays, fiction, forums, etc.), and the effect strength is analyzed for each. It is expected that topically coherent texts (e.g., academic papers focusing on a single subject) will show the strongest effects, while texts with frequent topic shifts (e.g., conversations) will show the weakest.
    - **Design Motivation**: If the salience effect operates through a topical expectation mechanism, then topical consistency should be a key moderator.

### Loss & Training

This is an analytical work and does not involve model training. Language models were used to calculate surprisal (negative log-probability), and statistical analyses were performed on the GUM v11 corpus (250K+ tokens, 16 genres).

## Key Experimental Results

### Main Results

| Research Question | Key Finding |
|----------|----------|
| RQ1: Surprisal of salient entities themselves | The surprisal of global salient entities is significantly higher than that of non-salient entities, even after controlling for position, length, and nesting. |
| RQ2: Impact on surrounding content | Salient entities systematically reduce the surprisal of subsequent content, creating a local "trough." |
| RQ3: Genre differences | The effect is strongest in topically coherent texts (academic papers) and weakest in conversational contexts. |

### Ablation Study

| Dimension of Analysis | Result |
|----------|------|
| Salience score vs. surprisal | Positively correlated—the higher the score, the higher the entity's own surprisal. |
| Minimal pairs: Salient vs. non-salient prompts | Surprisal of subsequent content is significantly lower under salient entity prompts. |
| Coherent vs. Shifted genres | The effect strength in topically coherent genres is approximately 2-3 times that of topically shifted genres. |

### Key Findings

- Global salient entities are "more unexpected" but "make the context more predictable"—these two seemingly contradictory findings actually reflect information organization at different layers.
- This pattern is similar to the "high surprisal at word beginnings" phenomenon in phonetics—information is non-uniform locally but serves overall uniformity on a larger scale.
- The genre effect aligns with the topical coherence hypothesis, adding the dimension of referential structure to the UID competing pressures framework.
- Approximately 84.5% of entities have a score of 0 (non-salient), indicating that most entities are "supporting characters."

## Highlights & Insights

- The insight that "salient entities are informational anchors" elegantly unifies findings from two directions—high intrinsic surprisal is due to carrying key information, while lowering surrounding surprisal is due to establishing strong topical expectations.
- The minimal-pair prompting method cleverly introduces causal reasoning into observational corpus analysis and can be generalized to the study of other discourse phenomena.
- The "competing pressures" in the UID framework are extended to the dimension of referential structure—previous work primarily considered phonetics, syntax, and discourse structure.

## Limitations & Future Work

- Only English data was used; cross-lingual generalizability remains unknown.
- Salience is based on summary consistency, which may be biased toward extractable information rather than deep thematic importance.
- Surprisal calculated by language models is not equivalent to human cognitive surprisal.
- Dynamic salience—how the local salience of an entity might change as the discourse progresses—was not explored.

## Related Work & Insights

- **vs Centering Theory**: The latter focuses on local attentional salience (grammatical roles, recency), while this paper focuses on global discourse salience—the two are complementary.
- **vs Clark et al. (2023)**: The latter found that syntactic constraints limit the degree to which UID is realized; this paper finds that referential structural constraints function similarly.
- **vs Tsipidi et al. (2024)**: The latter found that discourse structure predicts the non-uniformity of surprisal profiles; this paper extends this to the dimension of entity salience.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic study of the relationship between global entity salience and surprisal; novel minimal-pair method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 70K annotations across 16 genres provide broad coverage, though limited to English.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Research questions are clearly tiered, analysis logic is rigorous, and conclusions are clear.
- **Value**: ⭐⭐⭐⭐ Adds an important referential structure dimension to UID theory; provides insights for discourse processing and language model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [\[ICML 2026\] "I've Seen How This Goes": Characterizing LLM and Human Writing Diversity via Progressive Conditional Surprisal](../../ICML2026/llm_nlp/ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](../../NeurIPS2025/llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)

</div>

<!-- RELATED:END -->
