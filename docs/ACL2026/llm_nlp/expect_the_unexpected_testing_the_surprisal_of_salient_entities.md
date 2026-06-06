---
title: >-
  [Paper Note] Expect the Unexpected? Testing the Surprisal of Salient Entities
description: >-
  [ACL 2026][LLM/NLP][Uniform Information Density] This paper investigates the relationship between discourse-level salient entities and surprisal. Using 70K+ manually annotated entity mentions and a novel minimal-pair pro…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Uniform Information Density"
  - "discourse salience"
  - "surprisal"
  - "entity prominence"
  - "discourse structure"
date: 2026-05-08
content_hash: 972cdee229e14b93
---

# Expect the Unexpected? Testing the Surprisal of Salient Entities

**Conference**: ACL 2026
**arXiv**: [2604.10724](https://arxiv.org/abs/2604.10724)  
**Code**: None  
**Area**: Computational Linguistics / Information Theory
**Keywords**: Uniform Information Density, discourse salience, surprisal, entity prominence, discourse structure

## TL;DR

This paper investigates the relationship between discourse-level salient entities and surprisal. Using 70K+ manually annotated entity mentions and a novel minimal-pair prompting approach, the study finds that globally salient entities are themselves more surprising (higher surprisal), yet systematically reduce the surprisal of surrounding content. This effect varies by genre and is strongest in topically coherent texts.

## Background & Motivation

**Background**: The Uniform Information Density (UID) hypothesis posits that speakers tend to distribute information evenly across discourse, keeping surprisal approximately constant. However, multiple studies have identified systematic deviations—phonological constraints (high surprisal at word onsets), syntactic constraints, and discourse-structural constraints produce local non-uniformities through "competing pressures."

**Limitations of Prior Work**: (1) Prior UID research has largely overlooked the relative salience of discourse participants—i.e., which entities are the "protagonists" of a text. (2) Existing findings on whether salient entities are more or less predictable are contradictory. (3) Multiple factors (grammatical role, recency, referential form, etc.) affect entity predictability, making it difficult to isolate salience effects in naturalistic contexts.

**Key Challenge**: On one hand, salient entities are more predictable due to repeated mention; on the other hand, as primary information carriers they may convey higher information content. How do these two effects interact at the discourse level?

**Goal**: To systematically investigate, for the first time, the relationship between global entity salience and surprisal—distinguishing between the surprisal of entities themselves and the influence of entities on the surprisal of surrounding content.

**Key Insight**: The study leverages the manual annotations of the GUM-SAGE dataset (salience scores based on summary agreement) and the typological diversity of 16 genres, combined with a minimal-pair prompting method to control for confounds.

**Core Idea**: Globally salient entities function as "anchor points"—they carry more information themselves (high surprisal), but substantially reduce uncertainty in subsequent content by establishing topic expectations, creating local surprisal "valleys."

## Method

### Overall Architecture

The study proceeds at three levels: (1) RQ1—analyzing the surprisal characteristics of salient entities in naturalistic corpora, controlling for confounds such as position, length, and nesting; (2) RQ2—employing a minimal-pair prompting paradigm (substituting salient vs. non-salient entities) to measure the causal influence of entities on document content predictability; (3) RQ3—comparing effect sizes across 16 genres.

### Key Designs

1. **Summary-Agreement-Based Global Salience Measure**

    - Function: Provides a quantified discourse-level entity importance score.
    - Mechanism: Using the GUM-SAGE dataset, each document is accompanied by 5 independent summaries. An entity mentioned in all 5 summaries receives a score of 5 (most salient); one appearing in only 1 receives a score of 1; entities never mentioned receive a score of 0 (approximately 84.5% of entities). The dataset contains over 70K entity mentions covering 31K unique entities.
    - Design Motivation: Grounded in the intuition that "if an entity is salient, it is difficult to write a summary that omits it"—summary agreement provides a robust and operationalizable definition of salience.

2. **Minimal-Pair Prompting Paradigm**

    - Function: Controls for confounds and measures the causal effect of entities on the surprisal of subsequent content.
    - Mechanism: For the same document content, salient and non-salient entities are used alternately as prompt prefixes, and the language model's surprisal over the subsequent text is compared. If salient entities genuinely enhance document predictability, surprisal over subsequent content should be lower when a salient entity is used as the prompt.
    - Design Motivation: In naturalistic corpora, multiple factors operate simultaneously, making it impossible to isolate the independent contribution of salience. The minimal-pair design holds other factors constant, varying only entity identity, thereby enabling quasi-causal inference.

3. **Cross-Genre Analysis**

    - Function: Reveals moderating factors of the salience–surprisal relationship.
    - Mechanism: The GUM corpus spans 16 genres (academic papers, biographies, vlogs, conversations, court records, essays, fiction, forums, etc.); effect sizes are analyzed separately for each. Topically coherent texts (e.g., academic papers, which focus on a single theme) are expected to show the strongest effects, while texts with frequent topic shifts (e.g., conversations) are expected to show the weakest.
    - Design Motivation: If the salience effect operates through a topic-expectation mechanism, topical coherence should be the key moderating factor.

### Loss & Training

This is an analytical study and involves no model training. Surprisal (negative log-probability) is computed using language models, and statistical analyses are conducted on the GUM v11 corpus (250K+ tokens, 16 genres).

## Key Experimental Results

### Main Results

| Research Question | Core Finding |
|---|---|
| RQ1: Surprisal of salient entities themselves | Globally salient entities exhibit significantly higher surprisal than non-salient entities; this holds after controlling for position, length, and nesting. |
| RQ2: Influence on surrounding content | Salient entities systematically reduce the surprisal of subsequent content, creating local "valleys." |
| RQ3: Genre differences | The effect is strongest in topically coherent texts (academic papers) and weakest in conversational contexts. |

### Ablation Study

| Analysis Dimension | Result |
|---|---|
| Salience score vs. surprisal | Positive correlation—higher scores correspond to higher surprisal for the entity itself. |
| Minimal pair: salient vs. non-salient prompt | Surprisal of subsequent content is significantly lower when a salient entity serves as the prompt. |
| Topically coherent vs. topic-switching genres | Effect size in topically coherent genres is approximately 2–3× that in topic-switching genres. |

### Key Findings

- Globally salient entities are "more surprising" yet "make context more predictable"—two seemingly contradictory findings that in fact reflect information organization operating at different levels.
- This pattern resembles the "high surprisal at word onset" phenomenon in phonology—local non-uniformity serves overall uniformity at a larger scale.
- Genre effects are consistent with the topical coherence hypothesis, adding referential structure as a new dimension to the UID competing-pressures framework.
- Approximately 84.5% of entities receive a score of 0 (non-salient), indicating that most entities play a "supporting" role.

## Highlights & Insights

- The insight that "salient entities serve as information anchors" elegantly unifies findings in both directions—high surprisal in the entity itself reflects its role as a carrier of key information, while reduced surprisal in the surrounding content reflects the strong topic expectations it establishes.
- The minimal-pair prompting method ingeniously introduces causal reasoning into observational corpus analysis and is generalizable to other discourse phenomena.
- The UID framework's notion of "competing pressures" is extended to the dimension of referential structure—prior work considered only phonological, syntactic, and discourse-structural constraints.

## Limitations & Future Work

- Only English data are used; cross-linguistic generalizability remains unknown.
- Salience is operationalized via summary agreement, which may favor extractable information over deeper thematic importance.
- Surprisal computed by language models is not equivalent to human cognitive surprisal.
- Dynamic salience is not explored—the local salience of an entity may shift as discourse unfolds.

## Related Work & Insights

- **vs. Centering Theory**: Centering Theory addresses local attentional salience (grammatical role, recency), whereas this paper addresses global discourse salience—the two are complementary.
- **vs. Clark et al. (2023)**: That work finds that syntactic constraints limit the degree to which UID is realized; this paper finds that referential structure constraints operate similarly.
- **vs. Tsipidi et al. (2024)**: That work finds that discourse structure predicts non-uniformity in surprisal profiles; this paper extends the finding to the dimension of entity salience.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of the relationship between global entity salience and surprisal; the minimal-pair method is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 70K annotations and 16-genre coverage are broad, though limited to English.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are clearly layered, analytical logic is rigorous, and conclusions are well-articulated.
- Value: ⭐⭐⭐⭐ Adds an important referential-structure dimension to UID theory; offers insights for discourse processing and language model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal](an_existence_proof_for_neural_language_models_that_can_explain_garden-path_effec.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [\[NeurIPS 2025\] PluralisticBehaviorSuite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](../../NeurIPS2025/llm_nlp/pluralistic_behavior_suite_stress-testing_multi-turn_adherence_to_custom_behavio.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)

</div>

<!-- RELATED:END -->
