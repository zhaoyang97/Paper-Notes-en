---
title: >-
  [Paper Note] Concept-Level Explainability for Auditing & Steering LLM Responses
description: >-
  [NeurIPS 2025][Social Computing][Explainability] This paper proposes ConceptX, an LLM explainability method based on concept-level (rather than token-level) Shapley attribution. It measures the influence of input concepts on outputs via semantic similarity rather than token overlap, and can be used to audit bias and steer LLM outputs through prompt editing — reducing attack success rate from 0.463 to 0.242 in jailbreak defense.
tags:
  - NeurIPS 2025
  - Social Computing
  - Explainability
  - Concept-Level Attribution
  - LLM Safety
  - Shapley Values
  - Bias Auditing
date: 2026-05-08
content_hash: 2bc27b54faa399ce
---

# Concept-Level Explainability for Auditing & Steering LLM Responses

**Conference**: NeurIPS 2025
**arXiv**: [2505.07610](https://arxiv.org/abs/2505.07610)
**Code**: [https://github.com/k-amara/ConceptX](https://github.com/k-amara/ConceptX)
**Area**: Social Computing
**Keywords**: Explainability, Concept-Level Attribution, LLM Safety, Shapley Values, Bias Auditing

## TL;DR
This paper proposes ConceptX, an LLM explainability method based on concept-level (rather than token-level) Shapley attribution. It measures the influence of input concepts on outputs via semantic similarity rather than token overlap, and can be used to audit bias and steer LLM outputs through prompt editing — reducing attack success rate from 0.463 to 0.242 in jailbreak defense.

## Background & Motivation

**Background**: Attribution methods such as TokenSHAP can quantify the contribution of each input token to LLM outputs, aiding in understanding and modulating model behavior. However, existing methods operate exclusively at the token level.

**Limitations of Prior Work**: Token-level attribution suffers from three key problems: (a) it optimizes for token overlap rather than semantic similarity, failing to capture semantically equivalent paraphrases; (b) it often assigns high importance to uninformative function words (e.g., "the", "is") rather than semantically meaningful concept words; (c) processing tokens independently disrupts contextual coherence, leading to unstable generation results.

**Key Challenge**: Effective explainability requires both faithfulness and actionability, yet token-level methods tend to lack depth in semantic understanding, while humans naturally interpret explanations at the concept level.

**Goal**: How can attribution be performed at the concept level rather than the token level? How can attribution results not only explain model behavior but also guide prompt editing to steer LLM outputs?

**Key Insight**: Attribution is restricted to semantically rich "concepts" (content words with high degree in ConceptNet), semantic similarity replaces token overlap as the value function, and flexible explanation targets (e.g., gender bias, harmfulness) are supported.

**Core Idea**: ConceptNet is used to extract semantic concepts; Shapley-value attribution based on semantic similarity identifies the key concepts driving LLM outputs; editing these concepts then steers the output.

## Method

### Overall Architecture

ConceptX operates in two stages: (1) **Concept Extraction** — identifying semantically rich content words from the input (skipping function words), with ConceptNet node degree used to quantify semantic richness; (2) **Concept Importance Estimation** — computing the marginal contribution of each concept via Shapley-inspired Monte Carlo sampling, with a value function based on cosine similarity between the output and the explanation target.

### Key Designs

1. **Concept Extraction (Concepts as Features)**:

    - Function: Extracts high-semantic-value content words from the input prompt as attribution targets.
    - Mechanism: spaCy parsing retrieves nouns, verbs, adjectives, and adverbs; ConceptNet edge counts filter for semantically rich concepts.
    - Design Motivation: Function words (articles, prepositions, etc.) are frequent but semantically marginal; focusing on concept words aligns explanations more closely with human intuition.

2. **Three Substitution Strategies**:

    - Function: When a concept is absent from the current coalition, three strategies maintain sentence integrity.
    - **ConceptX-r**: Direct removal (identical to TokenSHAP).
    - **ConceptX-n**: GPT-4o-mini generates a semantically neutral replacement, preserving grammaticality.
    - **ConceptX-a**: Antonym substitution, providing a more explicit semantic contrast.
    - Design Motivation: Simple removal disrupts grammar and leads to unstable generation; neutral substitution isolates a concept's semantic influence while preserving sentence structure.

3. **Flexible Explanation Targets (Value Function)**:

    - Function: Cosine similarity measures the influence of a concept coalition on the output.
    - $v(S) = \cos(Emb \cdot f(S), Emb \cdot \mathbf{t})$
    - Three explanation targets: ConceptX_B (similarity to the original output), ConceptX_R (similarity to a reference text), ConceptX_A (similarity to a specific aspect such as "gender bias").
    - Design Motivation: Traditional methods only explain "why this output was generated"; ConceptX_A additionally answers "what in the input drives a specific aspect (e.g., bias)."

### Application Scenarios
- **Auditing**: Identifying key concepts in prompts that lead to biased or harmful outputs.
- **Steering**: Altering the sentiment or safety of LLM outputs by removing or replacing high-attribution concepts.

## Key Experimental Results

### Jailbreak Defense (Salad-Bench, Mistral-7B)

| Method | ASR↓ | HS↓ |
|--------|------|-----|
| No Defense | 0.463 | 2.51 |
| Random | 0.383 | 2.30 |
| TokenSHAP | 0.312 | 2.14 |
| SelfParaphrase | 0.328 | 2.14 |
| **ConceptX_B-r** | **0.242** | **1.92** |
| ConceptX_B-n | 0.281 | 2.01 |
| GPT-4o Mini (self-attr) | 0.233 | 1.86 |
| SelfReminder (prompt) | 0.223 | 1.79 |

### Gender Bias Auditing (GenderBias Dataset)

| Method | Proportion of Gender Words Ranked Top-1/Top-2 |
|--------|-----------------------------------------------|
| TokenSHAP | <10% |
| ConceptX_B-n | ~50%+ |
| ConceptX_A-n | ~70%+ (best) |

### Key Findings
- **Concept-level outperforms token-level**: ConceptX_B-r achieves an ASR of 0.242, substantially better than TokenSHAP (0.312), demonstrating that concept-level attribution more accurately identifies the critical components of harmful prompts.
- **Removal outperforms antonym substitution** (in harmful content scenarios): Harmful words tend to be nouns (e.g., "drug") with no direct antonyms, making removal more effective.
- **Antonym substitution outperforms removal** (in sentiment scenarios): Sentiment is often driven by adjectives, and antonym substitution enables sentiment reversal.
- **Cross-model variation**: GPT-4o mini is more robust to gender bias, exhibiting lower attribution scores for gender-related concepts.
- **Impact of explanation target**: ConceptX_A-n is most effective for bias auditing (targeted explanation), but offers no advantage over ConceptX_B-n in steering tasks.

## Highlights & Insights
- **Bridging explainability and actionability**: ConceptX not only explains model behavior but directly leverages attribution results to guide prompt editing — closing the gap between XAI research and practical safety requirements.
- **Aspect-Targeted Explanation**: ConceptX_A supports attribution along specific dimensions (gender bias, harmfulness), which is particularly valuable for auditing targeted safety properties.
- **Lightweight safety without retraining**: Identifying and editing key concepts in the prompt substantially reduces harmful outputs, offering a more transparent and controllable alternative to fine-tuning and prompt engineering.

## Limitations & Future Work
- **Exponential computational complexity remains**: Although restricting attribution to concept words roughly halves the number of tokens, the exponential complexity of Shapley value computation limits applicability to long prompts.
- **Semantic role of function words is neglected**: Certain function words (e.g., "not") carry critical semantics (negation), and ConceptX's concept filtering may miss these.
- **Model alignment differences**: Some models (e.g., Gemma) rely more heavily on token-level signals, in which case ConceptX underperforms TokenSHAP.
- **Dependency on external models**: The neutral substitution in ConceptX-n requires calls to GPT-4o-mini, introducing additional cost and dependency.

## Related Work & Insights
- **vs. TokenSHAP**: TokenSHAP performs Shapley attribution at the token level and holds an advantage on models driven by sentiment tokens (e.g., function words like "not"), but is substantially outperformed by ConceptX on concept-driven models and safety tasks.
- **vs. SelfReminder / Prompt Engineering**: Prompt engineering remains marginally superior in safety (ASR 0.223 vs. 0.242), but ConceptX provides an interpretable and reproducible alternative.
- **vs. Mechanistic Interpretability**: Mechanistic interpretability analyzes internal model mechanisms, while ConceptX provides model-agnostic input-level insights; the two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of concept-level attribution, semantic similarity value function, and aspect-targeted explanation is novel; the perspective connecting XAI with LLM safety is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLMs × three tasks (faithfulness / bias auditing / steering) × multiple variant comparisons constitute a comprehensive experimental design.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with detailed descriptions of methods and experiments, and notation is well-defined.
- Value: ⭐⭐⭐⭐ Demonstrates the practical value of attribution methods for LLM safety and offers a new direction for "safety alignment without retraining."

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](../../ACL2026/social_computing/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](../../ACL2026/social_computing/persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/social_computing/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)

<!-- RELATED:END -->
