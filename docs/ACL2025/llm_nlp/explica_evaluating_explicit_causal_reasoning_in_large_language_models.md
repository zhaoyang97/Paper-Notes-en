---
title: >-
  [Paper Note] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][causal reasoning] This paper proposes the ExpliCa dataset (4,800 questions containing causal and temporal connectives), which integrates causal and temporal relation evaluation for the first time along with crowdsourced human ratings. The study finds that even top-tier models struggle to exceed 0.80 accuracy, and models systematically misclassify temporal relations as causal relations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "causal reasoning"
  - "temporal reasoning"
  - "connectives"
  - "pairwise causal discovery"
  - "benchmark"
date: 2026-05-08
content_hash: c230e7ac0329cb25
---

# ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.15487](https://arxiv.org/abs/2502.15487)  
**Code**: [https://anonymous.4open.science/r/ExpliCa-6473/](https://anonymous.4open.science/r/ExpliCa-6473/)  
**Area**: LLM/NLP  
**Keywords**: causal reasoning, temporal reasoning, connectives, pairwise causal discovery, benchmark

## TL;DR
This paper proposes the ExpliCa dataset (4,800 questions containing causal and temporal connectives), which integrates causal and temporal relation evaluation for the first time along with crowdsourced human ratings. The study finds that even top-tier models struggle to exceed 0.80 accuracy, and models systematically misclassify temporal relations as causal relations.

## Background & Motivation

**Background**: LLM evaluation of causal reasoning primarily focuses on implicit causality (e.g., CLadder based on formal rules), while systematically lacking research on explicit causal markers (such as connectives like "because" and "so").

**Limitations of Prior Work**: Causal and temporal relations are tightly intertwined (effects typically occur after causes), but existing datasets do not annotate both simultaneously, making it impossible to evaluate models' discriminative capabilities.

**Key Challenge**: Can LLMs distinguish between "because" (causal) and "then" (temporal), two closely related but fundamentally different relations?

**Goal**: Construct an evaluation dataset that contains both causal and temporal relations expressed with explicit connectives.

**Key Insight**: Utilize four connectives (so/because for causal; then/after for temporal) $\times$ two directions (iconic/anti-iconic) $\times$ 600 sentence pairs = 4,800 entries, accompanied by crowdsourced acceptability ratings.

**Core Idea**: LLMs systematically mistake temporal relations for causal relations—they rely heavily on event order rather than true causal understanding.

## Method

### Overall Architecture
Constructing 600 sentence pairs $\times$ 4 connectives $\times$ 2 directions = 4,800 entries $\rightarrow$ Crowdsourced human acceptability ratings $\rightarrow$ Dual evaluation of 7 LLMs using PPL (competence) and prompting (performance).

### Key Designs

1. **Connective Design**

    - Causal forward (iconic): so (cause $\rightarrow$ effect)
    - Causal backward (anti-iconic): because (effect $\rightarrow$ cause)
    - Temporal forward (iconic): then (prior $\rightarrow$ subsequent)
    - Temporal backward (anti-iconic): after (subsequent $\rightarrow$ prior)
    - Design Motivation: Connectives are the sole relational cue, eliminating other linguistic hints.

2. **Three Categories of Sentence Pairs**

    - Causal category (200 pairs): The relation is primarily causal.
    - Temporal category (200 pairs): The relation is strictly temporal.
    - Unrelated category (200 pairs): Topically related but without causal/temporal relations.
    - Design Motivation: Three-way classification tests the discriminative capability.

3. **Dual Evaluation**

    - PPL evaluation (competence): Which connective yields lower perplexity.
    - Prompting evaluation (performance): Directly querying the model on which connective is more appropriate.
    - Design Motivation: Distinguishing between "knowing but unable to express" (tacit knowledge) and "genuinely not knowing".

4. **Control of Lexical Association Bias**

    - Verifying that lexical associations across the three categories of sentence pairs show no significant differences using PMI/LMI.
    - Design Motivation: Ensuring that models cannot rely on lexical co-occurrence to guess answers.

## Key Experimental Results

### Main Results -- Causal Reasoning Accuracy of 7 LLMs

| Model | Causal Identification (Prompting) | Temporal Identification (Prompting) | Overall |
|------|-------------------|-------------------|------|
| GPT-4o | ~78% | ~65% | ~72% |
| Claude-3.5 | ~75% | ~60% | ~68% |
| Llama-3.1-70B | ~70% | ~55% | ~63% |
| Average of Small Models | ~55% | ~45% | ~50% |

### PPL vs Prompting Comparison

| Evaluation Method | Average of Large Models | Average of Small Models |
|---------|-----------|-----------|
| PPL (Competence) | ~80% | ~70% |
| Prompting (Performance) | ~72% | ~55% |
| Gap | 8% | **15%** |

### Key Findings
- **Even the best models struggle to exceed 0.80 accuracy**
- **Causal-temporal confusion is systematic**: Models accept "then" entries as causal relations with high probability.
- **PPL > Prompting**: For small models, "knowing" (competence) far exceeds "doing" (performance), with a gap up to 15%.
- **Significant impact of order**: Iconic forward order is ~10% easier than anti-iconic backward order.
- **Model scale aids prompting but does not significantly affect PPL**

## Highlights & Insights
- **Causal-temporal confusion** is an overlooked but critical issue—essential in scenarios like medical or legal domains where distinguishing between "post hoc" (happening after) and "propter hoc" (happening because of) is crucial.
- **The gap between PPL and Prompting** reveals a chasm between the "implicit knowledge" (competence) and "explicit expression" (performance) of LLMs.
- **Pronoun-free design** eliminates confounding factors such as anaphora resolution.

## Limitations & Future Work
- Limited to English.
- The dataset scale of 600 sentence pairs is relatively small.
- Only 4 connectives are evaluated, without covering broader causal markers.

## Related Work & Insights
- **vs. COPA/e-CARE**: They evaluate implicit causality, whereas ExpliCa evaluates causality expressed through explicit connectives.
- **vs. BIG-Bench**: BIG-Bench uses "because" as a connective, whereas ExpliCa incorporates both causal and temporal connectives.

## Rating
- Novelty: ⭐⭐⭐⭐ First to integrate explicit relations of both causal and temporal dimensions into evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual evaluation using PPL + Prompting across 7 models.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely rigorous dataset construction.
- Value: ⭐⭐⭐⭐ Significant methodological contribution to causal reasoning research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] STEM-PoM: Evaluating Language Models Math-Symbol Reasoning in Document Parsing](stem-pom_evaluating_language_models_math-symbol_reasoning_in_document_parsing.md)
- [\[ACL 2025\] LR²Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems](lr2bench_evaluating_long-chain_reflective_reasoning_capabilities_of_large_langua.md)

</div>

<!-- RELATED:END -->
