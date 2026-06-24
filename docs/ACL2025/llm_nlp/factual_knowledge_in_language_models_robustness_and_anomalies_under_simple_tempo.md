---
title: >-
  [Paper Note] Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations
description: >-
  [ACL 2025][LLM (Other)][temporal robustness] This paper introduces the TimeStress dataset (521K statements, 2003 temporal facts) to evaluate the temporal robustness of factual knowledge in 18 LLMs under temporal context variations. The results find that even the best model achieves perfect robustness on only 11% of the facts, committing critical errors that humans would never make.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "temporal robustness"
  - "factual knowledge"
  - "TimeStress"
  - "temporal context"
  - "knowledge representation"
date: 2026-05-08
content_hash: b8d99fcceb34ac82
---

# Factual Knowledge in Language Models: Robustness and Anomalies under Simple Temporal Context Variations

**Conference**: ACL 2025  
**arXiv**: [2502.01220](https://arxiv.org/abs/2502.01220)  
**Code**: [https://github.com/Orange-OpenSource/TimeStress](https://github.com/Orange-OpenSource/TimeStress)  
**Area**: LLM/NLP  
**Keywords**: temporal robustness, factual knowledge, TimeStress, temporal context, knowledge representation

## TL;DR
This paper introduces the TimeStress dataset (521K statements, 2003 temporal facts) to evaluate the temporal robustness of factual knowledge in 18 LLMs under temporal context variations. The results find that even the best model achieves perfect robustness on only 11% of the facts, committing critical errors that humans would never make.

## Background & Motivation

**Background**: LLMs store a vast amount of factual knowledge, but this knowledge is not robust against textual perturbations (paraphrases, typos, negation, etc.). Research on robustness in the temporal dimension remains relatively sparse.

**Limitations of Prior Work**: Existing temporal reasoning works (such as TimeBench and TempReason) test temporal logical operation capabilities, but do not systematically test the robustness of a single fact under different temporal contexts.

**Key Challenge**: Can LLMs correctly associate a temporal context (e.g., "in 2011") with a past fact that has a clear validity period? Can they maintain consistency across all valid/invalid temporal contexts?

**Goal**: Quantify the robustness and anomalies of temporal representations in LLMs.

**Key Insight**: Formulate the evaluation as a "tournament" — letting LLMs judge preferences between statements of the same fact under correct and incorrect temporal contexts. Variations are explored along two dimensions: the distance between the incorrect context and the validity period, and the temporal granularity (year/month/day).

**Core Idea**: Fundamental flaws exist in the temporal representations of LLMs — even for "known" facts, they fail to remain consistent across all temporal context variations and commit long-distance temporal errors that humans would never make.

## Method

### Overall Architecture
Collect 2003 temporal facts with validity periods from Wikidata $\rightarrow$ Generate high-quality QA textualizations using GPT-4o $\rightarrow$ Sample correct/incorrect/transitional temporal contexts $\rightarrow$ Formulate pairwise tournaments $\rightarrow$ Evaluate preferences using the conditional probabilities of 18 LLMs $\rightarrow$ Calculate win rates and robustness.

### Key Designs

1. **Formalization of Temporal Facts**

    - Quintuple $(s, r, o, a, b)$, such as (Obama, president, USA, 2009-01-20, 2017-01-20).
    - Temporal contexts are categorized into correct (fully within $[a,b]$), incorrect (fully outside $[a,b]$), and transitional (partially overlapping).
    - Design Motivation: Strictly define correctness/incorrectness to prevent ambiguity.

2. **Relative Distance $\alpha$ Metric**

    - $\alpha$ = distance from the context midpoint to the validity period midpoint / duration of the validity period.
    - $|\alpha| < 0.5$ is correct, while $|\alpha| > 0.5$ is incorrect.
    - Design Motivation: Enable the analysis of distance effects — incorrect contexts further away should be easier to distinguish.

3. **Three Temporal Granularities**

    - Y (year), YM (year-month), and YMD (year-month-day).
    - Hierarchical structure from coarse to fine.
    - Design Motivation: Test the transferability of knowledge across different granularities.

4. **Two Core Indicators**

    - Win Rate $W(M,f)$: The rate at which model $M$ correctly prefers the correct context for fact $f$.
    - Robustness $R(M,f) = \mathbb{1}[W(M,f) = 1]$: Considered robust only if 100% correct.
    - Design Motivation: Robustness is a more stringent metric, where a single error renders a prediction non-robust.

### Dataset Construction
- 2003 temporal facts $\times$ multiple temporal contexts = 521K statements.
- An average of 11 correct contexts + 74 incorrect contexts per fact.
- Covers 86 relationships and 1883 unique entities.

## Key Experimental Results

### Main Results — Average Win Rate and Robustness (Top 5 Models)

| Model | Average Win Rate (Y) | Average Robustness (Y) | All-Granularity Robustness |
|------|-------------|---------------|-------------|
| Llama-3.1-70B-Instruct | **87%**| 14% | 9% |
| gemma-2-27b-it | 85% | **17%**| **11%**|
| gemma-2-9b-it | 83% | 12% | 7% |
| Mistral-Nemo | 80% | 8% | 4% |
| Llama-3.1-8B-Instruct | 78% | 5% | 3% |

### Distance Error Analysis (Among "Known" Facts with Win Rate >95%)

| Incorrect Context Distance \|$\alpha$\| | Error Rate (Raw Text) | Error Rate (Instruction) |
|--------------|--------------------------|--------------------------|
| $\ge 1$ | **19%**| **25%**|
| $\ge 2$ | 9% | 13% |
| $\ge 3$ | **6%**| **8%**|
| $\ge 4$ | 4% | 5% |

### Cross-Granularity Knowledge Transfer

| From $\rightarrow$ To | Success Rate | Explanation |
|------|--------|------|
| Y $\rightarrow$ YM | 74% | Hard to transfer from coarse to fine |
| Y $\rightarrow$ YMD | 68% | Finer granularity is more challenging |
| YM $\rightarrow$ Y | 88% | Relatively easy from fine to coarse |
| Average Failure Rate | **28%**| Nearly one-third of the facts fail to transfer across granularities |

### Key Findings
- **Even the best model (gemma-2-27b-it) is perfectly robust on only 11% of the facts** — the vast majority of facts have at least one temporal context misjudged.
- **Long-distance errors are systemic**: Even for facts that are "almost certainly known", 19% of errors stem from contexts more than $1\times$ the validity duration away — errors that humans would not make.
- **Cross-granularity transfer failure rate of 28%**: Knowing the correct context at the year level does not guarantee knowing it at the month level.
- **Instruction format unexpectedly increases critical errors**: Compared to raw text, the instruction format yields more long-distance errors.
- **Transitional contexts exhibit higher probability**: Likely because start/end years co-occur with the facts in the training data more frequently than years nested strictly within the validity period.
- **Instruction tuning improves robustness**: Llama-3.1-70B-Instruct is 3.6 times more robust than its base counterpart.

## Highlights & Insights
- **The massive gap between high average win rate and low robustness** is the most significant finding. An 87% average win rate seems strong, yet only 11% are perfectly robust, indicating that spatial-temporal knowledge in LLMs is "generally correct but highly unreliable in details."
- **Long-distance critical errors** directly challenge the reliability of LLMs as knowledge bases. If a model that knows Obama was president from 2009 to 2017 still assigns a high probability to a 1998 context, it reveals a fundamental flaw in its temporal representation.
- **Granularity transfer experiments** shed light on an overlooked issue — LLMs do not truly understand directory-like logical "containment" (e.g., that January 2018 is contained within the year 2018).

## Limitations & Future Work
- Fact select is biased toward high-popularity entities; temporal robustness for low-frequency knowledge might be even lower.
- Limited to past facts (ended before 2021); processing of future facts remains untested.
- The restriction of validity periods $>3$ years excludes short-term facts.
- Future directions: Constructing time-aware fine-tuning data, enhancing temporal logical representations, and analyzing the robustness of multi-hop temporal reasoning.

## Related Work & Insights
- **vs. ChronoSense**: ChronoSense tests qualitative reasoning about Allen interval relations, whereas TimeStress evaluates the consistency of a single fact under various temporal contexts — representing complementary research directions.
- **vs. TempReason (Tan et al.)**: TempReason evaluates via average performance, while TimeStress measures robustness (requiring 100% accuracy) — making the latter far more stringent.
- **vs. Knowledge Editing**: Instead of updating knowledge, TimeStress evaluates the logical consistency of existing temporal knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study on the temporal robustness of LLM factual knowledge; long-distance anomaly findings are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models, 521K statements, multi-dimensional analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formalization and high-quality visualizations.
- Value: ⭐⭐⭐⭐⭐ Important implications for LLM knowledge representations and RAG system reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions](uaqfact_evaluating_factual_knowledge_utilization_of_llms_on_unanswerable_questio.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)
- [\[ACL 2025\] ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving](arithmattack_evaluating_robustness_of_llms_to_noisy_context_in_math_problem_solv.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)

</div>

<!-- RELATED:END -->
