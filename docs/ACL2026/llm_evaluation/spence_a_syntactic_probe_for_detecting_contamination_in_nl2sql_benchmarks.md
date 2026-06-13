---
title: >-
  [Paper Note] SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks
description: >-
  [ACL 2026][LLM Evaluation][Data contamination detection] SPENCE detects and quantifies data contamination of LLMs on NL2SQL benchmarks by systematically rewriting benchmark queries syntactically and measuring the decay o…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Data contamination detection"
  - "NL2SQL"
  - "Syntactic probe"
  - "Benchmark leakage"
  - "Generalization evaluation"
date: 2026-05-08
content_hash: f993b8c44973ed22
---

# SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks

**Conference**: ACL 2026  
**arXiv**: [2604.17771](https://arxiv.org/abs/2604.17771)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Data contamination detection, NL2SQL, Syntactic probe, Benchmark leakage, Generalization evaluation

## TL;DR

SPENCE detects and quantifies data contamination of LLMs on NL2SQL benchmarks by systematically rewriting benchmark queries syntactically and measuring the decay of execution accuracy with syntactic distance. It reveals that older benchmarks like Spider exhibit stronger contamination signals, whereas the newer BIRD benchmark remains largely unaffected.

## Background & Motivation

**Background**: LLMs have achieved high execution accuracy on NL2SQL benchmarks such as Spider, SParC, CoSQL, and BIRD. However, the academic community increasingly questions whether these scores reflect true generalization capabilities. Data contamination and benchmark leakage have been widely studied, with methods like ConStat, Data Contamination Quiz, and Min-K% Prob detecting memorization behaviors from various perspectives.

**Limitations of Prior Work**: Existing contamination detection methods either rely on string overlap checks, which fail to capture structural memorization, or require access to internal model parameters, making them inapplicable to black-box models. In the NL2SQL context, there is a lack of systematic contamination detection frameworks specifically for SQL generation; for instance, Ranaldi et al. (2024) only compare seen vs. unseen databases at the dataset level without fine-grained probing on the syntactic axis.

**Key Challenge**: The high reported accuracy may stem from the model's memorization of the surface forms of benchmark queries rather than true compositional generalization. If a model merely remembers the syntactic patterns of original queries, it should perform worse on semantically equivalent but syntactically different rewrites. The core problem is how to distinguish "memorization" from "insufficient generalization" and whether this behavior correlates with the benchmark's release date.

**Goal**: Design a controlled syntactic probing framework to quantify the contamination sensitivity of NL2SQL models through generated rewrites with increasing distance and conduct a comparative analysis across four benchmarks.

**Key Insight**: Utilize Tree Edit Distance (TED) of syntactic dependency trees to rank rewrites, operationalizing the "contamination degree" as the "decay rate of accuracy relative to syntactic distance," and quantify the significance of the trend using the Kendall's $\tau$ statistic.

**Core Idea**: Syntactic rewriting + distance ranking + accuracy decay analysis = Contamination probe. A steeper decay in older benchmarks indicates more severe contamination.

## Method

### Overall Architecture

The SPENCE pipeline consists of: (1) Extracting natural language queries, gold SQL, and database schemas from test sets; (2) Using GPT-5 to generate 10 syntactic rewrites for each query; (3) Parsing dependency trees with spaCy and calculating Tree Edit Distance (TED) to rank them (rank 1 being the closest, rank 10 the farthest); (4) Filtering rewrites with semantic drift using embedding cosine similarity; (5) Executing both original queries and ranked rewrites on downstream NL2SQL models to compare changes in execution accuracy $\Delta \text{Accuracy}$; (6) Quantifying the rank correlation between accuracy and rewrite rank using Kendall’s $\tau$.

### Key Designs

1.  **TED-based Syntactic Distance Ranking**:
    - **Function**: Sorts rewrites by the degree of syntactic change to construct a controlled "distance gradient."
    - **Mechanism**: Each rewrite and original query are parsed into dependency trees using spaCy. The Tree Edit Distance is calculated as $\text{TED}_i = \text{TreeEditDistance}(T_{p_i}, T_q)$, where allowed operations include insertion, deletion, and substitution of dependency tree nodes. Low TED corresponds to surface lexical changes (synonym replacement), while high TED corresponds to deep structural changes (clause rearrangement).
    - **Design Motivation**: Unlike simple lexical overlap measures, TED captures true syntactic structural changes, ensuring the probe measures syntactic robustness rather than surface lexical similarity.

2.  **Embedding Similarity Filtering + Human Verification**:
    - **Function**: Ensures rewrites maintain semantic equivalence and excludes meaningless or drifted rewrites.
    - **Mechanism**: The E5-base-v2 embedding cosine similarity is calculated between each rewrite and the original query; rewrites below a threshold are filtered. Additionally, 100 Spider samples across ranks 1/5/10 underwent human verification, confirming that 85%+ of rewrites are semantically equivalent.
    - **Design Motivation**: If rewrites are not semantically equivalent, accuracy drops could stem from semantic changes rather than contamination signals; the filtering step eliminates this confounding factor.

3.  **Kendall’s $\tau$ Rank Correlation Analysis + Bootstrap Confidence Intervals**:
    - **Function**: Statistically quantifies the monotonic downward trend of accuracy relative to rewrite rank.
    - **Mechanism**: For each model-dataset pair, Kendall’s $\tau = (n_c - n_d) / \frac{1}{2}n(n-1)$ is calculated, where $n_c$ and $n_d$ are the numbers of concordant and discordant pairs. 95% confidence intervals are obtained through $B=100$ bootstrap resamples. A strong negative $\tau$ indicates monotonic accuracy decay (contamination signal), while values near zero indicate no sensitivity.
    - **Design Motivation**: Provides a statistically reliable measure of contamination beyond visual observation. Confidence intervals that do not cross zero indicate a statistically significant trend.

### Loss & Training
This work is an evaluation framework and does not involve model training.

## Key Experimental Results

### Main Results (Kendall's $\tau$, averaged across 6 models)

| Dataset | Release Date | $\tau$ (All ranks) | $\tau$ (rank $\ge 3$) |
| :--- | :--- | :--- | :--- |
| Spider | 2018.09 | -0.89 | -0.88 |
| SParC | 2019.06 | -0.76 | -0.63 |
| CoSQL | 2019.10 | -0.71 | -0.64 |
| BIRD | 2023.05 | -0.35 | -0.37 |

### Control Experiments to Exclude Alternative Explanations

| Control Variable | Method | Conclusion |
| :--- | :--- | :--- |
| Query Length | Compare length distributions of rank 1/5/10 | Distributions are similar; length effect excluded |
| Lexical Overlap | Recalculate $\tau$ after Jaccard stratification | Negative correlation persists; lexical effect excluded |
| Schema linking | Human inspection of failure cases | No broken schema linking found |
| Rewrite Generator | Substitute GPT-5 with LLaMA-4 Maverick | Decay patterns and slopes remain consistent |

### Key Findings
- **Clear Temporal Gradient**: Older benchmarks have larger absolute $\tau$ values, indicating stronger contamination signals. Spider (2018) is the strongest (-0.89), while BIRD (2023) is the weakest (-0.35).
- On Spider, all models show an accuracy drop exceeding 10-15 percentage points at ranks 8-10, whereas they remain stable or slightly improve on BIRD.
- Even when considering only distant rewrites (rank $\ge$ 3), the negative correlation for Spider/SParC/CoSQL remains significant ($\tau$ between -0.57 and -1.00).
- Larger models (e.g., LLaMA-3.1-405B) decay more slowly on SParC, but no model is immune.

## Highlights & Insights
- **Ingenious Syntactic Probe Design**: By controlling the syntactic distance of rewrites to build a continuous gradient, the method transforms contamination detection from a binary judgment (leaked or not) into a continuous measure (decay rate), which is more informative.
- **Comprehensive Control Experiments**: The influence of query length, lexical overlap, schema linking, and rewrite generator bias was ruled out one by one, demonstrating methodological rigor.
- The method is transferable to contamination detection in other NLP benchmarks: any task allowing for semantic-preserving rewrites (e.g., QA, text classification) can adopt the SPENCE framework.

## Limitations & Future Work
- The temporal gradient represents correlation, not necessarily causation; older benchmarks may differ from newer ones in difficulty, annotation style, or distributional characteristics.
- Only general LLMs were evaluated, excluding SQL-specific systems (e.g., OmniSQL, Arctic-Text2SQL-R1).
- Rewrite generation itself relies on an LLM (GPT-5); although results were consistent with LLaMA-4, generator bias cannot be entirely ruled out.
- For conversational benchmarks (SParC, CoSQL), only the last user query was rewritten, without extending to earlier turns.
- Future work could combine SPENCE with sample-specific probes like ConStat to capture deeper forms of contamination.

## Related Work & Insights
- **vs ConStat**: ConStat defines contamination as non-generalizable performance inflation across syntax/sample/benchmark dimensions. SPENCE focuses on the syntax axis but provides finer distance gradient control.
- **vs Ranaldi et al.**: They detect dataset-level contamination by comparing Spider vs. a new dataset (Termite). SPENCE detects it via syntactic transformations within the same dataset, offering finer granularity.
- **vs Min-K% Prob**: Requires internal model token probabilities (white-box). SPENCE is entirely black-box, requiring only the observation of outputs.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of syntactic distance gradients and Kendall's $\tau$ for NL2SQL contamination detection is novel, though the general framework of probe-based evaluation is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across four benchmarks, six models, four control experiments, bootstrap confidence intervals, and verification with alternative rewrite generators.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and informative charts, though it is quite long and could be more concise.
- Value: ⭐⭐⭐⭐ Provides an actionable contamination detection tool for the NL2SQL community, with a methodology generalizable to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems](clarity_a_framework_and_benchmark_for_conversational_language_ambiguity_and_unan.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ICML 2026\] When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](../../ICML2026/llm_evaluation/when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)

</div>

<!-- RELATED:END -->
