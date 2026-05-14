---
title: >-
  [Paper Note] SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks
description: >-
  [ACL 2026][Interpretability][data contamination detection] SPENCE detects and quantifies data contamination of LLMs on NL2SQL benchmarks by systematically generating syntactic paraphrases of benchmark queries and measuri…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "data contamination detection"
  - "NL2SQL"
  - "syntactic probe"
  - "benchmark leakage"
  - "generalization evaluation"
date: 2026-05-08
content_hash: 4fccb14ad45a0ee8
---

# SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks

**Conference**: ACL 2026
**arXiv**: [2604.17771](https://arxiv.org/abs/2604.17771)
**Code**: None
**Area**: Interpretability
**Keywords**: data contamination detection, NL2SQL, syntactic probe, benchmark leakage, generalization evaluation

## TL;DR

SPENCE detects and quantifies data contamination of LLMs on NL2SQL benchmarks by systematically generating syntactic paraphrases of benchmark queries and measuring the decay of execution accuracy as a function of syntactic distance. Older benchmarks (e.g., Spider) exhibit stronger contamination signals, while the more recent BIRD benchmark is largely unaffected.

## Background & Motivation

**Background**: LLMs have achieved high execution accuracy on NL2SQL benchmarks (Spider, SParC, CoSQL, BIRD), yet the research community has grown increasingly skeptical of whether these scores reflect genuine generalization. Data contamination and benchmark leakage have been widely studied, with various methods (e.g., ConStat, Data Contamination Quiz, Min-K% Prob) detecting memorization behavior from different perspectives.

**Limitations of Prior Work**: Existing contamination detection methods either rely on string-overlap checks—failing to capture structural memorization—or require access to model internals, making them inapplicable to black-box models. No systematic contamination detection framework specifically targeting SQL generation tasks exists for the NL2SQL setting. The closest prior work, Ranaldi et al. (2024), only compares seen vs. unseen databases at the dataset level, without fine-grained probing along a syntactic axis.

**Key Challenge**: High reported accuracy may stem from memorization of the surface forms of benchmark queries rather than true compositional generalization. If a model has merely memorized the syntactic patterns of original queries, it should perform worse on semantically equivalent but syntactically distinct paraphrases. The key question is how to distinguish memorization from insufficient generalization, and whether this behavior correlates with a benchmark's release date.

**Goal**: To design a controlled syntactic probing framework that quantifies contamination sensitivity in NL2SQL models by generating paraphrases of increasing syntactic distance, and to conduct a comparative analysis across four benchmarks.

**Key Insight**: Syntactic dependency tree edit distance (TED) is used to rank paraphrases, operationalizing "degree of contamination" as the rate of accuracy decay with syntactic distance, with Kendall's τ used to statistically quantify the trend.

**Core Idea**: Syntactic paraphrasing + distance ranking + accuracy decay analysis = contamination probe. Steeper decay on older benchmarks indicates heavier contamination.

## Method

### Overall Architecture

The SPENCE pipeline proceeds as follows: (1) extract natural language queries, gold SQL, and database schemas from the test set; (2) use GPT-5 to generate 10 syntactic paraphrases per query; (3) parse dependency trees with spaCy, compute tree edit distance (TED) for ranking (rank 1 = closest, rank 10 = farthest); (4) filter semantically drifted paraphrases using embedding cosine similarity; (5) execute both the original query and each ranked paraphrase on downstream NL2SQL models, comparing execution accuracy change $\Delta$ Accuracy; (6) quantify the rank correlation between accuracy and paraphrase rank using Kendall's τ.

### Key Designs

1. **TED-Based Syntactic Distance Ranking**:

    - Function: Rank paraphrases by degree of syntactic variation to construct a controlled "distance gradient."
    - Mechanism: Each paraphrase and the original query are parsed into dependency trees using spaCy; TED is computed as $\text{TED}_i = \text{TreeEditDistance}(T_{p_i}, T_q)$, where permitted operations are node insertion, deletion, and substitution in the dependency tree. Low TED corresponds to surface lexical changes (synonym substitution); high TED corresponds to deep structural changes (clause reordering).
    - Design Motivation: Unlike simple lexical overlap metrics, TED captures genuine syntactic structural change, ensuring that the probe targets syntactic robustness rather than surface lexical similarity.

2. **Embedding Similarity Filtering + Human Validation**:

    - Function: Ensure paraphrases remain semantically equivalent, excluding nonsensical or semantically drifted outputs.
    - Mechanism: E5-base-v2 embedding cosine similarity between each paraphrase and the original query is computed; paraphrases below the threshold are filtered. An additional human validation is performed on 100 Spider examples across ranks 1/5/10, confirming that 85%+ of paraphrases are semantically equivalent.
    - Design Motivation: If paraphrases are not semantically equivalent, accuracy drops may reflect semantic shift rather than a contamination signal; the filtering step eliminates this confound.

3. **Kendall's τ Rank Correlation + Bootstrap Confidence Intervals**:

    - Function: Statistically quantify the monotonic decline in accuracy with increasing paraphrase rank.
    - Mechanism: For each model–dataset pair, Kendall's τ is computed as $\tau = (n_c - n_d) / \frac{1}{2}n(n-1)$, where $n_c$ and $n_d$ are the numbers of concordant and discordant pairs, respectively. 95% confidence intervals are obtained via $B=100$ bootstrap resamplings. A strong negative τ indicates that accuracy decreases monotonically with syntactic distance (contamination signal); values near zero indicate no sensitivity.
    - Design Motivation: Provides a statistically reliable contamination measure rather than relying on visual inspection alone. Confidence intervals that do not cross zero confirm the trend is statistically significant.

### Loss & Training
This paper presents an evaluation framework and does not involve model training.

## Key Experimental Results

### Main Results (Kendall's τ, averaged across 6 models)

| Dataset | Release Date | τ (all ranks) | τ (rank ≥ 3) |
|---------|-------------|---------------|--------------|
| Spider | 2018.09 | -0.89 | -0.88 |
| SParC | 2019.06 | -0.76 | -0.63 |
| CoSQL | 2019.10 | -0.71 | -0.64 |
| BIRD | 2023.05 | -0.35 | -0.37 |

### Control Experiments Ruling Out Alternative Explanations

| Control Variable | Method | Conclusion |
|-----------------|--------|------------|
| Query length | Compare length distributions at ranks 1/5/10 | Distributions are similar; length effect ruled out |
| Lexical overlap | Recompute τ after Jaccard-based stratification | Negative correlation persists; lexical effect ruled out |
| Schema linking | Manual inspection of failure cases | No schema linking breakdowns detected |
| Paraphrase generator | Replace GPT-5 with LLaMA-4 Maverick | Decay patterns and slopes are consistent |

### Key Findings
- **Clear temporal gradient**: Older benchmarks exhibit larger absolute τ values and stronger contamination signals. Spider (2018) is the strongest (-0.89); BIRD (2023) is the weakest (-0.35).
- On Spider, all models show accuracy drops exceeding 10–15 percentage points at ranks 8–10, while on BIRD accuracy remains stable or even slightly improves.
- Even when considering only distant paraphrases at rank ≥ 3, the negative correlation for Spider/SParC/CoSQL remains significant (τ ranging from -0.57 to -1.00).
- Larger models (e.g., LLaMA-3.1-405B) exhibit slower decay on SParC, but no model is immune.

## Highlights & Insights
- **The syntactic probe design is elegant**: By controlling the syntactic distance of paraphrases to construct a continuous gradient, contamination detection is transformed from a binary judgment (leaked or not) into a continuous measure (decay rate), yielding substantially more information.
- **Four control experiments comprehensively rule out alternative explanations**: The effects of query length, lexical overlap, schema linking, and paraphrase generator bias are each eliminated in turn, demonstrating methodological rigor.
- The method is transferable to contamination detection in other NLP benchmarks: any task admitting semantically preserving paraphrases (QA, text classification, etc.) can adopt the SPENCE framework.

## Limitations & Future Work
- The temporal gradient is correlational rather than causal: older benchmarks may differ from newer ones in difficulty, annotation style, and distributional properties.
- Only general-purpose LLMs are evaluated; SQL-specialized systems (e.g., OmniSQL, Arctic-Text2SQL-R1) are not covered.
- Paraphrase generation itself relies on an LLM (GPT-5); although substituting LLaMA-4 yields consistent results, generator bias cannot be entirely excluded.
- For conversational benchmarks (SParC, CoSQL), only the final user utterance is paraphrased; earlier dialogue turns are not addressed.
- Future work could combine SPENCE with ConStat's sample-specific probes to capture deeper forms of contamination.

## Related Work & Insights
- **vs. ConStat**: ConStat defines contamination as non-generalizable performance inflation and detects it along syntax, sample, and benchmark dimensions. SPENCE focuses on the syntactic axis but provides finer-grained distance gradient control.
- **vs. Ranaldi et al.**: Their approach detects dataset-level contamination by comparing Spider against a new dataset, Termite. SPENCE operates within the same dataset via syntactic transformation, offering finer granularity.
- **vs. Min-K% Prob**: This method requires internal token probabilities and is thus white-box. SPENCE is fully black-box, requiring only observed outputs.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of syntactic distance gradients and Kendall's τ for NL2SQL contamination detection is novel, though the broader framework of probe-based evaluation is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, six models, four control experiments, bootstrap confidence intervals, and paraphrase generator substitution—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with informative figures and tables, though the paper is somewhat lengthy and could be condensed.
- Value: ⭐⭐⭐⭐ Provides the NL2SQL community with an actionable contamination detection tool, and the method generalizes to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do LLMs Know Tool Irrelevance? Demystifying Structural Alignment Bias in Tool Invocations](do_llms_know_tool_irrelevance_demystifying_structural_alignment_bias_in_tool_inv.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Evian: Towards Explainable Visual Instruction-tuning Data Auditing](evian_towards_explainable_visual_instruction-tuning_data_auditing.md)

</div>

<!-- RELATED:END -->
