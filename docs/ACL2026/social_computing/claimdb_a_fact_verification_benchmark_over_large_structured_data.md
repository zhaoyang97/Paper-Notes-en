---
title: >-
  [Paper Note] ClaimDB: A Fact Verification Benchmark over Large Structured Data
description: >-
  [ACL 2026][Social Computing][Fact verification benchmark] ClaimDB is the first benchmark to scale fact-verification evidence to 80 real-world databases, averaging 11 tables, 4.6 million rows…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Fact verification benchmark"
  - "large-scale structured data"
  - "tool-calling agent"
  - "NEI abstention capability"
  - "SQL reasoning"
date: 2026-05-08
content_hash: 5351dcaed1af203c
---

# ClaimDB: A Fact Verification Benchmark over Large Structured Data

**Conference**: ACL 2026  
**arXiv**: [2601.14698](https://arxiv.org/abs/2601.14698)  
**Code**: https://claimdb.github.io  
**Area**: Fact Verification / Structured Data / LLM Evaluation  
**Keywords**: Fact verification benchmark, large-scale structured data, tool-calling agent, NEI abstention capability, SQL reasoning

## TL;DR
ClaimDB is the first benchmark to scale fact-verification evidence to 80 real-world databases, averaging 11 tables, 4.6 million rows, and 110 million tokens per claim. This scale mandates the use of executable programs (SQL) for compositional reasoning. Evaluations of 30 SOTA LLM tool-calling agents reveal that over half have an accuracy below 55%. Furthermore, closed-source models rarely "abstain," while open-source models over-abstain—identifying NEI (Not Enough Information) processing as the most significant weakness.

## Background & Motivation

**Background**: Existing fact-verification benchmarks have evolved from FEVER (text) to TabFact and FEVEROUS (small tables) and then to SCITAB (scientific tables). However, they all assume a paradigm where evidence fits within the LLM's context window—allowing for a "read-then-reason" approach.

**Limitations of Prior Work**: Real-world high-impact claims (e.g., Biden's "U.S. inflation is the lowest in the world" or Trump's "Washington's murder rate is the highest in the world") rely on evidence from million-row CSVs or multi-table databases (like the BLS or police departments) that cannot be fully read. Even with Gemini's 1–2M token context, the gap remains 2–3 orders of magnitude compared to the 110M tokens found in these datasets.

**Key Challenge**: There is a massive scale gap between the true distribution of fact-checking evidence and the "small evidence" assumption of existing benchmarks. This causes models that appear SOTA on existing leaderboards to fail completely on real databases. Once the evidence cannot be "read in full," the paradigm must shift to "symbolic processing of massive data via executable programs + reasoning." However, this neuro-symbolic capability lacks both suitable benchmarks and systematic evaluation.

**Goal**: Construct a large-scale fact-verification benchmark where **(1) evidence far exceeds LLM context, (2) compositional reasoning like aggregation/sorting/multi-table joins is required, and (3) realistic NEI categories are included**, then systematically evaluate the performance of 30 LLMs as tool-calling agents.

**Key Insight**: Starting from BIRD (an NL2SQL benchmark with 11k NL/SQL pairs and real large databases), the SQL queries themselves indicate "questions requiring compositional reasoning." By executing these and using a GPT-5 generator followed by a multi-LLM judge panel, the authors produce high-quality entailed/contradicted/NEI claims.

**Core Idea**: Utilize SQL AST to filter for compositional queries (aggregation, sorting, multi-table joins, window functions), convert execution results into claims, and perform quality control via an LLM-as-a-judge panel. This forces evaluation methods to solve claims using SQL tool-calling agents.

## Method

### Overall Architecture
The ClaimDB construction pipeline consists of five stages:

- **Step 1 BIRD Origin**: Start with 11k NL/SQL pairs and 80 real databases from BIRD.
- **Step 2 Pre-Filtering**: Convert SQL to AST and retain only queries with `ORDER BY`, aggregates (`AVG`, `SUM`), window functions, or multi-table joins where the answer rows $\le 10$, leaving ~6.5k queries.
- **Step 3 Claim Generation**: For each Q/A pair, use GPT-5 to generate three types of claims: entailed, contradicted, and NEI (further categorized into out-of-schema, counterfactual, and subjective).
- **Step 4 Quality Control**: A judge panel (Phi-4 + grok-3-mini + mistral-small) evaluates each claim for label correctness, self-containment, and NEI validity. A **single-veto system** is used—any rejection results in a drop.
- **Step 5 NEI Grounding**: Use `gemini-embedding-001` to calculate similarity between claims and Q/A pairs. NEIs are ranked by conceptual distance to the database, and the test split samples hard cases from the top quartile.

The final dataset contains 53,368 claims (E: 12,855 / C: 16,529 / NEI: 23,984), with an average of 11.3 tables, 4.6M rows, and ~110M tokens per claim. Agents are provided a SQL execution tool (Google MCP toolbox) with a 20-call limit for 3-way classification (E/C/NEI).

### Key Designs

1. **AST-based Pre-Filtering (Enforced Compositional Reasoning)**:
    - **Function**: Filters the 11k BIRD pairs for a subset where the answer requires aggregating massive data, preventing the benchmark from degrading into simple single-row lookups.
    - **Mechanism**: Parses SQL into an AST and retains it if it meets 4 rules: (a) `ORDER BY`/superlatives (e.g., `MAX`, `TOP-K`); (b) aggregate functions (`AVG`, `SUM`, `COUNT`); (c) window functions (complex info flow across partitions); (d) joins involving 3+ tables. Answer rows are capped at $\le 10$ to ensure GPT-5 can track column/value structures during claim generation.
    - **Design Motivation**: This is the source of ClaimDB's difficulty. LLM prompting alone cannot guarantee that evidence is large or reasoning is compositional. AST rules provide mechanical, verifiable proof of compositional reasoning, ensuring that any method failing at cross-table aggregation cannot succeed.

2. **Three-Category + Three-Subcategory NEI Claim Generation**:
    - **Function**: Refines "Not-Enough-Info" (NEI) from a coarse category into three realistic failure modes, preventing NEI from being trivially identifiable.
    - **Mechanism**: Adopts a taxonomy defining three NEI subcategories—**Out-of-Schema** (concepts not in the DB); **Counterfactual** (what-if scenarios); **Subjective** (value judgments). E/C claims use 1-shot prompting (structure constrained by answers), while NEI claims use zero-shot prompting to maximize diversity. Metadata is provided to GPT-5 to ensure Out-of-Schema claims are conceptually relevant to the DB theme.
    - **Design Motivation**: Traditional benchmarks either lack NEI or use simple irrelevant samples. In real fact-checking, "insufficient evidence" is common and dangerous. This taxonomy allows for diagnostic evaluation (e.g., discovering closed models are "afraid" to predict NEI while open models over-predict it).

3. **LLM Judge Panel (Recall-oriented Quality Control) + STS Grounding**:
    - **Function**: Provides high-quality filtering at scale (64k claims) at low cost and layers NEI difficulty by conceptual proximity.
    - **Mechanism**: (a) **Panel Design**: Uses three small models from different families (Phi-4, grok-3-mini, mistral-small) to avoid self-enhancement bias from OpenAI models. The rubric is binary (label correct? self-contained? NEI subcategory correct?). The prompt explicitly instructs "If you are unsure, answer no" to **maximize recall on bad claims**. (b) **STS Grounding**: Filters for hard NEIs using embeddings to ensure they are semantically close to the DB content (e.g., "Commander has a law degree" vs. "Cases involve tourists"), forcing the model to query the DB rather than reject based on common sense.
    - **Design Motivation**: Human annotation is unfeasible at this scale. A multi-family small-judge panel with a recall-first prompt provides an optimal engineering solution. STS-based grounding prevents the benchmark from becoming trivial in the NEI dimension.

## Key Experimental Results

### Main Results (30 LLMs × 1000 Public Claims, SQL tool-calling agent, 20-call limit)

| Model | Acc. | Macro-F1 | F1_E | F1_C | F1_NEI |
|------|------|----------|------|------|--------|
| gpt-5-mini | **0.827** | **0.828** | 0.810 | 0.815 | 0.860 |
| claude-haiku-4-5 | 0.809 | 0.811 | 0.815 | 0.814 | 0.805 |
| gemini-3-flash | 0.801 | 0.800 | 0.776 | 0.832 | 0.792 |
| gpt-5-nano | 0.787 | 0.787 | 0.777 | 0.794 | 0.790 |
| gemini-2.5-flash | 0.793 | 0.793 | 0.755 | 0.777 | 0.849 |
| gpt-oss:20b (open) | 0.740 | 0.739 | 0.749 | 0.710 | 0.758 |
| qwen3-coder:30b | 0.672 | 0.672 | 0.691 | 0.641 | 0.685 |
| nemotron-3-nano:30b | 0.667 | 0.671 | 0.681 | 0.658 | 0.673 |
| ministral-3:14b | 0.623 | 0.623 | 0.605 | 0.608 | 0.655 |
| qwen3:32b | 0.574 | 0.561 | 0.512 | 0.544 | 0.626 |
| llama3.1:8b | 0.344 | 0.288 | 0.269 | 0.133 | 0.461 |
| qwen3:1.7b | 0.366 | 0.239 | 0.110 | 0.089 | 0.518 |

**Key Statistics**: **17 out of 30 models (>50%)** scored < 55% in both Acc and Macro-F1. Most open-source models do not exceed 68%.

### Analysis Experiments

| Dimension | Key Observation | Implication |
|---------|---------|------|
| Contamination Test | `gpt-5-mini` without tools yields Macro-F1=0.253, Acc=0.367 (near random). | ClaimDB cannot be solved by parametric knowledge alone, ruling out contamination. |
| Tool Call Frequency | Quadratic polynomial fit shows optimal performance at ~4–8 calls; more calls lead to degradation. | Long sessions cause loss of focus; a single bad query can dump hundreds of thousands of tokens. |
| SQL Success Rate | `gpt-5-mini` 93%, `claude-haiku-4.5` 99%. | Incorrect queries are evenly distributed among success/fail predictions, suggesting failures are in reasoning, not syntax. |
| Abstention Behavior (NEI) | `gpt-5-mini`/`claude` rarely predict NEI; `qwen3`/`nemotron` over-predict NEI. | Closed models are overconfident; open models "give up"—the gap is primarily in NEI handling. |

### Key Findings
- **Over half of SOTA LLMs score < 55% Acc on ClaimDB**, indicating "large-scale structured data + compositional reasoning" is a genuine blind spot not covered by existing benchmarks.
- **Scaling returns are log-linear and weak**: Increasing model size in open-source models yields marginal gains, suggesting data scaling alone won't solve ClaimDB.
- **Abstention behavior is polarized**: Closed-source models "hallucinate knowledge," while open-source models "refuse to try." Both behaviors are unacceptable, highlighting a calibration issue for trustworthy LLMs.
- **The tool-calling "sweet spot"**: 4–8 SQL calls are optimal. This provides a specific execution target for agent design.

## Highlights & Insights
- **Scale Leap**: Moving from TabFact’s kilobyte-scale evidence to 110M tokens increases difficulty by 3 orders of magnitude and forces a paradigm shift toward neuro-symbolic methods.
- **Reusable Pipeline**: The AST filtering + multi-LLM judge panel + STS grounding pipeline is highly reusable for any project synthesizing controlled-difficulty datasets from NL-to-SQL data.
- **Anti-Pattern Prompting for Judges**: Instead of "agreement with humans," the judge panel optimizes for "recall on bad samples" using a single-veto, conservative prompt strategy.
- **Quantifying Abstention**: Breaking NEI into subcategories and using confusion matrices reveals diagnostic insights into closed vs. open-source behaviors, providing more information than a simple abstention rate.

## Limitations & Future Work
- **Ours**: (1) **Dependency on BIRD**—annotation errors in BIRD propagate to ClaimDB; (2) **Single modality**—only structured data is covered; (3) **Snapshot validity**—DBs are static snapshots and may drift from the current world; (4) **SQL bias**—reliance on SQL might disadvantage models better at Python/pandas.
- **Future Work**: (1) Include multi-modal evidence (charts, PDFs); (2) Use multiple base models for generation to reduce single-model bias; (3) Compare coding agents (Python) vs. SQL agents to further probe capability differences.

## Related Work & Insights
- **Comparison with FEVER/TabFact/SCITAB**: In those benchmarks, evidence is "readable." ClaimDB pushes the task from "read+reason" to "search+reason."
- **Comparison with BIRD**: BIRD is an NL-to-SQL parsing benchmark; ClaimDB "recycles" it by using its SQL as the ground truth for compositional reasoning to generate claims.
- **Vision for the Future**: ClaimDB provides a stress test for neuro-symbolic methods like Program of Thoughts, where such methods are expected to show intrinsic advantages over in-context reading.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First 110M token-scale fact-checking benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 LLMs, 4-dimensional analysis, contamination checks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear pipeline, intuitive real-world examples, and taxonomies.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous metric for "LLM capability on real big data."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ACL 2026\] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)
- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)

</div>

<!-- RELATED:END -->
</div>

## Related Papers

- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[ACL 2026\] VeriTaS: The First Dynamic Benchmark for Multimodal Automated Fact-Checking](veritas_the_first_dynamic_benchmark_for_multimodal_automated_fact-checking.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[ACL 2026\] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)
- [\[ACL 2026\] Decide less, communicate more: On the construct validity of end-to-end fact-checking in medicine](decide_less_communicate_more_on_the_construct_validity_of_end-to-end_fact-checki.md)

</div>

<!-- RELATED:END -->
