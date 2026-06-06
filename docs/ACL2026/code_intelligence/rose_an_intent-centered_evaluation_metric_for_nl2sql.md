---
title: >-
  [Paper Note] ROSE: An Intent-Centered Evaluation Metric for NL2SQL
description: >-
  [ACL2026][Code Intelligence][Intent-centered evaluation] ROSE shifts NL2SQL evaluation from "predicting whether SQL matches a single reference SQL" to "predicting whether SQL satisfies user intent." Through a two-stage r…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Intent-centered evaluation"
  - "Text-to-SQL"
  - "Prover-Refuter"
  - "Execution Accuracy"
  - "Dataset Diagnosis"
date: 2026-05-08
content_hash: 7bd60b2fd434259f
---

# ROSE: An Intent-Centered Evaluation Metric for NL2SQL

**Conference**: ACL2026  
**arXiv**: [2604.12988](https://arxiv.org/abs/2604.12988)  
**Code**: https://github.com/CedricPei/ROSE  
**Area**: NL2SQL / Evaluation Metrics / Database QA  
**Keywords**: Intent-centered evaluation, Text-to-SQL, Prover-Refuter, Execution Accuracy, Dataset Diagnosis

## TL;DR
ROSE shifts NL2SQL evaluation from "predicting whether SQL matches a single reference SQL" to "predicting whether SQL satisfies user intent." Through a two-stage reasoning process involving an SQL Prover and an Adversarial Refuter, ROSE achieves nearly 24 percentage points higher Cohen's Kappa than the best existing metrics on ROSE-VEC and reveals an evaluation crisis caused by reference SQL errors and question ambiguity in benchmarks like BIRD.

## Background & Motivation
**Background**: The goal of NL2SQL is to convert natural language questions into executable SQL. Evaluation has long relied on Execution Accuracy (EX). The criterion for EX is straightforward: a predicted SQL is considered correct if its execution result on the database matches that of the annotated SQL; otherwise, it is incorrect. This metric is simple, automated, and scalable, making it the core standard for benchmarks like Spider and BIRD.

**Limitations of Prior Work**: As LLM generation capabilities enhance, the flaws of EX become increasingly apparent. First, the same semantics can have multiple SQL implementations or output representations, leading EX to misjudge non-standard but correct implementations. Second, user questions may inherently have multiple reasonable interpretations that a single reference SQL cannot cover. Third, annotated SQLs can also be incorrect; an erroneous reference will judge a correct prediction as wrong. The paper cites analysis showing that non-canonical but correct forms can lead to up to 28.9% false negatives, and approximately 6.91% ground-truth SQL errors are reported in BIRD Dev.

**Key Challenge**: What NL2SQL truly needs to evaluate is "whether the user question was answered," rather than "whether the reference SQL was replicated." Reference SQLs are useful but should not be the sole ground truth; however, completely discarding them might lead an LLM judge to be overly lenient. Therefore, an evaluation mechanism is needed that is intent-centered while still utilizing the reference SQL as evidence for refutation.

**Goal**: The authors propose ROSE as an intent-centered metric, construct the ROSE-VEC expert consensus verification set to validate its alignment with human experts, and re-evaluate 19 NL2SQL methods using ROSE to analyze the gap between EX and semantic correctness in the era of strong models.

**Key Insight**: Instead of simply asking an LLM to compare the predicted SQL and reference SQL, the paper decomposes evaluation into proof and refutation. The Prover first evaluates whether the predicted SQL satisfies the question intent independently (without looking at the reference SQL). The Refuter then uses the reference SQL as counter-evidence to challenge the Prover’s judgment and diagnose reference errors or question ambiguity.

**Core Idea**: Downgrade ground-truth SQL from the "sole truth" to "evidence for refutation," balancing the leniency of intent-centered evaluation with the constraints of reference signals through a Prover-Refuter cascade.

## Method
The evaluation objects of ROSE are the natural language question, database, predicted SQL, reference SQL, execution results, and a set of acceptance criteria. It first requires the predicted SQL to be executable; if not, it is judged incorrect. If the execution results of the predicted and reference SQLs differ, the SQL Prover independently judges whether the predicted SQL satisfies the user intent. Even if the results are identical, the Refuter still checks for accidental correctness or reference errors. A prediction only receives 1 point if it passes all stages.

### Overall Architecture
The main workflow of ROSE consists of three steps. The first step is a syntax and execution check to filter out non-runnable SQL. The second step is the SQL Prover: when execution results differ, the Prover considers only the question, database, predicted SQL, and predicted result to judge semantic correctness based on acceptance criteria. The third step is the Adversarial Refuter: it reads both predicted and reference SQLs, using the reference as evidence to challenge the Prover or check for cases where identical results occur despite logical errors. It outputs whether to overturn the Prover's judgment along with diagnostic labels.

ROSE-VEC is used to validate the metric itself. The dataset contains 585 NL-SQL pairs, with 263 from multiple system outputs on Spider Test and 322 from BIRD Dev. Each sample was independently judged by two out of five experts, and only samples with complete agreement were retained to obtain high-confidence expert labels.

### Key Designs
1.  **Reference-Agnostic Judgment of SQL Prover**:
    - **Function**: Reduces the anchoring effect of a single reference SQL on evaluation.
    - **Mechanism**: The Prover does not access the ground-truth SQL. It judges whether the prediction satisfies user intent based solely on the question, database schema/content, predicted SQL, and execution result, outputting a boolean judgment and reasoning.
    - **Design Motivation**: Many correct predictions differ from the reference result due to formatting, ordering, redundant columns, or reasonable ambiguity. Ignoring the reference initially prevents these answers from being prematurely disqualified.

2.  **Evidence-Based Adversarial Refuter**:
    - **Function**: Controls the potential over-leniency of the Prover and utilizes information within the reference SQL.
    - **Mechanism**: When the Prover accepts a prediction that differs from the reference result, the Refuter compares the reasoning logic of both and determines if the difference affects user intent. It can overturn the Prover and label ground-truth errors (GoldX) or question ambiguity (AmbQ). Even if execution results match, the Refuter checks if it was merely coincidental.
    - **Design Motivation**: Reference-free LLM judges often overlook erroneous SQL, whereas reference SQLs, though not always perfect, contain valuable counter-evidence signals.

3.  **Diagnostic Labels and Versioned LLM Judge**:
    - **Function**: Enables the metric to not only provide scores but also help identify dataset issues and ensure reproducibility over long periods.
    - **Mechanism**: The Refuter outputs labels like GoldX and AmbQ for conflict cases. The ROSE judge is named using `ROSE_model-time` (e.g., `ROSE_o3-2504`); new models must be re-verified before replacement.
    - **Design Motivation**: Part of the NL2SQL evaluation crisis stems from the benchmarks themselves. The metric should be able to back-trace annotation errors and ambiguities. Since LLM judges change with versions, explicit versioning is required to reduce leaderboard drift.

### Loss & Training
ROSE is not a trained model but an evaluation pipeline based on a reasoning backbone. The authors instantiated the Prover/Refuter using OpenAI o3-2504, Gemini-2.5 Pro-2506, and DeepSeek-R1-2505. To reduce costs, ROSE uses simplified prompts and decides whether to invoke the second-stage Refuter based on execution results. To improve throughput, evaluations can be executed in parallel using multiple threads.

## Key Experimental Results

### Main Results
Core results on ROSE-VEC show that ROSE's alignment with expert labels is significantly higher than EX, FLEX, and LLM-SQL-Solver.

| Backbone | Metric | Kappa (%) | Acc (%) | MCC (%) | F1 (%) |
|----------|--------|-----------|---------|---------|--------|
| Deterministic | EM | 0.51 | 27.86 | 5.07 | 1.86 |
| Deterministic | ETM | 6.60 | 35.56 | 18.47 | 20.63 |
| Deterministic | EX | 25.56 | 55.90 | 37.23 | 57.00 |
| OpenAI o3 | FLEX | 56.70 | 78.97 | 62.01 | 83.31 |
| OpenAI o3 | ROSE w/o Refuter | 60.74 | 85.47 | 61.46 | 90.40 |
| OpenAI o3 | ROSE | **80.43** | **91.79** | **81.04** | **94.16** |
| Gemini-2.5 Pro | ROSE | 69.68 | 86.84 | 71.01 | 90.41 |
| DeepSeek-R1 | ROSE | 64.49 | 84.62 | 65.68 | 88.81 |

### Ablation Study

| Backbone | Metric | Kappa (%) | Acc (%) | MCC (%) | F1 (%) | Description |
|----------|--------|-----------|---------|---------|--------|-------------|
| OpenAI o3 | Unified w/o GT | 53.35 | 80.43 | 54.00 | 86.09 | Single prompt without reference SQL |
| OpenAI o3 | Unified | 66.35 | 83.85 | 68.22 | 86.87 | Single prompt with reference SQL |
| OpenAI o3 | ROSE w/o GT | 71.01 | 86.34 | 72.25 | 89.11 | Phased but without reference SQL |
| OpenAI o3 | ROSE | 80.68 | 90.99 | 81.64 | 92.91 | Full Prover-Refuter cascade |
| Gemini-2.5 Pro | Unified | 59.90 | 81.06 | 61.02 | 84.86 | Merged stages perform worse |
| Gemini-2.5 Pro | ROSE | 64.79 | 82.92 | 67.15 | 85.93 | Phased reasoning is more stable |

### Key Findings
- The Kappa for EX is only 25.56%, indicating a massive gap compared to expert semantic judgment; ROSE_o3-2504 reaches 80.43%, roughly 23.73 percentage points higher than FLEX_o3-2504.
- The Refuter is a critical component. Under OpenAI o3, the Kappa for ROSE w/o Refuter is 60.74%, while the full ROSE improves to 80.43%, showing that a purely reference-free Prover is insufficient.
- ROSE's diagnostic labels have practical value: OpenAI o3 achieves 84.32% precision for GoldX and 91.23% for AmbQ, which can be used for automated benchmark auditing.
- After re-evaluating 19 methods on BIRD Mini-Dev, the authors found that the gap between ROSE and EX grows as models become stronger, suggesting that progress in methods might be underestimated by reference-matching metrics.
- Multi-threading significantly improves efficiency. ROSE_o3-2504 takes an average of 22.48s per question on ROSE-VEC-BIRD single-threaded, which effectively drops to 3.35s with 8 threads.

## Highlights & Insights
- The most significant contribution of the paper is the redefinition of the role of ground truth. Reference SQL is no longer the judge but rather the opposing evidence; this is more reasonable than "complete trust in the reference" or "complete ignorance of the reference."
- ROSE integrates evaluation metrics with dataset diagnosis. It not only provides a method score but also identifies if discrepancies arise from reference errors or question ambiguity. This is highly valuable for maintaining NL2SQL benchmarks.
- The results reveal a concerning trend: the stronger the model, the more likely it is to generate semantically correct but formally different SQL, making EX increasingly likely to underestimate progress. If evaluation metrics do not upgrade, research directions may be distorted.
- Versioned LLM judges are a practical engineering detail. Many LLM-as-judge metrics suffer from irreproducible scores due to backbone updates; ROSE explicitly requires recording the backbone and release date.

## Limitations & Future Work
- ROSE depends on the base reasoning model; performance varies significantly across different backbones. The Kappa for DeepSeek-R1 and Gemini is lower than that of OpenAI o3, indicating that metric reliability fluctuates with model capability.
- ROSE-VEC only includes samples where two experts completely agree, which reduces label noise but may underestimate the difficulty of borderline cases and truly ambiguous problems.
- Multi-stage LLM judges have higher costs and latency than EX. While multi-threading and conditional calls provide mitigation, budget control remains necessary for large-scale online leaderboards.
- The Refuter uses reference SQL as evidence. If the reference SQL contains a highly subtle error or if the database content is insufficient, the judgment may still be misled. Future work could incorporate multiple reference SQLs, counterfactual data, or execution test generation to strengthen the evidence chain.

## Related Work & Insights
- **vs Execution Accuracy**: EX is automatic, cheap, and reproducible but compresses semantic correctness into a single execution result match; ROSE is closer to user intent but entails LLM reasoning costs.
- **vs FLEX**: FLEX is also an LLM-based metric but still focuses on sufficiency judgments centered around the reference SQL; ROSE reduces reference anchoring by having the Prover judge independently before the Refuter provides counter-arguments.
- **vs LLM-SQL-Solver**: LLM-SQL-Solver directly judges SQL equivalence, whereas ROSE explicitly distinguishes between semantic satisfaction, reference errors, and question ambiguity, providing stronger diagnostic capabilities.
- **Insight**: Many generation tasks suffer from the "unreliable single reference" problem, such as code generation, data analysis, information extraction, and multi-hop QA. Reforming reference answers into adversarial evidence may be more suitable for the era of strong generative models than simple reference matching.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Restructures the NL2SQL evaluation paradigm; the Prover-Refuter design is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Metric validation, ablation, diagnosis, and large-scale re-evaluation are comprehensive, though the expert set size and retention strategy still involve selection bias.
- Writing Quality: ⭐⭐⭐⭐☆ Problems are clearly stated, and despite dense tables, the logic is sound; appendix information is critical for understanding costs and version management.
- Value: ⭐⭐⭐⭐⭐ High value for NL2SQL evaluation and leaderboard maintenance, and transferable to other tasks where references are not unique.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2026\] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)

</div>

<!-- RELATED:END -->
