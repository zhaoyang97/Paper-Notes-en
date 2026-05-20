---
title: >-
  [Paper Note] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications
description: >-
  [NeurIPS 2025][LLM Safety][SQL debugging] This paper proposes BIRD-CRITIC (the first SQL debugging benchmark) and the Six-Gym training environment, and develops the Bird-Fixer agent. Through the f-Plan Boosting strategy…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "SQL debugging"
  - "LLM Agent"
  - "database security"
  - "code repair"
  - "open-source models"
date: 2026-05-08
content_hash: 2ace9b9a321cc67d
---

# SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications

**Conference**: NeurIPS 2025
**arXiv**: [2506.18951](https://arxiv.org/abs/2506.18951)  
**Code**: [https://bird-critic.github.io/](https://bird-critic.github.io/)  
**Area**: AI Safety
**Keywords**: SQL debugging, LLM Agent, database security, code repair, open-source models

## TL;DR

This paper proposes BIRD-CRITIC (the first SQL debugging benchmark) and the Six-Gym training environment, and develops the Bird-Fixer agent. Through the f-Plan Boosting strategy, it elevates the SQL debugging capability of a 14B open-source model to surpass Claude-3.7-Sonnet and GPT-4.1, achieving efficient SQL issue resolution while preserving data privacy.

## Background & Motivation

Relational databases are the cornerstone of modern applications, and SQL is the standard language for database interaction. However, diagnosing complex SQL issues poses significant challenges for users at all experience levels. Communities such as StackOverflow are flooded with requests for SQL debugging assistance, making automation of this process highly valuable.

Current LLMs perform well on Text-to-SQL tasks, but **debugging and repairing existing erroneous SQL** is a substantially more complex problem. Unlike generating new SQL, debugging requires:
- Understanding the user's true intent within lengthy contexts
- Analyzing the underlying logic of queries
- Identifying subtle errors
- Deeply interacting with database schemas

Nevertheless, the capability of LLMs in SQL problem-solving has not been systematically studied, and no suitable evaluation benchmark exists.

Furthermore, open-source models are critical for database tasks—enterprises can deploy them locally to protect data privacy, rather than sending sensitive SQL queries to cloud-based large models.

## Method

### Overall Architecture

The paper's contributions consist of three components: (1) the BIRD-CRITIC benchmark—the first SQL debugging evaluation benchmark; (2) the Six-Gym training environment—automatically generating SQL debugging training data; and (3) the Bird-Fixer agent—training open-source models to serve as effective SQL debuggers.

### Key Designs

1. **BIRD-CRITIC Benchmark Construction**:

    - Curated from real user questions on StackOverflow, comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi, covering PostgreSQL, MySQL, SQL Server, and Oracle).
    - Each task undergoes rigorous reconstruction: extracting problem intent and error cause → mapping schemas to BIRD-SQL databases → reproduction and validation → annotating solution SQL and evaluation scripts → cross-validation and red-teaming.
    - Custom evaluation scripts replace simple execution accuracy (EX), as DML/DDL operations admit multiple functionally equivalent formulations.
    - **Design Motivation**: The text-to-SQL field lacks benchmarks tailored to debugging scenarios, and the standard EX metric produces a large number of false negatives in debugging contexts.

2. **Six-Gym Training Environment and SQL-Rewind Strategy**:

    - **Core Idea**: Inverting the debugging paradigm—starting from a correct SQL ($\sigma^*$), systematically introducing errors to generate issue SQL ($\sigma_{\text{issue}}$).
    - Pipeline: mining candidate SQL from StackOverflow → adapting to training databases using Gemini-2.0-Flash → execution validation → automatically generating issue SQL, evaluation scripts, and user problem descriptions.
    - Each step incorporates 3-round iterative refinement to reduce hallucinations.
    - Approximately 3,301 high-quality synthetic instances are generated in total.
    - **Design Motivation**: Manually collecting SQL debugging datasets is labor-intensive and difficult to scale; SQL-Rewind enables large-scale data generation without human annotation.

3. **SQL-Act Agent Scaffold**:

    - Based on the ReAct paradigm but with a key modification: arbitrary SQL commands are used directly as actions (rather than a predefined tool set), greatly expanding the action space.
    - Each step outputs a tuple $(t_i, \sigma_i, o_i)$: thought, SQL statement, and execution result.
    - **Design Motivation**: Compared to Tool-Act (predefined tool sets), SQL-Act is more flexible and can handle a wider variety of debugging scenarios.

4. **f-Plan Boosting Trajectory Augmentation**:

    - **Problem**: Under the standard approach, Gemini-2.0-Flash generates only 1,254 successful trajectories (38% utilization).
    - Two-stage pipeline:
        - **Backward Reasoning**: Given the issue SQL and the correct answer, the teacher model generates a step-by-step debugging plan $F = (f_1, \dots, f_k)$.
        - **Forward Validation**: The teacher model, using only the problem context and plan $F$, re-executes debugging via SQL-Act; only trajectories passing all test cases are accepted.
    - This yields 2,178 successful trajectories, a **73.7%** increase over the vanilla approach.
    - Open-source models are fine-tuned using LoRA.

5. **Generative Thought Mode (GTM)**:

    - Decouples thought prediction from SQL generation.
    - A fine-tuned model $M_O$ generates thoughts $t_i$; a base model $M_B$ generates SQL conditioned on the thought: $\sigma_i = M_B(H_{i-1}, t_i)$.
    - This preserves the debugging logic of $M_O$ while leveraging $M_B$'s broad SQL dialect knowledge, avoiding overfitting to SQL patterns seen during training.
    - **Design Motivation**: Analogous to the context–target decoupling in Word2Vec's Skip-gram model.

### Loss & Training

- Open-source models are supervised fine-tuned with LoRA on successful trajectories from Six-Gym.
- GTM inference proceeds in two stages: the fine-tuned model first generates thoughts, then the base model generates SQL.

## Key Experimental Results

### Main Results

| Model | BIRD-CRITIC-PG SR (%) | BIRD-CRITIC-Multi SR (%) |
|---|---|---|
| Meta-Llama-3.1-8B | 16.98 | 12.81 |
| GPT-4.1 | 37.36 | 29.12 |
| Claude-3.7-Sonnet | 32.08 | 27.89 |
| O3-Mini (strongest reasoning) | **38.87** | 33.33 |
| Bird-Fixer (Qwen-14B) | **38.11** | **29.65** |

Bird-Fixer, based on a 14B parameter model, achieves performance comparable to O3-Mini and surpasses Claude-3.7-Sonnet and GPT-4.1.

### Bird-Fixer Improvement

| Base Model | Base SR | Bird-Fixer SR | Gain |
|---|---|---|---|
| Llama-3.1-8B | 16.98 | 24.34 | +43.34% |
| Qwen-2.5-Coder-7B | 23.40 | 31.32 | +33.84% |
| Qwen-2.5-Coder-14B | 31.32 | 38.11 | +21.68% |
| Phi-4 | 30.19 | 38.11 | +26.23% |

### Ablation Study

| Configuration | BIRD-CRITIC-PG SR (%) | Notes |
|---|---|---|
| Full Bird-Fixer | **38.11** | All components |
| w/o GTM | 33.33 | Notable degradation without decoupling |
| w/o f-Plan | 32.45 | Trained on vanilla trajectories only |

### Key Findings

- **Advantage of reasoning models**: Reasoning models outperform general-purpose models by an average of 6.13% on PG and 8.03% on multi-dialect tasks.
- **Task type variation**: Query-type problems are the most difficult (highest token diversity, with a −0.89 correlation with performance); management-type problems are the easiest.
- **Dialect variation**: Different models exhibit substantial performance differences across SQL dialects.
- **Error analysis**: Logical errors account for the largest share at 44.5%, followed by chained errors at 27.3% and projection mismatches at 26.9%.

## Highlights & Insights

- **The first SQL debugging benchmark**, filling a critical gap in LLM evaluation for database debugging.
- The SQL-Rewind "reverse engineering" approach is elegant—introducing errors from correct answers to automatically generate training data.
- f-Plan Boosting achieves a 73.7% increase in trajectories at negligible additional cost (comparable time to the baseline).
- The GTM "thought–execution decoupling" design ensures cross-dialect generalization—Bird-Fixer is trained solely on PostgreSQL yet generalizes to MySQL, SQL Server, and Oracle.

## Limitations & Future Work

- Even the strongest models resolve only approximately 39% of problems, indicating that SQL debugging remains an extremely challenging task.
- Query-type problems (the most common and important category) achieve the lowest success rates, requiring stronger logical reasoning capabilities.
- Synthetic data from Six-Gym may not fully cover the diversity of real-world SQL issues.
- The effects of larger-scale models and additional training data remain to be explored.

## Related Work & Insights

- The evaluation methodology is analogous to SWE-Bench (code repair), but focuses on SQL as a specialized and important domain.
- The f-Plan approach resembles Chain-of-Thought but is more structured—employing functional plans rather than free-form reasoning chains.
- The framework has direct practical value for enterprise scenarios sensitive to data privacy.

## Rating

- Novelty: ⭐⭐⭐⭐ First SQL debugging benchmark and training framework, filling an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model, multi-dialect, extensive ablations, and detailed error analysis
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though the appendix is substantial and requires repeated reference
- Value: ⭐⭐⭐⭐⭐ Extremely high practical value; both the benchmark and code are open-sourced

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](../../ACL2026/llm_safety/two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)

</div>

<!-- RELATED:END -->
