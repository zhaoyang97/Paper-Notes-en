---
title: >-
  [Paper Note] STaR-SQL: Self-Taught Reasoner for Text-to-SQL
description: >-
  [ACL 2025][Code Intelligence][Text-to-SQL] This paper reformulates the Text-to-SQL task as a reasoning-driven process. By employing the STaR (Self-Taught Reasoner) bootstrapping approach, it enables LLMs to learn how to generate step-by-step rationales to assist in SQL generation. Integrated with an Outcome-supervised Reward Model (ORM) validator for best-of-N sampling, the framework achieves an 86.6% execution accuracy on the Spider benchmark.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Text-to-SQL"
  - "Chain-of-Thought"
  - "Self-Taught Reasoner"
  - "Test-time verification"
  - "Inference scaling"
date: 2026-05-08
content_hash: db4c311ccb620b54
---

# STaR-SQL: Self-Taught Reasoner for Text-to-SQL

**Conference**: ACL 2025  
**arXiv**: [2502.13550](https://arxiv.org/abs/2502.13550)  
**Code**: None  
**Area**: Text-to-SQL / NLP  
**Keywords**: Text-to-SQL, Chain-of-Thought, Self-Taught Reasoner, Test-time verification, Inference scaling

## TL;DR

This paper reformulates the Text-to-SQL task as a reasoning-driven process. By employing the STaR (Self-Taught Reasoner) bootstrapping approach, it enables LLMs to learn how to generate step-by-step rationales to assist in SQL generation. Integrated with an Outcome-supervised Reward Model (ORM) validator for best-of-N sampling, the framework achieves an 86.6% execution accuracy on the Spider benchmark.

## Background & Motivation

Existing Text-to-SQL methods primarily rely on the instruction-following capabilities of LLMs, generating SQL via meticulously designed prompts and schema selection optimizations. However, they suffer from several limitations:

1. **Limitations of Prompt Engineering**: Prompt templates are rigid, consume significant context tokens, and smaller models find it difficult to understand complex prompts.
2. **Failure on Complex Queries**: When facing hard and extra-hard-level queries, the performance of existing methods drops substantially, with even specialized code LLMs performing poorly.
3. **Neglect of Reasoning Core**: Prior works overemphasize prompt engineering while neglecting the inherent reasoning capabilities of LLMs.
4. **Lack of Transparency**: End-to-end SQL generation lacks interpretability, making it difficult for non-expert users to verify whether the generated SQL accurately captures their intent.

The core mechanism of this paper is to shift Text-to-SQL from an "instruction execution" process to a "reasoning process", allowing LLMs to comprehend query intents and progressively construct SQL through step-by-step rationales.

## Method

### Overall Architecture

STaR-SQL consists of three main steps: (1) Step-by-step rationale generation and self-improvement: generating reasoning steps via few-shot prompting, filtering correct rationales for SFT, and bootstrapping iteratively; (2) Validator training: training an ORM using both correct and incorrect rationale samples; (3) Test-time verification: scaling test-time compute using a best-of-N sampling strategy.

### Key Designs

1. **Self-Taught Reasoner Bootstrapping**: Using the pre-trained LLM as a generator, the model is guided by a few examples with chain-of-thought rationales to generate $k$ reasoning and SQL candidates for each question in the training set. Only rationales that yield correct execution results are preserved for SFT fine-tuning. A key design is the difficulty-based resampling strategy: for questions initially failed by the model, the golden SQL is provided as a hint to enable the model to generate the rationale chain backward. This addresses the tail narrowing problem and prevents the training set from biasing towards easy questions. Each iteration re-initializes from the original pre-trained model to prevent overfitting.

2. **Outcome-supervised Reward Model (ORM)**: A binary classifier validator is trained using both correct and incorrect rationale samples produced during the STaR iterations. Based on the LLM, a linear layer is appended to output a scalar value, which is trained using binary classification loss. The core idea is to utilize incorrect samples; while traditional methods discard failed rationales, the ORM utilizes correct/incorrect pairs to learn discrimination.

3. **Best-of-N Test-Time Compute Scaling**: During inference, the LLM is prompted to generate $N$ candidate rationale and SQL pairs, and the ORM scores them to select the highest-scoring candidate as the final output. This allows the model to scale performance purely by increasing test-time computational resources without modifying the model architecture.

### Loss & Training

- **Generator SFT Loss**: Standard negative log-likelihood loss $\mathcal{L}_{SFT} = -\mathbb{E} \sum \log \pi_\theta(t_i | t_{<i}, X)$
- **ORM Training Loss**: Binary cross-entropy $\mathcal{L}_{ORM} = A_T \log r_T + (1-A_T) \log(1-r_T)$
- **Base Model**: Llama-3.1-8B-Instruct
- **Training Data**: 7,000 tasks are selected from the Spider training set, with 8 solutions sampled per task. Training is conducted iteratively until a performance plateau is reached, re-initializing from the original pre-trained model in each iteration.

## Key Experimental Results

### Main Results

| Method | Model | EX (%) | EM (%) |
|------|------|--------|--------|
| Few-shot | Llama-3.1-8B | 55.0 | 34.2 |
| DIN-SQL | GPT-4 | 74.2 | 60.1 |
| DAIL-SQL | GPT-4 | 81.7 | 69.1 |
| ROUTE | Qwen2.5-7B | 83.6 | - |
| STaR-SQL | Llama-3.1-8B | 75.0 | 64.9 |
| **STaR-SQL ORM@16** | **Llama-3.1-8B** | **86.6** | **72.5** |

### Ablation Study

| Configuration | EX | EM | Description |
|------|-----|-----|------|
| STaR-SQL ORM@16 | 86.6 | 72.5 | Full method |
| w/o rationales | 68.6 | 57.9 | Remove rationale chains, -18.0% |
| w/o best-of-N | 75.0 | 64.9 | No sampling, -11.6% |
| Self-Consistency | 78.8 | 71.7 | Replace ORM with majority voting, -7.8% |

### Key Findings

- **Difficulty Level Analysis**: Achieves 69.3% execution accuracy on extra-hard queries, outperforming the runner-up by 5.8%; achieves 82.8% on hard queries, outperforming the runner-up by 9.1%.
- **Impact of Sample Budget**: Outperforms DAIL-SQL (GPT-4) with just 4 samples; outperforms ROUTE with 8 samples; peaks at 16 samples.
- Outperforms the few-shot baseline by 31.6% and direct fine-tuning for SQL prediction by 18.0%.
- Budgeting tokens for reasoning chains is more effective than elaborate prompt engineering—STaR-SQL outperforms DIN-SQL (which uses a 6k+ token prompt) by 41.4%.
- Combining an open-source 8B model with a reasoning-driven approach outperforms prompt engineering using closed-source GPT-4.

## Highlights & Insights

- **Paradigm Shift**: Shifts Text-to-SQL from an agent-like framework driven by prompt engineering to a reasoning-driven framework.
- **Inference Scaling**: Demonstrates the effectiveness of scaling test-time compute (rather than training compute or prompt length) in Text-to-SQL tasks.
- **Improved Transparency**: The step-by-step rationales make the entire SQL generation process interpretable, allowing users to verify alignment with their intents.
- **Data Efficiency**: Training data for the ORM is derived entirely from the STaR iteration process, requiring no additional human annotation.
- **Solution to Tail Narrowing**: The difficulty-based resampling strategy is simple yet highly effective.

## Limitations & Future Work

- Evaluated only on the Spider benchmark, lacking diverse cross-domain evaluations.
- Does not integrate schema encoding optimization techniques (such as Graph Neural Networks), which might further boost performance.
- The validator utilizes ORMs (outcome-level supervision); employing a stricter PRM (process-level supervision) validator might yield higher gains.
- Advanced search strategies such as Monte Carlo Tree Search (MCTS) to exploit test-time compute more efficiently were not explored.
- The impact of rationale chain lengths on queries of varying complexities remains uninvestigated.

## Related Work & Insights

- Drawing inspiration from the STaR (Zelikman et al., 2022) bootstrapping framework, this work is the first to apply it to structured output tasks.
- Offers a stark contrast to agent-like methods such as DIN-SQL: prioritizing reasoning capability over prompt engineering.
- Insight: Other structured output tasks (e.g., code generation, table understanding) could also benefit from this reasoning-driven paradigm.
- The simple yet effective combination of Best-of-N + ORM has broad applicability in NL2Code tasks.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First application of reasoning bootstrapping to Text-to-SQL, demonstrating a paradigm shift. |
| Practicality | 4 | The method is straightforward and effective; the 8B open-source model outperforms GPT-4. |
| Experimental Thoroughness | 4 | Includes difficulty analysis, sample budget analysis, ablation studies, and case studies. |
| Writing Quality | 4 | The structure is clear, and the comparative analysis is convincing. |
| Overall Score | 4 | An excellent piece of work applying inference scaling to structured tasks. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [\[ACL 2026\] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](../../ACL2026/code_intelligence/r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)
- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](../../ACL2026/code_intelligence/pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](../../ACL2026/code_intelligence/pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)

</div>

<!-- RELATED:END -->
