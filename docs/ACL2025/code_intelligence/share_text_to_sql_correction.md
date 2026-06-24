---
title: >-
  [Paper Note] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL
description: >-
  [ACL 2025][Code Intelligence] Proposed the SHARE framework, which uses three dedicated Small Language Models (SLMs) with <8B parameters to form a sequential pipeline. It translates declarative SQL into step-by-step action trajectories that expose the reasoning path, and then corrects schema linking errors and logical reasoning errors in stages, achieving self-correction for LLM Text-to-SQL at an extremely low cost.
tags:
  - "ACL 2025"
  - "Code Intelligence"
date: 2026-05-08
content_hash: 99d0d491939a4eb5
---

# SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL

**Conference**: ACL 2025  
**arXiv**: [2506.00391](https://arxiv.org/abs/2506.00391)  
**Code**: [GitHub](https://github.com/quge2023/SHARE)  
**Area**: Others

## TL;DR

Proposed the SHARE framework, which uses three dedicated Small Language Models (SLMs) with <8B parameters to form a sequential pipeline. It translates declarative SQL into step-by-step action trajectories that expose the reasoning path, and then corrects schema linking errors and logical reasoning errors in stages, achieving self-correction for LLM Text-to-SQL at an extremely low cost.

## Background & Motivation

1. **High Cost of LLM Self-Correction**: Existing self-correction methods rely on recursive calls to LLMs, requiring full inference in each round, which leads to multiplicative computational overhead; for instance, the Multiple-Prompt method costs up to \$62.86 per thousand queries.
2. **Unreliable and Restricted Execution Feedback**: Self-debugging relies on database execution feedback, but mainstream engines like SQLite return overly simplified information, making it difficult to pinpoint semantic errors precisely; additionally, in data privacy-sensitive scenarios, direct database access is strictly restricted.
3. **Black-box Dilemma of Declarative SQL**: When LLMs directly correct declarative SQL, they cannot exhibit the underlying reasoning process, leading to a self-enhancement bias — GPT-4o's native self-correction actually decreases BIRD accuracy from 55.87% to 55.28%.
4. **Heavy Reliance on Prompt Engineering**: Existing high-performance methods (such as MAGIC) rely on elaborately designed multi-stage prompts, which incur high labor costs and are difficult to generalize to other generator models or SQL dialects.

## Method

### Overall Architecture

SHARE adopts an assistant-based self-correction paradigm: after an arbitrary generator LLM generates an initial SQL, three LoRA-finetuned dedicated SLMs (all <8B parameters) form a sequential pipeline for correction. Finally, the corrected action trajectory is used as feedback to guide the LLM to regenerate the SQL.

### Key Designs

**1. Base Action Model (BAM) — Exposing Reasoning from Declarative to Procedural**

BAM decomposes declarative SQL queries into step-by-step action trajectories similar to pandas APIs (e.g., `where → groupby → orderby → select`), making the implicit reasoning paths explicit. The training data consists of 13K high-quality query-trajectory pairs distilled via GPT-4o, and reversibility verification (trajectories must be restorable back to the original SQL) is applied to filter out hallucinated samples. BAM also acts as a data factory for subsequent models, automatically synthesizing training data for SAM and LOM through a hierarchical self-evolution strategy, avoiding repeated calls to the teacher LLM.

**2. Schema Augmentation Model (SAM) — Two-stage Schema Linking Correction**

SAM focuses on detecting and correcting schema linking errors (mismatched table names, column names, etc.) in action trajectories. It adopts a two-stage training approach: Phase 1 learns to precisely mark all schema element positions in the trajectory using `[MASK]`; Phase 2 fills the masked positions with correct schema links based on the database schema, user questions, and the schema list of the initial SQL. This decoupled "locate-then-correct" design allows the model to optimize each subtask independently.

**3. Logic Optimization Model (LOM) — Logic Correction with Action Perturbation Augmentation**

LOM addresses logical reasoning errors (order of operations, conditional logic, aggregation methods, etc.). To expand the training data, an action perturbation strategy is proposed: three types of perturbations, ADD (inserting redundant actions), DELETE (deleting necessary actions), and SUBSTITUTE (replacing action types or parameters), are applied to correct trajectories to simulate real error patterns. Ultimately, 15K error-correct trajectory pairs are collected for training.

### Training & Inference

| Phase | Model | Data Source | Data Volume | Training Method |
|------|------|---------|--------|---------|
| Stage 1 | BAM | GPT-4o Distillation + Reversibility Verification | 13K | LoRA Fine-tuning |
| Stage 2 | SAM | BAM Hierarchical Self-Evolution + Schema Masking | 13K | LoRA Two-stage Fine-tuning |
| Stage 3 | LOM | BAM Self-Evolution + Action Perturbation Augmentation | 15K | LoRA Fine-tuning |

During inference, the three models execute sequentially: BAM generates action trajectories → SAM corrects schema → LOM corrects logic → the corrected trajectory is fed back to the LLM to regenerate SQL, achieving a single-round interaction overall.

## Key Experimental Results

### Main Results: GPT-4o Generator + Single-round Correction

| Method | External Feedback | BIRD EX(%) | SPIDER EX(%) | Cost per 1K Queries |
|------|---------|-----------|-------------|-----------|
| GPT-4o (baseline) | — | 55.87 | 77.10 | — |
| Self-Correction | ✗ | 55.28 ↓ | 75.90 | — |
| Self-Consistency | ✗ | 58.75 | 81.80 | — |
| Multiple-Prompt | ✗ | 58.80 | 81.50 | \$62.86 |
| Self-Debugging | ✓ | 58.28 | 81.20 | — |
| MAC-Refiner | ✓ | 58.74 | 80.40 | \$20.18 |
| MAGIC | ✗ | 59.53 | 85.66 | \$37.99 |
| +SHARE-3.8B | ✗ | 60.89 | 84.00 | — |
| **+SHARE-8B** | **✗** | **64.14** | **85.90** | **\$2.57** |

SHARE-8B achieves a 14.80% relative improvement on BIRD, with inference costs at only 1/10 of the most economical baseline.

### Cross-Model Generalization & Ablation Study

| Experimental Dimension | Setting | BIRD EX(%) | Gain |
|---------|------|-----------|------|
| **Cross-Model** | Claude-3.5-S → +SHARE-8B | 49.41 → 63.56 | +28.64% |
| | GPT-4o-mini → +SHARE-8B | 49.09 → 59.64 | +21.49% |
| | Llama-3.1-70B → +SHARE-8B | 53.91 → 61.93 | +14.88% |
| | DS-Coder-6.7B → +SHARE-8B | 34.57 → 51.24 | +48.22% |
| **Robustness** | DK Dataset +SHARE-8B | 64.10 → 75.30 | +11.20% |
| | Realistic Dataset +SHARE-8B | 73.40 → 81.50 | +8.10% |
| **Ablation** | w/o SAM (Schema Aug) | 60.02 | ↓ 4.08 |
| | w/o LOM (Logic Opt) | 56.98 | ↓ 7.16 |
| | w/o Hierarchical Self-Evolution | 60.55 | ↓ 3.59 |
| | w/o Action Perturbation | 61.38 | ↓ 2.76 |
| **Data Efficiency** | 50% Training Data | 60.71 | Outperforms MAGIC |
| **Open-source Teacher** | SHARE-llama (Llama-70B Teacher) | 65.19 | Outperforms SHARE-gpt |

### Key Findings

1. GPT-4o native self-correction leads to performance degradation (55.87→55.28), validating LLM's self-enhancement bias on declarative SQL.
2. SHARE achieves BIRD +14.80% and SPIDER +11.41% in a single round, with an inference cost of only \$2.57/1K queries.
3. Utilizing only 50% of the training data outperforms SOTA (MAGIC), demonstrating that the hierarchical self-evolution strategy substantially boosts data efficiency.
4. It generalizes effectively across different models (closed-source/open-source) and SQL dialects (MySQL/PostgreSQL), rather than status-fitting specific error patterns.
5. Using the open-source Llama-70B as a teacher model instead of GPT-4o, SHARE-llama yields performance comparable to or even better than SHARE-gpt.

## Highlights & Limitations

**Highlights**:

- The core innovation lies in the paradigm shift from "declarative to procedural", converting black-box SQL correction into white-box action trajectory debugging, which dramatically enhances error localization accuracy.
- The collaborative paradigm of SLM assisting LLM is exceptionally cost-effective, with inference costs at only 6.8% of MAGIC.
- The hierarchical self-evolution strategy decouples training data construction from reliance on teacher LLMs, reducing training phase costs to only 14.7% of MAGIC.

**Limitations**:

- Only single-round correction was validated; multi-round iterative correction scenarios remain unexplored.
- The predefined pandas-like action space might not cover all the complex features of various SQL dialects.
- The correction effect on mathematical reasoning errors (Mathematical Delusion) is limited (only ↓1.63%), constrained by the mathematical capabilities of the generator model.

## Rating

| Dimension | Score (1-10) | Description |
|------|:---------:|------|
| Novelty | 7 | The design of declarative-to-procedural translation + three-model sequential pipeline is innovative, but the action model and LoRA fine-tuning are mature techniques. |
| Effectiveness | 9 | Comprehensive improvements across four benchmarks, strong generalization across models/dialects, outperforming SOTA with only 50% data, and complete ablation studies. |
| Engineering Value | 8 | Extremely low inference cost (\$2.57/1K queries), plug-and-play to assist any LLM, and the open-source teacher model option further lowers the barrier to entry. |
| Reproducibility | 8 | The code is open-source, training details are sufficient, and hyperparameters and prompts are fully provided in the appendix. |
- Currently only evaluated on SQLite-related benchmarks, while effectiveness on other database systems remains unknown.

## Related Work & Insights

- Self-debugging: Iterative correction based on execution feedback (Zhong et al., 2023; Li & Xie, 2024)
- Self-correction: Autonomous correction without execution feedback (Liu & Tan, 2024; Askari et al., 2024)
- Action model: Decomposing tasks into procedural action trajectories (Zhang et al., 2024)
- MAGIC: Current SOTA self-correction method for text-to-SQL

## Rating

- **Novelty**: ★★★★☆ — The design of action trajectory transformation coupled with modular correction is highly creative.
- **Technical Depth**: ★★★★☆ — The three-stage pipeline and self-evolution training strategy are meticulously designed.
- **Experimental Thoroughness**: ★★★★★ — 4 benchmarks + multiple generators + resource-constrained analysis + cross-dialect testing.
- **Value**: ★★★★☆ — Low-cost auxiliary correction is highly valuable in privacy-restricted practical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] STaR-SQL: Self-Taught Reasoner for Text-to-SQL](star-sql_self-taught_reasoner_for_text-to-sql.md)
- [\[ACL 2026\] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](../../ACL2026/code_intelligence/r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](../../ACL2026/code_intelligence/pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](../../ACL2026/code_intelligence/pexa_parallel_exploration_agent_for_complex_text-to-sql.md)

</div>

<!-- RELATED:END -->
