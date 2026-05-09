---
title: >-
  [Paper Note] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment
description: >-
  [ACL 2026][Code Generation] This paper proposes CodeRL+, which integrates execution semantics alignment into the RLVR training pipeline. By training the model to infer variable-level execution traces, CodeRL+ bridges the gap between code text representations and execution semantics, achieving an average pass@1 improvement of 4.6% on code generation, 15.5% on code reasoning, and 4.4% on test output generation benchmarks.
tags:
  - ACL 2026
  - Code Generation
  - Execution Semantics Alignment
  - RLVR
  - GRPO
  - Program Execution Traces
date: 2026-05-08
content_hash: 7ee270099f050e69
---

# CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment

**Conference**: ACL 2026
**arXiv**: [2510.18471](https://arxiv.org/abs/2510.18471)
**Code**: [https://github.com/jiangxxxue/CODERLPLUS](https://github.com/jiangxxxue/CODERLPLUS)
**Area**: Code Generation / Reinforcement Learning
**Keywords**: Code Generation, Execution Semantics Alignment, RLVR, GRPO, Program Execution Traces

## TL;DR

This paper proposes CodeRL+, which integrates execution semantics alignment into the RLVR training pipeline. By training the model to infer variable-level execution traces, CodeRL+ bridges the gap between code text representations and execution semantics, achieving an average pass@1 improvement of 4.6% on code generation, 15.5% on code reasoning, and 4.4% on test output generation benchmarks.

## Background & Motivation

**State of the Field**: LLMs learn textual patterns of code through autoregressive pretraining, demonstrating strong code generation capabilities. RLVR (Reinforcement Learning with Verifiable Rewards) leverages deterministic feedback from test case execution to bridge the gap between textual patterns and functional correctness.

**Limitations of Prior Work**: RLVR relies solely on binary pass/fail signals, which are insufficient for establishing strong alignment between code text representations and execution semantics. Experiments show that models trained with RLVR improve by only 4% over the baseline on execution trace inference tasks, failing to track basic execution semantics such as variable changes within loops.

**Root Cause**: There exists a fundamental misalignment between the pretraining objective of LLMs (fitting text distributions) and the evaluation criterion (execution correctness). Sparse rewards derived exclusively from final execution outcomes cannot enable models to learn to understand runtime behavior.

**Paper Goals**: Introduce execution semantics alignment into RLVR so that the model can infer variable-level execution traces, providing a direct learning signal for execution semantics.

**Starting Point**: Repurpose failed code explorations as training data for execution semantics alignment—training the model to infer the final values of each variable in failing programs.

**Core Idea**: Code generation (synthesizing the state transition function $\Phi_p$) and execution semantics alignment (understanding $\Phi_p$) are complementary and bidirectional; joint optimization can surpass the learning of superficial textual patterns.

## Method

### Overall Architecture

CodeRL+ introduces a dual-objective optimization into the GRPO training pipeline: (1) code generation—producing code that solves programming problems and verifying it against test cases; and (2) execution semantics alignment—inferring the final value of each variable in a program. Both objectives are jointly trained via a mixed prompt distribution $\mathcal{B}_{\text{mixed}} = \alpha \cdot \mathcal{B}_{\text{code}} + (1-\alpha) \cdot \mathcal{B}_{\text{align}}$.

### Key Designs

1. **Execution Semantics Alignment Task**:

    - Function: Trains the model to learn the runtime behavior of code.
    - Mechanism: Given a program $p$ and input $x$, the model must infer the value of each variable $var_i$ at its last assignment in the execution trace. This is more tractable than inferring the full execution trace while implicitly encoding control flow paths and data dependencies.
    - Design Motivation: Full execution traces suffer from state explosion in scenarios such as loops; final variable values serve as a feasible approximation.

2. **Dynamic Data Construction from Failed Explorations**:

    - Function: Dynamically constructs execution semantics alignment data from the model's own failed code.
    - Mechanism: During the rollout phase of code generation, failing programs are repurposed to construct alignment prompts $q' = \langle p_{\text{fail}}, x, V \rangle$, using the ground-truth execution semantics $\mathcal{F}_{p_{\text{fail}}}(x)$ obtained by executing the failing programs as labels. The initial iterations consist entirely of code generation tasks, with alignment samples gradually introduced in subsequent iterations.
    - Design Motivation: No additional data sources are required; alignment data co-evolves with model capability, and failing programs directly expose the model's deficiencies in understanding execution semantics.

3. **Fine-Grained Variable-Level Reward**:

    - Function: Provides a more granular reward signal for execution semantics alignment than binary signals.
    - Mechanism: The reward is defined as the proportion of variables correctly inferred by the model: $R_{\text{sem}}^{(i)} = \frac{1}{|V|}\sum_{v_k \in V} \mathbb{1}[\hat{v}_k^{\text{final}} = v_k^{\text{final},*}]$, allowing partially correct inferences to receive positive rewards.
    - Design Motivation: Compared to the all-or-nothing reward of code generation, variable-level rewards provide a denser learning signal.

### Loss & Training

The joint optimization objective is $\mathcal{J}_{\text{CodeRL+}}(\theta) = \mathbb{E}[r(\theta) \cdot A_{\text{gen}}] + \mathbb{E}[r'(\theta) \cdot A_{\text{sem}}]$, trained under the GRPO framework. The training data ratio is $\alpha = 0.6$ (60% code generation, 40% semantics alignment). The model is based on Qwen2.5-Coder-7B-Instruct, trained with batch size 128, 8 rollout samples, on 8×A100 GPUs.

## Key Experimental Results

### Main Results

**pass@1 (%) on Qwen2.5-Coder-7B-Instruct**

| Method | HumanEval | LeetCode | LiveCodeBench | Avg | Code Reasoning | Test Output |
|--------|-----------|----------|---------------|-----|----------------|-------------|
| Base | 88.4 | 50.6 | 34.3 | 57.8 | 60.8 | 48.8 |
| GRPO | 87.2 | 60.0 | 35.4 | 60.9 | 66.0 | 48.4 |
| OlympicCoder | 75.6 | 45.3 | 30.9 | 50.6 | 68.5 | 31.1 |
| CodeReasoner | 88.4 | 50.0 | 34.8 | 57.7 | 78.5 | 65.1 |
| **CodeRL+** | **90.9** | **63.3** | **36.9** | **63.7** | **85.0** | 53.2 |

### Ablation Study

| Configuration | Code Gen Avg | Code Reasoning | Notes |
|---------------|-------------|----------------|-------|
| GRPO (baseline) | 60.9 | 66.0 | Code generation only |
| + Execution semantics alignment | **63.7** | **85.0** | Full CodeRL+ |
| Semantics alignment only | — | Improved | Alignment alone is effective |
| Different RL algorithms (REINFORCE++, DAPO) | Improved | Improved | Consistent across algorithms |

### Key Findings

- CodeRL+ achieves an average improvement of 4.6% over GRPO on code generation and 15.5% on code reasoning.
- CodeRL+ successfully bridges the performance gap between code generation and code reasoning—methods previously focused on code reasoning often degraded code generation performance, and vice versa.
- Consistent improvements are observed across multiple models (Qwen, DeepSeek, Llama) and RL algorithms (GRPO, REINFORCE++, DAPO).
- Probing experiments confirm that after CodeRL+ training, the model attends more to execution semantics when generating code.

## Highlights & Insights

- The repurposing of failed explorations is a key design highlight—no computation is wasted, as failing code directly becomes training data for semantics alignment.
- The dual-objective joint optimization establishes a virtuous cycle between "synthesizing $\Phi_p$" and "understanding $\Phi_p$."
- No additional data sources or teacher model distillation are required; alignment data is derived entirely from the model's own exploration.

## Limitations & Future Work

- The execution trace approximation (inferring only final variable values) may discard critical information from intermediate states.
- The approach relies on executable test cases for rewards and is not applicable to programming tasks that cannot be automatically verified (e.g., UI development).
- Evaluation is limited to Python code generation; generalization to other programming languages remains to be verified.

## Related Work & Insights

- **vs. CODEI/O**: CODEI/O learns execution via teacher distillation and SFT, whereas CodeRL+ learns through RL-based self-exploration, yielding better generalization.
- **vs. CodeReasoner/CodeBoost**: These methods optimize only for code reasoning and may harm code generation; CodeRL+ jointly optimizes both.
- **vs. Standard GRPO**: Standard GRPO yields limited improvement in execution semantics understanding (4%); CodeRL+ achieves substantial gains through explicit alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to integrate execution semantics alignment into RLVR, leveraging failed explorations to construct training data.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, multiple models, multiple RL algorithms, and probing analyses—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous formalization, though some notation is heavy.
- Value: ⭐⭐⭐⭐⭐ Provides an important execution semantics learning signal for RL-based code generation training.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](../../NeurIPS2025/code_intelligence/embedding_alignment_in_code_generation_for_audio.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)

<!-- RELATED:END -->
