---
title: >-
  [Paper Note] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment
description: >-
  [ACL 2026][Code Intelligence][Code Generation] This paper proposes CodeRL+, which integrates execution semantics alignment into the RLVR training pipeline. By enabling models to infer variable-level execution trajectorie…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Generation"
  - "Execution Semantics Alignment"
  - "RLVR"
  - "GRPO"
  - "Program Execution Trajectory"
date: 2026-05-08
content_hash: bee287c27b98ac94
---

# CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment

**Conference**: ACL 2026  
**arXiv**: [2510.18471](https://arxiv.org/abs/2510.18471)  
**Code**: [https://github.com/jiangxxxue/CODERLPLUS](https://github.com/jiangxxxue/CODERLPLUS)  
**Area**: Code Generation / Reinforcement Learning  
**Keywords**: Code Generation, Execution Semantics Alignment, RLVR, GRPO, Program Execution Trajectory

## TL;DR

This paper proposes CodeRL+, which integrates execution semantics alignment into the RLVR training pipeline. By enabling models to infer variable-level execution trajectories, it bridges the gap between code textual representation and execution semantics. The approach achieves an average pass@1 gain of 4.6% in code generation, 15.5% in code reasoning, and 4.4% in test output generation.

## Background & Motivation

**Background**: LLMs learn textual patterns of code through autoregressive pre-training, achieving strong code generation capabilities. RLVR (Reinforcement Learning with Verifiable Rewards) utilizes test case execution to provide deterministic feedback, attempting to bridge the semantic gap between textual patterns and functional correctness.

**Limitations of Prior Work**: RLVR relies solely on binary pass/fail signals, which is insufficient to establish a robust alignment between code textual representations and execution semantics. Experiments indicate that models trained with standard RLVR only improve by 4% on execution trajectory inference tasks compared to baselines, failing to track basic execution semantics such as variable changes in loops.

**Key Challenge**: There is a fundamental misalignment between the pre-training objective of LLMs (fitting text distributions) and their evaluation criteria (execution correctness). Sparse rewards based only on final execution results prevent models from learning the runtime behavior of code.

**Goal**: To introduce execution semantics alignment into RLVR, enabling models to infer variable-level execution trajectories and providing direct learning signals for execution semantics.

**Key Insight**: Treat failed code exploration as training data for execution semantics alignment—forcing the model to learn and infer the final values of each variable in failed programs.

**Core Idea**: Code generation (synthesizing the state transition function $\Phi_p$) and execution semantics alignment (understanding $\Phi_p$) are complementary. Joint optimization can transcend the learning of superficial textual patterns.

## Method

### Overall Architecture

CodeRL+ introduces a dual-objective optimization within the GRPO training pipeline: (1) Code generation—generating code for programming problems and verifying via test cases; (2) Execution semantics alignment—inferring the final values of each variable in a program. Both objectives are trained jointly using a mixed prompt distribution $\mathcal{B}_{\text{mixed}} = \alpha \cdot \mathcal{B}_{\text{code}} + (1-\alpha) \cdot \mathcal{B}_{\text{align}}$.

### Key Designs

1.  **Execution Semantics Alignment Task**:
    - **Function**: Enables the model to learn and infer the runtime behavior of code.
    - **Mechanism**: Given a program $p$ and input $x$, the model must infer the value of each variable $var_i$ at its last assignment in the execution trajectory. This is more feasible than inferring complete trajectories while implicitly encoding control flow paths and data dependencies.
    - **Design Motivation**: Complete execution trajectories suffer from state explosion in loops; final variable values serve as a practical approximation.

2.  **Dynamic Data Construction based on Failed Exploration**:
    - **Function**: Dynamically constructs alignment data from the model's own failed code attempts.
    - **Mechanism**: During the rollout phase of code generation, failed programs are repurposed to build alignment prompts $q' = \langle p_{\text{fail}}, x, V \rangle$. The ground-truth execution semantics $\mathcal{F}_{p_{\text{fail}}}(x)$ obtained from executing these programs are used as labels. Initial iterations focus entirely on code generation, with alignment samples gradually introduced.
    - **Design Motivation**: No extra data sources are required. Alignment data evolves with the model's capability, and failed programs specifically expose the model's deficiencies in understanding execution semantics.

3.  **Fine-grained Variable-level Rewards**:
    - **Function**: Provides denser rewards for execution semantics alignment than binary signals.
    - **Mechanism**: The reward is defined as the proportion of correctly inferred variables $R_{\text{sem}}^{(i)} = \frac{1}{|V|}\sum_{v_k \in V} \mathbb{1}[\hat{v}_k^{\text{final}} = v_k^{\text{final},*}]$, allowing partially correct inferences to receive positive reinforcement.
    - **Design Motivation**: Compared to the all-or-nothing rewards in code generation, variable-level rewards provide a more continuous learning signal.

### Loss & Training

The joint optimization objective is $\mathcal{J}_{\text{CodeRL+}}(\theta) = \mathbb{E}[r(\theta) \cdot A_{\text{gen}}] + \mathbb{E}[r'(\theta) \cdot A_{\text{sem}}]$ using the GRPO framework. The training data ratio is $\alpha = 0.6$ (60% code generation, 40% semantic alignment). Based on Qwen2.5-Coder-7B-Instruct, with a batch size of 128, 8 rollout samplings, and 8×A100 GPUs.

## Key Experimental Results

### Main Results

**Pass@1 (%) on Qwen2.5-Coder-7B-Instruct**

| Method | HumanEval | LeetCode | LiveCodeBench | Avg | Code Reasoning | Test Output |
|------|-----------|----------|---------------|-----|---------|---------|
| Base | 88.4 | 50.6 | 34.3 | 57.8 | 60.8 | 48.8 |
| GRPO | 87.2 | 60.0 | 35.4 | 60.9 | 66.0 | 48.4 |
| OlympicCoder | 75.6 | 45.3 | 30.9 | 50.6 | 68.5 | 31.1 |
| CodeReasoner | 88.4 | 50.0 | 34.8 | 57.7 | 78.5 | 65.1 |
| **Ours** | **90.9** | **63.3** | **36.9** | **63.7** | **85.0** | 53.2 |

### Ablation Study

| Configuration | Code Gen Avg | Code Reasoning | Description |
|------|-------------|---------|------|
| GRPO (Base) | 60.9 | 66.0 | Code generation only |
| + Execution Semantics Alignment | **63.7** | **85.0** | Full CodeRL+ |
| Execution Semantics Alignment only | - | Gain | Separate alignment is also effective |
| Different RL Algorithms (REINFORCE++, DAPO) | Gain | Gain | Consistent across algorithms |

### Key Findings

- CodeRL+ achieves a 4.6% relative improvement in code generation over GRPO and a 15.5% improvement in code reasoning.
- CodeRL+ successfully bridges the performance gap between code generation and code reasoning—traditionally, methods focusing on reasoning often degrade generation and vice versa.
- Stable improvements are observed across different models (Qwen, DeepSeek, Llama) and RL algorithms (GRPO, REINFORCE++, DAPO).
- Probing experiments demonstrate that after CodeRL+ training, the model considers execution semantics more deeply while generating code.

## Highlights & Insights

- The repurposing of failed explorations is a key design highlight—no computational resources are wasted as failed code directly becomes training data for semantic alignment.
- Dual-objective joint optimization creates a virtuous cycle between "synthesizing $\Phi_p$" and "understanding $\Phi_p$".
- No additional data sources or teacher model distillation are needed; alignment data is entirely self-generated through the model's exploration.

## Limitations & Future Work

- Approximation of execution trajectories (inferring only final variable values) might lose critical information regarding intermediate states.
- Dependence on executable test cases for rewards makes it inapplicable to programming tasks that cannot be automatically verified (e.g., UI development).
- Only Python code generation was evaluated; generalization to other programming languages remains to be verified.

## Related Work & Insights

- **vs CODEI/O**: While CODEI/O learns execution through teacher distillation and SFT, CodeRL+ learns via RL self-exploration, offering better generalization.
- **vs CodeReasoner/CodeBoost**: These methods optimize only code reasoning, which can harm generation; CodeRL+ jointly optimizes both.
- **vs Standard GRPO**: Standard GRPO shows limited improvement in execution semantics (4%), whereas CodeRL+ significantly enhances it through explicit alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to integrate execution semantics alignment into RLVR using failed explorations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across five benchmarks, multiple models, RL algorithms, and probing analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous formalization, though symbol-heavy in parts.
- Value: ⭐⭐⭐⭐⭐ Provides a crucial execution semantics learning signal for code generation RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](../../ICLR2026/code_intelligence/execution-grounded_credit_assignment_for_grpo_in_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](../../NeurIPS2025/code_intelligence/embedding_alignment_in_code_generation_for_audio.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)

</div>

<!-- RELATED:END -->
