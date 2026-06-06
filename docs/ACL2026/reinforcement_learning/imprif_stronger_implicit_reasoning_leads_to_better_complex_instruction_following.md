---
title: >-
  [Paper Note] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following
description: >-
  [ACL 2026][Reinforcement Learning][Complex Instruction Following] ImpRIF formalizes the implicit reasoning structure within complex instructions as a verifiable Explicit Reasoning Graph (ERG). Based on this…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Complex Instruction Following"
  - "Implicit Reasoning"
  - "Reasoning Graph"
  - "Process Verification"
date: 2026-05-08
content_hash: 41aae980226b3d10
---

# ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following

**Conference**: ACL 2026  
**arXiv**: [2602.21228](https://arxiv.org/abs/2602.21228)  
**Code**: None  
**Area**: Instruction Following / LLM Reasoning  
**Keywords**: Complex Instruction Following, Implicit Reasoning, Reasoning Graph, Process Verification, Reinforcement Learning

## TL;DR

ImpRIF formalizes the implicit reasoning structure within complex instructions as a verifiable Explicit Reasoning Graph (ERG). Based on this, it constructs large-scale single/multi-turn data and trains models via SFT and process-verified RL. This approach enables 4B-32B models to significantly outperform base models across five instruction-following benchmarks, with the 32B model even surpassing some larger commercial models.

## Background & Motivation

**Background**: The instruction-following capability of LLMs is critical for complex applications. Current research primarily focuses on explicit, structured combinations of multiple constraints, enhancing performance through data engineering and template expansion.

**Limitations of Prior Work**: Real-world user instructions are not flat, single, or entirely explicit; they often contain multi-step reasoning, conditional statements, nested logic, and implicit premises. Existing methods do not systematically address instructions involving implicit reasoning and complex logical dependencies. Models tend to ignore key conditions or misunderstand implicit constraints when "reading between the lines" is required.

**Key Challenge**: Reliable instruction following fundamentally depends on a deep understanding of the instructions themselves, particularly the accurate modeling of implicit reasoning requirements and complex constraint structures. Prior work has not yet approached the problem from the perspective of implicit reasoning.

**Goal**: (1) Formalize the structure of implicit reasoning instructions; (2) Construct controllable, large-scale training data; (3) Train models to reason along reasoning graphs through SFT and RL.

**Key Insight**: Abstract the implicit reasoning structure as a Directed Acyclic Graph (DAG), where nodes represent programmable and verifiable atomic operations (conditional logic, mathematical calculation, or factual reasoning) and edges encode dependencies. During data generation, the graph logic is woven into natural language while intermediate reasoning is hidden, forming implicit constraint instructions.

**Core Idea**: Explicitly model the implicit reasoning structure (ERG) in instructions and utilize it throughout the entire pipeline—data synthesis (controllable generation), SFT (graph-guided CoT), and RL (process-verified rewards)—to enhance implicit reasoning capabilities.

## Method

### Overall Architecture

The ImpRIF pipeline consists of: (1) Building a constraint pool (verifiable atomic constraints categorized into logical, mathematical, and knowledge-based); (2) Generating ERGs and synthesizing implicit reasoning instructions (single/multi-turn); (3) SFT with ERG-guided Chain-of-Thought (CoT); (4) GRPO RL training using multi-granularity rewards based on process verification.

### Key Designs

1.  **Explicit Reasoning Graph (ERG) and Implicit Reasoning Instructions**:
    - **Function**: Formalizes the structure of implicit reasoning to support automated data generation and verification.
    - **Mechanism**: Three types of atomic nodes are defined: Condition nodes (boolean checks and branches), Math nodes (arithmetic and numerical comparisons), and Knowledge nodes (factual reasoning, concept disambiguation). Nodes form chains or DAGs, each equipped with executable verification code. During instruction generation, the graph logic is embedded into natural language, and multi-hop dependencies are hidden. Multi-turn data includes system-instruction dialogues and user-cumulative dialogues, some featuring adversarial final-turn queries (conflicts, injection attacks).
    - **Design Motivation**: Programmable verification ensures controllable data quality, the graph structure allows for adjustable complexity (via constraint count), and the DAG formalization provides a theoretical basis for CoT and reward design.

2.  **ERG CoT Guided SFT**:
    - **Function**: Teaches the model to reason according to the graph structure.
    - **Mechanism**: ERG nodes and dependency edges are unfolded into natural language CoT, traversing dependencies in a "parent-to-child" order to ensure each step builds on previous results. The process includes five steps: (a) describing reasoning for each node; (b) traversing dependencies from root to leaf; (c) unfolding derivations in dependency order; (d) checking coordination between multiple constraints; (e) generating the answer based on reasoning and self-checking. Samples with perfect scores and correct answers are selected for SFT.
    - **Design Motivation**: Explicitly mapping the ERG structure to thought processes allows the model to learn "graph-guided reasoning" during SFT.

3.  **Multi-granularity RL Rewards for Process Verification**:
    - **Function**: Optimizes both constraint satisfaction and reasoning process quality during RL training.
    - **Mechanism**: A three-layer reward system is employed: (a) Task Reward $R_{\text{task}}$: the ratio of satisfied constraints (verified via code for single-turn and LLM-scored rubrics for multi-turn); (b) Thinking Process Supervision $R_{\text{think}}$: an LLM judge compares the model's reasoning with the reference ERG CoT to evaluate logic and correctness; (c) Partial Order Reward $R_{\text{ref}}$: a strong model is introduced as a quality anchor, providing extra rewards only when the student surpasses the anchor. Total reward $R_{\text{total}} = R_{\text{task}} + R_{\text{ref}} + R_{\text{think}}$.
    - **Design Motivation**: Relying solely on final outcomes (constraint satisfaction) is insufficient; process supervision ensures the correctness of reasoning paths, while the partial order reward accelerates convergence.

### Loss & Training

The SFT stage utilizes standard language modeling loss. The RL stage employs Group Relative Policy Optimization (GRPO) combined with multi-granularity rewards. Training is conducted on Qwen3-4B/8B/32B models.

## Key Experimental Results

### Main Results

**Performance across Five Instruction Following Benchmarks (ImpRIF-8B_SFT+RL vs Qwen3-8B)**

| Benchmark | Qwen3-8B | ImpRIF-8B | Gain |
|-----------|----------|-----------|------|
| ImpRIF-Test ISR | 19.87 | **51.85** | +32.0 |
| SysBench ISR | 66.52 | **79.08** | +12.6 |
| MultiChallenge | 42.00 | **59.60** | +17.6 |
| MedMT ISR | 34.39 | **48.07** | +13.7 |
| ComplexBench ISR | 81.37 | **83.29** | +1.9 |

### Ablation Study

| Configuration | ImpRIF-Test CSR | Description |
|---------------|-----------------|-------------|
| ImpRIF-8B_SFT+RL | **78.33** | Full Method |
| ImpRIF-8B_SFT | 68.63 | SFT Only |
| ImpRIF-8B_RL | 66.33 | RL Only |
| Qwen3-8B (Base) | 55.64 | No Training |

### Key Findings

- The combination of SFT and RL significantly outperforms either used alone—SFT provides a strong initialization, while RL further reinforces reasoning capabilities.
- ImpRIF-32B_SFT+RL surpasses Qwen3-235B-A22B and Qwen2.5-72B on multiple benchmarks, achieving the performance of much larger models with only 32B parameters.
- Significant improvements are also observed in the 4B model (ImpRIF-Test ISR: 17.70→49.11, +31.4), demonstrating the effectiveness of the method for small models.
- Thinking process supervision rewards are crucial for improving reasoning quality; removing them leads to a noticeable drop in logical consistency scores.

## Highlights & Insights

- The formal design of the ERG is the cornerstone of this work—a unified graph structure serves data generation, CoT construction, and reward design, ensuring consistency across the entire pipeline.
- Redefining the "instruction following" problem as an "implicit reasoning" problem provides a fresh theoretical perspective.
- The combination of process-supervised RL and partial order rewards provides a valuable paradigm for RL training on complex tasks.

## Limitations & Future Work

- ERG construction relies on LLMs and manually designed constraint pools; scaling to new domains may require additional engineering.
- The use of an LLM judge for process supervision introduces evaluation noise.
- The method was only validated on the Qwen3 series; its generalizability across different model families is unknown.
- The definition of implicit reasoning is limited to logic, math, and knowledge, excluding more complex linguistic phenomena such as rhetoric or irony.

## Related Work & Insights

- **vs RwG/RAIF**: While RwG uses graphs to enhance reasoning and RAIF rewards the reasoning process, ImpRIF unifies both within the ERG framework.
- **vs Traditional Instruction Data Scaling**: Traditional methods focus on scaling explicit constraint combinations, whereas ImpRIF focuses on dependencies in implicit reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The ERG formalization and the shift to the "implicit reasoning → instruction following" perspective are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes five benchmarks, three model scales, complete ablation studies, and both single/multi-turn evaluation.
- Writing Quality: ⭐⭐⭐⭐ Detailed methodological description, though the paper is long and could be more concise.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic solution to the problem of complex instruction following.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/reinforcement_learning/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Financial Instruction Following Evaluation (FIFE)](../../NeurIPS2025/reinforcement_learning/financial_instruction_following_evaluation_fife.md)
- [\[ACL 2026\] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)
- [\[ACL 2026\] Adaptive Instruction Composition for Automated LLM Red-Teaming](adaptive_instruction_composition_for_automated_llm_red-teaming.md)

</div>

<!-- RELATED:END -->
