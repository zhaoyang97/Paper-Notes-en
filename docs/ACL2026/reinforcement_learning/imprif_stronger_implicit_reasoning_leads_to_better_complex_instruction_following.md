---
title: >-
  [Paper Note] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following
description: >-
  [ACL 2026][Reinforcement Learning][Complex instruction following] ImpRIF formalizes the implicit reasoning structure in complex instructions as a verifiable Explicit Reasoning Graph (ERG)…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Complex instruction following"
  - "implicit reasoning"
  - "reasoning graph"
  - "process verification"
date: 2026-05-08
content_hash: 9ff0767d0e05d629
---

# ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following

**Conference**: ACL 2026
**arXiv**: [2602.21228](https://arxiv.org/abs/2602.21228)  
**Code**: None  
**Area**: Instruction Following / LLM Reasoning
**Keywords**: Complex instruction following, implicit reasoning, reasoning graph, process verification, reinforcement learning

## TL;DR

ImpRIF formalizes the implicit reasoning structure in complex instructions as a verifiable Explicit Reasoning Graph (ERG), constructs large-scale single-turn/multi-turn training data accordingly, and trains models via SFT combined with process-verified RL. Models ranging from 4B to 32B parameters significantly outperform their base counterparts across five instruction-following benchmarks, with the 32B model surpassing several large commercial models.

## Background & Motivation

**Background**: Instruction-following capability is critical for LLMs in complex applications. Existing research primarily focuses on explicit, structured multi-constraint compositional instructions, improving compliance through data engineering and template expansion.

**Limitations of Prior Work**: Real-world user instructions are not flat, atomic, or fully explicit—they frequently involve multi-step reasoning, conditional statements, nested logic, and implicit premises. Existing methods do not systematically address instructions involving implicit reasoning and complex logical dependencies. Models tend to overlook critical conditions or misinterpret implicit constraints when encountering requirements that demand inferring unstated implications.

**Key Challenge**: Reliable instruction following fundamentally depends on deep comprehension of the instruction itself, particularly accurate modeling of implicit reasoning requirements and complex constraint structures—yet no prior work has approached the problem from the perspective of implicit reasoning.

**Goal**: (1) Formalize the structure of implicit reasoning instructions; (2) construct controllable large-scale training data; (3) train models via SFT and RL to reason along the reasoning graph.

**Key Insight**: Abstract implicit reasoning structures as directed acyclic graphs (DAGs), where nodes represent programmatically verifiable atomic operations (conditional judgment / mathematical computation / knowledge inference) and edges encode dependency relations. During data generation, graph logic is woven into natural language while intermediate reasoning steps are concealed, yielding implicit-constraint instructions.

**Core Idea**: Explicitly model the implicit reasoning structure in instructions via ERG, and apply it uniformly across data synthesis (controllable generation), SFT (graph-guided CoT), and RL (process verification reward), thereby enhancing implicit reasoning capability across the entire training pipeline.

## Method

### Overall Architecture

The ImpRIF pipeline consists of: (1) constructing a constraint pool (three categories of programmatically verifiable atomic constraints: conditional, mathematical, and knowledge-based) → (2) generating ERGs and synthesizing implicit reasoning instructions (single-turn/multi-turn) → (3) SFT with ERG CoT → (4) GRPO RL training with multi-granularity process-verified rewards.

### Key Designs

1. **Explicit Reasoning Graph (ERG) and Implicit Reasoning Instructions**:

    - **Function**: Formalize the structure of implicit reasoning to support automated data generation and verification.
    - **Mechanism**: Three categories of atomic nodes — conditional nodes (Boolean checks and branching), mathematical nodes (arithmetic and numerical comparison), and knowledge nodes (factual reasoning, concept disambiguation). Nodes are composed into chains or DAGs, each equipped with executable verification code. During instruction generation, graph logic is woven into natural language while multi-hop reasoning dependencies are concealed. Multi-turn data supports two dialogue types — system-instruction-based and user-cumulative — with some instances including adversarial final-turn queries (conflicts, injection attacks).
    - **Design Motivation**: Programmatic verification enables controllable data quality; graph structure allows adjustable complexity (via constraint count); DAG formalism provides a principled basis for CoT construction and reward design.

2. **ERG CoT-Guided SFT**:

    - **Function**: Train the model to reason following the graph structure.
    - **Mechanism**: ERG nodes and dependency edges are unrolled into natural-language CoT by traversing dependencies in parent-to-child order, ensuring each step builds on prior results. The process involves five steps: (a) describe the reasoning at each node; (b) traverse dependencies from root to leaf; (c) unfold derivations in dependency order; (d) check coordination among multiple constraints; (e) generate the answer based on the reasoning and perform self-verification. Only samples with full scores and correct answers are selected for SFT.
    - **Design Motivation**: Explicitly mapping ERG structure to the reasoning process allows the model to learn graph-guided reasoning during SFT.

3. **Multi-Granularity RL Reward with Process Verification**:

    - **Function**: Jointly optimize constraint satisfaction and reasoning process quality during RL training.
    - **Mechanism**: Three-level rewards — (a) task reward $R_{\text{task}}$: proportion of satisfied constraints (programmatic verification for single-turn; additionally rubric-based LLM scoring for multi-turn); (b) thinking process supervision $R_{\text{think}}$: an LLM judge compares model reasoning against reference ERG CoT and evaluates logical coherence and correctness; (c) partial-order reward $R_{\text{ref}}$: a stronger model serves as a quality anchor, granting additional reward only when the student surpasses the anchor. Total reward: $R_{\text{total}} = R_{\text{task}} + R_{\text{ref}} + R_{\text{think}}$.
    - **Design Motivation**: Evaluating only final outcomes (constraint satisfaction) is insufficient; process supervision ensures the correctness of reasoning paths. The partial-order reward accelerates convergence.

### Loss & Training

The SFT stage uses standard language modeling loss. The RL stage employs GRPO with multi-granularity rewards. Training is conducted on Qwen3-4B/8B/32B.

## Key Experimental Results

### Main Results

**Five instruction-following benchmarks (ImpRIF-8B_SFT+RL vs. Qwen3-8B)**

| Benchmark | Qwen3-8B | ImpRIF-8B | Gain |
|-----------|----------|-----------|------|
| ImpRIF-Test ISR | 19.87 | **51.85** | +32.0 |
| SysBench ISR | 66.52 | **79.08** | +12.6 |
| MultiChallenge | 42.00 | **59.60** | +17.6 |
| MedMT ISR | 34.39 | **48.07** | +13.7 |
| ComplexBench ISR | 81.37 | **83.29** | +1.9 |

### Ablation Study

| Configuration | ImpRIF-Test CSR | Notes |
|---------------|----------------|-------|
| ImpRIF-8B_SFT+RL | **78.33** | Full method |
| ImpRIF-8B_SFT | 68.63 | SFT only |
| ImpRIF-8B_RL | 66.33 | RL only |
| Qwen3-8B (base) | 55.64 | No training |

### Key Findings

- The SFT+RL combination substantially outperforms either component alone—SFT provides a strong initialization while RL further reinforces reasoning capability.
- ImpRIF-32B_SFT+RL surpasses Qwen3-235B-A22B and Qwen2.5-72B on multiple benchmarks, achieving performance comparable to much larger models with only 32B parameters.
- The 4B model also achieves significant gains (ImpRIF-Test ISR: 17.70→49.11, +31.4), demonstrating the method's effectiveness at smaller scales.
- The thinking-process supervision reward is critical for reasoning quality; removing it leads to a marked drop in logical coherence scores.

## Highlights & Insights

- The formal design of ERG is the cornerstone of the entire paper—a unified graph structure simultaneously serves data generation, CoT construction, and reward design, achieving end-to-end consistency.
- Reframing "instruction following" as an "implicit reasoning" problem provides a novel theoretical perspective.
- The combination of process-supervised RL and partial-order rewards offers a replicable paradigm for RL training on complex tasks.

## Limitations & Future Work

- ERG construction relies on LLM-assisted generation and a manually designed constraint pool; extension to new domains may require additional engineering effort.
- Thinking-process supervision uses an LLM judge, introducing evaluation noise.
- Experiments are conducted solely on the Qwen3 family; generalizability across model families remains unknown.
- The definition of implicit reasoning is limited to three categories (conditional, mathematical, knowledge-based) and does not cover more complex linguistic phenomena such as rhetoric or irony.

## Related Work & Insights

- **vs. RwG/RAIF**: RwG enhances reasoning with graphs; RAIF rewards the reasoning process. ImpRIF unifies both within the ERG framework.
- **vs. traditional instruction-following data scaling**: Conventional approaches expand explicit constraint combinations, whereas ImpRIF focuses on implicit reasoning dependencies.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ ERG formalization and the perspective shift from "instruction following" to "implicit reasoning" are highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, three model scales, comprehensive ablations, single-turn and multi-turn settings.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are detailed, though the paper is lengthy and could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Provides a systematic solution to the problem of complex instruction following.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/reinforcement_learning/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[ACL 2026\] LENS: Less Noise, More Voice — Reinforcement Learning for Reasoning via Instruction Purification](less_noise_more_voice_reinforcement_learning_for_reasoning_via_instruction_purif.md)
- [\[NeurIPS 2025\] Financial Instruction Following Evaluation (FIFE)](../../NeurIPS2025/reinforcement_learning/financial_instruction_following_evaluation_fife.md)
- [\[ACL 2026\] Adaptive Instruction Composition for Automated LLM Red-Teaming](adaptive_instruction_composition_for_automated_llm_red-teaming.md)

</div>

<!-- RELATED:END -->
