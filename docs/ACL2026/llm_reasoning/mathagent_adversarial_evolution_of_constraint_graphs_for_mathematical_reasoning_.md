---
title: >-
  [Paper Note] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis
description: >-
  [ACL 2026][LLM Reasoning][Mathematical Reasoning] This paper proposes MathAgent, a hierarchical data synthesis framework based on adversarial evolution of constraint graphs. It reformulates data synthesis from a text generation task into an unsupervised optimization problem over constraint graphs. A three-agent Legislator system (Proposer-Critic-Moderator) evolves problem skeletons, which are then instantiated into natural language by an Executor. With only 1K synthetic samples, MathAgent surpasses LIMO and s1K across eight mathematical benchmarks.
tags:
  - ACL 2026
  - LLM Reasoning
  - Mathematical Reasoning
  - Data Synthesis
  - Constraint Graph
  - Adversarial Evolution
  - Legislator-Executor
date: 2026-05-08
content_hash: 7ae24fbeb1e7eca4
---

# MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis

**Conference**: ACL 2026
**arXiv**: [2604.11188](https://arxiv.org/abs/2604.11188)
**Code**: None
**Area**: Data Synthesis / LLM Reasoning
**Keywords**: Mathematical Reasoning, Data Synthesis, Constraint Graph, Adversarial Evolution, Legislator-Executor

## TL;DR
This paper proposes MathAgent, a hierarchical data synthesis framework based on adversarial evolution of constraint graphs. It reformulates data synthesis from a text generation task into an unsupervised optimization problem over constraint graphs. A three-agent Legislator system (Proposer-Critic-Moderator) evolves problem skeletons, which are then instantiated into natural language by an Executor. With only 1K synthetic samples, MathAgent surpasses LIMO and s1K across eight mathematical benchmarks.

## Background & Motivation

**State of the Field**: High-quality mathematical reasoning data is a key driver for improving LLM reasoning capabilities. As the scalability of human-annotated data increasingly becomes a bottleneck, synthetic data generation has emerged as a mainstream research direction.

**Limitations of Prior Work**: (1) Seed-expansion methods (e.g., Self-Instruct, WizardMath) are constrained by the "semantic radius" of the initial seeds, imposing an upper bound on diversity; (2) Zero-shot methods (e.g., Magpie) directly probe the model distribution without structural guidance, making them susceptible to mode collapse and logical hallucination; (3) Existing methods treat data synthesis as a direct text generation task, causing models to imitate surface-level narratives rather than acquire core reasoning capabilities.

**Root Cause**: Performing data synthesis directly in the token space fails to effectively control the logical complexity and structural diversity of problems. High-difficulty, high-quality long-tail samples are precisely what is needed to develop complex reasoning abilities, yet standard methods struggle to discover them.

**Paper Goals**: Design a synthesis framework that requires no human seed data and can automatically explore the structural space, generating mathematical reasoning data with both high complexity and high diversity.

**Starting Point**: Decouple data synthesis into two stages—structural evolution (meta-level) and semantic instantiation (base-level)—first optimizing the logical skeleton of a problem (the constraint graph), then converting the skeleton into a natural language question.

**Core Idea**: Represent the logical structure of mathematical problems using constraint graphs. A three-agent adversarial evolution mechanism (Proposer-Critic-Moderator) continuously optimizes graph complexity and diversity, after which an Executor generates natural language problems and reasoning chains.

## Method

### Overall Architecture
MathAgent consists of two decoupled stages: (1) **Meta-Level Structural Evolution**: the three-agent Legislator system adversarially evolves constraint graphs to produce an optimized problem skeleton $\mathcal{G}^*$; (2) **Base-Level Semantic Instantiation**: the Executor converts $\mathcal{G}^*$ and style token $\mathcal{S}$ into a natural language question $Q$ and reasoning chain $A$. Qualified samples are then selected through external model verification.

### Key Designs

1. **Constraint Graph Representation**:

    - **Function**: Formally describes the logical skeleton of a mathematical problem.
    - **Mechanism**: Models a problem as a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ augmented with style tokens $\mathcal{S}$. Nodes $\mathcal{V}$ represent mathematical concepts, edges $\mathcal{E}$ represent logical relations, and $\mathcal{S}$ controls global attributes (problem category, difficulty level, etc.). The optimization objective is $\mathcal{G}^* = \arg\max_{\mathcal{G}} \mathcal{H}(\mathcal{G})$, where $\mathcal{H}$ estimates complexity, and the constraint $\mathbb{I}_{\text{valid}}(\mathcal{G}|\mathcal{S})=1$ ensures solvability.
    - **Design Motivation**: Decoupling structural specification from textual realization allows the framework to focus on constructing complex and diverse logical structures, rather than being constrained by surface-level linguistic patterns.

2. **Legislator Three-Agent Evolution System**:

    - **Function**: Iteratively optimizes the constraint graph structure through adversarial dynamics.
    - **Mechanism**: Three agents collaborate—the Proposer ($\mathcal{A}_P$) refines $\mathcal{G}_t \to \mathcal{G}_{t+1}$ based on prior feedback, resolving logical contradictions and expanding structural depth; the Critic ($\mathcal{A}_C$) reviews the new graph along three dimensions (internal consistency, specification alignment, and optimization potential) and produces an improvement report; the Moderator ($\mathcal{A}_M$) acts as a strategic decision-maker, either terminating evolution upon convergence or directing the Proposer to implement improvements. The initialization phase also employs an adversarial mechanism to automatically construct a concept taxonomy and a style token pool.
    - **Design Motivation**: Adversarial evolution drives the system to continuously explore the frontier of the structural space, enabling the discovery of high-difficulty samples absent from standard datasets. Adaptive truncation prevents over-evolution.

3. **Executor Semantic Instantiation**:

    - **Function**: Converts the optimized constraint graph into a natural language mathematical problem and reasoning chain.
    - **Mechanism**: A conditional generative model $(Q, A) \sim P_{\text{executor}}(\cdot | \mathcal{G}^*, \mathcal{S})$ generates questions and answers from a linearized graph representation. Since complexity and diversity are guaranteed by the Legislator, the Executor focuses solely on language. Generated samples are verified by an external judge model for logical correctness and question-answer consistency.
    - **Design Motivation**: Decoupling relieves the Executor of the burden of exploring the complexity space, enabling more efficient generation of diverse textual formulations.

### Loss & Training
Synthetic data is used to fine-tune the target model via standard SFT. During validation, an external LLM serves as a judge to evaluate the logical correctness and consistency of synthetic QA pairs; only samples passing verification are retained.

## Key Experimental Results

### Main Results

| Model | Dataset | GSM8K | MATH500 | AIME24 | AIME25 | Avg |
|------|--------|-------|---------|--------|--------|------|
| Qwen3-14B | LIMO | 91.8 | 86.2 | 33.8 | 27.5 | 59.5 |
| Qwen3-14B | s1K | 87.5 | 86.4 | 37.9 | 25.0 | 60.3 |
| Qwen3-14B | **Ours** | **95.4** | **91.8** | **38.8** | **30.0** | **63.9** |
| Qwen2.5-Math-7B | LIMO | 87.4 | 72.2 | 10.8 | 14.6 | 45.6 |
| Qwen2.5-Math-7B | **Ours** | **91.6** | **82.2** | **18.8** | **18.3** | **53.5** |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Full MathAgent | 63.9 (Qwen3-14B Avg) | All components |
| w/o Critic | ~60.5 | No adversarial review; structural quality degrades |
| w/o Adaptive Truncation | ~61.2 | Fixed evolution rounds; reduced efficiency |
| Direct Text Generation | ~58.0 | No constraint graph; mode collapse |

### Key Findings
- Only 1K synthetic samples suffice to surpass LIMO and s1K at the same scale, demonstrating remarkable data efficiency.
- Gains are especially pronounced on high-difficulty competition benchmarks such as AIME, validating the framework's advantage in generating long-tail, high-difficulty samples.
- Strong cross-model generalization: effective across 10 models spanning four model families (Qwen, Llama, Mistral, Gemma).
- Smaller models benefit more from MathAgent data; Qwen3-4B improves from a base of 42.8 to 53.5.

## Highlights & Insights
- Elevating data synthesis from the text space to the structural space is the key innovation—constraint graphs serve as an intermediate representation that effectively separates the orthogonal dimensions of *how hard a problem is* and *how a problem is expressed*.
- The adversarial evolution mechanism requires no seed data; it bootstraps high-quality data from the model's intrinsic conceptual primitives, achieving genuine unsupervised synthesis.
- The adaptive truncation mechanism resembles early stopping, preventing over-evolution from producing unsolvable problems and reflecting a fine-grained balance between synthetic data quality and complexity.

## Limitations & Future Work
- Validation is currently limited to mathematical reasoning; generalizability to domains requiring structured data, such as code generation and logical reasoning, remains unexplored.
- The Legislator system requires multiple rounds of LLM interaction for evolution, potentially incurring higher synthesis costs than simple seed-expansion methods.
- External judge verification may have its own blind spots; correctness judgments on extremely difficult problems may not be fully reliable.

## Related Work & Insights
- **vs LIMO/s1K**: These methods rely on carefully curated seed data, whereas MathAgent is fully automated and surpasses them with fewer samples.
- **vs Self-Instruct**: Self-Instruct expands in the token space, with diversity bounded by the semantic radius of the seeds; MathAgent explores the structural space and can discover distributions far beyond that radius.
- **vs Magpie**: Magpie is zero-shot but lacks structural guidance and is prone to mode collapse; MathAgent provides a structural skeleton via constraint graphs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The hierarchical synthesis framework combining constraint graphs and adversarial evolution constitutes an entirely new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validated across 10 models, 8 benchmarks, and multiple model families.
- **Writing Quality**: ⭐⭐⭐⭐ Formalization is clear, though some notation is slightly heavy.
- **Value**: ⭐⭐⭐⭐⭐ Surpassing mainstream methods with 1K samples demonstrates remarkable data efficiency.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ACL 2026\] GanitLLM: Difficulty-Aware Bengali Mathematical Reasoning through Curriculum-GRPO](ganitllm_difficulty-aware_bengali_mathematical_reasoning_through_curriculum-grpo.md)

<!-- RELATED:END -->
