---
title: >-
  [Paper Note] MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis
description: >-
  [ACL 2026][LLM Reasoning][Mathematical Reasoning] The study proposes MathAgent, a hierarchical data synthesis framework based on the adversarial evolution of constraint graphs. It reframes data synthesis from a text gene…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Mathematical Reasoning"
  - "Data Synthesis"
  - "Constraint Graphs"
  - "Adversarial Evolution"
  - "Legislator-Executor"
date: 2026-05-08
content_hash: c6bb36b18393a41e
---

# MathAgent: Adversarial Evolution of Constraint Graphs for Mathematical Reasoning Data Synthesis

**Conference**: ACL 2026  
**arXiv**: [2604.11188](https://arxiv.org/abs/2604.11188)  
**Code**: None  
**Area**: Data Synthesis / LLM Reasoning  
**Keywords**: Mathematical Reasoning, Data Synthesis, Constraint Graphs, Adversarial Evolution, Legislator-Executor

## TL;DR
The study proposes MathAgent, a hierarchical data synthesis framework based on the adversarial evolution of constraint graphs. It reframes data synthesis from a text generation task to an unsupervised optimization problem of constraint graphs. By evolving problem skeletons through a three-agent Legislator system and instantiating them via an Executor into natural language, it outperforms LIMO and s1K on eight mathematical benchmarks using only 1K synthetic samples.

## Background & Motivation

**Background**: High-quality mathematical reasoning data is a critical driver for enhancing the reasoning capabilities of LLMs. As the scale bottleneck of human-annotated data becomes increasingly prominent, synthetic data generation has become a mainstream research direction.

**Limitations of Prior Work**: (1) Seed expansion methods (e.g., Self-Instruct, WizardMath) are limited by the "semantic radius" of the initial seeds, putting an upper bound on diversity; (2) Zero-shot methods (e.g., Magpie) directly probe model distributions but lack structural guidance, leading to mode collapse and logical hallucinations; (3) Existing methods treat data synthesis as a direct text generation task, where models often stay at the level of surface narrative imitation without mastering core reasoning abilities.

**Key Challenge**: Synthesizing data directly in the token space fails to effectively control the logical complexity and structural diversity of the problems. High-difficulty, high-quality long-tail samples are essential for forging complex reasoning abilities, yet standard methods struggle to discover these samples.

**Goal**: Design a synthesis framework that automatically explores the structural space without requiring human seed data, generating mathematical reasoning data with both high complexity and high diversity.

**Key Insight**: Decouple data synthesis into two stages: structural evolution (meta-level) and semantic instantiation (base-level). This involves first optimizing the logical skeleton (constraint graph) of the problem before converting the skeleton into natural language questions.

**Core Idea**: Represent the logical structure of mathematical problems using constraint graphs. Continuously optimize the complexity and diversity of the graph structure through a three-agent adversarial evolution mechanism (Proposer-Critic-Moderator), followed by an Executor that generates natural language problems and reasoning chains.

## Method

### Overall Architecture
MathAgent is divided into two decoupled phases: (1) Meta-Level Structural Evolution: The Legislator three-agent system undergoes adversarial evolution in the constraint graph space to produce an optimized problem skeleton $\mathcal{G}^*$; (2) Base-Level Semantic Instantiation: The Executor transforms $\mathcal{G}^*$ and style tokens $\mathcal{S}$ into natural language questions $Q$ and reasoning chains $A$. Finally, qualified samples are filtered through external model verification.

### Key Designs

1.  **Constraint Graph Representation**:
    - **Function**: Formally describes the logical skeleton of the mathematical problem.
    - **Mechanism**: Models the problem as a graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ plus style tokens $\mathcal{S}$. Nodes $\mathcal{V}$ represent mathematical concepts, and edges $\mathcal{E}$ represent logical relationships. The optimization goal is $\mathcal{G}^* = \arg\max_{\mathcal{G}} \mathcal{H}(\mathcal{G})$, where $\mathcal{H}$ estimates complexity, and the constraint $\mathbb{I}_{\text{valid}}(\mathcal{G}|\mathcal{S})=1$ ensures solvability.
    - **Design Motivation**: Decoupling structural specifications from text implementation allows the framework to focus on building complex and diverse logical structures rather than being bound by surface linguistic patterns.

2.  **Legislator Three-Agent Evolution System**:
    - **Function**: Iteratively optimizes the constraint graph structure through adversarial dynamics.
    - **Mechanism**: Three roles work collaboratively—the Proposer ($\mathcal{A}_P$) optimizes the graph $\mathcal{G}_t \to \mathcal{G}_{t+1}$ based on feedback from previous rounds, resolving logical contradictions and extending structural depth; the Critic ($\mathcal{A}_C$) reviews the new graph across internal consistency, specification alignment, and optimization potential to generate improvement reports; the Moderator ($\mathcal{A}_M$) acts as a strategic decision-maker, either terminating the evolution (if converged) or guiding the Proposer to implement improvements. The initialization phase also automatically constructs concept taxonomies and style token pools via adversarial mechanisms.
    - **Design Motivation**: Adversarial evolution drives the system to continuously explore the frontier of structural space, discovering high-difficulty samples missing in standard datasets. Adaptive truncation prevents over-evolution.

3.  **Executor Semantic Instantiation**:
    - **Function**: Converts optimized constraint graphs into natural language math problems and reasoning chains.
    - **Mechanism**: A conditional generation model $(Q, A) \sim P_{\text{executor}}(\cdot | \mathcal{G}^*, \mathcal{S})$ generates questions and answers based on linearized graph representations. Since complexity and diversity are already guaranteed by the Legislator, the Executor only needs to focus on the language itself. After generation, external models (judges) verify logical correctness and QA consistency.
    - **Design Motivation**: By decoupling, the Executor is freed from the burden of exploring the complexity space, enabling more efficient generation of diverse textual expressions.

### Loss & Training
Synthetic data is used to fine-tune target models via standard SFT. During the verification phase, an external LLM acts as a judge to evaluate the logical correctness and consistency of synthetic QA pairs, retaining only those that pass verification.

## Key Experimental Results

### Main Results

| Model | Dataset | GSM8K | MATH500 | AIME24 | AIME25 | Average |
|-------|---------|-------|---------|--------|--------|---------|
| Qwen3-14B | LIMO | 91.8 | 86.2 | 33.8 | 27.5 | 59.5 |
| Qwen3-14B | s1K | 87.5 | 86.4 | 37.9 | 25.0 | 60.3 |
| Qwen3-14B | **Ours** | **95.4** | **91.8** | **38.8** | **30.0** | **63.9** |
| Qwen2.5-Math-7B | LIMO | 87.4 | 72.2 | 10.8 | 14.6 | 45.6 |
| Qwen2.5-Math-7B | **Ours** | **91.6** | **82.2** | **18.8** | **18.3** | **53.5** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full MathAgent | 63.9 (Qwen3-14B Avg) | All components |
| w/o Critic | ~60.5 | No adversarial review; structural quality drops |
| w/o Adaptive Truncation | ~61.2 | Fixed evolution rounds; efficiency decreases |
| Direct Text Gen | ~58.0 | Mode collapse without constraint graphs |

### Key Findings
- Only 1K synthetic samples are needed to surpass LIMO and s1K of the same scale, demonstrating extreme data efficiency.
- Gains are particularly significant on high-difficulty competition benchmarks like AIME, verifying the framework's advantage in generating long-tail, high-difficulty samples.
- Strong cross-model generalization: Effective across 10 models from four series (Qwen, Llama, Mistral, Gemma).
- Smaller models benefit more from MathAgent data; Qwen3-4B improved from a base of 42.8 to 53.5.

## Highlights & Insights
- Elevating data synthesis from the token space to the structural space is a key innovation—the constraint graph serves as an intermediate representation that effectively separates the orthogonal dimensions of "problem difficulty" and "problem phrasing."
- The adversarial evolution mechanism does not rely on seed data. It constructs high-quality data starting from the model's inherent conceptual primitives, truly achieving "creation from nothing."
- The adaptive truncation mechanism, similar to early stopping, avoids unsolvable problems caused by over-evolution, reflecting a fine balance between synthetic data quality and complexity.

## Limitations & Future Work
- Currently only verified in the mathematical reasoning domain; whether it can generalize to scenarios requiring structured data like code generation or logical reasoning remains unexplored.
- The Legislator system requires multiple rounds of LLM interaction for evolution, which may have higher synthesis costs than simple seed expansion.
- External judge verification may have its own blind spots, and its judgment on the correctness of extremely difficult problems may not be completely reliable.

## Related Work & Insights
- **vs LIMO/s1K**: These methods rely on meticulously filtered seed data, whereas MathAgent is fully automated and surpasses them with less data.
- **vs Self-Instruct**: Self-Instruct expands in the token space, with diversity limited by the semantic radius of the seeds; MathAgent explores the structural space, discovering more distant areas of the distribution.
- **vs Magpie**: Magpie is zero-shot but lacks structural guidance, making it prone to mode collapse; MathAgent provides a structural skeleton through constraint graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The hierarchical synthesis framework of constraint graphs + adversarial evolution is a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 models, 8 benchmarks, cross-series verification.
- Writing Quality: ⭐⭐⭐⭐ Formalization is clear, though some notation is slightly heavy.
- Value: ⭐⭐⭐⭐⭐ Surpassing mainstream methods with 1K data; data efficiency is remarkable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[ACL 2026\] Self-Reinforcing Controllable Synthesis of Rare Relational Data via Bayesian Calibration](self-reinforcing_controllable_synthesis_of_rare_relational_data_via_bayesian_cal.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICML 2026\] An Information-Theoretic Criterion for Efficient Data Synthesis](../../ICML2026/llm_reasoning/an_information-theoretic_criterion_for_efficient_data_synthesis.md)
- [\[ACL 2026\] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)

</div>

<!-- RELATED:END -->
