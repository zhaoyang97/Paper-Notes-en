---
title: >-
  [Paper Note] A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm
description: >-
  [ACL 2025][LLM (Other)][Automatic Prompt Optimization] This paper presents a systematic survey of over 80 automatic prompt optimization (APO) methods based on heuristic search algorithms, proposing a five-dimensional taxonomy (Where/What/What criteria/Which operators/Which algorithms) to unify fragmented research into a comprehensive analytical framework.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Automatic Prompt Optimization"
  - "Heuristic Search"
  - "Instruction Optimization"
  - "Evolutionary Algorithm"
  - "Taxonomy"
date: 2026-05-08
content_hash: aa123980b4f740d8
---

# A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm

**Conference**: ACL 2025  
**arXiv**: [2502.18746](https://arxiv.org/abs/2502.18746)  
**Code**: [GitHub](https://github.com/jxzhangjhu/Awesome-LLM-Prompt-Optimization)  
**Area**: LLM/NLP  
**Keywords**: Automatic Prompt Optimization, Heuristic Search, Instruction Optimization, Evolutionary Algorithm, Taxonomy

## TL;DR
This paper presents a systematic survey of over 80 automatic prompt optimization (APO) methods based on heuristic search algorithms, proposing a five-dimensional taxonomy (Where/What/What criteria/Which operators/Which algorithms) to unify fragmented research into a comprehensive analytical framework.

## Background & Motivation
**Background**: Manual prompt engineering (e.g., CoT, "step by step," "take a deep breath") relies entirely on human intuition and trial-and-error, failing to systematically identify the optimal prompt. Automatic Prompt Optimization (APO) formulates prompt design as a search or optimization problem, iteratively refining prompts using algorithms to maximize performance on downstream tasks.

**Limitations of Prior Work**: Since the emergence of APE and OPRO in 2023, the APO field has experienced explosive growth, yet the research remains highly fragmented. Different methods utilize evolutionary algorithms (EvoPrompt), Bayesian optimization (InstructZero), LLMs as optimizers (OPRO), or gradient signals (GCG), lacking a unified taxonomy to understand their similarities and differences.

**Key Challenge**: How to integrate various APO methods with vastly different methodologies into a coherent classification system? Existing surveys are either too broad (covering all prompting techniques) or too narrow (focusing only on a specific class of methods), failing to provide a clear overview.

**Goal**: To provide a comprehensive and organized survey that unifies fragmented research through an orthogonal five-dimensional taxonomy, enabling researchers to quickly locate any new work within the classification coordinate system.

**Key Insight**: The authors focus on two scope limitations—(a) instruction-focused prompting rather than exemplar selection/ordering; (b) heuristic search rather than reinforcement learning (RL) or model ensemble methods—avoiding overly general discussions through precise scoping.

**Core Idea**: The five-dimensional taxonomy (Where × What × What criteria × Which operators × Which algorithms) serves as a complete orthogonal coordinate system for understanding automatic prompt optimization, allowing any APO method to be precisely positioned across these five dimensions.

## Method

### Overall Architecture
Instead of proposing a new method, this work constructs an analytical framework. The core contribution is a five-dimensional taxonomy that decomposes APO methods into five orthogonal dimensions, with subcategories nested within each. The overall logic: first determine where to optimize (Where), then specify what to optimize (What), define the evaluation criteria (What criteria), determine how to generate candidate prompts (Which operators), and finally choose the search strategy to navigate the solution space (Which algorithms).

### Key Designs

1. **Where — Optimization Space**:

    - **Soft Prompt Space**: Optimizing within the continuous embedding space, where gradient signals can be utilized. Subcategories include: Gradient-to-Embedding (Prefix Tuning and P-Tuning directly optimize continuous vectors), Gradient-to-Target Token (GCG uses gradients to locate optimal substitution positions and then searches within discrete tokens), Gradient-to-Vocabulary Mapping (AutoPrompt uses gradients to search for the optimal tokens in the vocabulary), and Zero-Order Optimization (ZOPO estimates gradients using Gaussian processes approximated via NTK, eliminating the need for backpropagation).
    - **Discrete Prompt Space**: Directly optimizing natural language text strings, which are inherently non-differentiable. This requires black-box optimization methods. For example, ProTeGi uses LLM feedback to generate "pseudo-gradients" (textual suggestions for improvement), and EvoPrompt uses genetic algorithms to perform mutation and crossover on prompts.
    - **Soft-to-Discrete Projection**: Some methods optimize in the soft space and subsequently project back into the discrete space (e.g., GCG), though the projection process suffers from information loss, which remains an open problem.

2. **What — Optimization Target**:

    - **Instruction-Only Optimization** (the most common paradigm): Directly refining the instruction text, such as OPRO iteratively rewriting system prompts.
    - **Joint Instruction and Exemplar Optimization**: Three sub-paradigms—
        - Exemplar-to-Instruction (MoP: clusters exemplars first, then generates specialized instructions for each cluster).
        - Instruction-to-Exemplar (MIPRO: optimizes instructions first, then generates matching few-shot exemplars).
        - Parallel Optimization (EASE: uses bandit algorithms to search for the best instruction-exemplar combination simultaneously).
    - **Instruction + Optional Exemplars** (PhaseEvo: dynamically decides whether to add exemplars based on task characteristics).
    - **Key Insight**: Joint optimization of instructions and exemplars significantly outperforms optimizing either in isolation, but the search space expands exponentially.

3. **What criteria — Optimization Criteria**:

    - **Task Performance**: Traditional metrics such as accuracy, F1 score, etc.
    - **Robustness**: Stability against prompt perturbations and adversarial inputs.
    - **Efficiency**: Number of API calls, convergence speed, and computational cost.
    - **Interpretability**: Whether the optimized prompt is understandable (soft prompts are usually uninterpretable).
    - **Safety**: Optimization should not lead to harmful outputs (the adversarial prompt optimization techniques in GCG can be reversely applied to red-teaming).
    - **Multi-Objective Optimization**: Simultaneously optimizing performance, safety, and cost is receiving increasing attention.

4. **Which operators — Candidate Generation Operators**:

    - **Zero-parent Operators**: Generating candidates from scratch without relying on existing prompts (Lamarckian generation directly from task descriptions, model-driven random initialization).
    - **Single-parent Operators**: Generating variants based on a single existing prompt—semantic paraphrasing (local/global), LLM feedback-driven (prompting LLMs to analyze error cases and suggest rewrites), human feedback-driven, gradient feedback-driven, and deletion/addition/substitution operations.
    - **Multi-parent Operators**: Combining multiple existing prompts—crossover (extracting complementary components of two prompts), difference (learning the difference between two prompts and applying it to a third), and EDA-style estimation of distribution sampling.

5. **Which algorithms — Search Algorithms**:

    - **Bandit Algorithms**: Treating prompts as arms, using UCB or Thompson Sampling to balance exploration and exploitation.
    - **Beam Search**: Maintaining a top-k candidate set and expanding the best candidates at each step (e.g., ProTeGi).
    - **Monte Carlo Tree Search (MCTS)**: Structuring prompts as tree nodes and navigating using UCT (e.g., PromptAgent).
    - **Evolutionary Algorithm Family**: GA, Differential Evolution, CMA-ES, Simulated Annealing, treating prompts as "individuals" for population evolution.
    - **Iterative Refinement**: The simplest yet effective approach, where LLMs repeatedly rewrite their own prompts (the OPRO paradigm).

## Key Experimental Results

### Representative Methods Comparison

| Method | Optimization Space | Optimization Target | Search Algorithm | Representative Contributions |
|------|---------|---------|---------|-----------|
| APE (Zhou et al. 2023) | Discrete | Instruction-Only | Iterative Refinement | First method to generate and filter instructions using LLMs |
| OPRO (Yang et al. 2024) | Discrete | Instruction-Only | Iterative Refinement | Foundational paradigm of using the LLM itself as an optimizer |
| ProTeGi (Pryzant et al.) | Discrete | Instruction-Only | Beam Search | "Pseudo-gradients" concept—textual improvement signals |
| EvoPrompt (Guo et al.) | Discrete | Instruction-Only | GA/DE | First to introduce evolutionary algorithms to prompt optimization |
| PromptBreeder (Fernando) | Discrete | Instruction-Only | Evolutionary Algorithm | Self-evolution—the mutation strategy itself also evolves |
| MIPRO (Opsahl-Ong et al.) | Discrete | Instruction + Exemplar | Bayesian | Joint optimization paradigm of Instruction-to-Exemplar |
| InstructZero (Chen et al.) | Soft-to-Discrete | Instruction-Only | Bayesian | Soft space optimization + discrete projection |
| GCG (Zou et al. 2023) | Soft-to-Discrete | Instruction-Only | Greedy Coordinate Gradient | Gradient localization + discrete search, used for adversarial prompts |
| PromptAgent (Wang et al.) | Discrete | Instruction-Only | MCTS | Monte Carlo Tree Search navigating prompt space |
| PhaseEvo (Cui et al.) | Discrete | Instruction + Optional Exemplar | Evolutionary Algorithm | Dynamically deciding whether to add few-shot exemplars |

### Toolkits and Frameworks Comparison

| Toolkit/Framework | Core Functions | Features |
|------|---------|------|
| DSPy | Declarative prompt programming | Treating prompts as compilable/optimizable programming modules |
| TextGrad | Textual gradient optimization | Simulating gradient descent using LLM feedback |
| PromptFoo | A/B testing framework | Systematic evaluation of prompt variants |
| PromptBench | Robustness testing | Evaluating prompt stability against perturbations |

### Key Findings
1. **Discrete Space Dominance**: Most recent methods choose to optimize directly in the natural language space because: (a) it is friendly to closed-source APIs, (b) the results are interpretable, and (c) it does not require access to model weights.
2. **OPRO Paradigm is Simple yet Effective**: Iterative refinement methods that use the LLM itself as an optimizer, despite having the simplest algorithm, perform comparably to complex search algorithms in many scenarios.
3. **Evolutionary Algorithms Excel in High-Dimensional Spaces**: When the prompt space is complex (long instructions, multiple constraints), the population diversity of evolutionary algorithms helps prevent getting trapped in local optima.
4. **Joint Optimization > Isolated Optimization**: Joint optimization of instructions and exemplars outperforms instruction-only optimization by 5-15% on average, though computational costs increase significantly.

## Highlights & Insights
- **Clear and Comprehensive Five-Dimensional Taxonomy**: Unifies fragmented research under a single framework; any new paper can be precisely located across five dimensions, which is highly practical for rapidly scanning new work.
- **LLMs as Optimizers (OPRO Paradigm)** breaks the traditional separation of optimization and evaluation. Since the optimizer and the optimized object are the same model, this self-referential architecture is likely the most promising direction.
- **Optimization Criteria Beyond Accuracy**: Robustness and safety are equally crucial. GCG was initially designed for adversarial attacks (finding prompts that bypass alignment), but its technology can be conversely used for red-teaming and defense, revealing the double-edged sword nature of APO.
- **Soft-to-Discrete Projection represents a Key Bottleneck**: Soft space optimization utilizes gradients but produces uninterpretable and non-transferable results, whereas discrete space is interpretable but non-differentiable. Efficiently bridging these two spaces remains an unresolved core problem.

## Limitations & Future Work
- **Lack of Unified Experimental Comparison**: The most significant limitation as a survey. It only classifies but does not compare various methods on a unified benchmark, leaving readers unable to answer the key question of "which method is best."
- **Multimodal Prompt Optimization is Not Covered**: The survey primarily focuses on textual prompts, leaving out optimization methods for visual prompts (e.g., SAM's point/box prompts) and audio prompts.
- **Underestimation of the Dynamic N-shot Problem**: Most methods assume a fixed number of few-shot exemplars, whereas the optimal number of exemplars should dynamically change based on the task and input.
- **Missing Cost Analysis**: The number of API calls and costs of different APO methods vary vastly (from dozens to tens of thousands), but a systematic cost comparison is missing.
- **A Rapidly Evolving Field**: Recent methods from late 2024 to early 2025 (e.g., AgentOptimizer, RL-based methods) might not be covered.

## Related Work & Insights
- **vs Other Prompt Surveys**: Sahoo et al. (2024) offers broader coverage of all prompting techniques but lacks depth. This paper maintains a more precise scope (heuristic search + instructions), thereby analyzing each type of method in greater depth.
- **Inspirations for Search Algorithm Selection**: Prompt optimization is essentially a black-box optimization/search problem, where the choice of search algorithm is crucial. Simple tasks can be handled via iterative refinement, while complex tasks (long instructions, multiple constraints) are better suited for evolutionary algorithms or MCTS.
- **Connections to Agent Tool Optimization**: The APO framework can be directly transferred to tool selection and parameter optimization in Agents, where the search space transitions from "prompt text" to "tool invocation sequences."
- **DSPy Perspective**: Treating prompts as compilable program modules (rather than static strings) is likely a more promising paradigm than word-by-word optimization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The contribution of the survey lies in its taxonomy, and the methods reviewed are retrospective; the five-dimensional taxonomy is highly original, although the individual dimensions are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐ No experiments, pure survey—the lack of quantitative comparison on a unified benchmark is the biggest regret.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear classification and structured presentation. The tree-like taxonomy diagram of Fig.1 is intuitive and practical, covering 80+ methods without feeling cluttered.
- **Value**: ⭐⭐⭐⭐⭐ Provides practical reference value for prompt engineering practitioners. The five-dimensional coordinate system serves as an effective tool for quickly positioning new work, and the GitHub awesome list itself holds community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)
- [\[ACL 2025\] Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection](erm_prompt_optimization_memory.md)

</div>

<!-- RELATED:END -->
