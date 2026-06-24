---
title: >-
  [Paper Note] Autoformulation of Mathematical Optimization Models Using LLMs
description: >-
  [ICML 2025][Medical LLM][autoformulation] This paper proposes a method that combines Large Language Models (LLMs) with Monte-Carlo Tree Search (MCTS) to automatically convert optimization problems described in natural language into mathematical programming models solvable by solvers, significantly improving search efficiency through symbolic pruning and LLM-based value estimation.
tags:
  - "ICML 2025"
  - "Medical LLM"
  - "autoformulation"
  - "LLM"
  - "Monte-Carlo Tree Search"
  - "mathematical optimization"
  - "symbolic pruning"
date: 2026-05-08
content_hash: c8f38272636307d1
---

# Autoformulation of Mathematical Optimization Models Using LLMs

**Conference**: ICML 2025  
**arXiv**: [2411.01679](https://arxiv.org/abs/2411.01679)  
**Code**: None  
**Area**: Optimization  
**Keywords**: autoformulation, LLM, Monte-Carlo Tree Search, mathematical optimization, symbolic pruning

## TL;DR
This paper proposes a method that combines Large Language Models (LLMs) with Monte-Carlo Tree Search (MCTS) to automatically convert optimization problems described in natural language into mathematical programming models solvable by solvers, significantly improving search efficiency through symbolic pruning and LLM-based value estimation.

## Background & Motivation

**Background**: Mathematical optimization is a core tool in decision science, with wide applications in operations research, supply chain management, healthcare scheduling, and many other fields. However, translating real-world problems into standard mathematical programming models (such as Linear Programming LP, Mixed-Integer Programming MIP) typically requires deep domain expertise.

**Limitations of Prior Work**: The traditional modeling process highly relies on human experts, making it time-consuming and prone to errors. Although a few studies have attempted to use LLMs to assist in modeling, these methods typically employ simple prompt engineering or end-to-end generation, lacking deep exploration of the problem structure, which results in the unstable quality of the generated models.

**Key Challenge**: The search space for optimization models is extremely large and highly problem-dependent, making it difficult for a single LLM generation to cover the correct modeling scheme. Meanwhile, evaluating the correctness of a modeling formulation (without ground-truth answers) also poses a major challenge.

**Goal**: Automatically translate natural language problem descriptions into complete and correct mathematical optimization models (the autoformulation problem).

**Key Insight**: Leveraging the hierarchical structure of optimization modeling (decision variables $\rightarrow$ objective function $\rightarrow$ constraints) to formulate autoformulation as a tree search problem, combined with MCTS for systematic exploration.

**Core Idea**: Combine the generation capabilities of LLMs with MCTS search strategies, leverage the hierarchy of modeling to guide the search, eliminate equivalent branches via symbolic pruning, and direct the search using LLM-based evaluations of partial modeling quality.

## Method

### Overall Architecture
The input is an optimization problem described in natural language, and the output is a mathematical programming model that can be directly solved by a solver (such as Gurobi). The overall pipeline is divided into three levels:
1. **Decision Variable Definition**: Determine variable types and meanings.
2. **Objective Function Construction**: Formulate the objective based on the variables.
3. **Constraint Generation**: Progressively add constraints.

MCTS searches at each level, with the LLM acting as a policy network to generate candidate nodes.

### Key Designs

1. **Hierarchical MCTS (Hierarchical MCTS)**:

    - **Function**: Organizes the search tree using the natural hierarchy of optimization modeling (variables $\rightarrow$ objective $\rightarrow$ constraints).
    - **Mechanism**: Each level of the tree corresponds to a modeling stage. Starting from the root node (problem description), it sequentially expands variable definitions, the objective function, and constraints. Each node represents a partial modeling formulation.
    - **Design Motivation**: Directly searching in the full model space faces a combinatorial explosion. Hierarchical decomposition breaks down the search space structurally, making the search space at each level smaller and more meaningful.

2. **Symbolic Pruning (Symbolic Pruning)**:

    - **Function**: Identifies and eliminates mathematically equivalent modeling branches during the search.
    - **Mechanism**: Uses symbolic computation tools (such as SymPy) to determine if two mathematical expressions are equivalent. If a newly generated branch is equivalent to an existing branch, it is pruned immediately.
    - **Design Motivation**: LLMs frequently generate expressions that are structurally different but mathematically equivalent (such as $2x + 2y$ and $2(x+y)$). Failing to prune these wastes significant search budget on duplicate formulations.

3. **LLM-based Value Estimation (LLM-based Value Estimation)**:

    - **Function**: Evaluates the quality of partial modeling formulations in the search tree to guide the MCTS selection strategy.
    - **Mechanism**: Employs an LLM to compare partial formulations against the original problem description, outputting a score in the range of 0-1 as a value estimation.
    - **Design Motivation**: Traditional MCTS requires rolling out to leaf nodes to obtain feedback, but for the autoformulation problem, a complete rollout is highly expensive. Evaluating intermediate nodes with an LLM allows for earlier signal acquisition.

### Loss & Training
This method does not involve training; it is an inference-time search method. MCTS utilizes UCB1 for node selection, LLM-based value estimation replaces traditional rollouts, and symbolic pruning is used to reduce the search space.

## Key Experimental Results

### Main Results

| Benchmark Dataset | Metric (Accuracy %) | MCTS+LLM (Ours) | Direct LLM Generation | OptiMUS | Gain |
|------------|---------------|-----------------|---------------|---------|------|
| NL4OPT (LP) | Accuracy | **78.5%** | 52.3% | 61.7% | +16.8% |
| ComplexOR (MIP) | Accuracy | **63.2%** | 31.5% | 42.1% | +21.1% |
| IndustryOR | Accuracy | **55.8%** | 28.4% | 38.6% | +17.2% |

### Ablation Study

| Configuration | NL4OPT Accuracy | ComplexOR Accuracy | Description |
|------|-------------|----------------|------|
| Full Method | **78.5%** | **63.2%** | All components |
| Remove Symbolic Pruning | 71.3% | 54.8% | Equivalent branches waste search budget |
| Remove LLM Value Estimation | 68.9% | 51.2% | Search direction lacks effective guidance |
| Random Search (No MCTS) | 59.1% | 38.5% | Systematic search far outperforms random search |

### Key Findings
- The MCTS framework significantly outperforms direct end-to-end LLM generation and previous SOTA methods.
- Symbolic pruning reduces the number of search nodes by 35-40% on average without sacrificing search quality.
- Introducing LLM-based value estimation improves search efficiency by approximately 30%.
- The performance gain of this method is more significant on MIP problems because MIP has a larger modeling space.

## Highlights & Insights
- **Innovatively combining MCTS with LLMs**: Fully utilizes the generative capabilities of LLMs and the systematic search advantages of MCTS.
- **Symbolic pruning as a highlight**: Performs pruning based on the equivalence of mathematical expressions, which represents a structural prior unique to the autoformulation task.
- **Hierarchical modeling**: Integrates domain knowledge of optimization modeling (variables $\rightarrow$ objective $\rightarrow$ constraints) into the design of the search tree.

## Limitations & Future Work
- It depends heavily on the generation quality of the LLM and may still struggle with highly complex non-linear optimization problems.
- The trade-off between search efficiency and the number of LLM calls requires further analysis.
- Currently, it only supports LP and MIP; support for non-linear programming and stochastic programming remains to be expanded.
- The evaluation metrics only consider the correctness of the modeling, neglecting modeling efficiency (such as the number of variables and constraints).

## Related Work & Insights
- OptiMUS (AhmaditeshnizI et al., 2024): LLM-based end-to-end optimization modeling.
- The idea of using MCTS to guide LLM inference presented in this paper is worth extending to other tasks requiring structured generation (such as code generation and mathematical proof).

## Personal Thoughts
- Decomposing the modeling process into a hierarchical tree search is a key insight—modeling itself is inherently hierarchical.
- Symbolic pruning exploits the equivalence of mathematical expressions, which may also be applicable to other mathematical reasoning tasks.
- Solver feedback could be introduced in the future—passing the model to a solver to run and using the solving results to verify the correctness of the modeling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of LLM + MCTS + symbolic pruning is highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on multiple benchmark datasets with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic description of methodology.
- Value: ⭐⭐⭐⭐ Significantly drives the automation of optimization modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](../../ACL2025/medical_nlp/biore_llm_judge_evaluation.md)
- [\[ICML 2025\] EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](evolve_evaluating_and_optimizing_llms_for_in-context_exploration.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[ACL 2025\] Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](../../ACL2025/medical_nlp/evaluation_of_llms_in_medical_text_summarization_the_role_of_vocabulary_adaptati.md)

</div>

<!-- RELATED:END -->
