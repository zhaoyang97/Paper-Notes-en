---
title: >-
  [Paper Note] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths
description: >-
  [NeurIPS 2025][LLM (Other)][Agent framework] This paper proposes the Probabilistic Angelic Nondeterminism (PAN) programming model and the EnCompass Python framework, which decouple an agent's core workflow logic from its inference-time search strategy. Programmers only need to insert `branchpoint()` markers at LLM call sites and switch among best-of-N, beam search, tree search, and other strategies via a few configuration parameters, reducing the amount of code modification b…
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "Agent framework"
  - "inference-time search"
  - "nondeterministic programming"
  - "Beam Search"
  - "program execution paths"
date: 2026-05-08
content_hash: 62110c76d381d9c6
---

# EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths

**Conference**: NeurIPS 2025
**arXiv**: [2512.03571](https://arxiv.org/abs/2512.03571)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: Agent framework, inference-time search, nondeterministic programming, Beam Search, program execution paths

## TL;DR
This paper proposes the Probabilistic Angelic Nondeterminism (PAN) programming model and the EnCompass Python framework, which decouple an agent's core workflow logic from its inference-time search strategy. Programmers only need to insert `branchpoint()` markers at LLM call sites and switch among best-of-N, beam search, tree search, and other strategies via a few configuration parameters, reducing the amount of code modification by 3–6×.

## Background & Motivation

**Background**: Inference-time compute scaling has become a key approach for improving LLM agent performance, encompassing strategies such as best-of-N sampling, refinement (Reflexion), and tree search (ToT, LATS).

**Limitations of Prior Work**:
   - Programmers typically **hard-code inference-time strategies into agent workflows**, deeply coupling search logic with business logic.
   - Switching search strategies (e.g., from best-of-N to beam search) requires extensive structural code refactoring.
   - Different agents must each re-implement the same search strategies independently, precluding reuse.
   - Complex search strategies are abandoned due to implementation difficulty, causing missed opportunities for better scaling laws.

**Key Challenge**: The coupling of agent workflows (what to do) with inference-time strategies (how to search) reduces code readability and constrains the space of strategies that can be explored.

**Key Insight**: Inspired by the "model–inference separation" paradigm from probabilistic programming—which separates model definition from inference algorithms—agent programming should similarly decouple workflows from search strategies.

**Core Idea**: Inference-time strategies are fundamentally searches over nondeterministic program execution paths; marking uncertainty points with `branchpoint()` allows the framework to automatically construct the search tree.

## Method

### Overall Architecture
EnCompass is a Python framework implementing the PAN programming model. Programmers annotate agent functions with the `@encompass.compile` decorator, insert `branchpoint()` calls before unreliable operations such as LLM calls, and invoke `record_score()` at evaluation points. At runtime, the decorator compiles the function into a search-space object, which is then explored via `.search(algo, **config)` using the desired search algorithm.

### Key Designs

1. **PAN Programming Model**:

    - **Function**: Models nondeterministic program execution as a Markov chain; `branchpoint()` serves as a marker, and the program state is defined as (location, variable values).
    - **Mechanism**: Angelic nondeterminism—programmers write code assuming unreliable operations always produce good outputs, while the runtime search is responsible for finding execution paths that actually do so.
    - **Distinction from classical search**: Unlike classical graph search, all child nodes cannot be enumerated; only random sampling is possible. Classical search algorithms are adapted by specifying a branching factor.

2. **Unified Inference-Time Strategies**:

    - **Best-of-N → depth-1 search tree**: A single `branchpoint()` is added at the start; N samples are drawn and the highest-scoring one is selected.
    - **Beam Search → multi-level search tree**: A `branchpoint()` is added before each step; beam width and branching factor control search breadth.
    - **Refinement → backtracking with memory**: A `NoCopy` shared feedback list allows different execution paths to share the history of previous attempts.
    - **Self-Consistency → group evaluation**: `record_score()` supports group evaluation functions such as majority voting.
    - **Key insight**: Global BoN (beam width = N, branching = 1) and Local BoN (beam width = 1, branching = N) are the two extremes of beam search; beam search interpolates between them.

3. **EnCompass Compiler**:

    - **Function**: Compiles Python functions into search-space objects that can be manipulated by search algorithms.
    - **Mechanism**: The `@encompass.compile` decorator transforms a function into a state machine, with `branchpoint()` becoming state-transition points.
    - **vs. hand-written state machines**: EnCompass requires 3–6× fewer code modifications than plain Python while preserving code readability.

## Key Experimental Results

### Main Results: Repository-Level Code Translation (Java → Python)

| Search Strategy | ps0 Performance | Notes |
|----------------|----------------|-------|
| No search (baseline) | ~50% | Temperature 0.0, single execution |
| Global BoN | ~65% | N = 16 |
| Local BoN (coarse) | ~68% | N = 16, file-level |
| Beam (coarse) + Beam (fine) | **~78%** | File-level + method-level beam search |

### Code Modification Volume Comparison

| Case Study | Without EnCompass (lines/words added) | With EnCompass (lines/words added) | Savings |
|-----------|--------------------------------------|-----------------------------------|---------|
| Code translation | +423 lines / +2735 words | +75 lines / +514 words | 5.3× |
| Hypothesis Search | +21 lines / +120 words | +8 lines / +27 words | 4.4× |
| Reflexion | +27 lines / +181 words | +9 lines / +32 words | 5.7× |

### Key Findings
- Beam search scales substantially better than simple BoN—beam search achieves the steepest slope in the performance vs. $\log(\text{cost})$ relationship.
- The best-performing strategy is precisely the most complex to implement manually, underscoring the value of lowering the implementation barrier.
- Switching strategies requires changing only 2 lines of parameters, whereas hand-written implementations require refactoring the entire state machine.
- On ps1–ps4 (larger repositories), beam search consistently outperforms BoN.

## Highlights & Insights
- **"Separation of concerns" reduces the cost of exploring search strategies**: The primary contribution is not a new algorithm but enabling programmers to experiment with diverse search strategies at minimal cost. This parallels the significance of PyTorch for deep learning—lowering the barrier to experimentation unleashes innovation.
- **Unified perspective**: Framing BoN, beam search, refinement, and self-consistency as different strategies for searching over an execution-path tree is conceptually elegant.
- **Probabilistic programming for agents**: PAN stands in relation to agent programming as Stan/PyMC stands to probabilistic inference—separating model specification from inference.

## Limitations & Future Work
- Applicable only to "program-in-control" style agents; not suited for scenarios where the LLM autonomously decides tool calls (e.g., SWE-Bench, WebArena).
- Case studies are relatively small in scale (MIT OCW assignment repositories); validation on large-scale industrial agents is lacking.
- The search space grows exponentially with the number of branchpoints, requiring effective pruning and scoring strategies.
- The evaluation metric (self-validation %) may not be fully reliable.
- Integration examples with existing agent frameworks such as LangGraph are absent.

## Related Work & Insights
- **vs. LangGraph**: LangGraph models workflows as state machines, but search must still be implemented manually; EnCompass elevates search strategies to first-class citizens.
- **vs. DSPy**: DSPy automates prompt engineering; EnCompass automates search strategies—the two are orthogonal and complementary.
- **vs. ToT/LATS**: These are concrete search algorithms; EnCompass is a framework that makes such algorithms easily applicable to arbitrary agents.

## Rating
- Novelty: ⭐⭐⭐⭐ The PAN model and the "separation of concerns" perspective are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐ Case studies are persuasive but limited in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Concepts are clearly articulated, code examples are abundant, and the paper is easy to follow.
- Value: ⭐⭐⭐⭐ Has direct implications for agent development practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](../../ACL2025/llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](../../ACL2025/llm_nlp/autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](../../ACL2025/llm_nlp/idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)

</div>

<!-- RELATED:END -->
