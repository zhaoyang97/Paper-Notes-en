---
title: >-
  [Paper Note] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search
description: >-
  [NeurIPS 2025][LLM/NLP][Test-time scaling] This paper proposes SolverLLM, a training-free framework that treats the mathematical modeling of optimization problems as a search problem. It employs an enhanced MCTS to explore optimal formulations within a six-element representation space, incorporating dynamic expansion, prompt backpropagation, and uncertainty backpropagation. SolverLLM surpasses both prompting-based and fine-tuning-based methods on 6 benchmarks without any training.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Test-time scaling
  - Monte Carlo Tree Search
  - optimization problems
  - dynamic expansion
  - uncertainty propagation
date: 2026-05-08
content_hash: 07ad7da530e0c61d
---

# SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search

**Conference**: NeurIPS 2025
**arXiv**: [2510.16916](https://arxiv.org/abs/2510.16916)
**Code**: None
**Area**: LLM/NLP
**Keywords**: Test-time scaling, Monte Carlo Tree Search, optimization problems, dynamic expansion, uncertainty propagation

## TL;DR

This paper proposes SolverLLM, a training-free framework that treats the mathematical modeling of optimization problems as a search problem. It employs an enhanced MCTS to explore optimal formulations within a six-element representation space, incorporating dynamic expansion, prompt backpropagation, and uncertainty backpropagation. SolverLLM surpasses both prompting-based and fine-tuning-based methods on 6 benchmarks without any training.

## Background & Motivation

**Background**: Solving optimization problems involves three stages—problem formulation (natural language → mathematical modeling), code generation (mathematical model → executable code), and program execution (invoking solvers such as Gurobi/Pyomo). Problem formulation requires both domain knowledge and mathematical programming expertise, making it the central bottleneck for automation. LLMs offer a promising avenue for automating this process.

**Limitations of Prior Work**: Existing approaches fall into two categories. Prompting-based methods (Chain-of-Experts, OptiMUS) rely on multi-agent collaboration but are highly sensitive to prompt design and generalize poorly to unfamiliar problem types. Learning-based methods (ORLM, LLMOPT) acquire domain capabilities via supervised fine-tuning but depend heavily on large-scale annotated data, struggle with cross-domain generalization, and incur high training costs.

**Key Challenge**: A fundamental tension exists among generality, high performance, and low training cost—prompting-based methods are training-free but performance-limited, while fine-tuned methods perform well but lack generality. The challenge is to achieve high performance and cross-domain generalization simultaneously without any training.

**Goal**: Replace training-time computation with test-time computation by constructing a general framework that improves optimization problem modeling quality through structured search at inference time, without requiring additional training.

**Key Insight**: Test-time scaling has demonstrated performance gains on tasks such as mathematical reasoning by allocating more inference computation. The modeling process for optimization problems is naturally suited to structured search—it can be decomposed into multiple semantic elements (type, sets, parameters, variables, objective, constraints), enabling layer-by-layer search for the optimal formulation.

**Core Idea**: Treat the mathematical modeling of optimization problems as a search problem, explore the six-element formulation space with an enhanced MCTS, and guide the search direction via reasoning signal feedback and uncertainty estimation.

## Method

### Overall Architecture

SolverLLM organizes the problem modeling process as an MCTS tree, where each node corresponds to a partial formulation (one of the six elements) and a root-to-leaf path defines a complete mathematical model. The search follows the four standard MCTS phases—selection, expansion, simulation, and backpropagation—each adapted for LLM-guided symbolic reasoning. The input is a natural-language description of an optimization problem; the output is a mathematical formulation and solver-ready code.

### Key Designs

1. **Six-Element Modeling Decomposition (with Type Element)**:

    - Function: Provides structured guidance for search, transforming the unstructured modeling task into a hierarchical decision sequence.
    - Mechanism: A Type element is introduced in addition to the prior five elements (Sets, Parameters, Variables, Objective, Constraints) to identify the high-level problem category (LP, IP, MIP, etc.). The Type element provides global guidance $G_g$, helping the LLM establish a correct mental model of the problem before detailed formulation begins. Each node encodes a partial formulation, and the complete path defines the full model.
    - Design Motivation: Analogous to a student reviewing key concepts before an exam—early identification of the problem type helps avoid fundamental modeling errors downstream, such as incorrectly modeling integer variables as continuous. Ablation studies show that the Type element yields especially significant gains on complex problems (e.g., graph-related optimization).

2. **Dynamic Expansion + Prompt Backpropagation**:

    - Function: Enables bidirectional correction of formulations within the search tree, overcoming the limitation of standard MCTS that only expands at leaf nodes.
    - Mechanism: Two innovations are introduced. (a) Non-leaf node expansion: the selection strategy is modified to allow the search to pause and expand at non-leaf nodes (triggered by $t_s$), enabling decisions in earlier layers to be revised based on feedback from later layers—e.g., feedback from the constraint layer can trigger re-modeling at the variable layer. (b) Prompt backpropagation: the evaluation phase produces a reasoning signal triple $\mathcal{S}_l = (t_{s_l}, E_{s_l}, G_l)$ for each layer, where $t_s$ is the activation trigger, $E_{s_l}$ is the evaluation rationale, and $G_l$ is the corrective guidance. These signals are filtered by local uncertainty ($U^{local}_{s_l}$ must exceed threshold $\eta$) before being propagated back into each layer's knowledge base $\mathcal{G}_l$ to guide subsequent expansion. Local uncertainty is estimated via predictive entropy.
    - Design Motivation: Modeling an optimization problem is not a strictly top-down linear process—constraints may depend on revisions to variable definitions. Dynamic expansion enables SolverLLM to "learn from mistakes" during search.

3. **Uncertainty Backpropagation**:

    - Function: Improves the robustness of reward propagation and reduces the impact of noise in LLM evaluations.
    - Mechanism: When using an LLM as a semantic evaluator, semantic uncertainty $U^{global}_s$ is estimated via multiple samples of the objective score. During backpropagation, rewards are down-weighted by uncertainty: $Q_{s'} \leftarrow Q_{s'} + \rho_s \cdot (\bar{R} - Q_{s'}) / N_{s'}$, where the weight factor $\rho_s = \exp(-U^{global}_s)$. High-confidence evaluations propagate strongly; noisy evaluations propagate weakly.
    - Design Motivation: Using an LLM as an evaluator inevitably introduces subjectivity and variance; directly propagating unreliable reward signals misleads the search. Uncertainty propagation makes the search more robust to evaluation noise and significantly reduces search time on complex problems.

### Loss & Training

SolverLLM is a training-free framework. The reward function is a weighted combination: $R(f_s, x^*) = \alpha \cdot \mathbb{I}_{feasible} + \beta \cdot \text{objective\_score}(f_s, x^*) - \gamma \cdot \mathbb{I}_{error}$. The selection strategy uses UCT: $s_{child} = \arg\max_{s'} [Q_{s'} + c \cdot \sqrt{2\log N_s / N_{s'}}]$.

## Key Experimental Results

### Main Results

Comparison with prompting-based methods (Solving Accuracy, SA):

| Dataset | GPT-4o Direct | OptiMUS | Chain-of-Experts | SolverLLM | Gain |
|--------|--------------|---------|-----------------|-----------|------|
| NL4Opt | 81.0% | 78.8% | 64.2% | **97.0%** | +18.2% |
| NLP4LP | 32.4% | 72.0% | 53.1% | **87.0%** | +15.0% |
| ComplexOR | 27.3% | 66.7% | 38.1% | **77.8%** | +11.1% |

Comparison with learning-based methods:

| Dataset | LLMOPT (SFT) | ORLM-LLaMa3 | SolverLLM | Gain |
|--------|-------------|------------|-----------|------|
| MamoEasy | **97.0%** | 82.3% | 96.0% | -1.0% |
| NL4Opt | 93.0% | 85.7% | **97.0%** | +4.0% |
| MamoComplex | 68.0% | 37.4% | **76.0%** | +8.0% |
| IndustryOR | 46.0% | 38.0% | **56.0%** | +10.0% |

### Ablation Study

| Variant | NL4Opt SA | MamoComplex SA | IndustryOR SA | Avg. AGT Change |
|---------|----------|---------------|---------------|----------------|
| SolverLLM (Full) | **97.0%** | **76.0%** | **56.0%** | Baseline |
| w/o Prompt Backprop | 93.0% (−4%) | 69.0% (−7%) | 46.0% (−10%) | Slightly faster |
| w/o Uncertainty Backprop | 97.0% | 75.0% (−1%) | 56.0% | Significantly slower on complex datasets |
| w/o Type Element | 96.0% (−1%) | 59.0% (−17%) | 48.0% (−8%) | Marginal change |

### Key Findings

- Prompt backpropagation contributes most on complex problems (−10% on IndustryOR), as complex problems require more bidirectional formulation correction.
- Uncertainty backpropagation primarily improves search efficiency rather than final accuracy—on MamoComplex, AGT decreases from 4.34 minutes to 3.85 minutes.
- The Type element is particularly effective for graph-related optimization problems, helping to avoid fundamental variable-type errors.
- Token efficiency analysis: over 20 search iterations, SolverLLM achieves 76% SA with 40,920 tokens, while AutoFormulation achieves only 37% SA with 43,150 tokens.
- Execution rate (ER) reaches nearly 100% across all datasets, attributable to error backpropagation in the code generation stage.

## Highlights & Insights

- **Applying test-time scaling to structured modeling problems is a compelling direction**: Unlike test-time scaling on free-form text reasoning, optimization problem modeling has a clear element decomposition structure that is naturally suited to tree search. The six-element schema provides a meaningful search granularity.
- **Prompt backpropagation enables "experience transfer" within the search tree**: Standard MCTS propagates only scalar rewards; SolverLLM additionally propagates semantic-level corrective guidance, allowing subsequent search steps to "learn from earlier mistakes." This mechanism is transferable to other tasks requiring structured generation by LLMs.
- **Uncertainty propagation is a best practice for LLM-as-judge settings**: Down-weighting unreliable evaluations via uncertainty estimation when using an LLM as a scorer is a generalizable and practically valuable technique.

## Limitations & Future Work

- Inference costs are non-trivial—each problem requires multiple LLM calls for MCTS search, with AGT of approximately 2–4 minutes, making the framework impractical in latency-sensitive settings.
- The framework relies on a strong backbone LLM at the GPT-4 level; performance on open-source models has not been evaluated.
- The objective score in the reward function depends on subjective LLM evaluation, which may be insufficiently accurate for certain problem types (e.g., nonlinear optimization).
- The six-element decomposition is manually designed and may require adjustment for scenarios beyond standard optimization problems (e.g., stochastic optimization, multi-objective optimization).
- SolverLLM is roughly on par with LLMOPT on simple datasets (−1% on MamoEasy), indicating that fine-tuning methods remain competitive within their training distribution; SolverLLM's advantage is primarily evident in out-of-distribution generalization.

## Related Work & Insights

- **vs. Chain-of-Experts**: Chain-of-Experts employs a fixed multi-agent workflow (interpreter → modeler → coder → reviewer) and relies on carefully engineered prompt templates. SolverLLM dynamically searches via MCTS rather than following a fixed pipeline, offering greater flexibility.
- **vs. LLMOPT**: LLMOPT excels within its training distribution via SFT and alignment, but is constrained by training data coverage. On complex OOD problems (MamoComplex, IndustryOR), SolverLLM's advantage is pronounced, suggesting that search-based reasoning generalizes better than memorization-based learning.
- **vs. AutoFormulation (another MCTS-based method)**: AutoFormulation applies standard MCTS over a four-element search space. SolverLLM enhances search efficiency and quality through dynamic expansion, prompt backpropagation, and uncertainty propagation, achieving higher SA with lower token consumption across all datasets.

## Rating

- Novelty: ⭐⭐⭐⭐ The MCTS+LLM combination is not novel per se, but the designs of dynamic expansion and reasoning signal backpropagation are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six datasets, diverse baselines, detailed ablation studies and case studies, and thorough token efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, figures are intuitive, and case studies effectively illustrate the role of each component.
- Value: ⭐⭐⭐⭐ Demonstrates the potential of test-time scaling in structured domains and offers practical guidance for real-world optimization problem solving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)
- [\[NeurIPS 2025\] Solving Inequality Proofs with Large Language Models](solving_inequality_proofs_with_large_language_models.md)
- [\[NeurIPS 2025\] EvoRefuse: Evaluating and Mitigating LLM Over-Refusal via Evolutionary Prompt Optimization](evorefuse_evolutionary_prompt_optimization_for_evaluation_and_mitigation_of_llm_.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](scaling_up_active_testing_to_large_language_models.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](../../ICCV2025/llm_nlp/fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)

</div>

<!-- RELATED:END -->
