---
title: >-
  [Paper Note] FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving
description: >-
  [ACL 2025][Reasoning][Deliberate Reasoning] Proposes FineReason—a logical puzzle-based reasoning benchmark that performs atomic-level evaluation of LLMs' deliberate reasoning capabilities (reflection, backtracking, error correction) through two tasks: "state checking" (determining if the current state is solvable) and "state transition" (deciding the next action). It demonstrates that training on puzzle data can transfer to improve mathematical reasoning performance (GSM8K im…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Deliberate Reasoning"
  - "Logical Puzzles"
  - "State Checking"
  - "State Transition"
  - "Backtracking"
  - "Intermediate Step Verification"
date: 2026-05-08
content_hash: 5524d64ff2acf709
---

# FineReason: Evaluating and Improving LLMs' Deliberate Reasoning through Reflective Puzzle Solving

**Conference**: ACL 2025  
**arXiv**: [2502.20238](https://arxiv.org/abs/2502.20238)  
**Code**: [https://github.com/DAMO-NLP-SG/FineReason](https://github.com/DAMO-NLP-SG/FineReason)  
**Authors**: Guizhen Chen, Weiwen Xu, Hao Zhang, Hou Pong Chan, Chaoqun Liu, Lidong Bing, Deli Zhao, Anh Tuan Luu, Yu Rong  
**Institutions**: Nanyang Technological University, DAMO Academy (Alibaba), Hupan Lab  
**Area**: LLM Reasoning  
**Keywords**: Deliberate Reasoning, Logical Puzzles, State Checking, State Transition, Backtracking, Intermediate Step Verification

## TL;DR

Proposes FineReason—a logical puzzle-based reasoning benchmark that performs atomic-level evaluation of LLMs' deliberate reasoning capabilities (reflection, backtracking, error correction) through two tasks: "state checking" (determining if the current state is solvable) and "state transition" (deciding the next action). It demonstrates that training on puzzle data can transfer to improve mathematical reasoning performance (GSM8K improved by 5.1%).

## Background & Motivation

**Background**: LLM reasoning capabilities are transitioning from System 1 (fast intuition) to System 2 (slow analysis). Reasoning models like OpenAI-o1 and DeepSeek-R1 have demonstrated strong reasoning capabilities through iterative reflection and correction.

**Limitations of Prior Work**:
   - Existing reasoning benchmarks (MATH, GSM8K, HumanEval) **focus only on final answer accuracy**, failing to evaluate the intermediate reasoning process.
   - Models might arrive at the correct conclusion via flawed reasoning (Zelikman et al., 2022; Lightman et al., 2024).
   - Models might "cheat" using surface patterns in training data (Roelofs et al., 2019).
   - They cannot distinguish whether the model is performing "genuine reasoning" or "pattern matching."

**Core Motivation**: There is a need for a benchmark that can evaluate every step of the reasoning process, specifically the capabilities of **reflection** (checking the current state) and **error correction** (backtracking to the correct path). Logical puzzles are naturally suited for this purpose—each operation can be decomposed into atomic steps, and clear rules allow for automatic validation.

## Method

### Overall Architecture

FineReason includes four types of logical puzzles, two evaluation tasks, and a training set:

**Four types of puzzles**:

| Puzzle | State Definition | Minimum Operation | Data Source |
|------|---------|---------|---------|
| Sudoku | Partial/complete 9×9 board | Add/remove a digit | Kaggle dataset |
| Graph Coloring | Partially/fully colored graph | Color/de-color a vertex | Random graph generation + backtracking algorithm |
| Game of 24 | Partial/complete arithmetic expression | Perform/undo an operation on two numbers | Yao et al. (2023) |
| Grid Puzzles | Partial/complete grid | Assign/remove an attribute based on clues | Tyagi et al. (2024) |

### Key Designs

#### Key Design 1: Tree-Based Puzzle Decomposition

- Represents the puzzle-solving process as a **search tree**: nodes are intermediate states, edges are state transitions.
- Depth-first search (DFS) is performed starting from the initial state $s_1$, executing only the minimum operation at each step.
- Edges are bidirectional—supporting forward exploration and backtracking.
- Converts rules into executable code to automatically validate the legitimacy of each state.
- For logic grid puzzles, defines three auxiliary functions $r(v)$, $c(v)$, $T(i,j)$ to encode text clues into verifiable constraints.

#### Key Design 2: Two Evaluation Tasks

**State Checking**:
- Given a current state $s_i$, determine if a reachable solution $s_n$ exists.
- Uniformly sample solvable and unsolvable states from the tree.
- Evaluation on two levels: (1) check if existing steps violate rules (retrospective), (2) anticipate whether future states will lead to dead ends (prospective).

**State Transition**:
- Given the current state and the state checking result, decide the next action.
- Solvable state $\rightarrow$ explore an unvisited sub-state.
- Unsolvable state $\rightarrow$ backtrack to the parent state.
- Ground-truth state checking labels are provided during evaluation to eliminate interference from state-checking errors.
- Some unsolvable sub-states are provided to test whether the model can effectively avoid them.

### Training Data

Constructs a puzzle training set containing state checking and state transition data to enhance general reasoning capabilities.

## Key Experimental Results

### Experimental Setup

- Test instances: 500 intermediate states per puzzle type $\times$ 4 types = 2000 instances/task
- Evaluation method: 0-shot CoT prompting, explicitly forbidding programmatic solvers
- Models: Reasoning models (o1, Gemini-2.0-Flash-Thinking) + General models (GPT-4o, GPT-3.5, Gemini-2.0-Flash, Qwen2.5-72B-Inst)

### End-to-End Puzzle Solving Accuracy

| Puzzle | GPT-4o | Gemini-F | Gemini-FT | o1 |
|------|--------|----------|-----------|-----|
| Sudoku | 0 | 5.9 | 0 | 0 |
| Graph Coloring | 7.8 | 35.3 | 80.4 | 78.4 |
| Game of 24 | 15.3 | 83.7 | 48.0 | 54.1 |
| Grid Puzzles | 2.2 | 10.9 | 34.8 | 45.7 |

End-to-end results are inconsistent—Gemini-F outperforms Gemini-FT on Sudoku and Game of 24, but lags significantly on other puzzles, indicating that end-to-end accuracy is insufficient to reliably evaluate reasoning capabilities.

### State Checking + State Transition Results (Core Results)

| Puzzle | Model | State Checking | State Transition | Average |
|------|------|---------|---------|------|
| Sudoku | Random | 50.0 | - | - |
| | GPT-4o | 52.4 | 38.8 | 45.6 |
| | Gemini-FT | 69.2 | 48.8 | 59.0 |
| | **o1** | **81.0** | **70.2** | **75.6** |
| Graph Coloring | GPT-4o | 56.4 | 49.4 | 52.9 |
| | Gemini-FT | 92.6 | 46.4 | 69.5 |
| | **o1** | **94.6** | **65.0** | **79.8** |
| Game of 24 | GPT-4o | 82.6 | 23.0 | 52.8 |
| | Gemini-FT | 96.0 | 48.6 | 72.3 |
| | **o1** | **97.4** | **86.6** | **92.0** |
| Grid Puzzles | GPT-4o | 52.4 | 10.0 | 31.2 |
| | Gemini-FT | 89.0 | 51.4 | 70.2 |
| | **o1** | **88.8** | **77.6** | **83.2** |

**Key Findings**:
- There is a significant **19.7%** gap between o1 and Gemini-FT, whereas the gap between them is small on other mathematical/code benchmarks.
- General models (e.g., GPT-4o) are close to random guessing for state checking in Sudoku and logic grid puzzles.
- Gemini-FT is close to o1 on state checking but lags significantly on state transition—revealing its weakness in error-correction capability.

### Model Behavior Analysis

**State Checking Precision/Recall** (with unsolvable states as positive instances):
- General models show extremely low recall in deep-tree puzzles (Sudoku, Graph Coloring), tending to be "overly optimistic"—defaulting to solvable when facing problems beyond their capability.
- GPT-4o and Qwen2.5 have high precision but low recall—extremely conservative, predicting unsolvable only when highly confident.
- Reasoning models perform well on both metrics.

### Training Transfer Effects

Experiments on DeepSeek-R1-Distill-Qwen-7B:

| Training Data | GSM8K | MATH-500 |
|---------|-------|----------|
| Math Data Only | 82.3% | - |
| Math + **Puzzle Data**| **87.4%** | Gain |

Puzzle data brings a **5.1%** improvement on GSM8K, proving that skills like backtracking and constraint validation can transfer from puzzles to general reasoning.

## Highlights & Insights

1. **Evaluation Paradigm Innovation**: Moving from "is the final answer correct" to "is every intermediate step correct"—representing the right direction for evaluating reasoning capabilities.
2. **Logical Puzzles = Perfect Evaluation Vehicle**: Clear rules, atomically decomposable steps, and automatically verifiable.
3. **Revealing the "Over-Optimism" Tendency of General Models**: Models like GPT-4o default to predicting states as solvable when facing difficult situations, and never backtrack.
4. **Much Higher Discriminative Power than Existing Benchmarks**: The o1 vs. Gemini-FT gap is 19.7%, whereas their performance on benchmarks like MATH is close to saturation.
5. **Training Transfer Effectiveness** proves that puzzle training can serve as a booster for general reasoning capabilities—analogous to how chess training improves human strategic thinking.

## Limitations & Future Work

1. The four puzzle types are all combinatorial optimization/constraint satisfaction problems, which might not fully represent all reasoning types (e.g., causal reasoning, analogical reasoning).
2. The "forbid programming" instruction relies on model instruction following; in practice, models might still implicitly leverage algorithms in their memory.
3. Training transfer experiments were only validated on a 7B model; the effect on larger/smaller models remains unknown.
4. Converting textual clues to code for logical grid puzzles relies on GPT-4o's one-shot translation + manual verification, limiting scalability.
5. Performance-stratified analysis for puzzles of different difficulties (e.g., different board sizes, different graph densities) is not discussed.

## Related Work & Insights

- **Reasoning Benchmarks**: GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), HumanEval (Chen et al., 2021)
- **Reasoning Models**: OpenAI o1 (2024), DeepSeek-R1 (2025), Gemini-2.0-Flash-Thinking (Google, 2024)
- **Process Reward / Intermediate Step Evaluation**: PRM (Lightman et al., 2024)
- **LLM + Search**: Tree-of-Thoughts (Yao et al., 2023)

## Rating

⭐⭐⭐⭐⭐ (5/5)

This is an outstanding benchmark construction work. The problem definition is clear (from the final answer to the intermediate process), the vehicle choice is ingenious (atomically verifiable nature of logical puzzles), the experimental findings are profound (revealing models' over-optimism and error-correction deficiencies), and the training transfer experiments provide strong evidence that 'puzzle training improves reasoning'. It has significant reference value for understanding and enhancing LLM reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Executable Counterfactuals: Improving LLMs' Causal Reasoning Through Code](../../ICLR2026/llm_reasoning/executable_counterfactuals_improving_llms_causal_reasoning_through_code.md)
- [\[ACL 2025\] CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought](cot-uq_improving_response-wise_uncertainty_quantification_in_llms_with_chain-of-.md)
- [\[ACL 2025\] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions](improving_chain-of-thought_reasoning_via_quasi-symbolic_abstractions.md)
- [\[ACL 2025\] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](../../ICML2025/llm_reasoning/improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)

</div>

<!-- RELATED:END -->
