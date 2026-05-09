---
title: >-
  [Paper Note] Atom of Thoughts for Markov LLM Test-Time Scaling
description: >-
  [NeurIPS 2025][LLM Reasoning][test-time scaling] This paper proposes Atom of Thoughts (AoT), which models LLM reasoning as a Markov chain where each state is a self-contained subproblem that is answer-equivalent to the original question but of strictly lower complexity. A two-phase transition mechanism based on DAG decomposition and contraction eliminates historical dependencies. AoT integrates seamlessly with existing methods such as ToT and reflection, achieving state-of-the-art performance across six benchmarks spanning mathematics, code, and multi-hop QA.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - test-time scaling
  - Markov process
  - atomic reasoning
  - DAG decomposition
  - reasoning framework
date: 2026-05-08
content_hash: c1f1c3bd01728e90
---

# Atom of Thoughts for Markov LLM Test-Time Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2502.12018](https://arxiv.org/abs/2502.12018)
**Code**: Submitted with the paper (to be released)
**Area**: LLM Reasoning
**Keywords**: test-time scaling, Markov process, atomic reasoning, DAG decomposition, reasoning framework

## TL;DR

This paper proposes Atom of Thoughts (AoT), which models LLM reasoning as a Markov chain where each state is a self-contained subproblem that is answer-equivalent to the original question but of strictly lower complexity. A two-phase transition mechanism based on DAG decomposition and contraction eliminates historical dependencies. AoT integrates seamlessly with existing methods such as ToT and reflection, achieving state-of-the-art performance across six benchmarks spanning mathematics, code, and multi-hop QA.

## Background & Motivation

**Background**: Test-time scaling (TTS) is a central paradigm for enhancing LLM reasoning. Existing approaches include chain-of-thought (CoT), tree search (ToT), graph-of-thought (GoT), and reasoning models such as O1/R1. Their shared logic is to invest more computation at inference time in exchange for better outcomes.

**Limitations of Prior Work — Accumulation of Historical Dependencies**: All existing frameworks rely heavily on historical context. CoT must retain the full reasoning trajectory to generate the next step, causing prompt token counts to grow linearly with the number of steps. ToT maintains parent-child and sibling relationships for branching decisions. GoT introduces even more complex dependencies through arbitrary inter-node connections. Figure 1(b) of the paper quantifies the prompt and completion token counts per LLM call across methods, clearly illustrating this overhead.

**Key Challenge**: The essence of TTS is "more computation yields better results," yet historical dependencies cause diminishing marginal efficiency at each step — the more tokens spent on reasoning, the longer the historical context each subsequent step must process, reducing the proportion of computation genuinely devoted to the current problem.

**Key Insight**: Drawing on the memoryless property of Markov processes — if each reasoning state depends only on the immediately preceding state rather than the full history, computation can be fully directed toward the current problem. The key challenge is making each state *self-contained* without losing information necessary for solving the original problem.

**Core Idea**: Redefine reasoning chain nodes from "intermediate thinking steps" to "independent subproblems that are answer-equivalent to the original question but of lower complexity." State transitions are realized via DAG decomposition and contraction, and the reasoning chain naturally converges to an irreducible "atomic" structure.

## Method

### Overall Architecture

Given an input question $Q_0$, AoT generates a state sequence $Q_0 \to Q_1 \to \cdots \to Q_N$ via iterative Markov state transitions. Two constraints are enforced: (a) **answer equivalence** — the answer to each $Q_i$ is the answer to $Q_0$; (b) **monotonically decreasing complexity** — $Q_{i+1}$ is simpler than $Q_i$. Each transition consists of a decomposition phase and a contraction phase, and the terminal state $Q_N$ can be solved directly.

### Key Design 1: Markov State Definition

Traditional CoT is modeled as $A \sim p(A|\mathcal{T}, Q_0) \prod_{i=0}^{N} p(T_i | \mathcal{T}_{<i}, Q_0)$, where each thinking step $T_i$ depends on all prior steps $\mathcal{T}_{<i}$. Least-to-Most Prompting replaces nodes with subproblems $Q_i$ but retains historical dependency $p(Q_i|\mathcal{Q}_{<i})$.

AoT models reasoning as a Markov chain:

$$A \sim p(A|S_N) \prod_{i=0}^{N} p(S_{i+1}|S_i)$$

where each state $S_i$ is represented by a subproblem $Q_i$ that is self-contained and answer-equivalent to $Q_0$. The key insight is that answer equivalence combined with decreasing complexity naturally yields the Markov property: $Q_{i+1}$ itself encapsulates all information needed for solving the problem, obviating the need to consult prior history.

### Key Design 2: Two-Phase DAG Decomposition-Contraction Transition

Directly prompting an LLM to "simplify a question into an equivalent but easier one" is non-trivial. AoT addresses this with a two-phase mechanism:

**Decomposition Phase**: A temporary DAG $\mathcal{G}_i = (\mathcal{N}, E)$ is constructed, where $E \subseteq \{(N_j, N_k) \mid j < k\}$. Nodes $N_k$ represent subproblems or reasoning steps, and a directed edge $(N_j, N_k)$ indicates that $N_j$ provides necessary information for $N_k$. This phase explicitly models the dependency structure among reasoning steps.

**Contraction Phase**: Nodes with no incoming edges in the DAG are identified as independent subproblems that can be solved and discarded (their answers are absorbed into dependent nodes). The remaining dependent nodes are reorganized into a new equivalent question $Q_{i+1}$.

The complete Markov transition probability is:

$$A \sim p(A|Q_N) \prod_{i=0}^{N} p(Q_{i+1}|\mathcal{G}_i) \cdot p(\mathcal{G}_i|Q_i)$$

### Key Design 3: Quality-Aware Termination Strategy

The memoryless property of the Markov chain is a double-edged sword — while CoT can correct early errors via context, a low-quality transition in AoT may cause subsequent reasoning to diverge. A termination mechanism is therefore introduced:

- After each transition $Q_i \to Q_{i+1}$, an LLM-as-a-judge selects the best answer among three candidates: $\text{solve}(Q_i)$, $\text{solve}(\mathcal{G}_i)$, and $\text{solve}(Q_{i+1})$.
- If $Q_{i+1}$ is not selected, the chain is terminated and the current best answer is returned.
- This implicitly enforces answer equivalence: if $Q_{i+1}$ has drifted semantically from $Q_0$, its solution will not constitute a valid answer to $Q_0$ and will not be selected by the judge.

### Key Design 4: Modular Integration and Emergent Atomic Structure

**Modular Integration**: Since each Markov state $Q_i$ is a self-contained, answer-equivalent simplification of $Q_0$, any $Q_i$ can serve as the entry point for other reasoning methods (e.g., ToT, reflection). AoT thus acts as a structured preprocessor for existing methods rather than a replacement.

**Emergent Atomic Structure**: When the Markov chain is extended with tree search and reflection, the reasoning token count at deeper states gradually converges to a stable lower bound — approaching the token count of the independent subproblem set in the minimal DAG. This indicates that the problem has been decomposed to an irreducible "atomic" granularity. Atomicity is not pre-engineered but emerges naturally during reasoning — different problems converge at different depths, and the same problem exhibits different atomic granularities across models of varying capability.

### Loss & Training

AoT is a purely inference-time method requiring no additional training or fine-tuning. Decomposition and contraction phases are implemented via prompt templates. Default hyperparameters: temperature 1.0, maximum Markov chain length 3. Due to the termination mechanism, increasing chain length raises the performance ceiling without linearly increasing cost.

## Key Experimental Results

### Main Results: Non-Reasoning Models

| Model | Method | MATH | GSM8K | MBPP | LongBench |
|-------|--------|------|-------|------|-----------|
| GPT-4o-mini | CoT | 78.3 | 90.9 | 72.4 | 57.6 |
| GPT-4o-mini | CoT-SC | 81.8 | 92.0 | 73.2 | 58.6 |
| GPT-4o-mini | AFlow | 83.0 | 93.5 | 74.0 | 61.0 |
| GPT-4o-mini | FoT | 82.6 | 94.2 | 74.8 | 60.8 |
| GPT-4o-mini | **AoT** | **83.6** | **95.0** | **75.2** | **68.5** |
| DeepSeek-V3 | CoT | 94.4 | 96.2 | 75.7 | 58.8 |
| DeepSeek-V3 | AFlow | 96.1 | 97.8 | 77.3 | 63.5 |
| DeepSeek-V3 | **AoT** | **96.5** | **98.2** | **79.6** | **71.0** |

### Main Results: Reasoning Models

| Model | Method | AIME | LiveCodeBench | LongBench |
|-------|--------|------|---------------|-----------|
| O3-mini | CoT | 79.6 | 23.6 | 56.3 |
| O3-mini | AFlow | 82.5 | 26.5 | 58.0 |
| O3-mini | FoT | 81.8 | 27.8 | 57.9 |
| O3-mini | **AoT** | **83.0** | **32.2** | **65.3** |
| DeepSeek-R1 | CoT | 78.3 | 24.5 | 55.1 |
| DeepSeek-R1 | AFlow | 81.2 | 27.4 | 58.7 |
| DeepSeek-R1 | **AoT** | **81.7** | **30.9** | **67.9** |

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| No decomposition (direct contraction) | Skips DAG construction and attempts direct simplification | Significant performance degradation |
| No DAG-guided contraction | Constructs DAG but does not exploit its structure during contraction | Worse than no decomposition; partial structural cues introduce interference |
| Full AoT | DAG decomposition + structure-guided contraction | Optimal |

DAG quality metrics: answer equivalence preservation rate >99% (all datasets); complexity reduction rate 74–82%.

### Integration Experiments (Scaling Up)

| Integration | MATH | AIME |
|-------------|------|------|
| ToT alone | 82.0 | 78.0 |
| FoT alone | 82.6 | -- |
| AoT state as ToT entry point | Performance gain + reduced computation cost | -- |
| AoT + ToT + Reflection (full) | **84.9** | **81.2** |

### Key Findings

1. **Largest gains on LongBench**: AoT outperforms AFlow by 7.5 points on GPT-4o-mini (68.5 vs. 61.0) and by 7.3 points on O3-mini (65.3 vs. 58.0). Multi-hop QA tasks exhibit the heaviest historical dependency and thus benefit most from Markovization.
2. **Sustained performance improvement without degradation**: Figure 3 demonstrates that AoT performance increases steadily as the number of reasoning iterations (chain length) grows, confirming that the termination strategy effectively prevents error propagation.
3. **Atomic convergence phenomenon**: Reasoning token counts in deeper Markov chain states naturally approach a lower bound corresponding to the size of the independent subproblem set in the minimal DAG, confirming that atomic structure emerges rather than being predefined.
4. **Win-win from state integration**: Using the intermediate state $Q_1$ from AoT as the entry point for FoT improves accuracy while reducing computational cost, since the simplified problem requires less search effort.
5. **Benefits extend to reasoning models**: AoT yields significant gains not only for non-reasoning models such as GPT-4o-mini but also for native reasoning models such as O3-mini and DeepSeek-R1, indicating that Markovization is orthogonal to intrinsic reasoning capability.

## Highlights & Insights

- **Novel perspective**: This is the first work to systematically introduce the memoryless property of Markov processes into general-purpose LLM reasoning — not as a loose analogy but as a rigorous probabilistic formulation (the derivation from Equations 1 to 3 is clearly presented).
- **Elegant DAG decomposition-contraction design**: The vague objective of "simplifying a problem" is decomposed into two structured operations, reducing demands on the LLM and making state transitions reliable.
- **Emergence of atomicity**: Rather than manually specifying atomic granularity, the reasoning process is allowed to converge naturally to irreducible states, which is more principled than hard-coding decomposition depth.
- **Plug-and-play architecture**: AoT can serve as a preprocessing module for any reasoning framework without modifying downstream methods; this orthogonality confers high practical value.
- **Asymmetric gains on multi-hop QA**: The large improvements on LongBench (+7 to +12 points) suggest that historical dependency is a primary bottleneck in multi-hop reasoning.

## Limitations & Future Work

1. **Ceiling of a purely inference-time method**: AoT currently relies entirely on prompt engineering. Internalizing the Markov reasoning pattern via SFT or RL (e.g., training on decomposition-contraction trajectories) may yield larger gains.
2. **DAG construction quality depends on model capability**: The decomposition phase requires the LLM to correctly identify inter-subproblem dependencies, which may be unreliable for weaker models.
3. **Conservative termination strategy**: LLM-as-a-judge may terminate promising chains prematurely, limiting the frequency with which deep atomic structures emerge.
4. **Maximum chain length of 3**: The main experiments fix the maximum chain length at 3; whether longer chains continue to yield benefits remains underexplored.
5. **Limited gains on code tasks**: Improvements on code generation benchmarks such as MBPP are relatively modest (+1 to +2 points), possibly because the nature of historical dependency in code tasks differs from that in natural language reasoning.

## Related Work & Insights

- **Least-to-Most Prompting**: AoT shares the spirit of subproblem sequencing with Least-to-Most Prompting, but the critical distinction is that AoT requires each state to be answer-equivalent to the original question (rather than addressing different subproblems), which is what enables the Markov property.
- **LLMs as Markov Chains (Zekri et al., 2024)**: This work models LLM token generation as a Markov chain from a theoretical standpoint; AoT exploits the Markov property at the higher "question" level — the two perspectives are complementary.
- **AFlow (Zhang et al., 2024)**: AFlow automatically searches for optimal agentic workflows; AoT surpasses it on multiple benchmarks, demonstrating that hand-crafted structural priors (Markovization) still offer value.
- **Forest-of-Thought (Bi et al., 2024)**: FoT adopts a parallel multi-tree reasoning structure; using AoT states as FoT entry points achieves a win-win (higher performance and lower cost), highlighting their complementarity.
- **Insight**: This work suggests that the essence of reasoning is the progressive simplification of a problem until it is irreducible — an idea closely aligned with recursive and divide-and-conquer paradigms in programming. Future work could explore more formal complexity measures to guide the decomposition strategy.

## Rating

- **Novelty**: 5/5 — The Markov modeling perspective combined with emergent atomicity is highly original; the probabilistic reformulation of reasoning structure is well-grounded in theory.
- **Experimental Thoroughness**: 4/5 — Comprehensive coverage across four models and six benchmarks, with cleverly designed integration experiments; however, ablations are conducted only on GPT-4o-mini and are absent for other models.
- **Writing Quality**: 5/5 — The mathematical derivations are self-consistent, figures are informative (Figure 1's token allocation comparison is particularly clear), and concepts are well-explained.
- **Value**: 4/5 — The plug-and-play design is practically valuable and the theoretical insights are substantive, though the performance gains of a purely inference-time method may be limited on stronger reasoning models.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] TTS-VAR: A Test-Time Scaling Framework for Visual Auto-Regressive Generation](tts-var_a_test-time_scaling_framework_for_visual_auto-regressive_generation.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)

<!-- RELATED:END -->
