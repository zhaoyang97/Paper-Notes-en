---
title: >-
  [Paper Note] Execution-Grounded Credit Assignment for GRPO in Code Generation
description: >-
  [ICLR 2026 Workshop (SPOT)][Code Intelligence][GRPO] This paper proposes EGCA (Execution-Grounded Credit Assignment), which leverages execution traces to localize the earliest semantic deviation in a program and concentr…
tags:
  - "ICLR 2026 Workshop (SPOT)"
  - "Code Intelligence"
  - "GRPO"
  - "code generation"
  - "credit assignment"
  - "reinforcement learning"
  - "execution tracing"
  - "RLVR"
date: 2026-05-08
content_hash: 942e8d66635779ba
---

# Execution-Grounded Credit Assignment for GRPO in Code Generation

**Conference**: ICLR 2026 Workshop (SPOT)
**arXiv**: [2603.16158](https://arxiv.org/abs/2603.16158)  
**Code**: Not released  
**Area**: Code Intelligence
**Keywords**: GRPO, code generation, credit assignment, reinforcement learning, execution tracing, RLVR

## TL;DR

This paper proposes EGCA (Execution-Grounded Credit Assignment), which leverages execution traces to localize the earliest semantic deviation in a program and concentrates GRPO gradients on the causal token span, addressing the coarse-grained credit assignment problem in code generation. EGCA achieves 82.1% pass@1 on HumanEval.

## Background & Motivation

As code generation models improve, modern models increasingly produce programs that are **syntactically correct, structurally sound, and fully executable**, yet fail unit tests due to subtle semantic errors. Conventional reinforcement learning from verifiable rewards (RLVR) approaches such as GRPO use unit tests as reward signals, but this signal is **temporally coarse-grained**—it applies to the entire program rather than to the specific decisions that cause failure.

GRPO's group-based policy gradient distributes reward signals uniformly across all tokens, causing "approximately correct" solutions to receive overly diffuse gradients that cannot rectify local reasoning errors. The central thesis of this paper is: **once a model can reliably produce executable, well-structured programs, credit assignment—rather than reward sparsity—becomes the primary bottleneck of critic-free RL for code generation.**

Limitations of prior work:
- **RLTF**: Enriches execution feedback but cannot localize where failures occur.
- **StepCoder**: Masks unexecuted tokens, but when a program executes in full, all tokens are executed and cannot be distinguished.
- **TEMPO/P2T**: Token-level updates based on textual branching points, but textual divergence does not necessarily correspond to the causal location of semantic failure.
- **CodeRL+**: Adds an execution-semantics alignment auxiliary objective, but departs from the critic-free paradigm.

## Method

### Overall Architecture

EGCA is a plug-and-play modification to GRPO that introduces no critic, auxiliary loss, or learned verifier. The pipeline:

1. Extract **algorithmic constraints** from a canonical reference solution.
2. Sample and execute a set of programs.
3. **Route each sample to one of four failure modes** via a deterministic gate.
4. **Localize the earliest execution deviation** for logic-mode candidates.
5. Concentrate GRPO advantages onto the causal token span and mask downstream tokens.

### Key Designs

**Canonical Reference Solution**: Each problem has an offline-curated reference solution $y^{\text{ref}}$, used not as an imitation target but solely to extract constraints, define reference execution behavior, and anchor semantic comparisons.

**Constraint-Guided Sampling**: A debugger LLM extracts algorithmic constraints $\mathcal{C} = \{c_1, \ldots, c_M\}$ from $(x, y^{\text{ref}})$ (permitted data structures, control-flow patterns, complexity targets, etc.) and injects them into the sampling prompt to bias the model toward structurally comparable programs:

$$y_i \sim \pi_\theta(\cdot \mid x \| \mathcal{C})$$

**Comparability Gate**: The candidate and reference solution are parsed into ASTs, normalized CFGs are constructed, structural similarity is computed, and a binary indicator $\mathbb{I}_{\text{cmp}}(y) \in \{0, 1\}$ is output to determine whether the candidate is "comparable" to the reference solution.

**Deterministic Classification into Four Failure Modes**:

$$m(y) = \begin{cases} \text{syntax} & \text{compilation/runtime error} \\ \text{constraint} & \mathbb{I}_\mathcal{C}(y)=0 \lor \mathbb{I}_{\text{cmp}}(y)=0 \\ \text{correct} & \hat{R}(y)=1 \land \mathbb{I}_\mathcal{C}(y)=1 \\ \text{logic} & \text{otherwise (executable, constraint-satisfying, but test-failing)} \end{cases}$$

**Token-Level Advantage Operator**:

$$a_{i,t} = \begin{cases} A_i / T_i & m(y_i) = \text{correct or constraint (uniform distribution)} \\ \frac{A_i}{|\mathcal{T}_{\text{err}}|} \mathbf{1}[t \in \mathcal{T}_{\text{err}}] & m(y_i) = \text{syntax (compiler-localized)} \\ \frac{A_i}{|\mathcal{T}_{k^*}|} \mathbf{1}[t \in \mathcal{T}_{k^*}] & m(y_i) = \text{logic (execution-deviation-localized)} \end{cases}$$

Key normalization guarantee: $\sum_{t=1}^{T_i} a_{i,t} = A_i$; total advantage is preserved and only redistributed onto the causal span.

### Execution Deviation Localization

For logic-mode candidates, the candidate and reference solution are executed in parallel on the first failing unit-test input $d$, yielding state traces:

$$\tau(y_i, d) = (S_1, \ldots, S_K), \quad \tau(y^{\text{ref}}, d) = (S_1^{\text{ref}}, \ldots, S_K^{\text{ref}})$$

The earliest semantic deviation boundary is: $k^* = \min\{k : S_k \neq S_k^{\text{ref}}\}$

The debugger LLM localizes $k^*$ on aligned structures and paired traces, then maps it to the token span $\mathcal{T}_{k^*}$. The debugger does not serve as a correctness judge; it only localizes the deviation.

### Loss & Training

Final objective:

$$\mathcal{L}(\theta) = -\sum_{i=1}^{G} \sum_{t=1}^{T_i} a_{i,t} \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})$$

No teacher gradients, auxiliary losses, or imitation terms are introduced. Training is based on DeepSeek-Coder-Instruct-6.7B with $G=16$, AdamW lr $= 5 \times 10^{-7}$, $\beta=0.05$, $\varepsilon=0.2$, 8×A100 80GB, 3 epochs.

## Key Experimental Results

### Main Results

| Method | HumanEval (pass@1) | MBPP (pass@1) |
|---|---|---|
| DeepSeek-Coder 6.7B base | 78.6 | 65.4 |
| SFT | 71.9 | 60.3 |
| Vanilla PPO | 78.0 | 65.6 |
| GRPO | 79.0 | 67.4 |
| RLTF | 77.9 | 64.5 |
| StepCoder-mask | 78.7 | 67.0 |
| CodeRL+ | 81.6 | 67.4 |
| **EGCA (Ours)** | **82.1** | **68.9** |

EGCA achieves gains of +3.1/+1.5 over GRPO, +3.4/+1.9 over StepCoder, and +0.5/+1.5 over CodeRL+, with only an 18% increase in wall-clock overhead.

### Ablation Study

**Ruling Out Teacher Leakage — Debugger Scale Ablation**:

| Debugger Model | Self pass@1 | Student HumanEval | Student MBPP |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | 70.7 | 78.9 | 66.1 |
| Qwen2.5-Coder-7B | 84.8 | 82.1 | 68.9 |
| Claude 4.5 Sonnet | 83.7 | — | 67.8 |

The student trained with the 1.5B debugger **surpasses the debugger itself by +8.2 points**, ruling out knowledge distillation. Scaling from 7B to Sonnet 4.5 yields only an additional +1.6 gain, indicating saturation in debugger capability benefit.

**Distillation Control**:

| Method | HumanEval | MBPP |
|---|---|---|
| Teacher SFT | 60.9 | 58.1 |
| Teacher-critique RL | 76.3 | 66.1 |
| **EGCA** | **82.1** | **68.9** |

### Key Findings

1. **Credit assignment is the bottleneck, not reward sparsity**: Random or late-deviation localization degrades to the uniform baseline.
2. **Soft mask monotonically degrades performance**: Confirming that binary masking outperforms gradual weighting.
3. Approximately **35% of training samples enter LOGIC mode** and are processed by EGCA; the remaining 65% use standard updates.
4. **Stage-dependent**: The method targets "approximately correct" scenarios; localization triggers less frequently under weak initialization.

## Highlights & Insights

1. **Incisive core insight**: "For approximately correct code, knowing *where* the error occurs is more valuable than knowing that an error occurred."
2. **Zero additional learned components**: No critic, auxiliary loss, or learned verifier is introduced; only the quality distribution of gradients is modified.
3. **Elegant integration of execution semantics and RL**: Runtime semantic information is injected via deterministic gates and execution traces.
4. **Rigorous experimental design**: A three-way teacher-leakage control experiment convincingly rules out the distillation hypothesis.
5. **Plug-and-play**: Only 18% overhead; applicable as a late-stage refinement technique for any GRPO training pipeline.

## Limitations & Future Work

1. **Dependency on reference solutions**: Restricts applicability to competitive programming and function synthesis with tests; open-ended generation is unsupported.
2. **Structural comparison limitations**: Valid solutions with structurally distinct implementations may be excluded by the comparability gate.
3. **Insufficient scale validation**: Only a 6.7B policy is evaluated; extension to larger models and multi-file generation remains unexplored.
4. **Workshop paper**: Experimental scale and benchmark coverage are relatively limited.
5. Extending execution deviation localization to other verifiable tasks such as mathematical reasoning warrants exploration.

## Related Work & Insights

- **StepCoder** (Dou et al., 2024): Masks unexecuted tokens; the closest prior work, but cannot handle programs that execute in full.
- **TEMPO/P2T** (Tran et al., 2025): Derives token-level updates from textual prefix trees, but textual divergence ≠ semantic divergence.
- **CodeRL+** (Jiang et al., 2025): Adds an execution-semantics alignment auxiliary objective but departs from the critic-free paradigm.
- Insight: Execution traces as credit assignment signals are generalizable to other domains (mathematical step verification, logical reasoning chains).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of execution tracing and credit assignment is novel; the four-failure-mode classification is elegantly designed.
- **Technical Depth**: ⭐⭐⭐⭐ — The normalized credit assignment operator carries theoretical elegance.
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited to a 6.7B model and two benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear; method description is precise.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play with reasonable overhead.
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](../../ACL2026/code_intelligence/coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](../../ACL2026/code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)

</div>

<!-- RELATED:END -->
