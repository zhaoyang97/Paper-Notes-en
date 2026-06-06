---
title: >-
  [Paper Note] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization
description: >-
  [ACL 2026][Code Intelligence][Automated Simulator Construction] Ours proposes SOCIA-EVO, an LLM agent framework that redefines automated simulator construction as a dual-anchored evolution process. By anchoring empirical…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Automated Simulator Construction"
  - "Dual-Anchored Evolution"
  - "Bi-Level Optimization"
  - "Strategy Playbook"
  - "Distributional Fidelity"
date: 2026-05-08
content_hash: 277115e16bbb0e76
---

# SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization

**Conference**: ACL 2026  
**arXiv**: [2604.17351](https://arxiv.org/abs/2604.17351)  
**Code**: [https://github.com/cruiseresearchgroup/SOCIA/tree/evo](https://github.com/cruiseresearchgroup/SOCIA/tree/evo)  
**Area**: Code Intelligence  
**Keywords**: Automated Simulator Construction, Dual-Anchored Evolution, Bi-Level Optimization, Strategy Playbook, Distributional Fidelity

## TL;DR

Ours proposes SOCIA-EVO, an LLM agent framework that redefines automated simulator construction as a dual-anchored evolution process. By anchoring empirical constraints with a static Blueprint, decoupling structural revision and parameter calibration via bi-level optimization, and managing repair hypotheses with a self-curated strategy Playbook using Bayesian-weighted retrieval from execution feedback, SOCIA-EVO significantly outperforms baselines such as Reflexion and G-SIM on three simulation tasks: user modeling, mask-wearing diffusion, and personal mobility.

## Background & Motivation

**Background**: Automatically constructing simulators from observational data (data-driven simulation) is a cornerstone for understanding complex systems. Unlike general software engineering where functional correctness suffices, simulator construction is essentially a scientific modeling task requiring distributional fidelity—generated programs must reproduce the statistical patterns, causal mechanisms, and emergent behaviors of real-world data.

**Limitations of Prior Work**: Standard LLM agents encounter two critical failure modes in long-cycle simulator construction: (1) Context drift—as simulator complexity increases, constraints established in initial data analysis lose salience, leading agents to hallucinate mechanisms non-existent in the data; (2) Unstable optimization—agents confuse structural errors (e.g., incorrect transition logic) with parameter mismatches (e.g., suboptimal rates), often rewriting correct logic when a simple parameter tune would suffice, leading to "Whac-A-Mole" oscillations.

**Key Challenge**: LLMs excel at discrete logical reasoning but struggle with high-dimensional continuous parameter search. Furthermore, they lack persistent verification mechanisms for repair strategies—previously attempted repairs and their quantitative outcomes are lost as the context window advances, causing repeated proposals of plausible but empirically invalid fixes.

**Goal**: Design an agent framework capable of maintaining long-term consistency, distinguishing structural from parametric issues, and accumulating effective repair experience.

**Key Insight**: Introduce a dual-anchor mechanism—a static Blueprint to prevent context drift and a dynamic Playbook to accumulate and verify repair hypotheses. Bi-level optimization strictly decouples structural modification (LLM-driven outer loop) from parameter calibration (numerical optimizer-driven inner loop).

**Core Idea**: Simulator construction = Blueprint-anchored search space + Bi-level optimization for structural/parametric decoupling + Self-curated Playbook for repair hypotheses.

## Method

### Overall Architecture

SOCIA-EVO adopts a closed-loop iterative workflow collaborated by six specialized agents. The process begins with a Data Analysis Agent generating a static Blueprint $\mathcal{B}$, followed by an evolution loop: a Code Generation Agent produces simulator code $P_t$ and a parameter calibrator $C_t$ based on $\mathcal{B}$ and current Playbook strategies $\mathcal{K}$; an Execution Agent runs the inner-loop optimization $\theta$ and retrieves metrics $\mathcal{M}_t$; a Feedback Agent diagnoses deviations; a Playbook Manager updates the strategy library; and an Iteration Control Agent evaluates convergence.

### Key Designs

1.  **Dual-Anchor Mechanism (Static Blueprint + Dynamic Playbook)**:

    - **Function**: Prevents context drift and accumulates effective repair experience.
    - **Mechanism**: The Blueprint $\mathcal{B}$ is an immutable, authoritative specification that strictly defines simulation topology, agent schemas, and evaluation metrics, locked after human verification. The Playbook $\mathcal{K}$ is a dynamic library of repair strategies, where each strategy $S_i = \langle R_i, I_i, \Sigma_i \rangle$ contains diagnostic content (with metric-bound sets $\Lambda_i$), metadata (usage/success/failure counts), and lifecycle stages. Strategies undergo transitions (Open→Queued→InProgress→Resolved), verified as successful ($\Delta\mathcal{M}/\mathcal{M}_t > \tau$), falsified ($< -\tau$), or uncertain based on metric delta thresholds.
    - **Design Motivation**: The Blueprint provides immutable constraints to prevent the generation process from drifting away from ground truth; the Playbook treats repair strategies as hypotheses rather than trusted fixes, using execution evidence for self-curated verification/falsification.

2.  **Bi-Level Optimization**:

    - **Function**: Decouples structural revision (discrete search) and parameter calibration (continuous optimization).
    - **Mechanism**: The outer loop involves the Code Generation Agent searching over the program space to generate structure $P_t$ and calibrator $C_t$: $(P_t, C_t) \leftarrow \pi_{code}(P_{t-1}, \mathcal{B}, \text{Knapsack}(\mathcal{K}))$. The inner loop executes calibrator $C_t$ using numerical methods such as Bayesian Optimization to solve for optimal parameters $\theta_t^* = \arg\min_\theta \mathcal{L}(\text{Sim}(P_t, \theta), \mathcal{D}_{obs})$. Crucially, the LLM does not guess parameter values but generates the optimization program to find them.
    - **Design Motivation**: Ensures that metrics observed by the Feedback Agent represent the intrinsic capability of structure $P_t$ at its parameter optimum, filtering out noise from untuned parameters and enabling precise attribution of flaws to structural logic.

3.  **Knapsack-Based Context Engineering**:

    - **Function**: Selects the most valuable repair strategies within a limited context budget.
    - **Mechanism**: Solves a 0-1 knapsack problem from the Open/Queued pools: $\max \sum_i v_i x_i$, subject to $\sum_i c_i x_i \leq L_{budget}$. Strategy value $v_i = w_{sev} \cdot U_i^{queue} \cdot \Phi_{rel}(S_i)$, where $w_{sev}$ is severity weight, $U_i^{queue}$ is a backlog reward to prevent starvation, and $\Phi_{rel}$ is a Bayesian reliability estimate based on a Beta-Bernoulli model $\Phi_{rel}(S_i) = (s_i+1)/(s_i+f_i+2)$. Prompts use a three-zone layout (System/Background/Instruction) to counter "Lost in the Middle" effects.
    - **Design Motivation**: Inserting all strategies dilutes attention (experiments show degradation comparable to memoryless setups at 3200 tokens); Bayesian reliability allows empirically falsified strategies to decay naturally.

### Loss & Training

SOCIA-EVO does not involve model training; instead, it optimizes the simulator through iterative evolution. The inner loop utilizes Optuna Bayesian Optimization or random calibrators. Outer-loop iteration control stops at improvement plateaus or regressions to prevent over-correction. All experiments use GPT-5.1 as the backbone, with results averaged over 5 random seeds with 95% confidence intervals.

## Key Experimental Results

### Main Results

**Performance Comparison across Three Simulation Tasks (Mean ± 95% CI, lower is better)**

| Method | User Modeling MAE↓ | Mask Sim RMSE↓ | Mobility N→A WD↓ |
|------|-------------|-------------|------------|
| Reflexion | 0.17±0.01 | 0.26±0.02 | 0.69±0.02 |
| YuLan-OneSim | 0.21±0.02 | 0.16±0.01 | 0.64±0.01 |
| G-SIM-SBI | 0.19±0.01 | 0.11±0.02 | 0.56±0.02 |
| ACE-OL | 0.14±0.02 | 0.24±0.01 | 0.61±0.02 |
| **SOCIA-EVO** | **0.11±0.01** | **0.07±0.01** | **0.53±0.02** |

### Ablation Study

| Configuration | ΔMAE↓ | ΔRMSE↓ | ΔWD↓ |
|------|-------|--------|------|
| SOCIA-EVO (Full) | — | — | — |
| w/o Inner-loop Calibration | +0.25 | +0.47 | +0.29 |
| w/o Blueprint | +0.20 | +0.37 | +0.23 |
| w/o HITL | +0.18 | +0.34 | +0.21 |
| w/o Memory Mechanism | +0.14 | +0.30 | +0.20 |
| w/o Strategy Playbook | +0.10 | +0.23 | +0.15 |
| Context Window +2200 tokens | +0.12 | +0.14 | +0.13 |

### Key Findings

- Inner-loop parameter calibration is the most critical component—removing it causes RMSE to spike by +0.47, as parameter updates degrade into LLM heuristic guessing.
- Excessive context windows (+2200 tokens to fit the entire Playbook) lead to performance degradation near memoryless levels, verifying the "attention dilution effect."
- Cumulative repeated errors decreased by 76% by the fifth iteration, proving that the value-based Knapsack mechanism effectively suppresses "Whac-A-Mole" behavior.
- Open-source backbones (Llama-3.3-70B, Qwen3-80B) achieve competitive performance, demonstrating the framework's independence from specific proprietary models.

## Highlights & Insights

- Treating repair strategies as hypotheses to be verified/falsified rather than trusted corrections introduces a "scientific methodology" mindset that is novel in LLM agent design—transferable to any agent system requiring long-term memory management.
- The core insight of bi-level optimization is profound: leveraging LLMs to generate programs that find parameters rather than guessing the values directly exploits LLM strengths in code generation while bypassing their weaknesses in continuous optimization.
- Knapsack + Bayesian reliability context engineering provides a general solution for maximizing information value within limited windows.

## Limitations & Future Work

- The strongest results rely on commercial LLM backbones (GPT-5.1); while open-source models are feasible, a performance gap remains.
- Currently focuses only on tasks expressible via iterative structural revision and bounded parameter calibration; more complex settings (long-term planning, strategic multi-agent interaction) may require stronger reasoning modules.
- Blueprint HITL verification is lightweight but introduces human dependency, and Blueprint quality directly impacts all subsequent iterations.
- Strategy matching currently uses thresholded text matching; finer-grained semantic matching might improve the quality of strategy merging.

## Related Work & Insights

- **vs Reflexion**: Reflexion uses episodic memory for verbal reinforcement learning but lacks metric-driven repair verification, leading to high RMSE (0.26 vs. 0.07 in mask sim).
- **vs G-SIM**: G-SIM effectively separates structure generation and parameter estimation but lacks long-term evidence tracking, leading to repeated exploration of falsified strategies.
- **vs Dynamic Cheatsheet**: DC accumulates successful strategies but may introduce negative transfer from irrelevant contexts.
- **vs YuLan-OneSim**: Generates behavior in predefined environments rather than inferring simulator logic from data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formalizes simulator construction as a dual-anchored evolution process; strategy hypothesis verification/falsification is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks, six baselines, detailed ablations, convergence analysis, and backbone portability verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous problem formalization, clear framework design, and tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Provides a systematic solution for LLM-driven scientific modeling; the Dual-Anchor + Bi-Level Optimization paradigm has broad potential for transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](benchmarking_testing_in_automated_theorem_proving.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)
- [\[ACL 2026\] Benchmarking Testing in Automated Theorem Proving](benchmarking_testing_in_automated_theorem_proving.md)
- [\[ACL 2026\] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)

</div>

<!-- RELATED:END -->
