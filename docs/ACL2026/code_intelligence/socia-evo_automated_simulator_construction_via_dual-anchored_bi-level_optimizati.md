---
title: >-
  [Paper Note] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization
description: >-
  [ACL 2026][Code Intelligence][Automated simulator construction] This paper proposes SOCIA-EVO, an LLM agent framework that reformulates automated simulator construction as a dual-anchored evolutionary process. It anchors…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Automated simulator construction"
  - "dual-anchored evolutionary framework"
  - "bi-level optimization"
  - "strategy playbook"
  - "distributional fidelity"
date: 2026-05-08
content_hash: e9046120d769e7dc
---

# SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization

**Conference**: ACL 2026
**arXiv**: [2604.17351](https://arxiv.org/abs/2604.17351)
**Code**: [https://github.com/cruiseresearchgroup/SOCIA/tree/evo](https://github.com/cruiseresearchgroup/SOCIA/tree/evo)
**Area**: Code Intelligence
**Keywords**: Automated simulator construction, dual-anchored evolutionary framework, bi-level optimization, strategy playbook, distributional fidelity

## TL;DR

This paper proposes SOCIA-EVO, an LLM agent framework that reformulates automated simulator construction as a dual-anchored evolutionary process. It anchors empirical constraints via a static Blueprint, decouples structural revision and parameter calibration through bi-level optimization, and manages repair hypotheses via a self-curated strategy Playbook with Bayesian-weighted retrieval guided by execution feedback. SOCIA-EVO significantly outperforms baselines such as Reflexion and G-SIM on three simulation tasks: user modeling, mask-wearing diffusion, and personal mobility.

## Background & Motivation

**Background**: Automatically constructing simulators from observational data (data-driven simulation) is foundational to understanding complex systems. Unlike general software engineering where functional correctness suffices, simulator construction is inherently a scientific modeling task requiring *distributional fidelity*—generated programs must reproduce the statistical regularities, causal mechanisms, and emergent behaviors present in real data.

**Limitations of Prior Work**: Standard LLM agents applied to long-horizon simulator construction exhibit two critical failure modes: (1) *context drift*—as simulator complexity grows, constraints established during initial data analysis lose salience and agents may hallucinate mechanisms absent from the data; (2) *optimization instability*—agents conflate structural errors (e.g., incorrect transition logic) with parameter mismatches (e.g., suboptimal rates), rewriting correct logic when simple parameter tuning would suffice, leading to oscillatory "whack-a-mole" revisions.

**Key Challenge**: LLMs excel at discrete logical reasoning but struggle with high-dimensional continuous parameter search. Furthermore, they lack a persistent mechanism for validating repair strategies—previously attempted fixes and their quantified outcomes are lost as the context window advances, causing repeated proposals of superficially plausible but empirically ineffective repairs.

**Goal**: Design an agent framework that maintains long-horizon consistency, distinguishes structural from parametric issues, and accumulates validated repair experience.

**Key Insight**: Introduce a dual-anchoring mechanism—a static Blueprint prevents context drift, while a dynamic strategy Playbook accumulates and validates repair hypotheses; bi-level optimization strictly decouples structural revision (LLM-driven outer loop) from parameter calibration (numerical optimizer inner loop).

**Core Idea**: Simulator construction = Blueprint-anchored search space + bi-level optimization decoupling structure and parameters + self-curated Playbook for repair hypotheses.

## Method

### Overall Architecture

SOCIA-EVO employs a closed-loop iterative workflow with six specialized agents. The process begins with a data analysis agent generating a static Blueprint $\mathcal{B}$, followed by an evolutionary loop: a code generation agent produces simulator code $P_t$ and calibrator $C_t$ conditioned on $\mathcal{B}$ and current Playbook strategies $\mathcal{K}$; an execution agent runs the inner-loop optimization over $\theta$ and collects metrics $\mathcal{M}_t$; a feedback agent diagnoses deviations; a Playbook manager updates the strategy pool; and an iteration controller evaluates convergence.

### Key Designs

1. **Dual-Anchoring Mechanism (Static Blueprint + Dynamic Playbook)**:

    - **Function**: Prevents context drift and accumulates validated repair experience.
    - **Mechanism**: The Blueprint $\mathcal{B}$ is an immutable authoritative specification that strictly defines simulation topology, agent modes, evaluation metrics, etc., and is locked after human expert validation. The Playbook $\mathcal{K}$ is a dynamic repair strategy pool, where each strategy $S_i = \langle R_i, I_i, \Sigma_i \rangle$ contains diagnostic content (with metric binding set $\Lambda_i$), meta-information (usage/success/failure counts), and a lifecycle state. Strategies undergo state transitions Open→Queued→InProgress→Resolved, with validation success ($\Delta\mathcal{M}/\mathcal{M}_t > \tau$), falsification ($< -\tau$), or inconclusive status determined via metric-change thresholds.
    - **Design Motivation**: The Blueprint provides immutable constraints to prevent generation from deviating from real data; the Playbook treats repair strategies as hypotheses rather than trusted corrections, verifying or falsifying them through execution evidence to enable self-curation.

2. **Bi-Level Optimization**:

    - **Function**: Decouples structural revision (discrete search) from parameter calibration (continuous optimization).
    - **Mechanism**: The outer loop has the code generation agent search the program space to produce structure $P_t$ and calibrator $C_t$: $(P_t, C_t) \leftarrow \pi_{code}(P_{t-1}, \mathcal{B}, \text{Knapsack}(\mathcal{K}))$. The inner loop executes calibrator $C_t$, using numerical methods such as Bayesian optimization to solve for optimal parameters $\theta_t^* = \arg\min_\theta \mathcal{L}(\text{Sim}(P_t, \theta), \mathcal{D}_{obs})$. Critically, the LLM does not guess parameter values but instead generates an optimization program that finds them.
    - **Design Motivation**: Ensures that metrics observed by the feedback agent reflect the intrinsic capability of structure $P_t$ at parameter optimality, filtering out noise from untuned parameters and enabling precise attribution of defects to structural logic.

3. **Knapsack-Based Context Engineering**:

    - **Function**: Selects the most valuable repair strategies within a limited context budget.
    - **Mechanism**: Solves a 0-1 knapsack problem over the Open/Queued pool: $\max \sum_i v_i x_i$ subject to $\sum_i c_i x_i \leq L_{budget}$. Strategy value is $v_i = w_{sev} \cdot U_i^{queue} \cdot \Phi_{rel}(S_i)$, where $w_{sev}$ is a severity weight, $U_i^{queue}$ is a backlog reward for starvation prevention, and $\Phi_{rel}$ is a Bayesian reliability estimate under a Beta-Bernoulli model: $\Phi_{rel}(S_i) = (s_i+1)/(s_i+f_i+2)$. Prompts are structured in a three-zone layout (system/context/instruction) to counteract the "lost-in-the-middle" effect.
    - **Design Motivation**: Injecting all strategies dilutes attention (experiments confirm that a full 3200-token Playbook degrades performance to near-amnesiac levels); Bayesian reliability causes empirically falsified strategies to naturally decay.

### Loss & Training

SOCIA-EVO does not perform model training; instead, it optimizes the simulator through iterative evolution. The inner loop uses Optuna Bayesian optimization or a random calibrator. The outer-loop iteration controller halts upon improvement plateaus or regressions to prevent overcorrection. All experiments use GPT-5.1 as the backbone, with means and 95% confidence intervals reported over 5 random seeds.

## Key Experimental Results

### Main Results

**Performance comparison across three simulation tasks (mean ± 95% CI, lower is better)**

| Method | User Modeling MAE↓ | Mask Simulation RMSE↓ | Mobility N→A WD↓ |
|--------|-------------------|----------------------|-----------------|
| Reflexion | 0.17±0.01 | 0.26±0.02 | 0.69±0.02 |
| YuLan-OneSim | 0.21±0.02 | 0.16±0.01 | 0.64±0.01 |
| G-SIM-SBI | 0.19±0.01 | 0.11±0.02 | 0.56±0.02 |
| ACE-OL | 0.14±0.02 | 0.24±0.01 | 0.61±0.02 |
| **SOCIA-EVO** | **0.11±0.01** | **0.07±0.01** | **0.53±0.02** |

### Ablation Study

| Configuration | ΔMAE↓ | ΔRMSE↓ | ΔWD↓ |
|---------------|-------|--------|------|
| SOCIA-EVO (full) | — | — | — |
| w/o inner-loop calibration | +0.25 | +0.47 | +0.29 |
| w/o Blueprint | +0.20 | +0.37 | +0.23 |
| w/o HITL | +0.18 | +0.34 | +0.21 |
| w/o memory mechanism | +0.14 | +0.30 | +0.20 |
| w/o strategy Playbook | +0.10 | +0.23 | +0.15 |
| Context window +2200 tokens | +0.12 | +0.14 | +0.13 |

### Key Findings

- Inner-loop parameter calibration is the most critical component—removing it causes RMSE to surge by +0.47, as parameter updates degrade to heuristic LLM guessing.
- Overly large context windows (+2200 tokens loading the full Playbook) cause performance degradation approaching amnesiac levels, validating the "attention dilution effect."
- Cumulative repeated errors decrease by 76% at the fifth iteration, demonstrating that the value-based knapsack mechanism effectively suppresses "whack-a-mole" oscillations.
- Open-source backbones (Llama-3.3-70B, Qwen3-80B) also achieve competitive performance, confirming that the framework is not dependent on any specific proprietary model.

## Highlights & Insights

- Treating repair strategies as hypotheses requiring verification or falsification rather than as trusted corrections represents a "scientific method" mindset that is highly novel in LLM agent design and is transferable to any agent system requiring long-horizon memory management.
- The core insight of bi-level optimization is elegant: having the LLM generate a program that *finds* parameter values rather than directly guessing them cleverly leverages LLM strengths in code generation while circumventing their weakness in continuous optimization.
- The knapsack + Bayesian reliability context engineering provides a general solution for maximizing information value within limited context windows.

## Limitations & Future Work

- The strongest results rely on a commercial LLM backbone (GPT-5.1); while open-source models are viable, a performance gap remains.
- The current framework only addresses tasks expressible through iterative structural revision and bounded parameter calibration; more complex settings (long-horizon planning, strategic multi-agent interaction) may require stronger reasoning modules.
- The HITL validation of the Blueprint, while lightweight, introduces human dependency, and Blueprint quality directly affects all subsequent iterations.
- Strategy matching relies on thresholded text matching; finer-grained semantic matching could improve strategy consolidation quality.

## Related Work & Insights

- **vs. Reflexion**: Reflexion uses episodic memory for verbal reinforcement learning but lacks metric-driven repair validation, resulting in RMSE as high as 0.26 on mask simulation vs. SOCIA-EVO's 0.07.
- **vs. G-SIM**: G-SIM effectively separates structural generation and parameter estimation but lacks long-term evidence tracking, leading to repeated exploration of already-falsified strategies.
- **vs. Dynamic Cheatsheet**: DC accumulates successful strategies but may introduce negative transfer from irrelevant contexts.
- **vs. YuLan-OneSim**: Generates behavior in predefined environments rather than inferring simulator logic from data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formalizes simulator construction as a dual-anchored evolutionary process; the strategy hypothesis verification/falsification mechanism is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks, six baselines, detailed ablations, convergence analysis, and backbone portability verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous problem formalization, clear framework design, and tight integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Provides a systematic solution for LLM-driven scientific modeling; the dual-anchor + bi-level optimization design paradigm has broad transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](../../ICLR2026/code_intelligence/the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [\[ICLR 2026\] A Problem-Oriented Perspective and Anchor Verification for Code Optimization](../../ICLR2026/code_intelligence/a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/code_intelligence/automated_multi-agent_workflows_for_rtl_design.md)

</div>

<!-- RELATED:END -->
