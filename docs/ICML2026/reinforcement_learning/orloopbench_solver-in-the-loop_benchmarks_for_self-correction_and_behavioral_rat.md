---
title: >-
  [Paper Note] ORLoopBench: Solver-in-the-Loop Benchmarks for Self-Correction and Behavioral Rationality in Operations Research
description: >-
  [ICML 2026][Reinforcement Learning][solver-in-the-loop] The authors formalize the task of "repairing an Infeasible operations research model" as a solver-in-the-loop MDP…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "solver-in-the-loop"
  - "IIS feedback"
  - "self-correction"
  - "GRPO"
  - "newsvendor bias"
date: 2026-05-08
content_hash: 2d1eaea463795dd0
---

# ORLoopBench: Solver-in-the-Loop Benchmarks for Self-Correction and Behavioral Rationality in Operations Research

**Conference**: ICML 2026  
**arXiv**: [2601.21008](https://arxiv.org/abs/2601.21008)  
**Code**: https://github.com/Archer222arc/ORLoopBench  
**Area**: LLM Evaluation / Self-Correction / Operations Research / Solver-in-the-Loop / RLVR  
**Keywords**: solver-in-the-loop, IIS feedback, self-correction, GRPO, newsvendor bias

## TL;DR
The authors formalize the task of "repairing an Infeasible operations research model" as a solver-in-the-loop MDP, where each modification step requires re-running Gurobi to obtain Irreducible Infeasible Subsystem (IIS) feedback. They release the ORLoopBench benchmark (5362 LP/MILP repair instances + inventory decision bias evaluation) and use RLVR to train an 8B model that outperforms closed-source APIs on LP repair with a 95.3% RR@5 (vs. 92.4%).

## Background & Motivation

**Background**: Existing LLM-for-OR benchmarks (NL4Opt, OptiBench, MAMO, ORLM, etc.) almost exclusively evaluate one-shot translation from "problem description $\rightarrow$ solver code." The evaluation ends once the model generates the code, lacking multi-turn interaction with the solver.

**Limitations of Prior Work**: In practical OR workflows, the first response an engineer receives after writing LP/MILP code is often `Status: Infeasible`. They must then rely on the Irreducible Infeasible Subsystem (IIS) provided by solvers like Gurobi to iteratively locate conflicting constraints and modify formulas. One-shot translation benchmarks bypass this debugging loop, thereby masking the true engineering capabilities of LLMs.

**Key Challenge**: In self-correction tasks, feedback signals are typically vague (e.g., compiler errors, unit test failures), leading works like CorrectBench to use "soft" metrics. Operations research scenarios possess a rare "triple certainty"—the solver output is a deterministic oracle, the IIS is a minimal certificate of conflict, and optimal solutions are mathematically verifiable. This should have been an ideal sandbox for studying self-correction, yet it had not been systematized.

**Goal**: (1) Formalize infeasible-model repair into a trainable MDP; (2) Provide a benchmark capable of distinguishing "accidental fixes" from "true diagnosis"; (3) Evaluate the behavioral rationality of LLMs in downstream operational decisions (inventory).

**Key Insight**: The authors focus on IIS as the "minimal conflict certificate." It is not a generic error but a mathematically compressible subset of conflicts. The model must perform targeted modifications based on the IIS rather than blind trial-and-error.

**Core Idea**: Pack the "problem description + current code + Gurobi status + IIS" into a state. Actions include "diagnosis query / repair edit / submit." Rewards are composed of "reaching Optimal, hitting true conflicts, and step count." Reinforcement Learning through Verifiable Rewards (RLVR) is then applied using GRPO under solver verification—naturally fitting the OR domain.

## Method

### Overall Architecture

ORLoopBench consists of two complementary components. **OR-Debug-Bench** is the core: a solver-in-the-loop MDP containing 5362 LP/MILP repair instances, covering 9 LP error types (A–I) and 8 MILP error types. **OR-Bias-Bench** is the supplement: 2000 newsvendor and 300 EOQ inventory decision problems, evaluating "pull-to-center" bias against closed-form optimal solutions $Q^{*}=F^{-1}(\text{CR})$ and $Q^{*}=\sqrt{2DK/h}$. Both components utilize a training pipeline based on Qwen3-8B with SFT + RL.

The evaluation loop is as follows: The solver reads code and outputs `Infeasible` with an IIS $\rightarrow$ The agent observes the state (NL problem + current code + Status + IIS + history + step number) $\rightarrow$ The agent outputs an action (Diagnosis: `Get_IIS` / `Check_Slack`; Repair: `Relax` / `Drop` / `Rewrite`; or Termination: `Submit`) $\rightarrow$ Gurobi 11.0 re-runs to produce the next state $\rightarrow$ The process continues until `Optimal` status is reached or the step limit (50) is exhausted.

### Key Designs

1.  **Solver-in-the-Loop MDP and Three-Stage Verifiable Reward**:
    - **Function**: Transforms multi-turn repair into an RL task where rewards are entirely determined by the solver, avoiding subjective noise from human or LLM evaluators.
    - **Mechanism**: Reward is defined as $\mathcal{R}=0.5\,R_\text{outcome}+0.3\,R_\text{diagnosis}+0.2\,R_\text{efficiency}$. Here, $R_\text{outcome}=+100$ if Optimal, else $-50$; $R_\text{diagnosis}=\text{DA}\cdot 100$, where Diagnosis Accuracy (DA) is $|\text{diagnosed}\cap \text{IIS}_\text{GT}|/|\text{IIS}_\text{GT}|$; $R_\text{efficiency}=-1$ per step. A "Loyalty Penalty" of $-20$ is added to suppress "detour repairs" where models modify non-IIS constraints just to reach feasibility.
    - **Design Motivation**: Relying solely on RR@5 risks rewarding "getting lucky." DA isolates diagnostic quality, while the loyalty penalty prevents cheating by masking conflicts with side effects, ensuring the reward measures "conflict understanding" rather than just "restoring feasibility."

2.  **Saboteur Pipeline and Verifiable Generation of 9 Error Types**:
    - **Function**: Automatically generates triplets of "Original LP + Sabotaged LP + Unique True Repair," ensuring each sabotaged LP is Infeasible, has a non-empty IIS, and the IIS contains the injected error.
    - **Mechanism**: A 4-stage pipeline (pool sampling $\rightarrow$ typed destruction $\rightarrow$ Gurobi verification $\rightarrow$ oracle labeling). 9 error types are categorized by difficulty: B/C involve minor sign or coefficient changes (Easy, baseline RR@5 $\ge$ 85%); H/I involve incomplete IIS or optimal selection (Medium, 70–85%); A/D/E/F/G involve direction flips, contradictory constraints, or cascading conflicts (Hard, <70%). Specific strategies are used for difficult injections, such as relaxation-based constraint selection for Type A to raise success rates from 30% to 95%.
    - **Design Motivation**: Synthetic data must be "difficult and unique." "Anti-pattern" measures—such as UUID-based naming, hiding dependencies so the IIS only shows symptoms, and cascading conflicts requiring multi-turn reasoning—ensure simple string matching fails.

3.  **Small-Model RLVR: GRPO + Process Reward Model**:
    - **Function**: Uses minimal compute (approx. 8 GPU-hours) on Qwen3-8B to push LP repair performance beyond closed-source APIs and enable zero-shot transfer to MILP.
    - **Mechanism**: Initially, SFT is performed on 696 successful trajectories from GPT-5.2-chat / o4-mini / DeepSeek-R1 (requiring $\le 5$ steps + hitting at least half of GT conflicts). This is followed by GRPO ($\beta=0$ for KL, asymmetric clip $[0.2, 0.28]$, LoRA $r=16, \alpha=32$). To mitigate reward sparsity, a Process Reward Model (PRM) is trained using step-level labeling (highest reward for hitting conflicts or reducing IIS). The PRM achieves an AUC-ROC of 0.94, further improving DA by 4.7pp.
    - **Design Motivation**: Result-based rewards arrive too late for short-trajectory LP repair tasks. The PRM decouples "doing the right thing" from "the final fix," allowing an 8B model to approach frontier API performance without scaling up.

### Loss & Training

Two parallel tracks: The Debug-Track uses SFT + GRPO + optional PRM. The Bias-Track uses SFT + three-stage curriculum learning (Stage 1 uses extreme CR 0.1/0.9 to teach direction; Stage 2 refines magnitude at boundaries 0.15–0.25 and 0.75–0.85; Stage 3 consolidates within [0.2, 0.8]). Both tracks are based on Qwen3-8B with inference via SGLang.

## Key Experimental Results

### Main Results

Large-scale evaluation of 26 models (LP test set: 450 cases, 50 per error type A–I):

| Model | RR | RR@5 | DA | Avg. Steps |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B-GRPO (Ours) | 100% | **95.3%** | **62.4%** | **2.25** |
| Qwen3-8B-Curriculum | 100% | 94.0% | 61.7% | 2.22 |
| Qwen3-8B-SFT | 99.8% | 93.1% | 60.8% | 2.34 |
| Claude Sonnet 4.6 (Prev. SOTA) | — | 92.4% | — | — |
| o4-mini | 97.8% | 86.2% | 47.8% | 3.15 |
| claude-sonnet-4 | 100% | 86.2% | 50.1% | 3.71 |
| gpt-5.2-chat | 99.8% | 81.8% | 40.9% | 3.72 |
| DeepSeek-R1 | 99.1% | 56.7% | 34.5% | 5.08 |

MILP Transfer (10 domains $\times$ 8 error types $\times$ 5 repeats): The LP-trained model achieves 78.8% RR@5 zero-shot; MILP-specific fine-tuning reaches 87.1%, outperforming Claude Sonnet 4.6 (71.0%) by 16.1pp.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| SFT only | RR@5 93.1% / DA 60.8% | Only imitates teacher trajectories |
| + GRPO (Outcome Reward) | RR@5 95.3% / DA 62.4% | Mainstream configuration |
| + PRM Process Supervision | DA 72.7% (+4.7pp) | Step-level rewards significantly boost diagnostic quality |
| + Curriculum (Bias-Track) | OOD bias 10.4% (-9.6%) | Only training scheme that reduces bias on OOD data |

### Key Findings
- **8B Outperforming Closed-Source**: On LP repair, Qwen3-8B-GRPO exceeds the strongest Claude Sonnet 4.6 by 2.9pp (RR@5), while improving DA from 47.8% to 62.4% and reducing steps from 3.15 to 2.25. This suggests that "solver-verifiable rewards" are exceptionally effective for small models on mathematically structured tasks.
- **Semantic Drift in Code Rewriting**: For OptiMUS-style full-model regeneration in MILP, GPT-5.4 reached Optimal in 90% of cases but preserved the original objective function in only 28.2%; for Claude Sonnet 4.6, these figures were 85% / 22.4%. Constraint-level repair naturally avoids objective drift by only touching constraints within the IIS.
- **Diagnostic Style Reversal**: Post-training models transitioned to a "diagnose once, repair precisely" mode, with only 1.3 diagnostic actions per episode (vs. 2.1 for APIs), yet achieving significantly higher repair hit rates—behavioral evidence of shifting from "trial-and-error" to "systematic elimination."
- **Directional Behavioral Bias**: All LLMs exhibited "pull-to-center" bias on the newsvendor problem—under-ordering when $\text{CR}>0.5$ and over-ordering when $\text{CR}<0.5$. Curriculum learning was the only solution to reduce bias on OOD data (-9.6%), suggesting a "learn extremes then boundaries" order is critical for correcting bias.

## Highlights & Insights
- **Reinventing the Solver as an RL Reward**: While using a solver as an oracle is not new, integrating it directly into the reward loop is. By triggering a solver re-run for every action, rewards are derived directly from solver state changes, allowing RLVR in OR to scale infinitely without human judges.
- **Decoupling DA and RR@5**: This design choice allows easy identification of models that "fix it without knowing why." High RR@5 with low DA indicates a lack of diagnostic reasoning—a 2D metric valuable for other self-correction benchmarks.
- **PRM AUC-ROC of 0.94**: This high value indicates that step-level labels generated from outcome rewards are sufficiently clean, suggesting that process supervision might be easier to implement in structured tasks than in general math or coding.
- **OR-Bias-Bench Line**: Porting the "pull-to-center" bias from behavioral economics to LLM evaluation provides a systemic bias probe. The closed-form solutions ensure unambiguous ground truths, which can be extended to other textbook problems like EOQ or pricing.

## Limitations & Future Work
- **Acknowledged Limitations**: While 8B + LoRA training is cost-effective, SGLang TP=2 still requires 2$\times$A100, which remains a barrier for some labs. The 87.1% score on MILP still leaves a gap for production environments.
- **Independent Observations**: The evaluation is tied to Gurobi 11.0; IIS algorithms vary by solver and version, meaning DA definitions might require recalibration for CPLEX or HiGHS. OR-Bias-Bench only covers Newsvendor and EOQ, far from multi-product/multi-period inventory reality. The saboteur pipeline, while unique, lacks statistical validation against the real-world bug distribution of OR code.
- **Future Directions**: (1) Replace IIS with a solver-agnostic "infeasibility certificate" interface; (2) Integrate objective function drift as a first-order metric for code regeneration; (3) Combine synthetic bias with real retail data to add an OOD real-world distribution layer to OR-Bias-Bench.

## Related Work & Insights
- **vs. CorrectBench (Tie et al., 2025)**: Both study LLM self-correction, but CorrectBench uses general programming with soft tests. This work uses the deterministic certificate of Gurobi's IIS, allowing precise quantification of diagnostic accuracy.
- **vs. SWE-bench (Jimenez et al., 2024)**: SWE-bench uses unit tests as an oracle, treating issues as black boxes. ORLoopBench breaks the oracle down into every step, providing readable IIS feedback for step-level RL.
- **vs. ORLM (Huang et al., 2025a) / OptiBench**: Previous works only evaluate one-shot "NL $\rightarrow$ Formula" translation. This work upgrades the task to "repairing incorrect code," mirroring the daily routine of OR engineers.
- **vs. AIM-Bench (Zhao et al., 2025)**: AIM-Bench first reported the "pull-to-center" bias in LLMs. This work proves this bias can be corrected via three-stage curriculum learning and provides a recipe for reducing bias under OOD conditions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using IIS as an RLVR signal source is clear and systematized for the first time, though the "solver-in-the-loop" concept has roots in the SWE-bench era.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablation across 26 models, dual track, MILP transfer, bias OOD, PRM, and curriculum.
- **Writing Quality**: ⭐⭐⭐⭐ Tables and reward formulas are clean, though some experimental details being relegated to the appendix slightly affects flow.
- **Value**: ⭐⭐⭐⭐⭐ Releasing both the benchmark and training pipeline, and proving 8B models can outperform closed-source APIs, provides a reusable recipe for small-model RLVR in structured tasks.

## Rating
- **Novelty**: TBD
- **Experimental Thoroughness**: TBD
- **Writing Quality**: TBD
- **Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dr. Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](dr_tulu_reinforcement_learning_with_evolving_rubrics_for_deep_research.md)
- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](../../NeurIPS2025/reinforcement_learning/research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICML 2026\] Making Expert Reasoning Learnable with Self-Distillation](making_expert_reasoning_learnable_with_self-distillation.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)

</div>

<!-- RELATED:END -->
