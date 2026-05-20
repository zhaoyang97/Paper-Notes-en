---
title: >-
  [Paper Note] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning
description: >-
  [ICML 2026][Robotics][Vision-Language-Action] This paper proposes CAPS: reinterpreting "instruction drift" as a systematic sampling error…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Vision-Language-Action"
  - "Long-Horizon Planning"
  - "Instruction Drift"
  - "Power Distribution Sampling"
  - "MCMC"
  - "Metacognitive Control"
date: 2026-05-08
content_hash: b58d12163d48cc80
---

# Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning

**Conference**: ICML 2026  
**arXiv**: [2605.09537](https://arxiv.org/abs/2605.09537)  
**Code**: None  
**Area**: Robotics / VLA / Inference-time Computation  
**Keywords**: Vision-Language-Action, Long-Horizon Planning, Instruction Drift, Power Distribution Sampling, MCMC, Metacognitive Control

## TL;DR
This paper proposes CAPS: reinterpreting "instruction drift" as a systematic sampling error, using SNR ($=\log|\mathcal{A}|-\mathcal{H}$) as a metacognitive switch. Only when entering high-entropy "Pivotal Windows" does it trigger Metropolis-Hastings iterative refinement based on power distributions $\pi\propto p^\alpha$. On RoboTwin, Simpler-WindowX, and Libero-long, it surpasses OpenVLA and TACO in a training-free manner.

## Background & Motivation

**Background**: VLA models (OpenVLA, $\pi_0$, $\pi_{0.5}$, etc.) perform well on short-horizon tasks but frequently fail in long-horizon operations (dual-arm collaboration, multi-step placement). A common failure mode is Instruction Drift—during task execution, background noise and irrelevant objects gradually dilute the initial instruction's attention weights, causing the robot to perform "locally reasonable but globally incorrect" actions.

**Limitations of Prior Work**: (1) Prompt engineering methods like Chain-of-Thought cannot break the unidirectionality of open-loop generation; (2) TACO's "generate-verify" paradigm relies on parallel sampling + reranker selection, but if a good trajectory is not sampled, it cannot be iteratively refined; (3) Global search methods such as Tree-of-Thoughts and RoboMonkey use a fixed high compute budget, which is unsuitable for real-time closed-loop robotic constraints.

**Key Challenge**: In continuous control spaces, single-step greedy sampling (low-temperature sampling) easily falls into a "Negative Pivotal Window"—locally high probability but physically irreversibly cuts off the path to global success. Brute-force enumeration (as in ToT) works for discrete symbolic reasoning but is infeasible in high-dimensional continuous action manifolds (**Topological Mismatch**).

**Goal**: (1) Redefine the drift problem from a sampling theory perspective; (2) Design a training-free inference-time computation framework to improve long-horizon robustness without retraining; (3) Invest extra computation only when necessary to maintain real-time performance.

**Key Insight**: Signal detection theory—viewing instruction drift as a process where effective signal decays to the noise level. In statistical physics, sampling from a power distribution $\pi(\tau)\propto p_\theta(\tau)^\alpha, \alpha\ge 1$ is equivalent to implicit future lookahead; Karan & Du 2026 have proven this in symbolic reasoning, and this work transfers it to continuous action manifolds.

**Core Idea**: Construct a "fast/slow dual-process"—System 1 executes greedily under high SNR; System 2 activates power-distribution-based MCMC iterative refinement when SNR drops below a threshold, dynamically trading inference time for long-horizon consistency.

## Method

### Overall Architecture
At each decision step, CAPS executes the following:
1. **Metacognitive Gating**: Uses policy entropy $\mathcal{H}(\pi_\theta(\cdot|H_t))$ to determine if currently in a Pivotal Window;
2. **If Low Entropy (High SNR)**: Directly greedily sample actions (System 1, nearly zero extra cost);
3. **If High Entropy (Low SNR)**: Enter System 2 iteration—
    - **Proposal**: Retain the current action chunk prefix, resample the suffix from $p_\theta$ with temperature $1/\alpha$ to obtain candidate $\tau_{new}$;
    - **Acceptance**: Use the MH acceptance rate under the power distribution to compare $\tau_{new}$ and $\tau_{old}$;
    - Iterate $N_{MCMC}$ times, output the final action chunk.

All computation is performed at inference time, with no parameter updates, fully plug-in.

### Key Designs

1. **Power Distribution Sampling (Global Trajectory Probability Sharpening)**:

    - **Function**: Replace the target distribution from the original $p_\theta(\tau|I,H_t)$ to $\pi(\tau)\propto p_\theta(\tau|I,H_t)^\alpha$ ($\alpha\ge 1$), equivalent to implicit lookahead planning.
    - **Mechanism**: $\alpha>1$ sharpens the probability peaks ("rich get richer"), suppressing Negative Pivotal Windows; MCMC sampling on $\pi$ does not require direct computation of $\pi$, only the likelihood ratio $(p_\theta(\tau_{new})/p_\theta(\tau_{old}))^\alpha$.
    - **Design Motivation**: Theorem 3.1 proves that under ideal lookahead, the effective horizon satisfies $T_{eff}(\text{CAPS})/T_{eff}(\text{Base})\approx \epsilon^{1-\alpha}$ ($\epsilon$ is the single-step drift rate), with $\alpha=2,\epsilon=0.1$ yielding a 10× horizon extension; Theorem 3.2 provides an asymptotic lower bound under finite MCMC steps, acknowledging sampling bias $O(\rho^N)$.

2. **SNR-based Metacognitive Gating**:

    - **Function**: Models "when to activate System 2" as an optimal control problem under resource constraints, automatically balancing accuracy and compute.
    - **Mechanism**: Defines contextual SNR as the KL divergence between the policy and uniform distribution, $\text{SNR}_t=D_{KL}(\pi_\theta\|\mathcal{U}_{\text{unif}})=\log|\mathcal{A}|-\mathcal{H}(\pi_\theta(\cdot|H_t))$. SNR is strictly linearly negatively correlated with policy Shannon entropy, so $\mathcal{H}>\gamma$ can serve as an efficient proxy criterion. Minimizing $\mathcal{L}(\pi)=\mathbb{E}[\text{Error}]+\lambda\cdot\mathcal{C}(\pi)$ yields hard-threshold switching: high entropy triggers CAPS, low entropy uses greedy.
    - **Design Motivation**: Avoids expensive global search at every step; from an information geometry perspective, high-entropy moments correspond to "bifurcation points" on the probability manifold, which are precisely where drift is most likely.

3. **Block-based Autoregressive MCMC**:

    - **Function**: Uses local chunk-level MH iteration under a receding-horizon paradigm to approximate the global trajectory-level target distribution.
    - **Mechanism**: Proposal $q(\tau_{new}|\tau_{old})$ is explicitly defined as "retain prefix, resample suffix with temperature $1/\alpha$"; Acceptance rate $A=\min(1, (p_\theta(\tau_{new})/p_\theta(\tau_{old}))^\alpha \cdot q(\tau_{old}|\tau_{new})/q(\tau_{new}|\tau_{old}))$. Each chunk runs $N_{MCMC}$ rounds of propose+accept, outputting actions only at chunk boundaries.
    - **Design Motivation**: Running MCMC on the full trajectory is computationally infeasible; chunk-based provides a compromise of "local acceptance + global scoring", and since the base model implicitly encodes long-horizon priors, local corrections can still maintain global consistency.

### Loss & Training
- No training, fully inference-time plug-in; base VLA uses $\pi_0$ / $\pi_{0.5}$ / OpenVLA.
- Key hyperparameters: $\alpha$ (sharpening factor, typically 2–4), $\gamma$ (entropy threshold), $N_{MCMC}$ (number of iterations, trading off inference latency), chunk size.
- Hardware: 4× A100; candidate count 50 (same as TACO for comparison).
- Autoregressive VLA ($\pi_{0.5}$) sampling temperature set to 1.

## Key Experimental Results

### Main Results

RoboTwin 1.0 average success rate:

| Method | Avg SR | Dual Bottles Hard | Mug Hanging Easy |
|--------|--------|-------------------|------------------|
| $\pi_0$ | 32.2 | 48.0 | 7.0 |
| $\pi_0$ + TACO | 41.3 | 52.0 | 12.0 |
| **CAPS** | **47.4** | **61.0** | **21.0** |

Simpler-WindowX OOD generalization:

| Task | $\pi_0$ | TACO | **CAPS** | SpatialVLA |
|------|--------|------|------|-------------|
| Spoon on Towel | 36 | 52 | **57** | 16.7 |
| Carrot on Plate | 42 | 52 | **61** | 25.0 |
| Stack Cubes | 34 | 30 | **37** | 29.2 |
| Eggplant in Basket | 80 | 88 | 87 | 100.0 |
| **Avg** | 48 | 55.5 | **60.5** | 42.7 |

RoboTwin 2.0 ($\pi_{0.5}$ baseline, 34 tasks average):

| Method | Avg SR |
|--------|--------|
| RDT | 34.6 |
| $\pi_{0.5}$ | 59.3 |
| $\pi_{0.5}$ + TACO | 64.0 |
| **$\pi_{0.5}$ + CAPS** | **66.2** |

On Libero-long, $\pi_{0.5}$ + CAPS achieves 98–100% on 5/6 subtasks, while OpenVLA only reaches 36–70%.

### Ablation Study

| Configuration | RoboTwin 1.0 Avg |
|---------------|-----------------|
| $\pi_0$ Baseline (System 1 only) | 32.2 |
| TACO (parallel sampling + rerank) | 41.3 |
| **CAPS Full** (dual-process + MCMC + power distribution) | **47.4** |

The appendix also provides sensitivity analysis for $\alpha$ and $N_{MCMC}$ (M, O), but the main table shows CAPS outperforms TACO by 6.1pp under the same candidate budget, indicating active iterative refinement is superior to passive screening.

### Key Findings
- **Drift is a sampling problem, not a modeling problem**: Keeping the base VLA unchanged and only switching the sampling strategy yields a 15pp improvement, validating the "drift = sampling error" hypothesis.
- **Dual-process is more efficient than parallel sampling**: CAPS triggers System 2 only in high-entropy (Pivotal Window) moments, with average compute cost much lower than TACO's global parallel + rerank.
- **Horizon extension matches theory**: Theorems 3.1/3.2 predict a 10× horizon extension at $\alpha=2$; in experiments, long-horizon tasks (Mug Hanging Easy 7→21, Blocks Ranking Size 36→46) show the most significant gains.
- **OOD robustness**: Carrot on Plate +19pp shows SNR gating triggers more aggressive MCMC under visual noise (low SNR), enabling escape from local optima.

## Highlights & Insights
- **Reframing the problem**: Elevates the engineering "drift" phenomenon to a sampling theory issue, providing the analytic relation $T_{eff}\propto \epsilon^{1-\alpha}$—a rare robotics work that is self-consistent from theory to empirical results.
- **Entropy = SNR bridge**: Uses $\text{SNR}_t=\log|\mathcal{A}|-\mathcal{H}$ to unify "signal detection" and "policy uncertainty", offering a zero-cost metacognitive trigger.
- **MCMC's utility in continuous control**: Uses the base model $p_\theta$ as proposal (resampling suffix with temperature $1/\alpha$) + power distribution as target, avoiding the fixed compute budget issue of Tree-of-Thoughts methods.
- **Training-free**: Pure inference-time plug-in, applicable to any base VLA, and engineering-friendly.

## Limitations & Future Work
- $\gamma$ is a fixed threshold; the appendix F provides a derivation but no adaptation. Entropy distributions vary greatly across tasks, possibly requiring per-task tuning.
- In dual-arm tasks, single-arm drift remains a bottleneck (Block Handover 67/100, Handover Block 38/100), indicating chunk-based MCMC has limited modeling of coordination constraints.
- Increasing $N_{MCMC}$ directly increases wall-clock time; real-time control scenarios require fine-tuning between accuracy and latency. The paper does not provide a complete latency profile.
- Theorems 3.1/3.2 assume constant single-step drift rate $\epsilon$, but in practice $\epsilon$ is strongly correlated with scene complexity, so theoretical gains may be overestimated.
- All experiments are in simulation; sim2real gap on real robots is not evaluated.

## Related Work & Insights
- **vs. TACO (Yang et al. 2025)**: TACO treats inference as a contextual bandit, selecting the best candidate via reranker; CAPS uses MCMC for active iterative refinement, able to converge even when the candidate set is of poor quality.
- **vs. Tree-of-Thoughts (Yao et al. 2023)**: ToT uses fixed compute for global search, while CAPS uses SNR gating for on-demand triggering, better suited for real-time control.
- **vs. RoboMonkey (Kwok et al. 2025)**: RoboMonkey trains a dedicated verifier, while CAPS is completely training-free.
- **vs. Karan & Du 2026 (symbolic reasoning power sampling)**: Transfers the power distribution theory from symbolic to continuous action manifolds, proving the same mathematical framework is effective across both modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The "drift = sampling error" redefinition + SNR-gated MCMC combination is indeed a new perspective in VLA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid comparison across three benchmarks × multiple baselines × training-free, but lacks real robot and latency profile.
- Writing Quality: ⭐⭐⭐⭐ Good integration of theory (Theorems 3.1/3.2) and algorithm (dual-process pseudocode), though some sections are verbose.
- Value: ⭐⭐⭐⭐ Provides a training-free path for long-horizon performance improvement for existing VLA models, immediately usable in engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)
- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)
- [\[CVPR 2025\] Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method](../../CVPR2025/robotics/towards_long-horizon_vision-language_navigation_platform_benchmark_and_method.md)

</div>

<!-- RELATED:END -->
