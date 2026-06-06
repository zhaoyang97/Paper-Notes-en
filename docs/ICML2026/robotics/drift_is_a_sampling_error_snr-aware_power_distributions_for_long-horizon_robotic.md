---
title: >-
  [Paper Note] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning
description: >-
  [ICML 2026][Robotics][Vision-Language-Action] This paper proposes CAPS: reinterpreting "instruction drift" as a systematic sampling error and using SNR ($=\log|\mathcal{A}|-\mathcal{H}$) as a metacognitive switch to trig…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Vision-Language-Action"
  - "Long-horizon planning"
  - "Instruction drift"
  - "Power distribution sampling"
  - "MCMC"
  - "Metacognitive control"
date: 2026-05-08
content_hash: a687b78274b176af
---

# Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning

**Conference**: ICML 2026  
**arXiv**: [2605.09537](https://arxiv.org/abs/2605.09537)  
**Code**: None  
**Area**: Robotics / VLA / Inference-time Computing  
**Keywords**: Vision-Language-Action, Long-horizon planning, Instruction drift, Power distribution sampling, MCMC, Metacognitive control

## TL;DR
This paper proposes CAPS: reinterpreting "instruction drift" as a systematic sampling error and using SNR ($=\log|\mathcal{A}|-\mathcal{H}$) as a metacognitive switch to trigger Metropolis-Hastings iterative refinement based on a power distribution $\pi\propto p^\alpha$ only during high-entropy "Pivotal Windows." It outperforms OpenVLA and TACO training-free on RoboTwin, Simpler-WindowX, and Libero-long.

## Background & Motivation

**Background**: VLA models (OpenVLA, $\pi_0$, $\pi_{0.5}$, etc.) perform excellently on short-horizon tasks but frequently fail in long-horizon manipulations (bimanual collaboration, multi-step placement). A common failure mode is Instruction Drift—as the task progresses, background noise and irrelevant objects gradually dilute the attention weights of the initial instruction, causing the robot to execute actions that are "locally reasonable but globally incorrect."

**Limitations of Prior Work**: (1) Prompt engineering like Chain-of-Thought cannot break the unidirectionality of open-loop generation; (2) The "generate-verify" paradigm of TACO relies on parallel sampling + reranker selection, but if a single trajectory is not sampled, it cannot be iteratively refined; (3) Global search methods like Tree-of-Thoughts and RoboMonkey use fixed high computational budgets, which are not suitable for real-time closed-loop robotic constraints.

**Key Challenge**: In continuous control spaces, single-step greedy sampling (low-temperature sampling) easily falls into "Negative Pivotal Windows"—where local probability is high but the path to global success is physically and irreversibly cut off. Conversely, brute-force enumeration search (e.g., ToT), which works in discrete symbolic reasoning, is infeasible on high-dimensional continuous action manifolds (**Topological Mismatch**).

**Goal**: (1) Redefine the drift problem from a sampling theory perspective; (2) Design a training-free inference-time computation framework that enhances long-horizon robustness without retraining; (3) Allocate additional computation only when necessary to maintain real-time performance.

**Key Insight**: Signal Detection Theory—view instruction drift as a process where the "effective signal decays to noise levels." In statistical physics, sampling from a power distribution $\pi(\tau)\propto p_\theta(\tau)^\alpha, \alpha\ge 1$ is equivalent to performing an implicit lookahead. While Karan & Du 2026 proved this in symbolic reasoning, this paper transfers it to continuous action manifolds.

**Core Idea**: Construct a "fast/slow dual-process"—System 1 executes greedily during high SNR; System 2 activates iterative MCMC refinement based on the power distribution when SNR drops below a threshold, dynamically trading inference time for long-horizon consistency.

## Method

### Overall Architecture
CAPS runs the following pipeline at each decision step:
1. **Metacognitive Gating**: Determines if the system is currently in a Pivotal Window based on policy entropy $\mathcal{H}(\pi_\theta(\cdot|H_t))$;
2. **If low entropy (high SNR)**: Directly sample the action greedily (System 1, near-zero overhead);
3. **If high entropy (low SNR)**: Enter System 2 iteration—
    - **Proposal**: Keep the current action chunk prefix and resample the suffix using $p_\theta$ with temperature $1/\alpha$ to obtain candidate $\tau_{new}$;
    - **Acceptance**: Compare $\tau_{new}$ with $\tau_{old}$ using the MH acceptance rate under the power distribution;
    - Iterate $N_{MCMC}$ times to output the final action chunk.

All computations are performed at inference time without updating any parameters, making it fully plug-in.

### Key Designs

1. **Power Distribution Sampling (Global Trajectory Probability Sharpening)**:
    - **Function**: Replaces the target distribution from the original $p_\theta(\tau|I,H_t)$ with $\pi(\tau)\propto p_\theta(\tau|I,H_t)^\alpha$ ($\alpha\ge 1$), equivalent to implicit lookahead planning.
    - **Mechanism**: $\alpha>1$ makes the probability peaks sharper ("rich get richer"), suppressing Negative Pivotal Windows. When sampling via MCMC on $\pi$, there is no need to compute $\pi$ directly; only the likelihood ratio $(p_\theta(\tau_{new})/p_\theta(\tau_{old}))^\alpha$ is required.
    - **Design Motivation**: Theorem 3.1 proves that under ideal lookahead, the effective horizon satisfies $T_{eff}(\text{CAPS})/T_{eff}(\text{Base})\approx \epsilon^{1-\alpha}$ (where $\epsilon$ is the single-step drift rate). For $\alpha=2, \epsilon=0.1$, this yields a $10\times$ horizon improvement. Theorem 3.2 provides an asymptotic lower bound for a finite number of MCMC steps, acknowledging a sampling bias of $O(\rho^N)$.

2. **SNR-based Metacognitive Gating**:
    - **Function**: Models "when to start System 2" as an optimal control problem under resource constraints, automatically balancing accuracy and computation.
    - **Mechanism**: Defines contextual SNR as the KL divergence of the policy relative to a uniform distribution: $\text{SNR}_t=D_{KL}(\pi_\theta\|\mathcal{U}_{\text{unif}})=\log|\mathcal{A}|-\mathcal{H}(\pi_\theta(\cdot|H_t))$. Since SNR is strictly linearly negatively correlated with Shannon entropy, $\mathcal{H}>\gamma$ serves as an efficient proxy criterion. Solving for the minimization of $\mathcal{L}(\pi)=\mathbb{E}[\text{Error}]+\lambda\cdot\mathcal{C}(\pi)$ results in hard-threshold switching: high entropy triggers CAPS, while low entropy follows greedy execution.
    - **Design Motivation**: Avoids expensive global searches at every step. From an information geometry perspective, high-entropy moments correspond to "bifurcation points" on the probability manifold, which are exactly the windows where drift is most likely to occur.

3. **Block-based Autoregressive MCMC**:
    - **Function**: Uses local chunk-level MH iterations to approximate the global trajectory-level target distribution under the receding-horizon principle.
    - **Mechanism**: The proposal $q(\tau_{new}|\tau_{old})$ is explicitly defined as "keep prefix, resample suffix at temperature $1/\alpha$." The acceptance rate $A=\min(1, (p_\theta(\tau_{new})/p_\theta(\tau_{old}))^\alpha \cdot q(\tau_{old}|\tau_{new})/q(\tau_{new}|\tau_{old}))$. Each chunk undergoes $N_{MCMC}$ rounds of propose+accept, outputting actions only at chunk boundaries.
    - **Design Motivation**: Directly running MCMC over the full trajectory is computationally infeasible. Chunk-based iteration offers a compromise between "local acceptance and global scoring." During acceptance evaluation, the foundation model has already implicitly encoded long-range priors, ensuring that local corrections maintain global consistency.

### Loss & Training
- No training required, fully inference-time plug-in. Base VLAs used are $\pi_0$, $\pi_{0.5}$, and OpenVLA.
- Key hyperparameters: $\alpha$ (sharpening coefficient, typically 2–4), $\gamma$ (entropy threshold), $N_{MCMC}$ (number of iterations, trading off inference latency), and chunk size.
- Hardware: 4× A100; number of candidates 50 (consistent with TACO benchmarks).
- Sampling temperature for Autoregressive VLA ($\pi_{0.5}$) is set to 1.

## Key Experimental Results

### Main Results

Average Success Rate on RoboTwin 1.0:

| Method | Avg SR | Dual Bottles Hard | Mug Hanging Easy |
|------|--------|-------------------|------------------|
| $\pi_0$ | 32.2 | 48.0 | 7.0 |
| $\pi_0$ + TACO | 41.3 | 52.0 | 12.0 |
| **CAPS** | **47.4** | **61.0** | **21.0** |

Simpler-WindowX OOD generalization:

| Task | $\pi_0$ | TACO | **CAPS** | SpatialVLA |
|------|--------|-------|------|-------------|
| Spoon on Towel | 36 | 52 | **57** | 16.7 |
| Carrot on Plate | 42 | 52 | **61** | 25.0 |
| Stack Cubes | 34 | 30 | **37** | 29.2 |
| Eggplant in Basket | 80 | 88 | 87 | 100.0 |
| **Avg** | 48 | 55.5 | **60.5** | 42.7 |

RoboTwin 2.0 ($\pi_{0.5}$ baseline, average over 34 tasks):

| Method | Avg SR |
|------|--------|
| RDT | 34.6 |
| $\pi_{0.5}$ | 59.3 |
| $\pi_{0.5}$ + TACO | 64.0 |
| **$\pi_{0.5}$ + CAPS** | **66.2** |

On Libero-long, $\pi_{0.5}$ + CAPS reached 98–100% on 5/6 subtasks, enquanto OpenVLA reached only 36–70%.

### Ablation Study

| Configuration | RoboTwin 1.0 Avg |
|------|-----------------|
| $\pi_0$ Baseline (System 1 only) | 32.2 |
| TACO (Parallel sampling + rerank) | 41.3 |
| **CAPS Full** (Dual-process + MCMC + Power Dist) | **47.4** |

The appendix includes sensitivity analyses for $\alpha$ and $N_{MCMC}$. The main results show that CAPS outperforms TACO by 6.1pp under the same candidate budget, indicating that active iterative refinement is superior to passive screening.

### Key Findings
- **Drift is a sampling problem, not a modeling problem**: Keeping the base VLA constant and only changing the sampling strategy yielded a 15pp gain, validating the "drift = sampling error" hypothesis.
- **Dual-process is more efficient than parallel sampling**: CAPS triggers System 2 only during high-entropy windows (Pivotal Windows), resulting in much lower average computational overhead than TACO's global parallel sampling + reranking.
- **Horizon extension aligns with theory**: Theorems 3.1/3.2 predict a $10\times$ horizon improvement for $\alpha=2$. In experiments, long-horizon tasks (Mug Hanging Easy 7→21, Blocks Ranking Size 36→46) showed the most significant gains.
- **OOD Robustness**: Carrot on Plate (+19pp) shows that SNR gating triggers more aggressive MCMC under visual noise (low SNR), allowing the system to escape local optima.

## Highlights & Insights
- **Reframing the Problem**: Elevates the engineering phenomenon of "drift" to a sampling theory problem, providing an analytical relationship $T_{eff}\propto \epsilon^{1-\alpha}$. This is a rare robotics work that is self-consistent from theory to empirical results.
- **Entropy-SNR Bridge**: Connects "signal detection" and "policy uncertainty" via $\text{SNR}_t=\log|\mathcal{A}|-\mathcal{H}$, providing a zero-overhead metacognitive trigger.
- **Innovation an MCMC for Continuous Control**: Uses the foundation model $p_\theta$ as a proposal (resampling suffix at temperature $1/\alpha$) + the power distribution as a target, avoiding the fixed computational budget issues of Tree-of-Thoughts methods.
- **Training-Free**: As a pure inference-time plug-in, it applies to any base VLA and is highly deployment-friendly.

## Limitations & Future Work
- $\gamma$ is a fixed threshold. Although a derivation is provided in Appendix F, adaptive thresholding was not implemented. Entropy distributions vary significantly across different tasks, potentially requiring per-task tuning.
- On bimanual tasks, single-arm drift remains a bottleneck (Block Handover 67/100, Handover Block 38/100), suggesting that chunk-based MCMC has limited modeling of coordination constraints.
- Increasing $N_{MCMC}$ directly increases wall-clock time. Real-time control scenarios require fine-tuning the precision-latency tradeoff; the paper does not provide a complete latency profile.
- Theorems 3.1/3.2 assume a constant single-step drift rate $\epsilon$, but in practice $\epsilon$ is strongly correlated with scene complexity, meaning theoretical gains might be overestimated.
- All experiments were conducted in simulation; the sim2real gap on real robots has not been evaluated.

## Related Work & Insights
- **vs. TACO (Yang et al. 2025)**: TACO treats inference as a contextual bandit, relying on a reranker to choose the best candidate. CAPS uses MCMC for active iterative refinement, which can converge even when the initial candidate set quality is poor.
- **vs. Tree-of-Thoughts (Yao et al. 2023)**: ToT uses fixed computation for global search, whereas CAPS uses SNR gating to trigger computation on demand, making it more suitable for real-time control.
- **vs. RoboMonkey (Kwok et al. 2025)**: RoboMonkey trains a specialized verifier, while CAPS is entirely training-free.
- **vs. Karan & Du 2026 (Power sampling in symbolic reasoning)**: Transfers power distribution theory from the symbolic domain to continuous action manifolds, proving the mathematical framework's validity across both modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "drift = sampling error" redefinition and SNR-gated MCMC provides a fresh perspective in the VLA field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robust comparison across three benchmarks and multiple baselines while remaining training-free, though missing real robot tests and latency profiles.
- Writing Quality: ⭐⭐⭐⭐ Good alignment between theory (Theorem 3.1/3.2) and algorithms (dual-process pseudocode), though some sections are wordy.
- Value: ⭐⭐⭐⭐ Provides a training-free path for performance enhancement in long-horizon tasks for existing VLA models, immediately applicable in engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](hdflow_hierarchical_diffusion-flow_planning_for_long-horizon_tasks.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[ICML 2026\] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation](tapsampling_inference-time_sampling_with_a_task-progress-understanding_verifier_.md)
- [\[ICML 2026\] EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents](embguard_constructing_hazard-aware_guardrails_for_safe_planning_in_embodied_agen.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)

</div>

<!-- RELATED:END -->
