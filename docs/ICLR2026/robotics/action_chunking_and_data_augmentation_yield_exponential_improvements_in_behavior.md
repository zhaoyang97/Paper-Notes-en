---
title: >-
  [Paper Note] Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control
description: >-
  [ICLR 2026][Robotics & Embodied AI][Paper Note] This paper provides the first theoretical guarantee for two empirical techniques in imitation learning—action chunking and expert noise-injection data augmentation—using "incremental stability" from control theory. It proves they suppress the compounding error that accumulates exponentially over time in continuous cont
tags:
  - ICLR 2026
  - Robotics & Embodied AI
date: 2026-05-08
content_hash: d858e3674c3674a4
---
# Action Chunking and Exploratory Data Collection Yield Exponential Improvements in Behavior Cloning for Continuous Control

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jiWXDvw1Lf](https://openreview.net/forum?id=jiWXDvw1Lf)  
**Code**: TBD  
**Area**: Robotics / Imitation Learning Theory  
**Keywords**: Imitation Learning, Behavior Cloning, Compounding Error, Action Chunking, Noise Injection, Incremental Stability  

## TL;DR
This paper provides the first theoretical guarantee for two empirical techniques in imitation learning—action chunking and expert noise-injection data augmentation—using "incremental stability" from control theory. It proves they suppress the compounding error that accumulates exponentially over time in continuous control behavior cloning (BC) to be "horizon-free" under various conditions.

## Background & Motivation
**Background**: In robotics and continuous control, learning policies from expert demonstrations (imitation learning / behavior cloning, BC) is a dominant paradigm. Recent works such as ACT and Diffusion Policy have significantly improved performance through three types of interventions: (1) predicting and executing a sequence of actions open-loop (action chunking), (2) meticulously filtering or augmenting expert data, and (3) using generative architectures (e.g., conditional diffusion) for policy parameterization.

**Limitations of Prior Work**: While the third point (generative architectures) has been extensively studied, the precise reasons why the first two are effective remain unexplained. Worse, Simchowitz et al. (2025) proved a negative result—in continuous state spaces, even if the expert and dynamics are "well-behaved," the compounding error of BC can grow **exponentially** with the task horizon $T$, and this cannot be avoided simply by modifying the learning algorithm (loss, randomization). This contrasts sharply with the discrete setting (language modeling) where errors grow only polynomially.

**Key Challenge**: Empirically, action chunking and data augmentation are clearly effective. However, existing theoretical tools such as "information-theoretic coverage" and "persistent excitation (PE)" neither explain their effectiveness nor provide bounds better than exponential. Existing methods to avoid compounding errors (DAGGER, DART) either require repeated interactive expert queries or rely on stability oracles/Jacobian information of the dynamics.

**Goal**: Under a minimal continuous control setting with a state-observable, deterministic expert, this paper aims to provide provable theoretical guarantees that action chunking and one-time noise injection can avoid compounding errors using "vanilla" BC, without relying on interactive expert queries or system priors.

**Key Insight**: **Control-theoretic stability is the underlying mechanism.** Action chunking induces closed-loop incremental stability by "non-Markovianizing" the policy. Noise injection precisely excites the "controllable directions" around expert trajectories that would otherwise lead to error explosion, making one-time collected data sufficient to cover the vulnerabilities of BC.

## Method

### Overall Architecture
The logical framework of the paper is as follows: first, formalize "compounding error" via incremental stability (EISS) as the amplification factor of "trajectory error $J_{\mathrm{TRAJ}}$ relative to the regression error $J_{\mathrm{DEMO}}$ on the expert distribution." Then, provide two complementary "positive counter-examples" for different environment difficulties to bypass the two branches of the negative theorem from Simchowitz et al. (2025).

```mermaid
flowchart TD
    A[Continuous Control BC<br/>Worst-case Exponential Error<br/>Simchowitz 2025 Lower Bound] --> B{Is Open-Loop<br/>Dynamics Stable?}
    B -->|Open-loop EISS holds| C[Practice 1: Action Chunking<br/>Modify policy parameterization only]
    B -->|Not necessarily stable| D[Practice 2: Noise Injection<br/>Must modify data distribution]
    C --> E[Thm 1: Sufficiently long chunk<br/>Induces closed-loop EISS<br/>→ J_TRAJ ≲ O*(1)·J_DEMO]
    D --> F[Thm 2: Mixed clean+noised data<br/>Excites controllable subspace<br/>→ J_TRAJ ≲ O*(T)·J_DEMO]
```

The objective is the squared trajectory error $J_{\mathrm{TRAJ},T}(\hat\pi)=\mathbb{E}\big[\sum_{t=1}^{T}\min\{1,\|x^{\hat\pi}_t-x^{\pi^\star}_t\|^2+\|u^{\hat\pi}_t-u^{\pi^\star}_t\|^2\}\big]$, while BC directly optimizes the on-policy expert error $J_{\mathrm{DEMO},T}$. The "compounding error problem" refers to $J_{\mathrm{TRAJ}}\gtrsim C^T\cdot J_{\mathrm{DEMO}}$ ($C>1$).

### Key Designs

**1. Translating Compounding Error into Controllable Quantities via Incremental Stability (EISS)**: The paper uses "Incremental Input-to-State Stability (EISS)" from control theory as the analytical core. A system is $(C_{\mathrm{ISS}},\rho)$-EISS if any two initial state/input sequences satisfy $\|x_t-x'_t\|\le C_{\mathrm{ISS}}\rho^{t-1}\|x_1-x'_1\|+C_{\mathrm{ISS}}\sum_{k=1}^{t-1}\rho^{t-1-k}\|u_k-u'_k\|$, meaning bounded input perturbations cause only bounded state deviations that decay over time. This is the continuous control version of "recoverability." A key insight is that expert closed-loop EISS does not eliminate compounding errors; if the learned $\hat\pi$ makes the system **unstable**, the "input perturbation" it causes to the expert system will grow exponentially—stability must hold for the learned policy to be effective. The paper also notes that end-effector control, due to underlying PD trackers, naturally makes the "desired position → system state" closed-loop system open-loop stable, providing a realistic basis for the chunking assumption on actual robotic arms.

**2. Action Chunking Induces Closed-Loop Stability via Non-Markovian Structure (Practice 1 / Theorem 1)**: A chunking policy outputs $\ell$ actions at once and executes them open-loop for $\ell$ steps. The induced policy can be written as a closed-loop rollout of a base policy $\hat\pi$ over some (potentially inaccurate) simulated dynamics $\hat f$ for $\ell$ steps: $\mathrm{chunk}[\tilde\pi](x)=\big(\hat\pi(x),\hat\pi(\hat f^{\hat\pi}(x)),\dots,\hat\pi((\hat f^{\hat\pi})^{\ell-1}(x))\big)$. The core proposition proves that as long as the true dynamics $f$ is open-loop EISS and the simulation is EISS for $(\hat\pi, \hat f)$ itself, a sufficiently long chunk ($\ell>\log(1/\rho)^{-1}\log(\mathrm{poly}(L_\pi,C_{\mathrm{ISS}}))$) forces $(\tilde\pi, f)$ to be EISS on the true system (with decay rate $\tilde\rho=\rho^{1/2}$). This yields $J_{\mathrm{TRAJ},T}(\tilde\pi)\le O^\star(1)\,J_{\mathrm{DEMO},T}(\tilde\pi;P_{\pi^\star})$—**horizon-free**. Three counter-intuitive points are noted: chunking changes previous perceptions (it was thought to handle partial observability, multimodality, or long-range planning), its key role is actually the stabilization brought by "open-loop execution." **Predicting multiple steps but still re-planning step-by-step (receding-horizon, $\ell=1$) does not help.** The required chunk length grows only logarithmically with system constants (very short), and marginal returns diminish as it grows longer. This holds even under full state observability and deterministic experts, showing the value of chunking is independent of non-Markovianity.

**3. Noise Injection Exciting the "Controllable/Excitable Subspace" is Sufficient (Practice 2 / Theorem 2)**: When the open-loop system is unstable, purely algorithmic changes fail (Theorem A.(ii) rules out any means that do not modify the data distribution). Data must be modified. The approach is simple: collect trajectories by adding isotropic white noise to expert actions $\tilde x_{t+1}=f(\tilde x_t,\pi^\star(\tilde x_t)+\sigma_u z_t)$ with noise scale $\sigma_u$, but **record expert labels as the clean $\pi^\star(\tilde x_t)$** (contrary to RL intuition, which often noises the policy labels). Then, fit the model on a **mixture** $P_{\pi^\star,\sigma_u,\alpha}$ of $\alpha$ clean trajectories and $(1-\alpha)$ noised trajectories. The theoretical elegance lies in the fact that pure noise or pure noised trajectories introduce a "drift error" lower bound $\Omega(C_\pi^2\sigma_u^4)$ (Prop 4.1). Mixing clean and noised data bypasses this, allowing $\sigma_u$ to be set to the maximum allowed by smoothness without sacrificing regression error. Key coverage analysis (Prop 4.3/4.4) proves that compounding errors enter only through input channels and fall primarily within the controllable subspace $\mathrm{range}(W^u_{1:t})$. One only needs to control first-order errors on the **excitable subspace** $R^{\pi^\star}_t(\lambda)=\mathrm{span}\{v_i:\lambda_i\ge\lambda\}$; errors in small eigenvalue directions that are hard to excite decay naturally. Final result: $J_{\mathrm{TRAJ},T}(\hat\pi)\lesssim O^\star(T)\,\sigma_u^{-2}\,J_{\mathrm{DEMO},T}(\hat\pi;P_{\pi^\star,\sigma_u,0.5})$. Setting $\sigma_u=O^\star(1)$ yields a linear horizon bound $O^\star(T)$. The disruptive takeaway is: **naive white noise is enough** (no full-dimensional PE as in control theory or strong coverage as in RL is needed) because the easiest directions to excite are precisely where errors compound fastest, ensuring automatic alignment.

## Key Experimental Results
The goal of the experiments is to verify theoretical predictions and the assertion that "control-theoretic stability is the key mechanism." Benchmarks include popular robot learning environments (robomimic, MuJoCo HalfCheetah-v5 / Humanoid-v5).

### Main Results (Qualitative Trends)

| Phenomenon | Setting | Observation |
|------|------|------|
| Action chunking saves open-loop stable systems | robomimic `tool_hang`, full state obs, 100 expert trajectories | Success rate **sharply increases** from $\ell=1$ (re-planning) to longer chunks; predictive horizon has only temporary effects, the decisive factor is the actual open-loop chunk length. |
| Noise injection matches iterative methods | HalfCheetah-v5 | Sufficiently large white noise injection brings significant improvements, performance is **comparable to complex iterative methods like DAGGER/DART.** |
| Naive noise is more robust | Humanoid-v5 | DAGGER/DART underperform due to poor learned policy rollouts or aggressive noise covariance shaping; naive noise injection reliably provides local exploration. |

### Ablation Study

| Ablation | Setting | Conclusion |
|------|------|------|
| Clean labels vs Noised labels | HalfCheetah-v5, $\sigma_u=1$ (~0.4 element-wise on $[-1,1]^6$) | Fitting noised labels leads to **catastrophic failure**; using clean labels (Practice 2) improves performance—confirming labels must be clean. |
| Mixing ratio $\alpha$ | Fixed $\sigma_u=0.5$, sweep $\alpha\in[0,1]$ | Performance difference is marginal as long as noised trajectories are sufficient (supporting Eq 4.1). |
| Blind chunking on unstable systems | HalfCheetah-v5 (open-loop unstable) | Direct chunking is **catastrophic**, contrasting with `tool_hang` (open-loop stable), confirming chunking depends on open-loop stability. |

### Key Findings
- The success is determined by the **actual open-loop execution chunk length** rather than the prediction horizon; chunking remains critical in state-observable deterministic control.
- Noise injection works only when both "recording clean labels" and "mixing clean trajectories" are satisfied.
- These techniques are not universal: chunking relies on open-loop stability (guaranteed by end-effector controllers in practice), while noise injection relies on smoothness.

## Highlights & Insights
- **First non-interactive positive guarantee**: Proves for the first time in continuous state-action IL that interventions exist to stop compounding errors without iterative expert feedback or system priors, reducing the cost of methods like DAGGER/DART to one-time data collection.
- **Reinterpreting action chunking**: Repositions chunking from an engineering trick to handle partial observability/multimodality/long-range planning to a mechanism for "inducing control-theoretic stability"—an orthogonal and more fundamental explanation.
- **Refined coverage/excitation theory**: Proposes a more granular "excitation-on-demand" concept for continuous state spaces—pay the statistical cost only for the level of excitation needed, where white noise naturally allocates supervision to the most dangerous directions.
- **Lower bounds guiding algorithm design**: The drift lower bound $\Omega(C_\pi^2\sigma_u^4)$ is not a negative result but directly suggests the algorithmic prescription of "mixed distribution + clean labels," tightly linking theory and algorithm.

## Limitations & Future Work
- Chunking guarantees depend on the structural assumption that $(\hat\pi, \hat f) \in \mathcal{P}$ is EISS; how to explicitly (regularization/hierarchy) or implicitly (architectural inductive bias) ensure this remains an open problem.
- Section 4 relies heavily on smoothness, which is not strictly met by many applications like MPC; the lower bound itself is built on $C_\pi$ smoothness, indicating it is an inherent requirement for noise injection. Extension to piecewise smooth cases is for future study.
- While the theory suggests white noise is sufficient, it may not be ideal for high-dexterity robotics; robust perturbative data collection recipes still need design.
- The marginal benefits of iterative interaction (DAGGER, etc.) versus one-time collection and sharp characterization of stability constants in continuous space are left for future work.

## Related Work & Insights
- **Direct antecedents**: Simchowitz et al. (2025) provided the exponential lower bound for continuous IL compounding error (this paper's "Motivating Theorem A"). This paper is the "positive inverse proposition." Tu et al. (2022) introduced the "incremental stability scale," and Pfrommer et al. (2022) gave sufficient conditions for mild compounding error but required stability oracles/Jacobian information—this paper removes these strong requirements.
- **Sources of empirical tricks**: Action chunking from ACT (Zhao et al. 2023) and Diffusion Policy (Chi et al. 2023); data augmentation from DAGGER (Ross et al. 2011) and DART (Laskey et al. 2017).
- **Insights**: (1) When designing IL pipelines, "open-loop execution + end-effector stability" should be viewed as a stability guarantee rather than just a latency/bandwidth tradeoff; (2) "noising execution but labeling clean + clean/noise mixing" is a theoretically sound and safe recipe for data augmentation reachable for general BC pipelines; (3) control-theoretic stability serves as a unified lens for analyzing compounding errors in other sequential decision-making (and even generative policy) tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Provides the first non-interactive horizon-free theoretical guarantee for two major empirical tricks and introduces a refined "excitation-on-demand" concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematically verifies predictions on chunking length, noise labels, and mixing ratios on robomimic and MuJoCo; experiments are clearly positioned for a theory paper but small-scale and lack high-dimensional real robotics.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous logic; clear "Key Findings" summaries and derivation chains from lower bounds to algorithms. However, dense with theorems and control-theoretic terminology.
- **Value**: ⭐⭐⭐⭐⭐ Provides a solid theoretical foundation for two widely used but poorly explained tricks in robot imitation learning, with direct practical implications for method selection and data collection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning](../../AAAI2026/robotics/continuous_vision-language-action_co-learning_with_semantic-.md)
- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](real-time_robot_execution_with_masked_action_chunking.md)
- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICML 2026\] Mixture of Horizons in Action Chunking](../../ICML2026/robotics/mixture_of_horizons_in_action_chunking.md)
- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](../../AAAI2026/robotics/test-driven_reinforcement_learning_in_continuous_control.md)

</div>

<!-- RELATED:END -->
