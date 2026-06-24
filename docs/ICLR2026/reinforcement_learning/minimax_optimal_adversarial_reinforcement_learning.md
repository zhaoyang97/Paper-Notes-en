---
title: >-
  [Paper Note] Minimax Optimal Adversarial Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Adversarial transition kernels] This paper provides the first proof that sublinear regret remains achievable in episodic MDPs where transition kernels are chosen arbitrarily by an adversary (fully adversarial). It proposes the AD-FTRL algorithm, which reduces regret to $\tilde{O}(\sqrt{(|S||A|)^K T})$, and establishes minimax optimality by constructing a matching lower bound.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Adversarial transition kernels"
  - "history-dependent policies"
  - "FTRL"
  - "trajectory occupancy measures"
  - "minimax optimal"
  - "sublinear regret"
date: 2026-05-08
content_hash: b9617a15647c04ef
---

# Minimax Optimal Adversarial Reinforcement Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QEcSLhfOoQ](https://openreview.net/forum?id=QEcSLhfOoQ)  
**Code**: Not provided (Theoretical paper)  
**Area**: Reinforcement Learning / Online Learning / Adversarial MDPs  
**Keywords**: Adversarial transition kernels, history-dependent policies, FTRL, trajectory occupancy measures, minimax optimal, sublinear regret  

## TL;DR
This paper provides the first proof that sublinear regret remains achievable in episodic MDPs where transition kernels are chosen arbitrarily by an adversary (fully adversarial). It proposes the AD-FTRL algorithm, which reduces regret to $\tilde{O}(\sqrt{(|S||A|)^K T})$, and establishes minimax optimality by constructing a matching lower bound.

## Background & Motivation

**Background**: Extensive research exists on adversarial reinforcement learning, but the majority assumes that only the loss functions $\ell_t$ change adversarially while the transition kernel $P$ remains fixed. In such settings, the problem can be reduced to "adversarial average losses $\bar\ell$," where Markov optimal policies exist, yielding a sublinear regret of $\tilde{O}(\sqrt{|S||A|T})$.

**Limitations of Prior Work**: The problem becomes significantly more difficult when the transition kernel $P_t$ is also chosen arbitrarily by an adversary in each episode. Existing works (Jin et al. 2023; Wei et al. 2022) attempt to estimate a "center/average" transition kernel $\bar P$ and update policies based on this estimate, resulting in bounds of $\tilde{O}(\sqrt{|S||A|T}+C_P)$, where $C_P$ measures the cumulative adversarial perturbation of the transition kernels.

**Key Challenge**: In the fully adversarial setting, $C_P$ can be as large as $O(T)$, causing these bounds to degenerate into **linear regret**, implying no learning occurs. The fundamental reason is that average occupancy measures do not equal the occupancy measures under the average transition kernel ($\bar p_\pi \ne p_{\bar P,\pi}$). The path of estimating $\bar P$ inherently introduces an unavoidable $C_P$ penalty, and $\bar p_\pi$ may not even correspond to any physically realizable transition model.

**Goal**: To answer a fundamental question: Is it possible to achieve sublinear regret in a setting with fully adversarial transitions and bandit feedback (where only losses of visited state-action pairs are observed), and what are the information-theoretic limits of this setting?

**Core Idea**: ① **Abandon transition kernel estimation** and instead use importance sampling to directly obtain unbiased estimates of **trajectory-level occupancy measures** $q_{P,\pi}$, bypassing the $C_P$ penalty at the root. ② Execute FTRL over the **history-dependent policy class** (the paper first proves that optimal policies under adversarial transitions must be history-dependent, as Markov policies are strictly suboptimal). ③ Construct a hard MDP instance and provide a matching lower bound using composite hypothesis testing to confirm minimax optimality.

## Method

### Overall Architecture
AD-FTRL reformulates the adversarial transition MDP as an "online linear optimization problem over the trajectory space." In each episode, a trajectory $\tau_t$ is sampled using the current history-dependent policy $\pi_t$. An unbiased estimate of the trajectory occupancy measure and loss is obtained via importance sampling. Subsequently, FTRL (with a carefully designed entropy/KL regularization) updates to the next history-dependent policy. This entire process **completely avoids transition kernel estimation**, thereby eliminating the $C_P$ term.

```mermaid
flowchart LR
    A[History-dependent Policy πt] --> B[Sample Trajectory τt<br/>Bandit feedback loss]
    B --> C[Importance Sampling<br/>Unbiased estimate of q̂ and loss L̂t]
    C --> D[Accumulate Υt+1 = Υt + L̂t]
    D --> E[FTRL Update<br/>⟨Ππ,Υ⟩ + 1/η·Φ(Ππ)]
    E --> A
```

### Key Designs

**1. History-dependent Policy Class: Markov Suboptimality under Adversarial Transitions.** While optimal policies in standard MDPs with fixed transitions are Markovian, the paper uses a counterexample with an alternating transition sequence $\{P_1,P_2,P_1,P_2,P_1\}$ (Figure 1) to show that when initial states "leak" information about the adversary's choice of transition, the optimal policy must depend on the history. In this example, $\bar V(\pi^*_{\text{Markov}})=\tfrac{2}{5}$ while $\bar V(\pi^*_{\text{his}})=0$, indicating a non-vanishing gap. Consequently, the authors expand the search space to the history space $H_k$ (much larger than the state space), where a policy $\pi^k_t: H_k \to \Delta(A)$ maps the full history to an action distribution—the fundamental premise for subsequent algorithms and lower bounds.

**2. Trajectory Occupancy Measures + Importance Sampling: Bypassing Kernel Estimation.** Instead of estimating state-action occupancy $p_{P,\pi}$ (which depends on known transitions), the paper defines trajectory-level occupancy $q_{P,\pi}(\tau)=P(s_0)\prod_{k=0}^{K-1}\pi(a_k\mid h_k)\,P(s_{k+1}\mid s_k,a_k)$. Using the trajectory sampled by the behavior policy $\pi_t$, an unbiased estimate is obtained via importance weighting (Lemma 3.1): $\mathbb{E}_{\tau_t}\big[\frac{\prod_k \pi(a_k\mid h_k)}{\prod_k \pi_t(a_k\mid h_k)}\mathbb{I}_{\tau_t=\tau}\big]=q_{\pi,P_t}(\tau)$. The estimated return $\hat V_t(\pi)=\langle \hat q_{\pi,P_t},\ell_t\rangle$ satisfies $\mathbb{E}[\hat V_t(\pi)]=V_t(\pi)$ without requiring knowledge of $P_t$. This is the key to eliminating the $C_P$ penalty—estimating the "visited trajectory distribution" directly rather than through the biased proxy of an "average transition kernel." Remark 1 further argues that the problem cannot be simplified to a (contextual) bandit: actions change subsequent state distributions, violating the "exogenous context" assumption.

**3. Policy-Environment Decoupled FTRL Regularization: Avoiding Unknowns in Regularizers.** Plugging estimates containing unknown transitions directly into regularizers leads to high variance and unstable updates. The authors decompose the average trajectory occupancy as $\bar q_{\pi,t}(\tau)=\Pi_\pi(\tau)\cdot\bar{\mathsf{F}}_t(\tau)$, where $\Pi_\pi(\tau)=\prod_{k}\pi(a_k\mid h_k)$ depends only on the policy and $\bar{\mathsf F}_t$ depends only on the (unknown) transition distribution. The regularizer **only uses the policy-related part** $\Phi_t(\Pi_\pi)=\sum_\tau \Pi_\pi(\tau)\log\Pi_\pi(\tau)$ (Shannon entropy / KL form), allowing the FTRL update to be written as $\pi_t=\arg\min_\pi \langle \Pi_\pi,\Upsilon_t\rangle+\frac{1}{\eta_t}\Phi_t(\Pi_\pi)$, where $\Upsilon_t=\sum_{\iota<t}L_\iota$ and $L_\iota(\tau)=\mathsf F_\iota(\tau)\ell_\iota(\tau)$. Loss estimates utilize clipped importance weights $\hat L_t=\frac{\ell_{\tau_t}}{\gamma+\prod_k \pi_t(a_k\mid h_k)}[\mathbb{I}_{\tau=\tau_t}]$ (with $\gamma$ for numerical stability). Updated policies are constrained within the $\epsilon$-greedy history-dependent class $C_{\pi,\epsilon}$ (typically $\epsilon=1/T$) to ensure unbiased importance sampling.

**4. Composite Hypothesis Testing Lower Bound: Confirming Minimax Optimality.** To prove the bound is unimprovable, the authors construct a hard MDP instance and reduce regret minimization to **composite hypothesis testing** (instead of the binary/multiple hypothesis testing common in standard RL)—a necessity for adversarial transitions. Using information-theoretic tools, they derive $\mathrm{Reg}_T\ge \Omega(\sqrt{(|S||A|)^K T})$, which is of the same order as the upper bound. Compared to the $\Omega(\sqrt{2^K T})$ lower bound in Markov games (Tian et al. 2021), this bound is tighter and explicitly characterizes dependence on state/action space dimensions.

## Key Experimental Results

This work is purely theoretical with no numerical experiments; the following table summarizes the core results and comparisons.

### Main Results (Regret Upper and Lower Bounds)

| Result | Setting | Regret Bound |
|---|---|---|
| Theorem 5.1 (Upper Bound) | Adversarial bandit loss + Unknown adversarial transitions, with $\eta=(\tfrac{|S|}{|A|})^{K/2}\tfrac{1}{\sqrt{T\log(|S||A|)}}$, $\gamma=\tfrac{1}{2(|S||A|)^{K/2}\sqrt{T}}$ | $\tilde{O}\big((|S||A|)^{K/2}\sqrt{T}\big)$ |
| Theorem 5.2 (Lower Bound) | Any algorithm, $T>4(|S||A|)^K\log T$ | $\Omega\big(\sqrt{(|S||A|)^K T}\big)$ |

### Comparison with Prior Work

| Work | Transition Setting | Regret |
|---|---|---|
| Jin et al. 2020a/2021 | Adv. loss + **Fixed** transitions | $\tilde{O}(\sqrt{|S||A|T})$ |
| Jin et al. 2023 / Wei et al. 2022 | Adv. transitions (Estimate $\bar P$) | $\tilde{O}(\sqrt{|S||A|T}+C_P)$, $\to O(T)$ (linear) in fully adversarial case |
| Tian et al. 2021 (Lower Bound) | Markov games | $\Omega(\sqrt{2^K T})$, lacks $|S|,|A|$ dependence |
| **Ours (AD-FTRL)** | **Fully adversarial** unknown transitions | $\tilde{O}(\sqrt{(|S||A|)^K T})$, **sublinear and minimax optimal** |

### Key Findings
- In the fully adversarial case ($C_P=O(T)$), previous methods degrade to linear regret, whereas AD-FTRL remains sublinear for sufficiently large $T$—providing the first affirmative answer to this problem.
- The exponential term $(|S||A|)^{K/2}$ in regret regarding state-action dimensions is unimprovable: the upper and lower bounds match, characterizing the intrinsic difficulty of the problem.
- This $K$-th power dependence is inherent to the problem (arising from the history space expanding exponentially with the horizon) and is not a looseness in algorithm analysis.

## Highlights & Insights
- **Changing Coordinates over Estimators**: Completely abandoning "adversarial transition kernel estimation" (doomed by $C_P$ penalties) in favor of unbiased estimation in the trajectory occupancy measure space is the key move to break the linear regret barrier.
- **Clever Policy-Environment Decoupled Regularization**: The decomposition $\bar q=\Pi_\pi\cdot\bar{\mathsf F}$ allows the regularizer to depend only on computable policy parts, effectively isolating unknown transitions within the linear loss term.
- **Composite Hypothesis Testing Framework for Lower Bounds**: This is an independent technical contribution, providing a unified template for lower bound analysis in broad classes of "adversarial/time-varying transition" problems.

## Limitations & Future Work
- **Exponential Growth in Horizon $K$**: The authors acknowledge that $(|S||A|)^{K/2}$ is unfavorable for long horizons; while proven minimax optimal, it implies that fully adversarial transitions are inherently difficult over long horizons, limiting practicality.
- **Purely Theoretical, No Experiments**: The algorithm requires FTRL updates over a trajectory space of size $|A|^K$. Computational/storage feasibility is not discussed, and numerical validation is absent.
- Results are limited to **tabular layered MDPs**; optimal bounds under function approximation, continuous states/actions, and structured adversaries (between fully adversarial and fixed) remain open.
- The practical impact of the $\epsilon$-greedy ($\epsilon=1/T$) constraint and importance weight clipping $\gamma$ on finite-sample performance is not fully characterized.

## Related Work & Insights
- **Adversarial Loss + Fixed Transitions**: Works like Even-Dar et al. 2009, Zimin & Neu 2013, and Jin et al. 2020a/2021 established the occupancy measure + FTRL paradigm; this paper upgrades it from state-action occupancy to trajectory occupancy.
- **Adversarial/Perturbed Transitions**: Jin et al. 2023, Wei et al. 2022, Chen et al. 2021, and Lykouris et al. 2021 take the "estimated center kernel + $C_P$" route; this work highlights its fundamental limitations in fully adversarial settings and provides an alternative.
- **Lower Bounds**: Domingues et al. 2021 ($\Omega(\sqrt{|S||A|T})$ for standard RL) and Tian et al. 2021 (Markov games); this paper extends lower bounds to adversarial transitions with explicit dimensional dependence via composite hypothesis testing.
- **Inspiration**: When estimates of an intermediate variable (like a transition kernel) possess unavoidable bias, switching to an equivalent representation that allows unbiased estimation (trajectory occupancy) is often superior—a strategy applicable to other adversarial online learning problems with "hard-to-estimate kernels."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Breaks the linear regret barrier under fully adversarial transitions for the first time; the combination of trajectory occupancy, history-dependent policies, and composite hypothesis testing is a major conceptual breakthrough.
- **Experimental Thoroughness**: ⭐⭐ A purely theoretical paper with complete theorems and matching bounds, but entirely lacks numerical experiments and discussion on computational feasibility in exponential trajectory spaces.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, step-by-step exposition of challenges and solutions, and thorough explanations of the counterexample and decomposition. Notation (e.g., trajectory distribution symbols) is slightly heavy.
- **Value**: ⭐⭐⭐⭐ Solves a long-standing fundamental open problem in adversarial RL and provides minimax limits; high theoretical value, though practical value is constrained by exponential $K$ dependence and lack of experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[ICLR 2026\] Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation](robust_deep_reinforcement_learning_against_adversarial_behavior_manipulation.md)
- [\[ICLR 2026\] Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning](potentially_optimal_joint_actions_recognition_for_cooperative_multi-agent_reinfo.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)

</div>

<!-- RELATED:END -->
