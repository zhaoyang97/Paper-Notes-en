---
title: >-
  [Paper Note] Distributional Inverse Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline IRL] This paper proposes DistIRL: a framework for offline inverse reinforcement learning that models the reward itself as a conditional distribution and upgrades the constraint…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline IRL"
  - "Reward Distribution"
  - "First-order Stochastic Dominance"
  - "Distortion Risk Measures"
  - "Neural Behavioral Modeling"
date: 2026-05-08
content_hash: d59bef6e17e62f50
---

# Distributional Inverse Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2510.03013](https://arxiv.org/abs/2510.03013)  
**Code**: Not disclosed  
**Area**: Reinforcement Learning / Inverse Reinforcement Learning / Distributional RL / Risk-Sensitive Imitation  
**Keywords**: Offline IRL, Reward Distribution, First-order Stochastic Dominance, Distortion Risk Measures, Neural Behavioral Modeling  

## TL;DR
This paper proposes DistIRL: a framework for offline inverse reinforcement learning that models the reward itself as a conditional distribution and upgrades the constraint "the expert is superior to the learner" from expectation to First-order Stochastic Dominance (FSD). By relaxing the intractable 0/1 indicator function of FSD into an optimizable risk-weighted objective using Distortion Risk Measures (DRM), it achieves the first systematic simultaneous recovery of complete reward distributions and distribution-aware policies from offline demonstrations.

## Background & Motivation

**Background**: Classical offline IRL follows the MaxEntIRL/IQ-Learn/ML-IRL paradigm, treating rewards as a deterministic function $r(s,a)\in\mathbb{R}$ and recovering parameters by matching occupancy measures or expected returns. Bayesian IRL introduces a posterior over reward parameters, but the likelihood remains driven by expectation-based terms like soft-$Q$.

**Limitations of Prior Work**: Rewards in many real-world scenarios are inherently random variables—for instance, in robotic contact-rich tasks, returns for the same $(s,a)$ pair exhibit high variance; in neuroscience, dopamine signals show significant trial-to-trial skewed jitter under identical behaviors. Matching only the expectation treats two reward distributions with the same mean but completely different variance/skewness/tails as equivalent, making higher-order structures "invisible" to the IRL objective.

**Key Challenge**: Distribution matching (e.g., Wasserstein distance) can measure the similarity between two distributions but does not imply the "expert is better than learner" ordinal relationship required for IRL. Conversely, looking only at the ordinal relationship of expectations discards higher-order moments. Thus, a target is needed that preserves both "expert dominance" semantics and propagates to the full distribution.

**Goal**: (1) Recover the **reward distribution** $q_\phi(r\mid s,a)$ in an offline setting without environment interaction; (2) Learn a **distribution-aware/risk-sensitive** policy based on this; (3) Provide convergence rate proofs rather than just empirical validation.

**Key Insight**: The authors observe that FSD (First-order Stochastic Dominance) exactly upgrades "$X$ is better than $Y$" from the mean level to the CDF level—$F_X(z)\le F_Y(z),\forall z$ not only implies $\mathbb{E}[X]\ge\mathbb{E}[Y]$ but also holds for any monotonic utility function. FSD is therefore naturally suited as a distributional version of the "expert is better than learner" constraint.

**Core Idea**: Replace the expectation difference in MaxEntIRL with the FSD violation $\int [F_{Z^E}(z)-F_{Z^\pi}(z)]_+\,dz$. This is formulated with an energy-based model and variational inference to learn the reward distribution. On the policy side, the unobservable FSD indicator function $\mathcal{I}(v)$ is relaxed into a computable distortion function $\tilde\xi(v)$, restoring it as a risk-sensitive policy objective in DRM form.

## Method

### Overall Architecture
The input consists of offline expert trajectories $\mathcal{D}=\{(s_t^E,a_t^E)\}$, a reward prior $p_0(r)$, and a chosen distortion function $\xi$ (defaulting to CVaR$_{0.05}$ in experiments). The output is a variational reward distribution $q_\phi(r\mid s,a)$ and a distribution-aware policy $\pi_\varphi(a\mid s)$, supplemented by a quantile regression critic $\theta$ to estimate return quantiles. The entire pipeline involves three-way updates in each outer iteration: starting from mini-batch states, return samples $Z^E, Z^\pi$ are constructed via Monte Carlo accumulation from expert and policy actions; the reward distribution $\phi$ is updated by FSD violation; the policy $\varphi$ is updated by the DRM objective; and the critic $\theta$ is updated via quantile Huber loss.

### Key Designs

1.  **FSD-form Reverse Objective $\mathcal{L}_{\text{FSD}}$**:
    - **Function**: Upgrades "expert better than learner" from expectation to the full distribution, enabling reward learning to perceive higher-order moments.
    - **Mechanism**: Defines the violation as $\mathcal{L}_{\text{FSD}}=\int [F_{Z^E}(z)-F_{Z^\pi}(z)]_+\,dz$. Using change of variables, this is rewritten in quantile space as $\int_0^1 [F_{Z^\pi}^{-1}(v)-F_{Z^E}^{-1}(v)]_+\,dv$, approximated via Monte Carlo and order statistics $z_{(k)}$ for empirical quantiles $F_{Z^\pi}^{-1}(k/N)\approx z_{(k)}$, avoiding the need for an explicit CDF.
    - **Design Motivation**: Symmetric distances like Wasserstein lose the "dominance" order. FSD provides a differentiable violation while automatically implying mean dominance (Corollary 4.3), representing the minimal modification to generalize MaxEntIRL to the distributional level.

2.  **Energy-Based Model + Variational Inference for Reward Distribution**:
    - **Function**: Interprets the FSD violation as a log-likelihood, allowing the learning of a full $q_\phi(r \mid s, a)$ rather than a point estimate within a Bayesian framework.
    - **Mechanism**: Constructs an energy-based likelihood $p(\mathcal{D}\mid r)\propto\exp(-\mathcal{L}_{\text{FSD}}(\pi,r))$, introduces a variational posterior $q_\phi$, and maximizes the ELBO to obtain $\mathcal{L}_r(\phi)=\mathbb{E}_{q_\phi}[\mathcal{L}_{\text{FSD}}]+\mathrm{KL}(q_\phi\Vert p_0)$. $q_\phi$ is instantiated as an Azzalini skew-normal (for neuroscience) or a quantile function (for robotics) to support efficient sampling, closed-form KL, and differentiable gradients.
    - **Design Motivation**: FSD itself is an energy function without an explicit probabilistic model. EBM + VI provides a full Bayesian learning interface for reward distributions and naturally aligns the convex regularizer $\psi(r)$ in MaxEntIRL with the KL term.

3.  **DRM Relaxation for Tractable FSD Indicator**:
    - **Function**: Resolves the unobservability of the indicator function $\mathcal{I}(v)=\mathbb{1}_{F_{Z^\pi}^{-1}(v)\ge F_{Z^E}^{-1}(v)}$ in the FSD policy objective, turning the policy side into a differentiable risk-sensitive objective.
    - **Mechanism**: Substitutes $\mathcal{I}(v)$ with a non-decreasing distortion function $\tilde\xi(v)\ge 0$. The policy objective becomes $\mathcal{L}_\pi(\varphi)=\int_0^1 F_{Z^\pi}^{-1}(v)\,d\tilde\xi(v)+\mathcal{H}(\pi_\varphi)$, which is a DRM expectation plus maximum entropy. Proposition 4.6 further proves that "DRM dominance for all $\xi$" is equivalent to FSD dominance, ensuring the relaxation can theoretically reach the optimal solution of the original FSD objective.
    - **Design Motivation**: Direct optimization of $\mathcal{I}(v)$ is intractable due to unobservability. The DRM relaxation preserves the risk signal regarding which quantile the expert trajectories prefer (e.g., CVaR$_{0.05}$ emphasizes the lower tail) and automatically embeds policy learning into the existing quantile regression critic toolchain.

### Loss & Training
The algorithm employs alternating optimization: the reward network is updated via $\phi_{k+1}\leftarrow\phi_k-\eta^\phi\nabla\mathcal{L}_r(\phi_k)$ (Eq. 3); the critic performs off-policy evaluation using quantile Huber loss $\mathcal{L}_{QR}$; the policy is updated via $\varphi_{k+1}\leftarrow\varphi_k+\eta^\varphi\nabla\mathcal{L}_\pi(\varphi_k)$ with a KKT-form KL constraint $\min_\pi \mathrm{KL}(\pi\,\Vert\,\tfrac{1}{Z}\exp\{M_\xi(Z^\pi)\})$ (Ziebart et al.). Theoretically, with a step size $\eta_k=\eta_0 k^{-1/2}$, the algorithm achieves an iteration complexity of $\mathcal{O}(\varepsilon^{-2})$ (Theorem 5.6). The entire process is purely offline, requiring no environment model or online rollouts.

## Key Experimental Results

### Main Results
On risk-sensitive D4RL tasks with sparse heavy penalties (HalfCheetah high-speed triggers $-70$, Walker2D/Hopper large pitch triggers $-30/-50$), 10 expert trajectories were collected using risk-averse DSAC for offline IRL. Mean scores across 5 random seeds:

| Method | HalfCheetah | Hopper | Walker2d |
|------|-------------|--------|----------|
| **DistIRL (Gauss)** | **3469±59** | **886±1** | **1526±148** |
| DistIRL-qrt (Quantile) | 3294±172 | 747±79 | 1211±182 |
| BC | 2828±281 | 346±1 | 1321±26 |
| ValueDICE | 1259±78 | 260±10 | 798±311 |
| Offline ML-IRL | 826±231 | 192±56 | 240±50 |
| **Expert** | 3540±44 | 892±3 | 1478±200 |

DistIRL approaches expert performance across all three risk-sensitive tasks, significantly outperforming BC and expectation-matching methods like ValueDICE/ML-IRL. The latter fail due to risk-neutral assumptions or misaligned pre-trained transition models. On risk-neutral D4RL (Table 2), DistIRL remains SOTA in Hopper/Walker2d and is second only to ML-IRL (which uses extra non-expert data) in HalfCheetah, demonstrating the framework's broad applicability beyond distributional rewards.

### Ablation Study
HalfCheetah + Right-skewed Gaussian reward + Risk-averse expert (Normalized scores):

| Configuration | Score | Note |
|------|------|------|
| **DistIRL (Dis-QR-FSD)** | **1.00±0.02** | Full model: Dist. reward + QR critic + FSD loss |
| Dis-TD-FSD | 0.67±0.31 | TD critic instead of QR; high variance |
| Dis-TD-Mean (≈BIRL) | 0.33±0.01 | Dist. reward but mean-matching only; performance halved |
| Dis-QR-Mean | 0.22±0.02 | Dist. reward + mean-matching; dropped performance |
| Det-TD-Mean (≈ValueDICE) | 0.22±0.00 | No distributional signals |
| Det-QR-Mean (≈RIZE) | 0.00±0.01 | Worst performance |

### Key Findings
- **FSD loss is the core of the performance jump**: Moving from Dis-QR-Mean → Dis-QR-FSD increases the score from 0.22 to 1.00, which is far more effective than adding a quantile critic or distributional reward model alone.
- **BIRL is equivalent to Dis-TD-Mean (score 0.33)**: This validates the motivation that having a distributional reward assumption without a distribution-aware objective fails to recover true variance.
- **Spontaneous rodent behavior experiments (§6.2)**: S-DistIRL (skew-normal reward) achieved the highest Pearson correlation (~0.3) with dopamine signals and estimated reward means, alongside the lowest W-1 distance. This shows skewness is vital for asymmetric tails in neural data. In a 5x5 gridworld, the same model recovered both means and the variance $\sigma^2=1$ at the top-right corner, while BIRL only recovered means and hallucinated false values in the bottom-left.

## Highlights & Insights
- **Upgrading IRL's "Expert Dominance" to FSD**: This defines superiority using the entire CDF rather than a single point. This generalization, coupled with a differentiable quantile space approximation, solves both the invisibility of reward higher-order moments and the lack of policy risk-awareness without increasing engineering complexity.
- **Dual Role of DRM Relaxation**: The distortion function $\tilde\xi$ is both an engineering trick to make the FSD indicator tractable and a knob to control policy risk preference (CVaR/Wang/POW…). Proposition 4.6 provides a theoretical closed loop where the intersection of all $\xi$ recovers FSD.
- **Transferable Design**: The EBM + VI reward learning skeleton is decoupled from specific distribution families. The paper presents skew-normal and quantile function instantiations; the same logic could be extended to Diffusion priors or Normalizing Flows for OOD robust reward modeling, RLHF preference modeling, or RL fine-tuning of LLMs.

## Limitations & Future Work
- The algorithm remains fundamentally part of the MaxEntIRL family; reward identifiability only holds under the chosen prior, variational family, and FSD inductive bias. The paper does not claim unique recovery of ground-truth rewards.
- The distortion function $\xi$ is currently manually selected (default CVaR$_{0.05}$). Performance may degrade if the expert's true risk preference significantly deviates from the chosen DRM. Future work should learn $\xi$ from demonstrations.
- Current experiments model the reward distribution independently for each $(s,a)$, ignoring correlations across states. This assumption might be too strong for contact-rich robotics or multi-step games.

## Related Work & Insights
- **vs MaxEntIRL / IQ-Learn / Offline ML-IRL**: These define expert constraints at the expectation level. This work pushes constraints to the CDF level to perceive higher-order moments and eliminates reliance on pre-trained transition models.
- **vs Bayesian IRL (BIRL)**: BIRL learns the posterior of deterministic reward parameters driven by soft-$Q$. DistIRL learns the conditional distribution of the reward itself, driven by an FSD energy function, allowing it to distinguish between rewards with the same mean but different variance/skewness.
- **vs Distributional RL (C51/QR-DQN/IQN)**: DistRL models return distributions for forward RL with known rewards. DistIRL solves the inverse problem for unknown reward distributions and explicitly links policy risk preferences to distortion functions.
- **vs Risk-aware Imitation (Singh 2018, Lacotte 2019, Cheng 2023)**: These works handle risk-sensitive policies but use point-estimate rewards. DistIRL learns both reward distributions and risk-sensitive policies with a proven $\mathcal{O}(\varepsilon^{-2})$ convergence.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Introducing FSD + DRM into IRL is the first framework to systematically learn reward distributions in an offline setting, with self-consistent theory and engineering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers gridworld, real neuroscience data, and risk-sensitive/risk-neutral D4RL. However, it only compares against 5 baselines and uses a default DRM, and the lack of public code affects reproducibility.
- **Writing Quality**: ⭐⭐⭐⭐ Math definitions and the derivation chain (EBM → VI → FSD → DRM) are clear. The recovery of the original problem via Proposition 4.6 is particularly elegant.
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical and provable tool for scientific and robotic scenarios where rewards are inherently stochastic. The FSD/DRM paradigm can be applied to RLHF, animal behavior modeling, and safe robotic imitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Convergence Theorems for Entropy-Regularized and Distributional Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/convergence_theorems_for_entropy-regularized_and_distributional_reinforcement_le.md)
- [\[ICML 2026\] Safe In-Context Reinforcement Learning](safe_in-context_reinforcement_learning.md)
- [\[ICML 2026\] Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality](reinforcement_learning_for_reachability_guaranteeing_asymptotic_optimality.md)
- [\[NeurIPS 2025\] Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](../../NeurIPS2025/reinforcement_learning/inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [\[ICML 2026\] EchoRL: Reinforcement Learning via Rollout Echoing](echorl_reinforcement_learning_via_rollout_echoing.md)

</div>

<!-- RELATED:END -->
