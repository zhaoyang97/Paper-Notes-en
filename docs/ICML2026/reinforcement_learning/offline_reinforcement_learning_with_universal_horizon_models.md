---
title: >-
  [Paper Note] Offline Reinforcement Learning with Universal Horizon Models
description: >-
  [ICML 2026][Reinforcement Learning][universal horizon model] The authors remove the limitation that "Geometric Horizon Models (GHM) can only sample from a fixed discounted distribution" and propose the Universal Horizon…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "universal horizon model"
  - "geometric horizon model"
  - "n-step TD"
  - "winsorized geometric"
  - "flow matching"
date: 2026-05-08
content_hash: 2eaa4e511c3a07c7
---

# Offline Reinforcement Learning with Universal Horizon Models

**Conference**: ICML 2026  
**arXiv**: [2605.15603](https://arxiv.org/abs/2605.15603)  
**Code**: https://rllab-snu.github.io/projects/UHM/  
**Area**: Offline Reinforcement Learning / Model-based RL / Value Learning / World Models  
**Keywords**: universal horizon model, geometric horizon model, n-step TD, winsorized geometric, flow matching

## TL;DR
The authors remove the limitation that "Geometric Horizon Models (GHM) can only sample from a fixed discounted distribution" and propose the Universal Horizon Model (UHM), which can directly sample future states at an arbitrary horizon $n$. By truncating excessively long horizons using a Winsorized geometric distribution, the method achieves an average success rate improvement of approximately 14% over the strongest baseline across 100 OGBench tasks.

## Background & Motivation

**Background**: Offline reinforcement learning aims to learn strategies from static datasets, with Temporal Difference (TD) learning being the mainstream approach. Recently, it has been observed that $n$-step TD can significantly reduce bias in long-horizon tasks. Consequently, works such as Park et al.'s $n$-step series, action chunking, and hierarchical policies have focused on "compressing the effective horizon." Simultaneously, model-based offline RL employs dynamics models to generate synthetic on-policy trajectories for value expansion, theoretically addressing "out-of-distribution" issues.

**Limitations of Prior Work**: (1) Single-step dynamics models in offline settings suffer from compounding error explosion due to repeated inference on self-generated states; (2) Although the Geometric Horizon Model (GHM) avoids repeated inference by "jumping to the discounted future" in one step, it compresses all futures into a single geometric distribution $\text{Geom}(1-\tilde\gamma)$, making the long-tail portion extremely difficult to learn accurately; (3) GHM cannot specify exactly how many steps ahead a sampled state is, making it impossible to perform $n$-step TD or TD($\lambda$).

**Key Challenge**: To eliminate compounding errors, one must "jump directly," but "jumping directly to a future fixed-weighted by a geometric distribution" loses horizon granularity. Conversely, $n$-step TD requires knowing $n$. These two aspects are mutually exclusive within the GHM framework.

**Goal**: Construct a generative model that can "directly sample the future" to avoid repeated inference while allowing an explicit arbitrary horizon $n$ to be provided during sampling, where $n$ can follow any distribution. Based on this, provide a truly scalable offline value learning algorithm that is not destabilized by long-tail horizons.

**Key Insight**: The authors start from a mathematical observation—GHM is a "mixture of future distributions where $n\sim\text{Geom}(1-\tilde\gamma)$," while a single-step model is where $n\sim\delta(1)$." Both are special cases of the same family $m^\pi(x|s,a,n)$ across different $n$.

**Core Idea**: Learn the $n$-step transition measure $m^\pi(x|s,a,n)=\Pr(s_n=x\mid s_0=s,a_0=a,\pi)$ itself as a generative model. Subsequently, use a "Winsorized geometric distribution" for horizon sampling—retaining the shape of TD($\lambda$) while setting a hard upper bound for the long tail.

## Method

### Overall Architecture

UHM is a generative model conditioned on $(s,a,n)$ implemented via flow matching, outputting the distribution of states after $n$ steps $m^\pi(\cdot|s,a,n)$. It is trained recursively via bootstrapping: step 1 corresponds to the real transition $\mathcal{P}(\cdot|s,a)$ from the dataset; for $n>1$, the model bootstraps from the $n-1$ step model. The accompanying critic learning utilizes a $\nu$-Bellman operator framework, unifying $n$-step TD, TD($\lambda$), and $\gamma$-MVE as different choices of $\nu$, with a Winsorized geometric measure selected as $\nu$ to stabilize training. The complete algorithm (Algorithm 1) uses TD3+BC as the backbone, supplemented by a reward network, actor, critic, UHM vector field $v_\theta$, target network EMA, and a behavior mixing coefficient $\beta=0.3$.

### Key Designs

1. **Universal Horizon Model: Horizon-Conditioned Generative Model**:

    - **Function**: Allows a single model to act as a single-step dynamics model (set $n=1$), a GHM (sample $n\sim\text{Geom}$), or jump to an arbitrary $n$-step future.
    - **Mechanism**: Defines the target as $m^\pi(x|s,a,n)=\Pr(s_n=x|s_0=s,a_0=a,\pi)$ and learns via bootstrapping for any $n$. For $n=1$, it fits the dataset transition $\mathcal{P}(x|s,a)$; for $n+1$, it uses $\mathbb{E}_{s'\sim\mathcal{P},a'\sim\pi}[m^\pi(x|s',a',n)]$ as the target. Implementation uses coupled flow matching: a target network $v_{\bar\theta}$ performs $N_\text{flow}$ ODE steps from noise $s_e^0\sim\mathcal{N}(0,I)$ to produce an $n-1$ step sample $s_e^1$. Then, a conditional OT path $s_e^\tau=(1-\tau)s_e^0+\tau s_e^1$ is used to regress $\|v_\theta(s_e^\tau|s,a,n,\tau)-(s_e^1-s_e^0)\|_2^2$ at flow timestep $\tau\sim\text{Unif}[0,1]$.
    - **Design Motivation**: GHM couples "uncertainty of $n$" with "uncertainty of the future state," making predictions for the long tail difficult because the model must simultaneously predict "how large $n$ is" and "where that $n$ lands." By treating $n$ as a condition, the model focuses solely on predicting the future given $n$, and training signals are distributed more uniformly across all $n$.

2. **$\nu$-Bellman Operator + Winsorized Geometric Measure: Bounded Consistency for TD($\lambda$)**:

    - **Function**: A Bellman operator that converges to $Q^\pi$ can be defined using any sub-probability measure $\nu$ on $\mathbb{N}$. It allows the sampling horizon distribution $p_H$ to differ from $\nu$, adjusted via importance ratios.
    - **Mechanism**: The $\nu$-Bellman operator is defined as $\mathcal{T}^\nu Q(s,a)=\mathbb{E}[R(s,a)+\gamma\sum_{k\ge 1}[\xi^\nu(k)R(s_k,a_k)+\nu(k)Q(s_k,a_k)]]$, where $\xi^\nu(k)=\gamma^{k-1}-\sum_{\kappa=0}^{k-1}\gamma^\kappa\nu(k-\kappa)$. Selecting $\nu(k)=\gamma^{n-1}\mathbf{1}[k=n]$ yields $n$-step TD; selecting $\nu(k)=(1-\lambda)(\lambda\gamma)^{k-1}$ yields TD($\lambda$). Ours selects a Winsorized geometric $\nu(k)=(1-\lambda)(\lambda\gamma)^{k-1}$ for $k<k_\text{max}$, and $\nu(k_\text{max})=(\lambda\gamma)^{k_\text{max}-1}$, with zeros beyond $k_\text{max}$. The corresponding sampling distribution is $n=\min(\text{Geom}(1-\lambda\gamma),k_\text{max})$, with importance ratios $w_\xi,w_\nu$ taken in their respective closed forms.
    - **Design Motivation**: In original TD($\lambda$), $\nu$ is never zero on the geometric tail, meaning the probability of sampling extremely long horizons is non-negligible. UHM is least accurate at those horizons, which would contaminate the critic target. Winsorizing sets a hard upper bound $k_\text{max}$ (taken as the $q$-quantile of $\text{Geom}(1-\lambda\gamma)$, here $q=0.2$). This maintains the bias-variance tradeoff of TD($\lambda$) while cutting off the "explosive long tail" and ensuring $\sum_k\nu(k)\le 1$ to satisfy sub-probability conditions for convergence proofs (Proposition 4.1).

3. **$\lambda$ Scheduling + Behavior Mixing: Stabilizing Bootstrap Training**:

    - **Function**: Mitigates issues where UHM is inaccurate for large $n$ early in training, and prevents UHM from hallucinating on OOD state-actions when the actor drifts from the data support.
    - **Mechanism**: (a) **$\lambda$ scheduling**: Utilizes $\lambda=\frac{r\lambda_f}{1-(1-r)\lambda_f}$ where $r\in[0,1]$ is training progress and final $\lambda_f=0.8$ (0.9 for long-horizon tasks). This causes the effective horizon $1/(1-\lambda\gamma)$ to grow linearly, allowing the model to learn $n=1$ accurately before $n=2,3,\dots$, forming a natural curriculum. (b) **Behavior mixing**: Generates bootstrap targets using a mixed strategy $\pi^\text{mix}=(1-\beta)\pi_\theta+\beta\delta(a')$ to sample the next action ($a'$ from dataset, $\beta=0.3$). this restricts UHM queries to state-actions with bounded TV distance from the behavior strategy. (c) Uses augmented states to concatenate a terminal indicator to $s$, letting UHM explicitly model termination to prevent the critic from bootstrapping past terminal states.
    - **Design Motivation**: Bootstrapping in "model-predicting-model" training is susceptible to vicious cycles: "early inaccurate large $n \to$ critic learns wrong target $\to$ policy drifts $\to$ UHM queries further OOD." $\lambda$ scheduling is a curriculum in the time dimension, while behavior mixing is a constraint in the state-action dimension; together they break this loop.

### Loss & Training

Total loss $L=L^v+L^Q+L^R+L^\pi$. UHM loss $L^v=\|v_\theta(s_e^\tau|s,a,n,\tau)-(s_e^1-s_e^0)\|^2$; critic loss $L^Q=(Q_\theta(s,a)-G^\nu)^2$, where $G^\nu=r+\gamma(w_\xi R_{\text{sg}(\theta)}(s_e^1,a_e)+w_\nu Q_{\bar\theta}(s_e^1,a_e))$; reward loss $L^R=(R_\theta(s,a)-r)^2$; actor uses TD3+BC: $L^\pi=\alpha\|\mu_\theta(s)-a\|_2^2-Q_{\text{sg}(\theta)}(s,\mu_\theta(s))$. All baselines share network architectures, trained for 1M gradient steps, with results averaged over the last three epochs across 5 random seeds.

## Key Experimental Results

### Main Results (Average Success Rates for Selected Representative Environments in OGBench 50)

| Environment (5 tasks/group) | ReBRAC | FQL | MAC | DTD($\lambda$) | GHM | **UHM (Ours)** |
|------------------|--------|-----|-----|----------------|-----|---------|
| antmaze-large-navigate | 81 | 79 | 18 | 93 | 90 | 89 |
| antmaze-giant-navigate | 26 | 9 | 0 | 52 | 33 | 36 |
| humanoidmaze-medium-navigate | 22 | 58 | 2 | 81 | 90 | **95** |
| humanoidmaze-large-navigate | 2 | 4 | 0 | 27 | 16 | **33** |
| antsoccer-arena-navigate | 0 | 60 | 29 | 0 | 20 | 26 |
| cube-double-play | 12 | 29 | 53 | 4 | 29 | 30 |
| puzzle-3x3-play | 21 | 30 | 20 | 99 | 51 | **99** |
| puzzle-4x4-play | 14 | 17 | 78 | 1 | 13 | 11 |
| **50 Task Average** | 31 | 44 | 40 | 48 | 55 | **52** |

For long-horizon reasoning tasks (25 tasks), UHM averages 22 vs. GHM 16 vs. DTD($\lambda$) 13. For noisy tasks (25 tasks), UHM averages 39 vs. GHM 38 vs. DTD($\lambda$) 23. Overall, the authors claim a 14% improvement over the "strongest baseline" across 100 tasks.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Full UHM | Significantly highest 100-task average | $\lambda$ scheduling + mixing $\beta=0.3$ + winsorize $q=0.2$ + terminal handling |
| w/o $\lambda$ scheduling | antmaze-giant fails completely | Early large-$n$ bootstrap collapse |
| w/o terminal augmentation | Massive degradation across all tasks | Continued bootstrap after terminal states destabilizes critic |
| $\beta=0.0$ vs $0.3$ vs $1.0$ | Avg 0.63 / 0.66 / 0.59 | Task-dependent; $0.3$ is a robust compromise |
| winsorize $q=10^{-8}$ vs $0.1$ vs $0.2$ | $q=0.1, 0.2$ significantly better than no truncation | Long-tail truncation necessary |
| MBTD($\lambda$) (single-step model + TD($\lambda$)) | Far lower than UHM | Confirms "direct jump to $n$" is more stable than "iterative $n$-step rollout" |
| GHM (same framework but fixed geometric) | Weaker than UHM | Horizon flexibility provides real gains |

### Key Findings
- **Horizon reduction is a key lever in offline RL**: All methods using $n$-step / TD($\lambda$) (DTD, GHM, UHM) significantly outperformed single-step TD (ReBRAC/FQL) on long-horizon tasks, transferring Park et al.'s model-free observations to the model-based side.
- **DTD($\lambda$) is a surprisingly strong baseline, but only on standard data**: In noisy data, DTD($\lambda$) manipulation tasks plummeted to <10% because it directly uses suboptimal trajectories for targets. UHM outperformed DTD($\lambda$) by 16pp on noisy tasks, verifying the necessity of "synthetic on-policy" sampling in noisy scenarios.
- **The gap between UHM and GHM is concentrated in long-horizon reasoning**: 22 vs 16 (a 38% relative gain). On standard tasks, they performed comparably, indicating UHM's primary win comes from fine-grained $n$ control.
- **Training time is nearly equivalent to GHM**: While a single UHM update is slightly slower, it is much faster than MBTD($\lambda$) (which requires $n$ model inferences) and stays within 1.1× of DTD($\lambda$) on long tasks. The computational advantage of direct sampling architectures scales better with task length.
- **Terminal handling is severely underrated**: Removing terminal augmentation caused massive degradation across all tasks, suggesting the offline MBRL community has previously overlooked how terminal states enter generative models.

## Highlights & Insights
- **Unifying "GHM vs Single-step" into a single generative model family**: This is an elegant conceptual abstraction where $n\sim\text{Geom}\Rightarrow$ GHM, $n\sim\delta(1)\Rightarrow$ single-step, and $n\sim$ arbitrary $p_H\Rightarrow$ UHM. Horizon distribution for RL can now be chosen freely without retraining models for each choice.
- **The $\nu$-Bellman operator is a neglected tool**: it clarifies exactly what kind of horizon-weighted backup converges to $Q^\pi$, providing a theoretical entry point for future anti-geometric, heavy-tail, or adaptive $\nu$ backups.
- **The engineering philosophy of Winsorizing is valuable**: Directly clipping low-probability long tails—a technique used by statisticians for decades—was rediscovered here for generative-model-driven RL. It highlights that "model errors on low-probability events" is a real risk in large pipelines.
- **Curriculum $\lambda$ scheduling is transferable**: The $\lambda$ scheduling that grows the effective horizon linearly can be applied to any bootstrap-trained world model to gradually relax the difficulty of bootstrapping.

## Limitations & Future Work
- **Acknowledged Limitations**: UHM is limited by data sparsity and extrapolates outside data support; a single MLP/transformer struggles to model all $n$ simultaneously; currently limited to state space without extension to visual observations or action chunking.
- **Own Observations**: (1) The Winsorize threshold $q$ is explicitly task-dependent ($q=0.2$ for cube-quadruple but $q=0.1$ for puzzle-4x6); auto-selecting $q$ is an open question. (2) Algorithm 1 learns a reward network, but the stability of stop-gradient reward coupling with critic targets was not ablated. (3) DTD($\lambda$) still outperforms model-based methods in humanoidmaze-giant, suggesting UHM lacks precision in high-dimensional, large-scale environments. (4) Using flow matching ODE steps as part of a bootstrap target is theoretically a nested fixed-point problem; gradient bias was not investigated.
- **Future Directions**: (1) Hierarchical UHM—using different networks for coarse and fine horizons; (2) Adaptive $q$ learning based on critic loss; (3) Integration with action chunking and visual world models (e.g., Dreamer series) as a latent dynamics layer.

## Related Work & Insights
- **vs GHM (Janner et al., 2020) / Thakoor 2022**: Both jump to the future, but GHM hides $n$ implicitly in a geometric distribution, making the long tail hard to learn and preventing TD($\lambda$). UHM is a strict generalization of GHM.
- **vs $\gamma$-MVE**: Ours proves $\gamma$-MVE is equivalent to TD($\lambda$) with $\lambda=\tilde\gamma/\gamma$ on on-policy trajectories, mathematically aligning GHM with TD($\lambda$) and unlocking $\lambda$ as a selectable hyperparameter via UHM.
- **vs MBTD($\lambda$) / LEQ (Park & Lee 2025)**: Both perform model-based TD($\lambda$), but iterative $n$-step prediction in offline settings causes severe compounding errors. UHM's direct sampling is significantly faster on long horizons.
- **vs action-chunk dynamics (Zhang 2023; Lin 2025; Park 2026a)**: Those methods learn $\Pr(s_{t+n}|s_t,a_t,\dots,a_{t+n-1})$ conditioned on fixed action sequences. UHM learns $\Pr(s_{t+n}|s_t,a_t,\pi)$ induced by the policy, generating truly "on-policy" futures.
- **vs MOPO / MOBILE**: These rely on uncertainty penalties on rewards to combat model error. Ours addresses the root cause via "direct jumping + Winsorizing the tail," proving more effective in sparse-reward, long-horizon tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Generalizing GHM by explicitly parameterizing $n$ is a beautiful incremental step, formalized via the $\nu$-Bellman framework with clear conceptual lineage.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 100 OGBench tasks across standard/noisy/long-horizon categories with complete ablations (scheduling, $\beta$, $q$, terminal).
- **Writing Quality**: ⭐⭐⭐⭐ Mathematically rigorous definitions and propositions, though some details in Algorithm 1 (ODE nesting, reward stop-gradient) require careful reading.
- **Value**: ⭐⭐⭐⭐ A directly plug-and-play improvement for offline MBRL that unifies GHM, single-step, and TD($\lambda$) into one framework, benefiting both theory and practice.

## Rating
- **Novelty**: To be evaluated
- **Experimental Thoroughness**: To be evaluated
- **Writing Quality**: To be evaluated
- **Value**: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[ICML 2026\] InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](inftythink_effective_and_efficient_infinite-horizon_reasoning_via_reinforcement_.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] Learning to Bet for Horizon-Aware Anytime-Valid Testing](learning_to_bet_for_horizon-aware_anytime-valid_testing.md)

</div>

<!-- RELATED:END -->
