---
title: >-
  [Paper Note] Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns
description: >-
  [ICLR 2026][Reinforcement Learning][Soft Actor-Critic] The MLP critic in SAC is replaced with a lightweight causal Transformer, allowing the critic to evaluate all prefixes of a "state + short action sequence" simultaneously. By using multi-horizon N-step returns for supervision without requiring importance sampling, the method maintains a strictly single-step policy while significantly outperforming standard SAC and episodic baselines on long-range, sparse-reward tasks.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Soft Actor-Critic"
  - "Transformer Critic"
  - "N-step Returns"
  - "Sequence Modeling"
  - "Long-range Credit Assignment"
  - "Target-free Training"
date: 2026-05-08
content_hash: b2ef8f7d30dd136e
---

# Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=rb5eTktqbc](https://openreview.net/forum?id=rb5eTktqbc)  
**Code**: Yes (GitHub + Weights & Biases logs mentioned in the paper)  
**Area**: reinforcement learning  
**Keywords**: Soft Actor-Critic, Transformer Critic, N-step Returns, Sequence Modeling, Long-range Credit Assignment, Target-free Training  

## TL;DR
The MLP critic in SAC is replaced with a lightweight causal Transformer, allowing the critic to evaluate all prefixes of a "state + short action sequence" simultaneously. By using multi-horizon N-step returns for supervision without requiring importance sampling, the method maintains a strictly single-step policy while significantly outperforming standard SAC and episodic baselines on long-range, sparse-reward tasks.

## Background & Motivation
- **Background**: Soft Actor-Critic (SAC) is a staple for continuous control due to its sample efficiency and stability. However, its critic is an MLP that evaluates pairs of $(s,a)$, lacking the capability to model temporal structures.
- **Limitations of Prior Work**: (1) In off-policy settings, accelerating credit assignment with N-step returns typically requires stepwise Importance Sampling (IS) to correct the distribution mismatch between the behavior policy $\mu$ and the target policy $\pi$. IS introduces high variance and can cause training divergence, limiting the effective horizon. (2) Another approach, "action chunking," enables policies to output open-loop action sequences, but fixed chunk lengths reduce control frequency and reactivity, tying performance to a specific horizon hyperparameter.
- **Key Challenge**: Achieving the benefits of long-range credit assignment (multi-step returns and sequential structure) without the high variance of IS or sacrificing the reactivity of a single-step policy.
- **Goal**: Introduce temporal structure modeling into the critic while maintaining a strictly single-step policy and an IS-free update rule, specifically targeting long-range, sparse-reward tasks.
- **Core Idea**: **"Strengthen the critic instead of the policy."** The critic is designed as a Transformer utilizing causal attention over short trajectory segments. It predicts values for all action prefixes $(s_t, a_t, \dots, a_{t+i-1})$ simultaneously and is supervised by N-step targets of "realized prefixes." Since rewards strictly follow the prefixes recorded in the replay buffer, the predictions represent the values of prefixes under the replay distribution, eliminating the need for IS at its root.

## Method

### Overall Architecture
T-SAC (Transformer-based SAC) retains the general framework of SAC (stochastic policy, automatic temperature adjustment, single-step policy updates) but replaces the MLP critic with a causal Transformer. The critic takes state $s_t$ and a sequence of $n$ subsequent actions $a_t, \dots, a_{t+n-1}$ as input, outputting $n$ **prefix-conditioned** values $\{Q_\psi(s_t, a_t, \dots, a_{t+i-1})\}_{i=1}^n$. Causal masking ensures that position $i$ only attends to timesteps $\le i$ to prevent leakage of future information. Training is supervised by non-soft N-step targets of variable horizons, where gradients from different horizons are averaged during backpropagation. A snapshot of the critic with frozen parameters is used as a temporary target network.

```mermaid
flowchart LR
    A["Replay Buffer: Short Segments<br/>(s_t, a_t...a_{t+n-1})"] --> B["Causal Transformer Critic<br/>(causal mask, 2 layers 128-256)"]
    B --> C["n Prefix Values<br/>Q(s_t, a_t..a_{t+i-1}), i=1..n"]
    D["Multi-horizon N-step Targets<br/>G^(i) (No IS)"] --> E["Per-horizon Loss L_i"]
    C --> E
    E --> F["Gradient-level Average<br/>∇L̄ = mean_i ∇L_i"]
    F --> B
    G["Policy π_θ (Single-step, Gaussian+LayerNorm)"] -->|Sample Action| A
    B -->|Q-guidance| G
    H["Critic Snapshot ϕ←ψ<br/>Cache V_ϕ, Frozen for K steps"] -.-.|Target-free Bootstrapping| D
```

### Key Designs

**1. Prefix-conditioned Critic + IS-free N-step Supervision: Eliminating Importance Sampling at the source.** Standard off-policy N-step TD assumes actions after $a_t$ follow the current policy $\pi_\theta$, but replay data comes from behavior policy $\mu$, requiring correction via stepwise importance ratios $\rho_{t+k}=\pi_\theta/\mu$, which causes high variance. T-SAC changes the prediction target: the critic directly predicts values for the **realized** prefixes in the replay buffer. The $i$-th step target is $G^{(i)}(s_t, a_{t:t+i-1}) = \sum_{j=0}^{i-1}\gamma^j r_{t+j} + \gamma^i V_\phi(s_{t+i})$, where $V_\phi(s):=\mathbb{E}_{a\sim\pi_\theta}[Q_\phi(s,a)]$. Since rewards follow the recorded action prefix $a_{t:t+i-1}$, there is no need to assume these actions came from $\pi_\theta$. Thus, **the entire multi-step supervision requires no IS**—only the terminal bootstrap term $V_\phi(s_{t+i})$ depends on the current policy. Theoretically, given $s_t$ and a realized prefix, the future reward distribution is determined by environmental dynamics and is independent of how the prefix was generated.

**2. Gradient-level Averaging vs. Target-level Averaging: Preserving long-range signals in sparse rewards.** A classic variance reduction technique is to average N-step targets $\bar G^{(n)}=\frac1n\sum_{i=1}^n G^{(i)}$. However, the authors found that this "target-level averaging" dilutes sparse long-range rewards (Appendix F), leading to poor performance on sparse tasks. T-SAC instead constructs a loss $L_i(\psi)=\frac12(Q_\psi^{(i)}-G^{(i)})^2$ for each horizon and averages their **gradients**: $\nabla_\psi\bar L=\frac1n\sum_{i=1}^n\nabla_\psi L_i$. Because adjacent horizons correspond to adjacent decoding positions in the same network and have overlapping targets, their per-parameter gradients are positively but not perfectly correlated. Averaging gradients reduces update variance without smoothing out sparse signals like target averaging does. During training, a starting point $t$ is sampled from a mini-batch, $n$ is sampled uniformly from $\{\text{min\_length},\dots,\text{max\_length}\}$ (default $n\sim\text{Unif}\{1,\dots,16\}$), and MSE is calculated across all horizons.

**3. Non-soft Critic + Policy-side Entropy Regularization: Decoupling entropy from the critic target.** Unlike standard SAC, the T-SAC critic estimates **standard (non-soft) action values**; the targets contain no entropy terms. Maximum entropy regularization is handled entirely on the policy side (policy objective $J_\pi=\mathbb{E}[\alpha\log\pi_\theta(a|s)-Q_\psi(s,a)]$, with temperature $\alpha$ automatically tuned). This decoupling is consistent with methods like MPO, AWAC, and IQL, allowing the critic to focus on unbiased value regression. Layer Normalization is also added to the policy network (following continuous control practices by Plappert et al.) to stabilize exploration noise.

**4. Critic Parameter Freezing: Target-free training via "Hard-copy Snapshots."** While CrossQ uses Batch Renorm and bounded activations to remove the target network, T-SAC takes a simpler path. At the start of a critic update segment, the online critic is snapshotted ($\phi \leftarrow \psi$). Bootstrapped targets $V_\phi(s)$ for all windows in that segment are pre-computed and cached. This snapshot is then frozen while the online critic is optimized for $K$ consecutive steps ($K=20$ for MuJoCo). This "segment-level target caching" replaces Polyak averaging, suppressing target drift without requiring BRN. The scheme introduces only one hyperparameter $K$; sensitivity tests on Walker2d with $K \in \{20, 100, 1000, 10000\}$ showed stable performance, indicating $K$ is robust.

## Key Experimental Results

### Main Results
Evaluation spans 57 tasks: Meta-World ML1 (50), Gymnasium MuJoCo (5), and Box-Pushing (dense/sparse, 2). Default 8 seeds with 95% bootstrap confidence intervals, UTD max of 1. The critic uses only a 2-layer Transformer with 128–256 hidden units.

| Benchmark | Metric | T-SAC | Comparison |
|------|------|-------|------|
| Meta-World ML1 (50 tasks) | Aggregated Success IQM | Solves most in ~5M steps; best IQM | TOP-ERL requires ~20M steps for similar aggregate |
| Box-Pushing (dense, ±5cm/±0.5rad) | Success Rate | **96.8%** | Previous baselines ≤85% |
| Box-Pushing (sparse) | Success Rate | 60% (hard-copy critic) | TOP-ERL 70% |
| Complex Multi-stage (Assembly/Hammer/etc.) | Success IQM | Significantly leading | SAC/CrossQ/PPO/GTrXL significantly lag |
| Gymnasium MuJoCo (Ant/Hopper/Walker2d) | Return IQM | Comparable or better than SAC | — |
| HumanoidStandup / HalfCheetah | Return IQM | Largest gains | — |

T-SAC maintains single-step policy updates yet surpasses Transformer-based episodic methods (GTrXL policy, TOP-ERL) on long-range tasks while matching standard SAC on locomotion, partially bridging the gap between "step-based for locomotion" and "episodic for long-range."

### Ablation Study
Ablations performed on Box-Pushing (dense) and MuJoCo Walker2d:

| Ablation Item | Setting | Conclusion |
|--------|------|------|
| Transformer Components | Remove ResNet / Causal Mask / Self-attention | Removing **self-attention** alone breaks segment-conditioned targets/causes divergence; removing all three degrades to MLP. |
| Multi-horizon Windows | $n\sim\text{Unif}\{1,16\}$ vs. fixed horizon | Default variable $n$ outperforms fixed single horizons. |
| Target-free Scheme | T-SAC Hard-copy vs. Soft-copy vs. CrossQ/SAC/TD3 | Hard-copy freezing matches or exceeds Polyak on locomotion and sparse tasks. |
| Averaging Method | Target-level vs. Gradient-level (Appendix F) | Target-level averaging degrades significantly in sparse rewards; validates necessity of gradient-level averaging. |

### Key Findings
- **Critic-side sequence modeling** (rather than action chunking on the policy side) provides long-range credit assignment benefits while maintaining single-step reactivity.
- "Predicting values of realized prefixes" makes N-step supervision **naturally IS-free**, which is critical for stable long-range learning.
- Target networks can be replaced by a simple "segment-level target cache + short freeze," where $K$ is a robust hyperparameter.

## Highlights & Insights
- **Key Insight**: Solving long-range problems does not strictly require changing the policy (action chunking) or adding IS. Internalizing temporal structure within the critic's conditions and targets is a lighter, more stable path.
- **IS-free Multi-step TD**: By predicting the value of realized prefixes under the replay distribution, importance sampling is bypassed. The future reward distribution depends only on dynamics, not on how the prefix was generated.
- **Gradient-level vs. Target-level Averaging**: A subtle but practical insight—for sparse rewards, averaging gradients reduces variance while preserving long-range signals, whereas averaging targets tends to smooth them out.
- **Minimalist Design**: Achieving SOTA across multiple benchmarks with a 2-layer Transformer, UTD $\le 1$, and a single freezing hyperparameter makes this very engineering-friendly.

## Limitations & Future Work
- Still lags behind TOP-ERL on Sparse Box-Pushing (60% vs. 70%), indicating that pure critic-side sequence modeling may not fully replace episodic methods for extremely sparse tasks.
- Windows are of fixed maximum length; variable-length episodes require action masks to avoid bootstrapping across episodes (treated as an implementation detail).
- Horizon upper bounds and $K$ still require task-specific tuning, although $K$ is argued to be robust.
- The work only modifies the critic; combining it with more expressive policies (Energy-based, Diffusion) or sequence policies like Decision Transformer remains future work.

## Related Work & Insights
- **TOP-ERL**: Also uses Transformer critic + truncated N-step targets, but operates with episodic, open-loop trajectory policies (ProDMP + TRPL, ~20M interactions). T-SAC brings these ideas to standard step-based, closed-loop SAC.
- **CrossQ**: Proved that BRN + bounded activations can remove target networks; T-SAC offers an alternative target-free route via "critic parameter freezing + segment-level caching."
- **Q-chunking / Action Chunking**: Directly learns $Q(s_t, a_{t:t+H-1})$ in a chunked action space, but typically uses MLP critics and open-loop policies. T-SAC puts the "chunking" inside the critic while keeping the policy single-step.
- **Decision Transformer**: Complements this work—while DT focuses on policy-side sequence modeling, T-SAC uses Transformers only for the critic; both could theoretically be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of critic-side sequence modeling, realized prefix prediction to bypass IS, and gradient-level averaging is a clever and distinct shift from TOP-ERL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 57 tasks, 8 seeds, and deep ablation of components.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation and design are clearly articulated; the transition from theory to implementation is logical.
- **Value**: ⭐⭐⭐⭐ — Provides a lightweight, reproducible, and IS-free strong baseline for off-policy long-range control.

## Related Papers

- [\[ICLR 2026\] DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty](dr-sac_distributionally_robust_soft_actor-critic_for_reinforcement_learning_unde.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning](neuralsymbolic_approaches_for_interpretable_actor-critic_reinforcement_learning.md)
- [\[ICLR 2026\] Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation](finite-time_analysis_of_actor-critic_methods_with_deep_neural_network_approximat.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty](dr-sac_distributionally_robust_soft_actor-critic_for_reinforcement_learning_unde.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning](neuralsymbolic_approaches_for_interpretable_actor-critic_reinforcement_learning.md)
- [\[ICLR 2026\] Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation](finite-time_analysis_of_actor-critic_methods_with_deep_neural_network_approximat.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->
