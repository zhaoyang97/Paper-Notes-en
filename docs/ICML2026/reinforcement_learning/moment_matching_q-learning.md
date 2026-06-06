---
title: >-
  [Paper Note] MoMa QL: Accelerating Diffusion/Flow Matching Policies for Offline and Offline-to-Online RL via Moment Matching
description: >-
  [ICML 2026][Reinforcement Learning][Offline RL] MoMa QL replaces the standard BC loss with Maximum Mean Discrepancy (MMD), compressing the multi-step sampling of diffusion/flow matching policies into a single-step or few…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Diffusion Policy"
  - "Flow Matching"
  - "Maximum Mean Discrepancy"
  - "Stochastic Interpolants"
  - "Consistency Models"
date: 2026-05-08
content_hash: 13fb533c7ad19c79
---

# MoMa QL: Accelerating Diffusion/Flow Matching Policies for Offline and Offline-to-Online RL via Moment Matching

**Conference**: ICML 2026  
**arXiv**: [2605.29033](https://arxiv.org/abs/2605.29033)  
**Code**: Not yet public  
**Area**: Reinforcement Learning / Offline RL / Diffusion Policy  
**Keywords**: Offline RL, Diffusion Policy, Flow Matching, Maximum Mean Discrepancy, Stochastic Interpolants, Consistency Models  

## TL;DR
MoMa QL replaces the standard BC loss with Maximum Mean Discrepancy (MMD), compressing the multi-step sampling of diffusion/flow matching policies into a single-step or few-step "marginal-preserving interpolation" sampler. It achieves a leading average normalized Gym score of 95.5 on D4RL, comprehensively outperforming Diffusion-QL (87.9). Due to significantly faster sampling, it shows greater gains during offline-to-online fine-tuning compared to Consistency-AC and Diffusion-QL.

## Background & Motivation

**Background**: Offline RL aims to learn an optimal strategy from static datasets. The challenge lies in unreliable value estimation for OOD actions and increasingly complex multimodal behavior distributions. Limited expressivity in GMMs or VAEs has led to the adoption of generative policies (Diffusion-QL, IDQL, FQL), which excel at representing arbitrary multimodal behaviors.

**Limitations of Prior Work**: Sampling from diffusion/flow matching policies is iterative—inference requires dozens to hundreds of denoising steps. For actor-critic training in offline RL, each actor update requires online sampling of actions to input into the critic, causing training costs to explode. More critically, online rollouts in the offline-to-online stage require step-by-step sampling, where sampling latency severely hinders diffusion policies in real environments.

**Key Challenge**: There is a natural conflict between expressivity (multi-step iteration) and computational efficiency (single-step sampling). Consistency models suggest a solution (learning a "any time $\to$ clean sample" mapping), but they primarily focus on distillation and are difficult to learn jointly with critic signals within an actor-critic framework.

**Goal**: To find a sampler $p^\theta_{s|t}(\mathbf{x}|\mathbf{x}_t)$ that can jump from any time $t$ in a diffusion trajectory to any earlier time $s$ in one step, while being directly optimizable via the critic's Q-signal and maintaining training stability.

**Key Insight**: Align the "target distribution the sampler aims to approximate" with the "intermediate distribution of the diffusion process" using MMD. MMD is a distance measure based on RKHS that implicitly captures all moments in the feature space. It is more stable than single-moment constraints (like the log-likelihood in BC) and more friendly to generative models than likelihood-based constraints such as KL/JS.

**Core Idea**: Replace the BC loss in the BRAC framework with the MMD between marginal distributions $p^{\theta^-}_{s|r}$ and $p^\theta_{s|t}$ for different intermediate steps $r<s<t$. This allows the policy to learn one-step denoising from any noise level to any cleaner level. Combined with Q-loss for synchronous critic training, the process functions as a hybrid of consistency training and actor-critic.

## Method

### Overall Architecture

Built upon a dual Q-learning structure within BRAC: the critic uses classic TD($\lambda=0$) + double Q, while the actor consists of two components: (1) Q-loss: recursively generates actions $\mathbf{a}^\pi$ from a prior using the learned sampler and inputs them into the critic to maximize $-\eta Q$; (2) BC-loss: utilizes MMD consistency constraints to ensure sampler self-consistency across different intermediate time steps. The sampling phase follows a DDIM-style recursion: starting from $\mathbf{a}_N \sim \mathcal{N}(0,\sigma_d^2 I)$, the learned $f_\theta(t_{i-1}, t_i, \mathbf{a}_{t_i})$ is called at each step to jump to the next time step.

### Key Designs

1.  **Marginal-Preserving Stochastic Interpolant + Cross-time One-step Sampler**:
    - **Function**: Transitions the diffusion process from "step-by-step denoising" to a learnable mapping that jumps from any intermediate time $t$ to any earlier time $s$.
    - **Mechanism**: Defines marginal interpolants $q_t(\mathbf{x}_t) = \iint q_t(\mathbf{x}_t|\mathbf{x}, \boldsymbol{\varepsilon}) q(\mathbf{x}) p(\boldsymbol{\varepsilon}) d\mathbf{x} d\boldsymbol{\varepsilon}$ and conditions $q_{s|t}(\mathbf{x}_s|\mathbf{x}, \mathbf{x}_t) = \mathcal{N}(I_{s|t}(\mathbf{x}, \mathbf{x}_t), \gamma^2_{s|t} I)$ based on stochastic interpolants. It requires "marginal preservation"—for any $s \le t$, $q_s = \iint q_{s|t} q_t(\mathbf{x}|\mathbf{x}_t) q_t(\mathbf{x}_t) d\mathbf{x}_t d\mathbf{x}$. The model learns an implicit sampler $p^\theta_{s|t}(\mathbf{x}|\mathbf{x}_t) = \delta(\mathbf{x} - G_\theta(\mathbf{x}_t, s, t))$, which predicts a clean sample $\tilde{\mathbf{x}}$ given $\mathbf{x}_t$, followed by DDIM interpolation to obtain $\tilde{\mathbf{x}}_s$.
    - **Design Motivation**: Traditional diffusion/flow matching only learns $t \to t-1$ mappings, making multi-step sampling expensive. Learning an "any $t \to s$ mapping" is equivalent to learning a consistency model. By using a marginal-preserving framework within stochastic interpolants, this method naturally supports both single-step and multi-step inference modes, adapting to different compute budgets in online/offline scenarios.

2.  **MMD Self-Consistency Loss**:
    - **Function**: Replaces log-likelihood/KL constraints in BC to provide a stable training signal with high-order moment alignment for the sampler.
    - **Mechanism**: Directly matching $q_s(\mathbf{x}_s)$ and $p^\theta_{s|t}(\mathbf{x}_s)$ is unstable when $t$ and $s$ are far apart. Borrowing from consistency training, a middle time $r$ ($s < r < t$) is introduced, and the loss $\mathcal{L}_D(\theta) = \mathbb{E}_{s,r,t}[\mathrm{MMD}^2(p^{\theta^-}_{s|r}(\mathbf{x}_s), p^\theta_{s|t}(\mathbf{x}_s))]$ is used to self-consistently bridge "short jumps" and "long jumps." The RBF kernel $k(x,y) = e^{-\|x-y\|^2 / (2\sigma^2)}$ is characteristic and implicitly captures all moments, being more robust than log-likelihood. MMD is calculated using an unbiased estimator, requiring no density estimation.
    - **Design Motivation**: (1) High-order moment matching accurately captures multimodal behavior, unlike Gaussian-likelihood BC which only matches the first moment. (2) MMD is sample-based and density-free, naturally compatible with implicit generative models like diffusion. (3) The self-consistency form is homologous to consistency models; theoretically, CM is a special case of this framework as $r \to s$.

3.  **Consistency Induction + Dual Q for Overestimation Mitigation**:
    - **Function**: Jointly trains for "short-jump consistency" and "stable Q-estimation."
    - **Mechanism**: The critic uses dual Q from TD3+BC: $\mathcal{L}(\phi) = \mathbb{E}[(\,r + \gamma \min_{i \in \{1,2\}} Q_{\phi^T_i}(\mathbf{s}', \mathbf{a}') - Q_{\phi_i}(\mathbf{s}, \mathbf{a}))^2]$, where $\mathbf{a}' \sim \pi_{\theta^T}(\cdot|\mathbf{s}')$ is sampled via the actor's inference algorithm. The actor uses an EMA target $\theta^T \leftarrow \alpha\theta^T + (1-\alpha)\theta$. The target $\theta^-$ is a stop-gradient EMA policy, providing a stable reference for the MMD consistency constraint.
    - **Design Motivation**: During actor training, actions $\mathbf{a}^\pi$ must be sampled frequently. If sampling requires 100 steps, a single actor update requires 100 forward passes. This framework significantly reduces this overhead through few-step sampling, allowing the "actor update + critic update" cycle to converge within a reasonable timeframe.

### Loss & Training

The total actor loss is $\mathcal{L}_\pi(\theta) = -\eta \mathbb{E}[Q_\phi(\mathbf{s}, \mathbf{a}^\pi)] + \mathcal{L}_D(\theta)$, where $\mathbf{a}^\pi$ is generated by recursively calling $f_\theta(t_{i-1}, t_i, \mathbf{a}_{t_i})$. $\eta$ controls the trade-off between the Q-signal and MMD consistency. The critic follows standard dual Q TD learning. The training loop progresses through critic updates, actor updates, and EMA target updates, synchronized as in BRAC.

## Key Experimental Results

### Main Results: D4RL Task Comparison

| Task Suite | BC | Diffusion-BC | Consistency-BC | CQL | IQL | Diffusion-QL | Consistency-AC | **MoMa QL** |
|---------|-----|--------------|-----------------|-----|-----|--------------|-----------------|-------------|
| Gym BC Avg (9 tasks) | 51.9 | 76.3 | 69.7 | — | — | — | — | **89.8** |
| Gym Offline RL (9 tasks) | — | — | — | 77.6 | 77.0 | 87.9 | 85.1 | **95.5** |
| Adroit (12 tasks) | 48.3 | — | — | — | 53.4 | — | 42.9 | **56.7** |
| Kitchen (3 tasks) | — | — | — | 48.2 | 53.3 | 69.0 | 45.3 | **73.1** |

On Gym BC, Ours improves by 1.18× over Diffusion-BC (76.3) and 1.73× over vanilla BC (51.9). On Gym Offline RL, Ours improves by 1.09× over Diffusion-QL (87.9), with a 14% lead in HalfCheetah and 8.5% in Walker2D. Performance on Adroit matches or slightly exceeds ReBRAC (55.4), ranking first on the door task (38.5).

### Offline-to-Online Fine-tuning

| Task | Offline Score | Online Score | Gain |
|------|-------|-------|------|
| HalfCheetah-m | 72.6 | 83.1 | +10.5 |
| Hopper-m | 104.2 | 104.3 | +0.1 |
| Walker2d-m | 95.6 | 99.1 | +3.5 |
| HalfCheetah-mr | 63.3 | 80.9 | +17.6 |

The largest improvement is seen in medium-replay tasks (HalfCheetah-mr +28% relative), confirming the causal chain: "fast sampling $\to$ efficient online interaction $\to$ more thorough fine-tuning."

### Key Findings

- **Multimodal Behavior Capture**: MoMa QL shows the greatest relative advantage on medium-replay data (the most multimodal). Walker2d-mr leads the strongest baseline by 1.26×, and HalfCheetah-m leads by 1.44×, validating the advantage of MMD high-order moment matching for multimodal distributions.
- **Trading Sampling Efficiency for Online Capability**: Diffusion-QL is strong offline but often suffers from performance degradation during online fine-tuning due to slow sampling. MoMa QL's single-step sampling makes online interaction nearly as fast as Gaussian policies, allowing "continuous growth" during the offline-to-online transition.
- **Anomaly on Hopper-me**: MoMa QL achieved 67.9 while BC-based baselines reached 100+. This suggests that on data already near optimal, overly strong high-order moment matching might turn the MMD self-consistency constraint into noise interference; this represents a trade-off in actor signal design.
- **Discriminative vs. Generative**: On Adroit, ReBRAC is slightly stronger on expert datasets, but MoMa QL is more stable in settings with varied data quality, indicating its advantage lies in "covering multimodal behavior distributions" rather than "pure expert imitation."

## Highlights & Insights

- **MMD as a Tool-Object Match**: The high-order moment properties of MMD precisely address the needs of multimodal behavior distributions in offline RL, while its sample-based nature aligns with implicit generative models like diffusion. This "natural fit" makes the method's selling point more than just performance numbers—it is a paradigm alignment.
- **Fusion of Consistency Training and Actor-Critic**: Previously, consistency models were mainly used for unsupervised accelerated sampling. This work integrates them into the Q-learning loop and proves CM is a special case, providing a unified path for "accelerating generative policies = consistency distillation + RL signal joint training."
- **Flexibility of Marginal-Preserving Interpolation**: This enables switching between single-step and multi-step sampling within a unified framework—multi-step for expressivity during offline training, and single-step for speed during online rollouts. This decoupling of "precise training, fast inference" is very deployment-friendly.
- **Stability from Kernel Functions**: The boundedness of the RBF kernel and the non-negativity of MMD prevent the actor loss from exploding. Compared to the numerical issues of KL-based positive/negative infinity, it is better suited for RL scenarios where rewards fluctuate across orders of magnitude.

## Limitations & Future Work

- **Dependence on Kernel Selection**: Experiments primarily use RBF with adaptive bandwidth, but kernel choice significantly affects MMD performance. The optimal kernel may vary by task, a strategy not deeply discussed.
- **Q-Signal vs. MMD Trade-off**: The hyperparameter $\eta$ controls the balance between Q and MMD. The appendix shows sensitivity to this value; the anomaly in hopper-me may be a side effect of $\eta$ lacking adaptive scaling on expert data.
- **MMD Estimation Variance**: Sample-based MMD has high variance with small batches, which might require larger batches or more complex unbiased estimators for high-dimensional action tasks (e.g., robotics).
- **Asymptotic Theoretical Convergence**: The conclusion that CM is a special case as $r \to s$ requires more refined discretization error analysis under finite step sizes.

## Related Work & Insights

- **vs. Diffusion-QL / IDQL**: These use full multi-step diffusion sampling, offering high expressivity but high training/online costs. MoMa QL uses MMD consistency to compress sampling to single/few steps, often with better performance.
- **vs. Consistency-AC / FQL**: Also target accelerated generative policies—CAC uses CM distillation and FQL uses flow matching distillation, but both separate "acceleration" from the "RL signal." This work joins them in a single MMD self-consistency loss, achieving superior training efficiency and final performance.
- **vs. QGPO / EDP / QIPO**: Energy-guided diffusion treats the RL objective as reward weighting. Ours follows an "explicit multi-step distribution matching" route, which does not rely on energy importance sampling and is more stable during training.
- **Inspiration**: This three-part approach (marginal-preserving interpolation + MMD consistency + task loss) can be applied to any scenario requiring "iterative multi-step generation + task signals," such as robotic control, text generation RL, or visual agents.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Unifying MMD, consistency training, and actor-critic via stochastic interpolants is a novel paradigm synthesis. The proof regarding CM as a special case is a theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three major D4RL suites, offline/offline-to-online transitions, and matches up against strong baselines. Ablations on kernels and $\eta$ are slightly lean.
- **Writing Quality**: ⭐⭐⭐⭐ Derivations are clear with a complete notation system. The correspondence between "training-inference" in Algorithms 1 and 2 is slightly compressed and requires some attention to follow.
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical solution to the deployment bottleneck (slow online rollouts) of diffusion-based policies. Leads the D4RL boards and the method is transferable to other generative RL scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICML 2026\] Adaptive Bandit Algorithms for Contextual Matching Markets](adaptive_bandit_algorithms_for_contextual_matching_markets.md)
- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
