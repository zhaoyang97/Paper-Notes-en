---
title: >-
  [Paper Note] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Continuous Normalizing Flow] PolicyFlow seamlessly integrates a continuous normalizing flow (CNF) policy into the PPO framework: it approximates the importance ratio via velocity field differences along an interpolated path (avoiding full ODE path backpropagation), and introduces a Brownian motion-inspired implicit entropy regularizer to prevent mode collapse. The method matches or surpasses Gaussian PPO and flow-based baselines (FPO/DPPO) across MultiGoal, PointMaze, IsaacLab, and MuJoCo environments.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Continuous Normalizing Flow
  - PPO
  - Multimodal Policy
  - Importance Ratio Approximation
  - Brownian Motion Entropy Regularization
date: 2026-05-08
content_hash: e067888a4484f6cd
---

# PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2602.01156](https://arxiv.org/abs/2602.01156)
**Code**: [Project Page](https://policyflow2026.github.io/)
**Area**: Reinforcement Learning / Policy Optimization
**Keywords**: Continuous Normalizing Flow, PPO, Multimodal Policy, Importance Ratio Approximation, Brownian Motion Entropy Regularization

## TL;DR

PolicyFlow seamlessly integrates a continuous normalizing flow (CNF) policy into the PPO framework: it approximates the importance ratio via velocity field differences along an interpolated path (avoiding full ODE path backpropagation), and introduces a Brownian motion-inspired implicit entropy regularizer to prevent mode collapse. The method matches or surpasses Gaussian PPO and flow-based baselines (FPO/DPPO) across MultiGoal, PointMaze, IsaacLab, and MuJoCo environments.

## Background & Motivation

**Background**: PPO is the dominant policy gradient method for online reinforcement learning, widely applied in robotic control and LLM alignment. Its core mechanism relies on the importance ratio to update the policy, typically under a Gaussian distribution assumption to simplify likelihood computation. However, Gaussian policies can only represent unimodal distributions and fail to model complex multimodal action distributions.

**Limitations of Prior Work**:

- **Insufficient expressiveness of Gaussian policies**: In tasks requiring multi-goal reaching or multi-path planning, Gaussian policies can only cover a single mode.
- **High computational cost of likelihood estimation for generative policies**: CNFs and diffusion models offer sufficient expressiveness, but computing importance ratios requires backpropagation through the full ODE trajectory—memory-intensive and gradient-unstable.
- **Bias issues in FPO**: FPO estimates importance ratios via ELBO, but suffers from asymmetric estimation bias (reliable when the ratio increases, unreliable when it decreases), requiring larger batch sizes for stability.
- **Limitations of DPPO**: DPPO treats the diffusion process as an internal MDP, suited for fine-tuning but prone to performance degradation when trained from scratch due to the lack of off-manifold exploration.
- **Difficulty of entropy regularization**: Computing action log-likelihoods for flow-based policies is non-trivial, making conventional entropy regularization methods inapplicable.

## Key Designs

1. **Interpolated Path Importance Ratio Approximation**: Rather than computing $\delta_{\varphi_1}(\mathbf{z};\mathbf{s})$ along the ODE flow trajectory, the method approximates the terminal displacement difference using the velocity field difference $\delta_{v_t}$ along the linear interpolation path $\mathbf{x}_t = (1-t)\mathbf{z} + t\hat{\varphi}_1(\mathbf{z};\mathbf{s})$. The theoretical approximation error is $\mathcal{O}(\epsilon)$, naturally controlled by the PPO clipping range. The ODE is only executed during sampling, not during training.
2. **Brownian Motion Entropy Regularizer**: Exploiting the monotonically increasing entropy of Brownian motion, the method aligns the velocity field with the negative score direction of a reference flow via $\eta_t = (1-t)v_t(\mathbf{x}_t;\mathbf{s},\theta) - (\mathbf{x}_t - t\hat{v}_t(\mathbf{x}_t;\mathbf{s}))$, penalizing $\|\eta_t\|_2^2$ to encourage trajectory diffusion—entirely avoiding log-likelihood computation.
3. **Conditional Flow Policy Architecture**: Actions are generated as $\mathbf{a} = \varphi_1(\mathbf{z};\mathbf{s}) + \mathbf{n}$, inducing a Gaussian mixture policy $\pi(\mathbf{a}|\mathbf{s}) = \int \mathcal{N}(\mathbf{a};\varphi_1(\mathbf{z};\mathbf{s}),\boldsymbol{\sigma}^2)p_z(\mathbf{z})d\mathbf{z}$, which is strictly more expressive than a Gaussian policy.

## Method

### Continuous Normalizing Flow Policy

A conditional flow $\varphi:[0,1]\times\mathbb{R}^d\times\mathbb{R}^n\to\mathbb{R}^d$ is defined, governed by the ODE:

$$\frac{d}{dt}\varphi_t(\mathbf{z};\mathbf{s}) = v_t(\varphi_t(\mathbf{z};\mathbf{s});\mathbf{s}), \quad \varphi_0(\mathbf{z};\mathbf{s}) = \mathbf{z}$$

where $v$ is a time-dependent velocity field parameterized by a neural network. Actions are generated as $\mathbf{a}=\varphi_1(\mathbf{z};\mathbf{s})+\mathbf{n}$, with $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ and $\mathbf{n}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\sigma}^2)$. The added Gaussian noise promotes exploration and enables a closed-form importance ratio.

### Importance Ratio Approximation

Leveraging the translation invariance of Gaussian likelihood ratios, the conditional likelihood ratio is reformulated to depend only on the terminal displacement difference $\delta_{\varphi_1}$. The key approximation replaces this terminal displacement with velocity field differences along the interpolated path:

$$\rho \approx \mathbb{E}_{p(t)}\left[\frac{p_n(\mathbf{a}-\hat{\varphi}_1; \delta_{v_t}(\mathbf{x}_t;\mathbf{s}), \boldsymbol{\sigma}^2)}{p_n(\mathbf{a}-\hat{\varphi}_1; \mathbf{0}, \hat{\boldsymbol{\sigma}}^2)}\right]$$

where $\delta_{v_t} = v_t(\mathbf{x}_t;\mathbf{s},\theta) - \hat{v}_t(\mathbf{x}_t;\mathbf{s})$. The approximation error is theoretically shown to be $\mathcal{O}(\epsilon)$, negligible under PPO clipping. The clipped surrogate objective is:

$$J^{\text{Flow}}(\theta,\boldsymbol{\sigma}) = \mathbb{E}\left[\min(\rho \hat{A}, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A})\right]$$

### Brownian Motion Entropy Regularization

Brownian motion particles naturally diffuse toward a uniform distribution with monotonically increasing entropy. The associated probability path satisfies the heat equation $\partial p_t/\partial t = \nabla^2 p_t$, corresponding to a velocity field equal to the negative score $v_t = -\nabla\log p_t$. Using the result of Liu et al. (2025), the explicit relationship between the score and velocity field is:

$$\nabla_{\mathbf{x}}\log\hat{p}_t(\mathbf{x}_t;\mathbf{s}) = \frac{1}{1-t}(t\hat{v}_t(\mathbf{x}_t;\mathbf{s}) - \mathbf{x}_t)$$

The regularization loss comprises two terms: (1) the Brownian regularizer $-w_b\|\eta_t\|_2^2$, which encourages alignment of the velocity field with the negative score; and (2) the Gaussian noise entropy $\frac{w_g}{2}\sum_i\log(2\pi e\sigma_i^2)$, which encourages stochasticity. The total training objective is $J^{\text{Flow}} + J^{\text{Reg}}$.

### Training Procedure

Each iteration proceeds as: (1) collect trajectories using the reference policy (requiring ODE execution to generate $\hat{\varphi}_1$); (2) compute GAE advantage estimates; (3) sample time $t\sim U[0,1]$ over mini-batches, compute the approximate importance ratio and Brownian regularization term along the interpolated path; (4) jointly optimize policy parameters $\theta$ and noise variance $\boldsymbol{\sigma}$. Notably, the ODE is only invoked during the sampling phase; no ODE simulation or path-wise backpropagation is required during training.

## Key Experimental Results

### Main Results: IsaacLab Benchmark

| Environment | PPO | PolicyFlow | p-value |
|-------------|-----|-----------|---------|
| Lift-Cube | $153.1\pm3.0$ | $\mathbf{154.6\pm0.6}$ | 0.32 |
| Navigation | $3.5\pm0.3$ | $\mathbf{4.2\pm0.1}$ | **0.0027** |
| Open-Drawer | $\mathbf{99.8\pm1.7}$ | $99.1\pm0.7$ | 0.41 |
| Quadcopter | $\mathbf{141.8\pm0.5}$ | $141.0\pm0.09$ | 0.099 |
| Anymal-D | $24.5\pm0.1$ | $\mathbf{24.6\pm0.2}$ | 0.26 |
| G1 | $25.4\pm1.2$ | $\mathbf{30.0\pm1.1}$ | **0.00026** |
| H1 | $\mathbf{29.3\pm0.9}$ | $27.3\pm0.2$ | **0.0069** |
| Go2 | $\mathbf{27.9\pm0.3}$ | $27.4\pm0.9$ | 0.33 |

PolicyFlow matches or outperforms PPO on most IsaacLab tasks, achieving a statistically significant improvement of +18% on the G1 humanoid robot and a significant advantage on Navigation.

### Computational Efficiency

| Environment | Embedding Dim | PPO (ms) | PolicyFlow (ms) | Overhead |
|-------------|--------------|----------|-----------------|----------|
| Lift-Cube | 64 | 43.0 | 57.7 | +34% |
| Navigation | 64 | 36.9 | 54.1 | +47% |
| Open-Drawer | 64 | 81.3 | 104.1 | +28% |
| Quadcopter | 64 | 37.8 | 55.6 | +47% |
| Anymal-D | 64 | 41.2 | 57.1 | +39% |
| G1 | 256 | 66.9 | 90.6 | +35% |
| H1 | 512 | 63.4 | 115.5 | +82% |
| Go2 | 512 | 63.9 | 111.5 | +74% |

When model capacity is comparable to PPO, per-iteration training time increases by less than 50%; even with an 8× increase in embedding dimension, the computational cost remains below 2× that of PPO.

### Multimodal Capability (MultiGoal)

In the MultiGoal environment with six equally spaced goals: PPO (Gaussian policy) covers only a subset of goals; FPO/DPPO suffer from mode collapse due to the lack of effective entropy regularization; PolicyFlow with the Brownian regularizer achieves the most balanced coverage across all six goals, demonstrating the multimodal expressiveness of CNFs. Ablation studies show: uniform noise injection alone still leads to mode collapse → adding Gaussian entropy regularization partially alleviates the issue → incorporating the Brownian regularizer yields optimal performance.

### Ablation Study

- **Clipping range $\epsilon$**: Smaller $\epsilon$ reduces approximation error but restricts update step size (slower learning); $\epsilon=0.2$ provides the best trade-off.
- **Network initialization**: Glorot initialization with zero output layer (GI+ZOL) > standard Glorot (GI) > zero initialization (ZI).
- **Time sampling**: Continuous uniform (USC), discrete uniform (USD), and multi-point discrete uniform (Multi-USD) strategies show negligible differences; USD is used as default for simplicity.
- **Interpolation path**: Rectified Flow and TrigFlow paths outperform the Stochastic Interpolant path on MultiGoal; no significant difference is observed across the three on locomotion control tasks.

## Highlights & Insights

- ⭐⭐⭐ **Elegant importance ratio approximation**: By exploiting the translation invariance of Gaussian likelihood ratios and the interpolated path approximation, costly ODE path integration is replaced by a simple forward computation of velocity field differences, with error naturally controlled by the PPO clipping range.
- ⭐⭐⭐ **Brownian motion entropy regularizer**: A conceptually elegant design grounded in physical intuition—implicit entropy maximization is achieved through speed-score alignment, without computing log-likelihoods or relying on heuristic noise injection.
- ⭐⭐ **Comprehensive experimental coverage**: Evaluations span from simple 2D multi-goal tasks to the full IsaacLab robot suite (manipulation, navigation, locomotion, quadrotor), broadly validating the method's generality.
- ⭐⭐ **Honest computational cost analysis**: Training time overhead (+28%~+82%) is explicitly reported, without concealing the additional cost at higher embedding dimensions.

## Limitations & Future Work

- ⭐⭐ **Limited theoretical grounding of the Brownian regularizer**: The authors acknowledge that the derivation is not theoretically rigorous—the policy velocity field is not obtained via flow matching gradients and does not fully correspond to rectified flow dynamics; the design is largely heuristic.
- ⭐⭐ **Significant computational overhead at high embedding dimensions**: Training time for H1/Go2 with embedding size 512 approaches twice that of PPO, potentially becoming a bottleneck for large-scale deployment.
- ⭐ **No direct comparison with FPO/DPPO on IsaacLab**: Framework incompatibilities (JAX vs. PyTorch) preclude direct comparison, leaving the relative advantage on these important benchmarks unconfirmed.
- ⭐ **Evaluation limited to moderate action-space dimensionalities**: Scalability to very high-dimensional action spaces (e.g., dexterous hand manipulation) remains to be validated.

## Personal Reflections

The core contribution of PolicyFlow lies in identifying a pathway to **use highly expressive flow models for online RL without sacrificing training efficiency**. The importance ratio approximation is particularly clever—it exploits the fact that the PPO clipping range inherently bounds the magnitude of policy updates, making a first-order approximation sufficiently accurate in practice. While the Brownian regularizer is not fully rigorous theoretically, its design intent—orienting the velocity field toward the direction of entropy increase—is well-motivated and empirically effective.

Implications for future research: (1) This framework could be directly applied to RLHF for LLMs—if language models are viewed as flow-based policies, PolicyFlow's importance ratio approximation may enable more expressive policy updates than standard PPO; (2) The Brownian regularization idea could be generalized to other settings requiring mode diversity (e.g., diverse text generation); (3) The sensitivity to interpolation path choice suggests an interesting direction—different interpolation families may be better suited to different task characteristics.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](../../NeurIPS2025/reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

<!-- RELATED:END -->
