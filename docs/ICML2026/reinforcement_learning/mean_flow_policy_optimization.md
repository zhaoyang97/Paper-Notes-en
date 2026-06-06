---
title: >-
  [Paper Note] Auto temperature
description: >-
  [ICML 2026][Reinforcement Learning][MeanFlow] MFPO utilizes MeanFlow models (which learn average velocity instead of instantaneous velocity) as RL policies to reduce diffusion policy sampling from 20+ steps to 2 steps. I…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "MeanFlow"
  - "MaxEnt RL"
  - "soft policy iteration"
  - "average divergence network"
  - "importance sampling"
date: 2026-05-08
content_hash: 6cfc876697613bf7
---

# MFPO: Running MaxEnt RL with Few-step MeanFlow Policy at Nearly Gaussian Policy Speeds

**Conference**: ICML 2026  
**arXiv**: [2604.14698](https://arxiv.org/abs/2604.14698)  
**Code**: https://github.com/dongxiaoyi-xyz/MFPO  
**Area**: Reinforcement Learning / Diffusion Policy / Flow Matching  
**Keywords**: MeanFlow, MaxEnt RL, soft policy iteration, average divergence network, importance sampling

## TL;DR
MFPO utilizes MeanFlow models (which learn average velocity instead of instantaneous velocity) as RL policies to reduce diffusion policy sampling from 20+ steps to 2 steps. It introduces an average divergence network to solve action likelihood computation and uses ESS-weighted SNIS combining Gaussian and policy proposals for soft policy improvement. On MuJoCo, DMC, and HumanoidBench, it achieves performance $\geq$ diffusion baselines while reducing training time by ~50%.

## Background & Motivation

**Background**: Online RL for continuous control often relies on Gaussian or deterministic policies plus noise, which are overly unimodal. When complex tasks present multi-modal reward landscapes, these models prone to local optima. Diffusion and flow matching policies (DIPO, QVPO, DACER, DIME) model multi-modal actions through iterative generation, but inference requiring 10-20 steps makes training one to two orders of magnitude slower.

**Limitations of Prior Work**: MaxEnt RL requires balancing exploration and exploitation via action likelihood evaluation and soft policy improvement (matching the Boltzmann distribution). Both are challenging for diffusion policies: likelihood requires integrating instantaneous velocity divergence (intractable), and soft improvement requires Boltzmann samples (unavailable). Existing methods (DIME lower bounds, MaxEntDP numerical integration, SAC-Flow GRU/Transformer) suffer from loose bounds or large ODE discretization errors in few-step regimes.

**Key Challenge**: Achieving multi-modal expressiveness requires diffusion, but training efficiency requires few steps; however, few steps destroy likelihood accuracy and policy improvement precision.

**Goal**: To achieve precise likelihood and soft improvement for a multi-modal policy within 2 sampling steps, bringing diffusion policy training time close to that of Gaussian policies.

**Key Insight**: MeanFlow models (Geng et al. 2025) learn average velocity $\boldsymbol{u}(\boldsymbol{x}_t, r, t) = \frac{1}{t-r} \int_r^t \boldsymbol{v}(\boldsymbol{x}_\tau, \tau) d\tau$ rather than instantaneous velocity. Once learned accurately, 2-step sampling eliminates discretization error. However, applying MeanFlow to MaxEnt RL still requires solving the likelihood and soft improvement challenges.

**Core Idea**: (1) Construct an average divergence network $\delta_\omega$ to approximate $\frac{1}{t-r} \int_r^t \nabla \cdot \boldsymbol{v}_\theta d\tau$, reusing the sampling pipeline for likelihood calculation; (2) Use SNIS to estimate the marginal velocity of the Boltzmann distribution, adaptively combining policy and Gaussian proposals via ESS weighting; (3) Train the MeanFlow policy using soft policy iteration with a distributional critic and automatic temperature tuning.

## Method

### Overall Architecture

The framework consists of two core networks: a critic $Q_\phi$ and a MeanFlow policy $\boldsymbol{u}_\theta$ (supplemented by an ADN $\delta_\omega$). Soft policy iteration alternates between policy evaluation and policy improvement. Inference takes 2 steps: $\boldsymbol{a}_{t_{i-1}} = \boldsymbol{a}_{t_i} - \frac{1}{T} \boldsymbol{u}_\theta(\boldsymbol{s}, \boldsymbol{a}_{t_i}, t_{i-1}, t_i)$.

### Key Designs

1. **MeanFlow Policy + Average Divergence Network for Likelihood**:

    - **Function**: Enables precise estimation of action likelihood for a 2-step MeanFlow policy with only 5% additional computational cost.
    - **Mechanism**: The MeanFlow policy learns average velocity, where sampling is defined as $\boldsymbol{a}_r = \boldsymbol{a}_t - (t-r) \boldsymbol{u}_\theta$. Action likelihood utilizes the change-of-variable formula $\log \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s}) = \log p_1(\boldsymbol{a}_1) + \int_0^1 \nabla \cdot \boldsymbol{v}_\theta dt$. Native Jacobian computation with numerical integration is too expensive. The average divergence network $\delta_\omega(\boldsymbol{s}, \boldsymbol{a}_t, r, t) \approx \frac{1}{t-r} \int_r^t \nabla \cdot \boldsymbol{v}_\theta d\tau$ is trained using the Skilling-Hutchinson trace estimator $\widehat{\text{div}} = \frac{1}{N} \sum \boldsymbol{\epsilon}_i^\top \frac{\partial \boldsymbol{v}_\theta}{\partial \boldsymbol{a}_t} \boldsymbol{\epsilon}_i$. During inference, $\log \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s}) = \log p_1(\boldsymbol{a}_1) + \frac{1}{T} \sum_i \delta_\omega(\boldsymbol{s}, \boldsymbol{a}_{t_i}, t_{i-1}, t_i)$ reuses the sampling trajectory.
    - **Design Motivation**: MaxEnt entropy requires likelihood, but ODE divergence integration is intractable. ADN is the divergence-counterpart to the MeanFlow concept—both accurate (trained to match) and cheap (5% overhead). Skilling-Hutchinson ensures the trace does not require $d$ backward passes.

2. **ESS-weighted SNIS for Soft Policy Improvement**:

    - **Function**: Accurately estimates the marginal velocity field for policy updates despite the absence of Boltzmann samples.
    - **Mechanism**: The Boltzmann distribution is defined as $\pi(\boldsymbol{a}_0|\boldsymbol{s}) \propto \exp(\frac{1}{\alpha} Q)$. The marginal velocity field is $\boldsymbol{v}_t(\boldsymbol{a}_t|\boldsymbol{s}) = \mathbb{E}_{\pi(\boldsymbol{a}_0|\boldsymbol{a}_t, \boldsymbol{s})}[\frac{\boldsymbol{a}_t - \boldsymbol{a}_0}{t}]$. MaxEntDP/SDAC use a Gaussian proposal $q^2(\boldsymbol{a}_0)$ for SNIS, but the Effective Sample Size (ESS) drops sharply as $t \to 1$. Ours adds a policy proposal $q^1(\boldsymbol{a}_0) = \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s})$ (likelihood computed via ADN), which maintains high ESS as $t \to 1$. The final estimate $\hat{\boldsymbol{v}}_t = \sum_k \frac{\text{ESS}_k}{\sum_l \text{ESS}_l} \hat{\boldsymbol{v}}_t^k$ is an ESS-weighted combination.
    - **Design Motivation**: Single proposals perform differently across $t$—Gaussian is effective at small $t$ (concentrated target, Gaussian match), while the Policy proposal is effective at large $t$ (target dominated by Q). ESS-adaptive weighting allows the high-effective-sample estimator to dominate, resulting in lower variance than a single proposal.

3. **Distributional Critic + Auto Temperature + Action Selection**:

    - **Function**: A combination of engineering techniques to improve training stability and evaluation performance.
    - **Mechanism**: (a) Distributional critic uses C51 to treat the Q-function as a categorical distribution, using the mean for policy updates; (b) Auto-tuned temperature ensures $\alpha$ matches the target entropy $\mathcal{H}_{\text{target}} = -\rho \cdot \dim(\mathcal{A})$, where $\rho = 0.5$ is found to be generally optimal; (c) Action selection during evaluation samples multiple candidates from the policy and selects the deterministic action with the highest Q-value.
    - **Design Motivation**: MaxEnt random policies assist exploration during training, but deterministic selection is better for testing; distributional Q-learning has been proven effective in diffusion RL; auto temperature makes the method robust to reward scales.

### Algorithm

```python
Initialize Q_φ, π_θ (MeanFlow), δ_ω, α
for each step:
    # Policy evaluation
    L(φ) = (Q_φ(s,a) - (r + γ(Q(s',a') - α log π(a'|s'))))²
    # Policy improvement  
    Estimate v̂_t via ESS-weighted SNIS combining q^1 = π_θ + q^2 = Gaussian
    L(θ) = ||u_θ - sg(u_tgt)||²
    # ADN update via Eq. 17
    L(ω) = ||δ_ω - sg(δ_tgt)||²
    # Auto temperature
    L(α) = α (H(π_θ) - H_target)
```

## Key Experimental Results

### Main Results: MuJoCo (5 locomotion)

| Algorithm | Sampling Steps | Inference Time (ms) | Avg Performance |
|---|---|---|---|
| **MFPO (ours)** | **2** | **0.46** | best/tied |
| DIME | 16 | 0.97 | comparable |
| FlowRL | 11 | 0.42 | comparable |
| SAC-Flow | 4 | 0.96 | comparable |
| MaxEntDP | 20 | 1.56 | slightly lower |
| DACER | 20 | 1.06 | comparable |
| QVPO | 20 | 1.68 | slightly lower |
| TD3 (Gaussian) | 1 | 0.14 | lower (unimodal) |
| SAC (Gaussian) | 1 | 0.15 | lower (unimodal) |

MFPO achieves 0.46ms inference time with 2-step sampling, which is 2-3.5$\times$ faster than other diffusion methods; performance is $\geq$ all diffusion baselines. Training time is reduced by ~50%.

### Ablation Study (HalfCheetah-v3)

| Ablation | Impact |
|---|---|
| MeanFlow $\to$ Flow Matching policy | Performance degradation; average velocity is necessary for few-steps |
| Remove ADN | Performance degradation; likelihood estimation is necessary |
| Only Gaussian proposal | Low ESS as $t \to 1$ |
| Only policy proposal | Failed; proposal not effective |
| $K_1:K_2 = 1:2$ (More Gaussian) | Optimal |
| Fixed temperature | Worse than auto-tuning |
| $\rho = 0.5$ for target entropy | Optimal |

### Key Findings

- **2-step sampling matches 20-step diffusion performance**: MFPO with MeanFlow 2-step catches up with MaxEntDP/QVPO 20-step while reducing inference time by 3-4$\times$.
- **Training time reduced by 50%**: Compared to DACER/DIME/MaxEntDP, training time is nearly halved.
- **ADN adds accurate likelihood with 5% overhead**: Compared to naive numerical integration (requiring $d$ backward passes per step), ADN is almost cost-free.
- **Two-proposal SNIS is critical**: The combined variance is significantly lower, which is the key to stable policy updates.
- **HumanoidBench performance**: MFPO also matches SOTA on high-dimensional tasks (>50 dim action), scaling to complex control.

## Highlights & Insights

- **MeanFlow + MaxEnt RL is a perfect combination**: MeanFlow solves "few-step expressiveness," while MaxEnt solves "exploration." The resulting combination is both fast and stable.
- **ADN is an elegant methodological analogy**: Transferring the "MeanFlow learns average velocity" idea to "learning average divergence" provides methodological consistency.
- **Engineering value of ESS-weighted SNIS**: This is not ad-hoc tuning but an automatic weighting using ESS, with theoretical variance reduction guarantees.
- **Practical significance of 50% faster training**: This makes diffusion-based RL viable for engineering projects—where previously 20-step training took days to match SAC results from a few hours, MFPO reaches diffusion performance in just a few hours.
- **Recycling MeanFlow**: Moving latest advances in generative modeling (average velocity) to RL is a prime example of cross-pollination between sub-fields.

## Limitations & Future Work

- **Expressivity limits at 2 steps**: Although MeanFlow reduces discretization error, there remains a theoretical gap in multi-modality between 2 steps vs. 20 steps; the sweet spot for extremely complex tasks might be 4-8 steps.
- **ADN training stability**: The ADN training target relies on stop-gradients and recursive structures; whether it drifts during long-term training has not been fully verified.
- **Ratio tuning for the two proposals**: $K_1:K_2 = 1:2$ was best for HalfCheetah, but whether this requires cross-task tuning is not systematically ablated.
- **Choice of Distributional Critic**: C51 is the default, but QR-DQN or IQN might be more stable; the impact of the distributional choice was not isolated in ablations.
- **Missing sample efficiency comparisons**: While training time is faster, the sample efficiency (performance per environment step) vs. baselines is not explicitly contrasted.
- **Lack of Atari / pixel-based validation**: All experiments were state-based; the effectiveness of MeanFlow policy under pixel observations remains unknown.

## Related Work & Insights

- **vs. DACER / DIME / MaxEntDP / SDAC**: These use diffusion + MaxEnt but require 10-20 sampling steps; MFPO uses MeanFlow to reduce steps to 2 while retaining MaxEnt precision.
- **vs. FPMD / QVPO / FlowRL**: These use diffusion-based RL but not the MaxEnt framework; MFPO possesses both multi-modal expressiveness and MaxEnt principled exploration.
- **vs. SAC-Flow**: They use GRU/Transformer to stabilize diffusion policy training; MFPO does not rely on specific architectures, making its methodology more general.
- **vs. MeanFlow (Geng et al. 2025)**: The original work on generative modeling; this paper represents the first application of MeanFlow to RL.
- **Insights**: (1) MeanFlow can be applied to any scenario where diffusion models are difficult to deploy due to slow sampling; (2) The "learning average via consistency" idea in MeanFlow can be extended to other time-integrated quantities like divergence, Lyapunov functions, or value functions; (3) Multi-proposal SNIS is a general technique for handling intractable distributions, with ESS-weighting providing adaptivity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ MeanFlow + MaxEnt RL combination + ADN analogy + ESS-weighted SNIS; a comprehensive methodological suite.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks + 8 baselines + 4-dimensional ablation + ESS/variance visualization; complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, intuitive ADN analogy, and effective visualization of motivation in Figure 1.
- Value: ⭐⭐⭐⭐⭐ Brings diffusion-based RL close to Gaussian policies in terms of inference speed and training cost; a key step toward practical diffusion RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[ICML 2026\] Learning to Route Languages for Multilingual Policy Optimization](learning_to_route_languages_for_multilingual_policy_optimization.md)
- [\[ICML 2026\] EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance](eapo_enhancing_policy_optimization_with_on-demand_expert_assistance.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](../../AAAI2026/reinforcement_learning/behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)

</div>

<!-- RELATED:END -->
