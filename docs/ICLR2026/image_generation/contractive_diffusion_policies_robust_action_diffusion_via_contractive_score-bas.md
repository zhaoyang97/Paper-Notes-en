---
title: >-
  [Paper Note] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations
description: >-
  [ICLR2026][Image Generation][Diffusion Policy] This paper proposes Contractive Diffusion Policies (CDPs), which introduce contraction regularization into the diffusion sampling ODE to suppress the accumulation of score matching errors and solver errors. With minimal modification and a single hyperparameter $\gamma$, CDPs improve the robustness of diffusion-based policies in offline learning settings.
tags:
  - ICLR2026
  - Image Generation
  - Diffusion Policy
  - Contraction Theory
  - Offline Reinforcement Learning
  - Imitation Learning
  - Score-Based Models
date: 2026-05-08
content_hash: f1a4d5d0466187e6
---

# Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations

**Conference**: ICLR2026
**arXiv**: [2601.01003](https://arxiv.org/abs/2601.01003)
**Code**: To be confirmed
**Area**: Image Generation
**Keywords**: Diffusion Policy, Contraction Theory, Offline Reinforcement Learning, Imitation Learning, Score-Based Models

## TL;DR
This paper proposes Contractive Diffusion Policies (CDPs), which introduce contraction regularization into the diffusion sampling ODE to suppress the accumulation of score matching errors and solver errors. With minimal modification and a single hyperparameter $\gamma$, CDPs improve the robustness of diffusion-based policies in offline learning settings.

## Background & Motivation
1. Diffusion policies achieve strong performance in offline RL and imitation learning, yet their iterative sampling process causes score estimation errors, discretization errors, and numerical integration errors to accumulate progressively.
2. Unlike image generation, in robotic control even small action deviations compound and amplify, pushing the policy out of the data distribution support and causing task failures.
3. Existing Diffusion Policies exhibit insufficient consistency in generating actions from the same state, and their performance degrades significantly in data-scarce scenarios.
4. Contraction Theory studies the convergence of solutions to differential equations, enabling systems to rapidly forget initial perturbations and naturally suppress error growth.
5. Prior contractive DDPM methods enforce contraction globally, which may reduce diversity and is difficult to integrate efficiently into offline learning frameworks.
6. A theoretically grounded, implementation-simple, and computationally tractable approach is needed to enhance robustness without sacrificing diversity in the action distribution.

## Method

### Overall Architecture
The reverse diffusion ODE $d\mathbf{a}_t = F_\theta(\mathbf{a}_t, t) dt$ in offline policy learning has its Jacobian decomposed into a drift term $f(t)I$ and a score Jacobian $h(t)J_{\epsilon_\theta}$, where $h(t) = g(t)^2 / (2\sigma_t)$. By constraining the maximum eigenvalue of the score Jacobian, the method promotes contractive behavior in the sampling process, causing nearby diffusion flow trajectories to converge and thereby suppressing error accumulation. During training, the score Jacobian is computed across all denoising steps for each batch and a contraction penalty is applied; at deployment, the frozen policy directly generates actions via ODE sampling.

### Key Design 1: Theoretical Derivation of the Contraction Condition
Theorem 3.1 is proved: the diffusion ODE is contractive if and only if the maximum eigenvalue of the symmetric part of the score Jacobian satisfies $\lambda_{\max}(J_{\epsilon_\theta}^{\text{sym}}) < -f(t)h(t)^{-1}$. Corollary 3.1.1 is then derived, providing an upper bound on action variance with respect to initial seed sensitivity.

### Key Design 2: Efficient Eigenvalue Computation
Power Iteration is used, requiring only $K=3\sim4$ Jacobian-vector products to stably estimate the maximum eigenvalue $\hat{\lambda}_{\max}$, avoiding the prohibitive cost of computing the full spectrum.

### Key Design 3: Contraction Loss Design
Two loss formulations are provided: (a) a truncated penalty based on $\max(-\beta, \hat{\lambda}_{\max} + f(t)h(t)^{-1})$; (b) an alternative based on the Frobenius norm $\|J_{\epsilon_\theta}^{\text{sym}} + \beta I\|_F$. The parameter $\beta$ prevents over-contraction from causing mode collapse.

### Loss & Training
The total loss is $\mathcal{L}(\theta) = \mathcal{L}_d(\theta) + \gamma \mathcal{L}_c(\theta)$, where $\mathcal{L}_d$ is the standard score matching loss and $\gamma$ is the sole newly introduced hyperparameter (default $\gamma=0.1$). The score matching loss ensures accurate denoising and prevents the contraction penalty from pushing the score toward a trivial contractive field; the two terms form a beneficial tension. Offline RL builds on EDP (Efficient Diffusion Policy) and IL builds on DBC (Diffusion Behavior Cloning). The backbone network uses a residual MLP for low-dimensional observations and a Diffusion Transformer for image observations. Training runs for 200k–500k steps, with checkpoints saved every 20k steps and evaluated in simulation.

## Key Experimental Results

### Main Results — D4RL Offline RL (Table 1)

| Environment | EDP | IDQL | CDP |
|---|---|---|---|
| ME-HalfCheetah | 93.4 | 88.9 | **94.8** |
| M-Hopper | 61.1 | 54.2 | **62.8** |
| M-Walker2D | 81.7 | 80.9 | **86.5** |
| MR-Hopper | 55.1 | 51.5 | **63.5** |
| Complete-Kitchen | 32.9 | 31.6 | **51.0** |
| **Overall Average** | 61.2 | 60.3 | **65.7** |

### Ablation Study / Low-Data Experiment (Figure 5)

| Data Ratio | EDP | CDP |
|---|---|---|
| 100% | Baseline | Marginally better |
| 10% | Significant degradation | **Decisive advantage** |

### Key Findings
- Robomimic IL success rate: CDP-Unet averages 0.90 vs. DP-Unet 0.88, reaching 1.00 vs. 0.95 on Can-H and 0.75 vs. 0.68 on Transport-H.
- Physical Franka arm experiments: CDP successfully completes 3/4 tasks (Slide/Stack/Peg), significantly outperforming DBC on challenging tasks such as Peg.
- The Kitchen-Complete environment shows the most prominent improvement: CDP 51.0 vs. EDP 32.9, an absolute gain of +18.1.
- MR-Walker2D: CDP 89.7 vs. IDQL 84.6 vs. EDP 82.0, with a clear advantage in scenarios with poor replay data quality.
- Training overhead: CDP takes approximately 5,236 seconds per 100k steps vs. 4,594 seconds for EDP, incurring roughly 14% additional computational cost.
- The hyperparameter $\gamma$ is searched over $\{0.001, 0.01, 0.1, 1, 10, 100\}$; setting $\gamma=0.1$ directly yields stable performance.

## Highlights & Insights
- **Theoretical elegance**: Contraction theory is precisely linked to the diffusion sampling ODE; Theorem 3.1 establishes a necessary and sufficient condition relating the eigenvalues of the score Jacobian to sampling contraction.
- **Practicality**: Only one hyperparameter $\gamma$ and one efficiently computable contraction loss are added, enabling seamless integration into existing diffusion policy architectures.
- **Significant low-data advantage**: CDP decisively surpasses all baselines with 10% data, demonstrating that contraction effectively suppresses the amplification of score matching errors under data scarcity.
- **Real-robot validation**: The practical value of the method is verified across four manipulation tasks on a physical Franka arm.

## Limitations & Future Work
- The contraction loss coefficient $\gamma$ is sensitive to tuning; improper adjustment degrades policy performance, and no adaptive adjustment mechanism is available.
- The method is implemented only on top of EDP and DBC; its combination with other offline learning methods such as DiffusionQL and IDQL remains unexplored.
- Experiments in image observation spaces are relatively limited, and no real-robot experiments with low-dimensional observations are conducted.
- In environments already near-optimal (e.g., ME-Walker2D), the gain from contraction regularization is marginal or even slightly negative.

## Related Work & Insights
- **vs. DiffusionQL**: DQL applies behavior regularization via diffusion loss plus value function maximization, whereas CDP enhances robustness at the level of sampling dynamics; on D4RL, CDP averages 65.7 vs. DQL 58.8.
- **vs. Contractive DDPM**: The latter enforces contraction globally and may sacrifice diversity; CDP controls contraction strength via the truncation parameter $\beta$ to avoid mode collapse, making it better suited for policy learning.
- **vs. Diffusion Policy (DP-Unet)**: DP-Unet leads in IL thanks to the UNet architecture and long action sequence advantages; CDP-Unet further improves to an average success rate of 0.90 after integrating contraction.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic incorporation of contraction theory into the diffusion policy sampling process.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers D4RL, Robomimic, and real-robot settings across both offline RL and IL paradigms.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and experimental descriptions are clear.
- Value: ⭐⭐⭐⭐ Delivers consistent performance gains at minimal implementation cost, particularly suited to data-scarce scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](../../NeurIPS2025/image_generation/real-time_execution_of_action_chunking_flow_policies.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid.md)

<!-- RELATED:END -->
