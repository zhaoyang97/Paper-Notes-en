---
title: >-
  [Paper Note] Offline Reinforcement Learning with Generative Trajectory Policies
description: >-
  [ICML2026][Reinforcement Learning][Offline Reinforcement Learning] This paper unifies Diffusion Policies, Flow Matching, and Consistency Policies into a single family of "Generative Trajectory Policies (GTP)" using a "co…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Offline Reinforcement Learning"
  - "ODE Flow"
  - "Consistency Trajectory"
  - "Flow Matching"
  - "Advantage Weighted"
date: 2026-05-08
content_hash: 9cc99632c47c529f
---

# Offline Reinforcement Learning with Generative Trajectory Policies

**Conference**: ICML2026  
**arXiv**: [2510.11499](https://arxiv.org/abs/2510.11499)  
**Code**: https://github.com/wmd3i/gtp  
**Area**: Reinforcement Learning / Offline RL / Generative Policy  
**Keywords**: Offline Reinforcement Learning, ODE Flow, Consistency Trajectory, Flow Matching, Advantage Weighted

## TL;DR
This paper unifies Diffusion Policies, Flow Matching, and Consistency Policies into a single family of "Generative Trajectory Policies (GTP)" using a "continuous-time ODE solution mapping". Combined with a closed-form score approximation to align with offline samples and an advantage-weighted training objective, the policy achieves near-perfect scores on challenging tasks like AntMaze while maintaining few-step sampling efficiency.

## Background & Motivation

**Background**: Offline RL prohibits interaction with the environment and requires mining a generalizable policy from a fixed dataset. Since behavior in data often exhibits strong multi-modality, "using generative models as policies" has become mainstream recently—leading to a proliferation of Diffusion Policies, Consistency Policies, and Flow Matching policies.

**Limitations of Prior Work**: This family of methods has long struggled with a sharp trade-off: Diffusion Policies have high expressivity but require dozens of iterative sampling steps, making single-step inference too costly; Consistency Policies compress inference to one or two steps, but the policy quality drops significantly and performance saturates quickly.

**Key Challenge**: Diffusion and Consistency appear to be two different routes, but they essentially learn the same "noise $\to$ data" trajectory described by an ODE. The former learns the instantaneous velocity field, while the latter learns large-span jumps. Each only touches one extreme of the ODE solution mapping $\Phi(\boldsymbol{x}_t, t, s)$, and no one has attempted to learn the entire solution mapping.

**Goal**: (i) Put Diffusion, Flow Matching, Consistency, CTM, Shortcut, and MeanFlow into a unified ODE solution mapping framework; (ii) Design an offline RL policy class within this framework that balances expressivity and efficiency; (iii) Solve the implementation hurdles of "unstable bootstrapping supervision" and the "mismatch between BC objectives and policy improvement."

**Key Insight**: Instead of choosing between "Diffusion vs. Consistency," it is better to directly learn the complete ODE solution mapping $\Phi(\boldsymbol{x}_t, t, s)$—it naturally enables jumping across any step length, retaining the expressivity of diffusion while gaining the efficiency of consistency.

**Core Idea**: Use two complementary objectives—"instantaneous anchors + global self-consistency"—to joint-learn the solution mapping. Replace bootstrapping supervision with a closed-form score proxy based on offline samples and push the generative loss toward high-value actions using advantage exponential weights.

## Method

### Overall Architecture

GTP implements the policy $\pi_\theta(s)$ as a parameterized ODE solution mapping $\Phi_\theta(s, a_t, t, \tau)$: taking state $s$, noisy action $a_t$, current time $t$, and target time $\tau$ as inputs, and outputting a cleaned action $a_\tau$. During inference, it starts from $a_T \sim \mathcal{N}(0, T^2 I)$ and repeatedly calls $\Phi_\theta$ along an arbitrary time grid $T = t_0 > t_1 > \dots > t_K = 0$ to obtain the final action, allowing a free trade-off between 1 step and dozens of steps. During training, an Actor-Critic framework is used: the Critic is a twin Q-network learned via standard TD error; the Actor optimizes both "instantaneous flow loss" and "trajectory consistency loss," driven toward policy improvement by the advantage-weighted coefficient $w(s,a)$.

### Key Designs

1.  **Unified ODE Solution Map with Inst Map + Consistency Dual Objectives**:
    *   **Function**: Compresses diffusion denoising and consistency trajectories into the same learning objective, requiring the model to equate to a denoiser/velocity field at "infinitesimal steps" and satisfy trajectory additivity at "arbitrary large spans."
    *   **Mechanism**: Introduces a proxy function $\phi(\boldsymbol{x}_t, t, s) = \boldsymbol{x}_t + \frac{t}{t-s}\int_t^s f(\boldsymbol{x}_\tau, \tau) d\tau$, and restores the solution map via $\Phi = (1 - s/t)\phi + (s/t)\boldsymbol{x}_t$. The **instantaneous flow loss** takes the limit $s \to t$ as $\lim_{s\to t}\phi(\boldsymbol{x}_t, t, s) = \boldsymbol{x}_t - t f(\boldsymbol{x}_t, t)$, equivalent to letting the network learn denoising/velocity as a local anchor; the **trajectory consistency loss** enforces $\Phi(\boldsymbol{x}_t, t, s) \approx \Phi(\Phi(\boldsymbol{x}_t, t, u), u, s)$ for any $t > u > s$ as a global regulator.
    *   **Design Motivation**: A standalone instantaneous loss only reproduces local behavior (requiring many integration steps); a standalone consistency loss without a local anchor is merely copying a teacher network. Optimizing both together achieves the "entire solution map" GTP needs for both "few-step quality" and "multi-step upper bounds."

2.  **Stable Score Approximation**:
    *   **Function**: Eliminates "bootstrapping supervision" common in diffusion/consistency—where the early, poor network itself acts as the ODE right-hand term $f_\theta$ to integrate the training target, leading to bad targets driving bad updates in the actor-critic loop.
    *   **Mechanism**: Replaces the true score $f^\star(\boldsymbol{x}_t, t) = (\boldsymbol{x}_t - \mathbb{E}[\boldsymbol{x}|\boldsymbol{x}_t])/t$ on the ODE right-hand side with a closed-form proxy $\tilde{f}(\boldsymbol{x}_t, t) = (\boldsymbol{x}_t - \boldsymbol{x})/t$ anchored to the current offline sample $\boldsymbol{x}$. Theorem 4.1 proves that when the ODE solver is zero-stable of order $p$ and the maximum step size is $h$, the difference between the ideal target $\mathcal{L}_{\text{ideal}}$ and the practical target $\mathcal{L}_{\text{prac}}$ is $O(h^p)$. In practice, intermediate samples in the trajectory consistency loss are generated in one step by $\boldsymbol{x}_u = \boldsymbol{x} + u \cdot \boldsymbol{z},\ \boldsymbol{z} \sim \mathcal{N}(0, I)$, bypassing the ODE solver.
    *   **Design Motivation**: Bootstrapping is the root cause of why offline RL cannot train diffusion/consistency policies deeply; replacing multi-step integration with a single perturbation saves computation and breaks the vicious cycle of "bad scores $\to$ bad targets $\to$ worse scores."

3.  **Advantage-Weighted Value-Driven Objective**:
    *   **Function**: Pushes the BC-flavored generative objective toward true policy improvement, allowing GTP to retain diffusion-like expressivity while biasing toward high-reward actions within the data distribution, similar to IQL/AWR.
    *   **Mechanism**: Under the KL-regularized RL objective, the optimal policy takes the form $\pi^*(a|s) \propto \pi_{\text{BC}}(a|s)\exp(\eta A(s,a))$, leading to the "advantage-weighted generative loss" $\max_\theta \mathbb{E}_{(s,a)\sim\mathcal{D}}[\exp(\eta A(s,a)) \cdot \ell_{\text{gen}}(\pi_\theta; a|s)]$. The practical weights are normalized and truncated: $w(s,a) = \exp\left(\eta \cdot \frac{\max(0, A(s,a))}{\text{std}(A) + \epsilon}\right)$, injected into the expectations of both flow and consistency losses.
    *   **Design Motivation**: Pure generative objectives only replicate the data distribution; hard truncation of negative advantage prevents low-quality actions from pulling weights to negative values; standard deviation normalization ensures $\eta$ does not require extensive retuning across tasks.

### Loss & Training

The total Actor loss is $\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{Consistency}} + \lambda_{\text{Flow}} \cdot \mathcal{L}_{\text{Flow}}$, with both terms multiplied by $w(s, a)$; the Critic follows the standard twin Q TD objective $r + \gamma \min_{j=1,2} Q_{\bm{\varphi}_j^-}(s', \pi_{\theta'}(s'))$; Actor and Critic target networks are updated via EMA. The inference steps $K$ can be chosen between 1–8, allowing a single model to provide a continuous spectrum from "extremely fast but coarse" to "multi-step refinement."

## Key Experimental Results

### Main Results

Compared with the strongest contemporary generative policies (including Diffusion/Consistency/Flow) and classic offline RL on D4RL, GTP achieves SOTA on both Locomotion and AntMaze, nearly reaching a perfect score on the AntMaze large map tasks known for multi-modal trajectories.

| Dataset | Metric | GTP (Ours) | Prev. SOTA Generative | Gain |
| :--- | :--- | :--- | :--- | :--- |
| AntMaze-Large-Diverse | Normalized Score | ≈ 100 (Perfect) | Significantly lower | Substantial lead, essentially "solved" |
| AntMaze-Large-Play | Normalized Score | ≈ 100 (Perfect) | Significantly lower | Substantial lead |
| D4RL Locomotion (mean) | Normalized Score | Highest Total | Slightly lower | Matched or exceeded across the board |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full GTP | Highest | Stable Score + Consistency Loss + Advantage Weighting |
| w/o Stable Score Approx | Significant Drop | Degenerates to bootstrapping; unstable training, huge drop in AntMaze |
| w/o Consistency Loss | Moderate Drop | Only instantaneous loss remains; few-step quality collapses, needs more steps |
| w/o Advantage Weighting| Moderate Drop | Degenerates to generative BC; no policy improvement |

### Key Findings

*   The **Stable Score Approximation** is the watershed for "whether or not" AntMaze can be solved—it both saves computation and stabilizes training, consistent with the $O(h^p)$ error bound in Theorem 4.1.
*   The **Trajectory Consistency Loss** is crucial for "few-step inference"; the version without it drops most significantly when inference steps $K$ decrease from 8 to 1.
*   **Advantage Weighting** shows more pronounced gains in tasks with lower data diversity like Locomotion, and slightly less gain in AntMaze, which is primarily limited by expressivity bottlenecks.

## Highlights & Insights

*   **"Learning the whole solution map" is a neglected middle ground**: The authors move beyond the "single-step vs. multi-step" binary opposition to learn the mapping $\Phi(\boldsymbol{x}_t, t, s)$ itself. This perspective conveniently incorporates CTM, Shortcut, and MeanFlow into a single family, inspiring applications in both generative models and policy learning.
*   **Closed-form scores save computation and stabilize training**: Replacing the network-predicted score with $(\boldsymbol{x}_t - \boldsymbol{x})/t$ essentially "anchors supervision back to data," consistent with the spirit of anchoring scores to conditional paths in Flow Matching, but more aggressively removing the ODE solver. Its $O(h^p)$ bound provides explicit guidance for practitioners on "how large a step size is sufficient."
*   **Transferable trick**: The $\max(0, A)/\text{std}(A)$ normalization + truncation paradigm for advantage-weighted generative loss is highly "plug-and-play" and can be applied to any offline RL framework combining generative BC and value refinement.

## Limitations & Future Work

*   **Theory depends on Lipschitz + Zero-stability assumptions**: The $O(h^p)$ bound assumes both $f^\star$ and $\Phi_\theta$ are Lipschitz with respect to $\boldsymbol{x}$; practical networks (with ReLU, attention) only satisfy this approximately, and the singularity of $\tilde{f}$ near $t \to 0$ is not fully discussed.
*   **Evaluated only on D4RL**: It does not cover robotic control from vision (RoboMimic / real arms) or multi-task policies; whether the victory in AntMaze translates to high-dimensional pixel observations is unknown.
*   **Future directions**: Combining GTP with "classifier guidance" from Diffusion Q-learning or upgrading advantage weighting from "multiplying by loss" to "multiplying on sampling trajectories" might enable explicit policy improvement during the sampling phase, further reducing reliance on the weight $\eta$.

## Related Work & Insights

*   **vs. Diffusion Policy (Wang et al. 2023, Janner et al. 2022)**: They learn instantaneous denoising, requiring dozens of inference steps; GTP extends the same network into a solution map, enabling jumps at arbitrary step lengths, with 1–4 steps approaching the multi-step performance ceiling.
*   **vs. Consistency Policy (Ding & Jin 2024)**: They rely on distillation to force inference into 1 step, leading to rapid performance saturation; GTP avoid teacher distillation by learning local velocity fields and global consistency simultaneously, resolving the expressivity-efficiency trade-off at its root.
*   **vs. IQL / AWR**: Classic advantage-weighted methods use Gaussian/MLP policies, struggling to fit multi-modal behavior; GTP grafts the same advantage-weighted logic onto generative trajectory policies, cleanly decoupling "value function guidance" and "distribution expressivity" for the first time.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Bringing CTM/Flow/Consistency into a unified ODE solution mapping framework and applying it to RL is a rare "perspective as contribution" work.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Full SOTA on D4RL and perfect scores on AntMaze; three ablations clearly explain the contribution of key designs; lacks pixel-based tasks.
*   Writing Quality: ⭐⭐⭐⭐⭐ The narrative from unified framework $\to$ three hurdles $\to$ three countermeasures is clean and sharp. Theorem 4.1 + two Remarks tightly link intuition with bounds.
*   Value: ⭐⭐⭐⭐⭐ Resolves the long-standing "expressivity vs. efficiency" trade-off for generative policies in offline RL, providing a universal blueprint for future continuous-time generative policies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Trajectory-Level Data Augmentation for Offline Reinforcement Learning](trajectory-level_data_augmentation_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[ICML 2026\] Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](beyond_the_proxy_trajectory-distilled_guidance_for_offline_gflownet_training.md)
- [\[ICML 2026\] MoMa QL: Accelerating Diffusion/Flow Matching Policies for Offline and Offline-to-Online RL via Moment Matching](moment_matching_q-learning.md)
- [\[ICML 2026\] PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](pac-bayesian_reinforcement_learning_trains_generalizable_policies.md)

</div>

<!-- RELATED:END -->
