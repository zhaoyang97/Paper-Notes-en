---
title: >-
  [Paper Note] Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals
description: >-
  [ICML 2025][Image Generation][Hierarchical Reinforcement Learning] Proposes a hierarchical reinforcement learning framework combining conditional diffusion models with Gaussian process priors. Through an uncertainty-aware subgoal generation mechanism, it addresses the core challenge of high-level policies struggling to generate effective subgoals amid dynamic changes in low-level policies.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Hierarchical Reinforcement Learning"
  - "Diffusion Models"
  - "Gaussian Processes"
  - "Subgoal Generation"
  - "Uncertainty"
date: 2026-05-08
content_hash: bbe6c44a9b363ed4
---

# Hierarchical Reinforcement Learning with Uncertainty-Guided Diffusional Subgoals

**Conference**: ICML 2025  
**arXiv**: [2505.21750](https://arxiv.org/abs/2505.21750)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Hierarchical Reinforcement Learning, Diffusion Models, Gaussian Processes, Subgoal Generation, Uncertainty

## TL;DR

Proposes a hierarchical reinforcement learning framework combining conditional diffusion models with Gaussian process priors. Through an uncertainty-aware subgoal generation mechanism, it addresses the core challenge of high-level policies struggling to generate effective subgoals amid dynamic changes in low-level policies.

## Background & Motivation

**Background**: Hierarchical reinforcement learning (HRL) decomposes complex decision-making into two levels—high-level goal setting and low-level execution—and represents a mainstream paradigm for solving long-horizon sparse-reward problems. Existing methods such as HIRO and HAC use deterministic or simple Gaussian distributions to generate subgoals.

**Limitations of Prior Work**: HRL faces a fundamental non-stationarity issue: the low-level policy continuously changes during training, meaning that the same subgoal can correspond to completely different behaviors at different training stages. The high-level policy must make decisions in a continuously drifting goal space, but existing methods lack uncertainty modeling for subgoal generation.

**Key Challenge**: The high-level policy needs to generate sufficiently diverse subgoals to explore effective policies, while simultaneously ensuring that these subgoals are actually reachable for the low-level policy. These two demands consistently conflict during training due to the dynamic evolution of the low-level policy.

**Goal**: To design a high-level policy that can both capture complex subgoal distributions and quantify generation uncertainty, maintaining stable and efficient learning in hierarchical systems during the evolution of the low-level policy.

**Key Insight**: Utilizing the powerful distribution-modeling capabilities of diffusion models to generate diverse subgoals, while introducing a Gaussian process prior to provide uncertainty quantification and trajectory alignment constraints.

**Core Idea**: Regularizing the conditional diffusion model with a Gaussian process prior, combining the expressiveness (multimodal modeling of diffusion models) and reliability (uncertainty quantification of GPs) of subgoal generation.

## Method

### Overall Architecture

The system consists of three core components: (1) a conditional diffusion subgoal generator, which generates candidate subgoals conditioned on the current state; (2) a Gaussian process prior module, which models feasible subgoal trajectories and provides uncertainty estimations; and (3) a hybrid subgoal selection mechanism, which combines the GP prediction mean (feasibility) and diffusion samples (diversity) to determine the final subgoal. Upon receiving the subgoal $g_t$, the low-level policy executes action steps of fixed length and is trained using the intrinsic reward $r_t^l = -\|s_t + g_t - s_{t+1}\|_2$, guiding the agent to move toward the subgoal. The high-level policy updates the subgoal every $c$ steps based on environmental feedback.

### Key Designs

1. **GP-Regularized Conditional Diffusion**:

    - Function: Generating diverse and feasible subgoal distributions
    - Mechanism: Training a conditional diffusion model $p_\theta(g | s)$ to generate subgoals, while imposing regularization constraints on the diffusion process using a GP prior $\mathcal{GP}(\mu, k)$. A GP log-likelihood term is integrated into the ELBO objective of the diffusion model, aligning the generated subgoal distribution with the GP posterior of historical successful trajectories
    - Design Motivation: Although a standalone diffusion model can model complex distributions, it may generate numerous unfeasible subgoals in early training phases. The GP prior provides 'soft constraints' leveraging experiences, guiding the diffusion model to generate subgoals within feasible regions without sacrificing the multimodal representation capacity of the distribution

2. **Uncertainty-Guided Subgoal Selection**:

    - Function: Dynamically balancing exploration and exploitation
    - Mechanism: Sampling $N$ candidate subgoals from the diffusion model, and evaluating the predictive mean $\mu(g)$ (feasibility score) and predictive variance $\sigma^2(g)$ (uncertainty) for each candidate using the GP posterior. The subgoal with the highest comprehensive score $\mu(g) + \beta \sigma(g)$ is selected, where $\beta$ controls the exploration-exploitation balance
    - Design Motivation: In the early stages of training, the uncertainty of the GP prior is high, making $\sigma(g)$ dominate the selection and encouraging exploration. As experience accumulates, the GP posterior tightens, making $\mu(g)$ dominate the selection, which shifts the policy toward exploitation. This adaptive mechanism avoids manually designing exploration schedules

3. **Intrinsic Reward & Trajectory Alignment**:

    - Function: Training the low-level policy to move toward subgoals and ensuring consistency between the GP prior and actual trajectories
    - Mechanism: The low-level policy uses a distance-driven intrinsic reward $r_t^l = -\|s_t + g_t - s_{t+1}\|_2$. Meanwhile, the GP kernel function is constructed based on actual trajectory distances in the state space, allowing the GP posterior to accurately reflect the reachability relations between different subgoals
    - Design Motivation: Intrinsic rewards provide dense training signals to resolve the sparse reward problem; GP trajectory alignment ensures that uncertainty quantification is based on real dynamics rather than arbitrary metric spaces

### Loss & Training

The loss of the high-level policy consists of two components: the denoising loss of the diffusion model $L_{\text{diff}} = \mathbb{E}_{t, g_0, \epsilon}[\|\epsilon - \epsilon_\theta(g_t, t, s)\|^2]$, and the GP regularization term $L_{\text{GP}} = -\log p_{\text{GP}}(g_0 | \mathcal{D})$, where $\mathcal{D}$ is the historical trajectory buffer. The total loss is $L = L_{\text{diff}} + \lambda L_{\text{GP}}$. The low-level policy is trained using standard off-policy RL algorithms (TD3/SAC), optimizing the weighted sum of intrinsic and environmental rewards. The GP hyperparameters are updated online via marginal likelihood maximization.

## Key Experimental Results

### Main Results

| Environment | Ours | HIRO | HAC | HRAC | RIG | Metric |
|:--|:--|:--|:--|:--|:--|:--|
| Ant Maze | **0.92** | 0.68 | 0.71 | 0.78 | 0.62 | Success Rate |
| Ant Push | **0.85** | 0.52 | 0.58 | 0.64 | 0.48 | Success Rate |
| Ant Fall | **0.73** | 0.31 | 0.38 | 0.45 | 0.29 | Success Rate |
| Reacher | **-3.2** | -5.8 | -4.9 | -4.1 | -6.2 | Negative Distance |
| Pusher | **-12.5** | -22.3 | -18.7 | -15.6 | -24.1 | Cumulative Reward |

*All results are the average of 5 independent runs. Ours achieves the best performance across all continuous control benchmarks.*

### Ablation Study

| Variant | Ant Maze Success Rate | Ant Push Success Rate | Sample Efficiency Gain | Description |
|:--|:--|:--|:--|:--|
| Full (Diffusion + GP) | **0.92** | **0.85** | **1.0x (Baseline)** | Full method |
| Diffusion only (No GP) | 0.81 | 0.72 | 0.7x | Over-exploration, feasibility decreases |
| GP only (No Diffusion) | 0.74 | 0.65 | 0.6x | Insufficient distribution modeling |
| Deterministic Subgoals | 0.68 | 0.55 | 0.5x | Fails to capture multimodal distribution |
| No Intrinsic Reward | 0.45 | 0.32 | 0.3x | Deficient low-level policy training |

### Key Findings

- The diffusion model provides the largest sample efficiency gains, validating the importance of modeling complex subgoal distributions.
- The primary contribution of the GP prior lies in learning stability, reducing the variance across different random seeds.
- The hybrid subgoal selection (GP mean + diffusion samples) significantly outperforms any single-source selection strategy.
- In the most challenging Ant Fall environment (which requires crossing a gap), the advantages of Ours are the most pronounced, achieving over a 130% improvement compared to HIRO.

## Highlights & Insights

- Innovatively unifies generative models (diffusion) and Bayesian methods (GP) in HRL, leveraging the strengths of both.
- Uncertainty-aware subgoal selection achieves a natural exploration-exploitation balance without manual annealing schedules.
- The GP prior acts like a curriculum learning mechanism in diffusion training: constraining the generation space in early stages, and degrading into a weak regularization in later stages.
- The methodological contribution can be transferred to other generative decision-making problems requiring a balance between 'diversity' and 'feasibility'.

## Limitations & Future Work

- The $O(n^3)$ computational complexity of GP can become a bottleneck as historical trajectories grow, necessitating sparse GP approximations.
- Multi-step sampling from the diffusion model may introduce latency when the subgoal generation frequency is high, requiring integration with acceleration methods like consistency models.
- The experiments are primarily validated in MuJoCo continuous control environments, and the performance in discrete action spaces or more complex robotic tasks remains unexplored.
- The impact of GP kernel selection (such as RBF vs. Matérn) on performance is not fully discussed.

## Related Work & Insights

- **vs HIRO**: HIRO uses a deterministic high-level policy, resulting in limited subgoal generation capability. Ours models multimodal distributions via the diffusion model, showing a clear advantage in complex environments.
- **vs HAC**: HAC introduces hindsight subgoal correction but still utilizes a simple Gaussian policy. The GP uncertainty quantification in Ours provides a more principled exploration mechanism.
- **vs Diffusion Policy (Chi et al.)**: Diffusion Policy directly generates action sequences without a hierarchical structure. Ours uniquely applies the diffusion model to the subgoal generation layer, maintaining the temporal abstraction advantages of HRL.

## Rating

- Novelty: ⭐⭐⭐⭐ First combining diffusion models and GP in HRL; uncertainty-guided subgoal selection is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison and detailed ablation studies in multiple continuous control environments.
- Writing Quality: ⭐⭐⭐⭐ Clean problem motivation with sufficient explanation of the role of each component in the method.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for subgoal generation in HRL, and the combination of diffusion and Bayesian methods has broad reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](../../ICLR2026/image_generation/hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](../../ICLR2026/image_generation/rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)
- [\[CVPR 2026\] HiCoGen: Hierarchical Compositional Text-to-Image Generation in Diffusion Models via Reinforcement Learning](../../CVPR2026/image_generation/hicogen_hierarchical_compositional_text-to-image_generation_in_diffusion_models_.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](../../CVPR2025/image_generation/uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[ICLR 2026\] SoftCFG: Uncertainty-guided Stable Guidance for Visual Autoregressive Model](../../ICLR2026/image_generation/softcfg_uncertainty-guided_stable_guidance_for_visual_autoregressive_model.md)

</div>

<!-- RELATED:END -->
