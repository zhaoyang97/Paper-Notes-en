---
title: >-
  [Paper Note] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration
description: >-
  [ICML2025][Reinforcement Learning][Unsupervised skill pre-training] This paper proposes SUPE, a method that uses unlabeled offline trajectory data "twice"—both for VAE skill pre-training and as high-level off-policy data via UCB pseudo-labels to accelerate online exploration, comprehensively outperforming prior methods on 42 sparse-reward tasks.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Unsupervised skill pre-training"
  - "hierarchical reinforcement learning"
  - "offline data utilization"
  - "exploration strategy"
  - "pseudo-labeling"
date: 2026-05-08
content_hash: af1b5342aacdd4fb
---

# Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration

**Conference**: ICML2025  
**arXiv**: [2410.18076](https://arxiv.org/abs/2410.18076)  
**Authors**: Max Wilcoxson, Qiyang Li, Kevin Frans, Sergey Levine (UC Berkeley)
**Code**: [rail-berkeley/supe](https://github.com/rail-berkeley/supe)  
**Area**: Reinforcement Learning  
**Keywords**: Unsupervised skill pre-training, hierarchical reinforcement learning, offline data utilization, exploration strategy, pseudo-labeling

## TL;DR

This paper proposes SUPE, a method that uses unlabeled offline trajectory data "twice"—both for VAE skill pre-training and as high-level off-policy data via UCB pseudo-labels to accelerate online exploration, comprehensively outperforming prior methods on 42 sparse-reward tasks.

## Background & Motivation

Unsupervised pre-training has achieved great success in NLP and CV, but transferring it to reinforcement learning (RL) presents unique challenges: online fine-tuning in RL is not about mimicking data, but finding solutions and iteratively self-improving through **exploration**. Consequently, the core issue of RL pre-training is not only learning good representations but also acquiring **efficient exploration strategies**.

Unlabeled trajectory data (collected by unknown policies or task-agnostic behaviors) is the easiest to obtain but suffers from the **entanglement problem**: general environmental knowledge is mixed with task-specific behaviors. For example, in locomotion data, the agent needs to learn how to move in the environment, but does not necessarily need to visit the specific locations present in the training data.

Prior methods typically utilize offline data in only one phase:
- **Skill pre-training methods** (such as SPiRL, OPAL): Discard the offline data after extracting skills from it, and learn high-level policies from scratch online.
- **Offline-to-online RL** (such as ExPLORe): Treat offline data directly as off-policy data without performing skill pre-training.

The key insight of this paper: Unlabeled trajectories provide **dual value**—(1) learning diverse low-level skills, and (2) serving as off-policy transitions to compose these skills. Combining both yields a compounding effect.

## Method

### Overall Architecture

SUPE consists of two phases:

**Phase 1: Offline Skill Pre-training**
- Extract trajectory segments of length $H$ from unlabeled offline data $\mathcal{D}$.
- Learn a low-level skill space using a trajectory-segment VAE.

**Phase 2: Online RL + Pseudo-labeling**
- Use the pre-trained trajectory encoder to assign high-level action labels $\hat{z}$ to the offline data.
- Use UCB reward estimation to assign optimistic reward labels $\hat{r}$ to the offline data.
- Combine the pseudo-labeled offline data with the online replay buffer to train the high-level policy $\pi_\psi(z|s)$.

### Trajectory VAE Pre-training

A trajectory segment $\tau_{[H]} = \{s_0, a_0, s_1, \ldots, s_{H-1}, a_{H-1}\}$ is fed into the encoder $f_\theta(z|\tau)$ to output a distribution over the skill latent variable $z$. The skill policy $\pi_\theta(a|s,z)$ reconstructs the action sequence. Simultaneously, a state-conditioned prior $p_\theta(z|s_0)$ is learned.

**VAE Loss Function (Core Formula):**

$$\mathcal{L}_\theta(\tau) = \beta D_{\mathrm{KL}}(f_\theta(z|\tau) \| p_\theta(z|s_0)) - \mathbb{E}_{z \sim f_\theta(z|\tau)} \left[\sum_{h=0}^{H-1} \log \pi_\theta(a_h|s_h, z)\right]$$

The first term is the KL regularization (aligning the encoder posterior with the state prior), and the second term is the action reconstruction log-likelihood.

### Online Pseudo-Labeling Mechanism

For each offline trajectory segment $\tau_{[H]}$, high-level transition tuples are constructed:

$$(s_0, \quad \hat{z} \sim f_\theta(z|\tau), \quad \hat{r} = r_{\mathrm{UCB}}(s_0, \hat{z}), \quad s_H)$$

- **High-level action label $\hat{z}$**: Inferred via the VAE encoder, computed once prior to the start of online learning.
- **Optimistic reward label $\hat{r}$**: Estimated using UCB, recomputed before each update.

**UCB Reward Estimation:**

$$r_{\mathrm{UCB}}(s, z) = r_\zeta(s, z) + \alpha \|g_\phi(s, z) - \bar{g}(s, z)\|_2^2$$

Where $r_\zeta$ is a reward prediction model learned from online data, the second term is the RND (Random Network Distillation) exploration bonus, and $\alpha$ controls the degree of optimism.

### Implementation Details

- **High-level Policy**: SAC (Soft Actor-Critic) with an ensemble of 10 critics, using the RLPD framework.
- **Policy Parameterization**: Employs tanh transformation + entropy regularization (replacing SPiRL's KL constraint, resulting in better and more stable performance).
- **Sampling Strategy**: 128 samples each from online and offline sources are drawn to balance training.
- **Skill Space**: Latent dimension of 10, trajectory segment length $H=4$.
- **UTD Ratio**: 20 for state-based domains, 40 for visual domains.

## Key Experimental Results

### Experimental Setup

Evaluated across **8 domains and 42 sparse-reward tasks**:

| Domain Category | Environment | Number of Tasks | Features |
|---------|------|-------|------|
| State-based Navigation | antmaze (medium/large/ultra × 4 goals) | 12 | D4RL standard benchmark |
| State-based Navigation | humanoidmaze (medium/large/giant × navigate/stitch) | 6 | High-dimensional action space (21 vs. 8) |
| State-based Navigation | antsoccer (arena/medium) | 2 | Requires controlling the ball to the target position |
| State-based Manipulation | kitchen (mixed/partial/complete) | 3 | Multi-step compositional manipulation |
| State-based Manipulation | cube-single (5 tasks) | 5 | Pick-and-place arrangement |
| State-based Manipulation | cube-double (5 tasks) | 5 | Dual-cube manipulation |
| State-based Manipulation | scene (5 tasks) | 5 | Multi-object interaction (drawer/window/lock/cube) |
| Visual Navigation | visual-antmaze (4 goals) | 4 | 64×64 pixel observations |

### Baselines

- **ExPLORe**: Does not pre-train skills; directly labels offline data with RND+UCB for online RL.
- **DBC+JSRL**: Employs a diffusion policy to mimic offline data $\to$ JSRL-style initialization for online exploration.
- **Trajectory Skills**: VAE skill pre-training $\to$ discard offline data $\to$ purely online learning of high-level policy.
- **HILP Skills**: HILP skill pre-training $\to$ discard offline data $\to$ purely online learning of high-level policy.
- **SUPE (HILP)**: HILP skills + offline pseudo-labeling (the HILP variant of Ours).

### Main Results

**Aggregate Performance (Figure 3):** SUPE performs best in 6 out of 8 domains, while SUPE(HILP) performs better in antsoccer and scene.

**Key Findings:**

1. **Double utilization of offline data is crucial**: SUPE > Trajectory Skills, and SUPE(HILP) > HILP Skills, showing that utilizing offline data during the online phase significantly accelerates learning.
2. **Skill pre-training is indispensable**: SUPE > ExPLORe, especially in highly challenging environments where ExPLORe completely fails.
3. **Absolute advantage in HumanoidMaze**: SUPE is the only method that achieves non-zero rewards in large and giant mazes.
4. **Exploration efficiency (antmaze first-goal-time)**: SUPE reaches the goal the fastest across all layouts, verifying its superior exploration policy.

**Hyperparameter Sensitivity:**
- RND coefficient $\alpha$: Performance is stable within the range $\{2, 8, 16\}$, but drops significantly when $\alpha=0$.
- Skill length $H=4$ is globally optimal; $H=2$ or $H=8$ yields worse performance in most tasks.

### Data Robustness (Appendix K)

Tested on antmaze-large with offline data of varying quality:
- **Navigate/Stitch data**: SUPE maintains its advantage, particularly on the more difficult Stitch subset.
- **Purely random Explore data**: All skill-based methods fail due to poor skill quality.
- **Insufficient data (5% data / removing data near the goal)**: The learning of SUPE slows down, but its asymptotic performance remains unaffected, still outperforming the baselines.

## Highlights & Insights

1. **Simple yet profound core idea**: Utilizing offline data "twice"—this idea is intuitive but previously overlooked. The authors effectively realize this via a carefully designed pseudo-labeling mechanism.
2. **UCB reward pseudo-labels**: Elevates ExPLORe's state-action level optimistic estimation to a high-level state-skill level, matching the temporal scale of skill abstraction.
3. **Removing KL constraint**: Replaces SPiRL's KL penalty with tanh + entropy regularization, which is simpler and yields better performance—a significant engineering simplification.
4. **Large-scale and systematic experiments**: 42 tasks across 8 domains, multiple baselines, sensitivity analysis, and data quality analysis—extremely thorough.

## Limitations & Future Work

1. **Skill freezing issue**: The pre-trained skills remain fixed during the online phase, which can be restrictive if skills are poorly learned or need adaptation to online distribution shifts. Allowing online fine-tuning of low-level skills is a natural future direction.
2. **Reliance on RND**: UCB estimation relies on RND, which might be unreliable in high-dimensional observation spaces. Although it works without ICVF in visual-antmaze, more complex visual domains may require more robust exploration signals.
3. **Fixed skill length**: $H=4$ is used globally, but different environments might require different granularities of temporal abstraction. Variable-length skills (options framework) could offer more flexibility.
4. **Unusable purely random offline data**: When offline data quality is extremely poor (e.g., the Explore dataset), all skill-based methods fail, indicating a lower bound requirement on data quality.
5. **Computational cost**: A total of approximately 16,600 GPU hours (A5000/V100), posing a high barrier for reproducibility at this experimental scale.

## Related Work & Insights

- **SPiRL / OPAL** (Pertsch et al. 2021; Ajay et al. 2021): Direct predecessors of SUPE. The VAE skill pre-training framework is conceptually identical, but they only utilize data in the offline phase.
- **ExPLORe** (Li et al. 2024): Proposes utilizing offline data online via UCB pseudo-labels, but does not pre-train skills. SUPE scales this to the high-level skill space.
- **HILP** (Park et al. 2024b): An alternative skill discovery method that learns directional skills via Hilbert representations. SUPE(HILP) demonstrates that the framework is agnostic to the type of skill representation.
- **RLPD** (Ball et al. 2023): The underlying algorithm for SUPE's online RL, providing an efficient off-policy framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The core idea (double utilization of offline data) is simple and effective. Scaling pseudo-labels from the action space to the skill space is a reasonable and sound innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 42 tasks across 8 domains, with systematic ablation and sensitivity analyses, and an extremely comprehensive appendix.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete method description, and well-designed figures/tables.
- **Value**: ⭐⭐⭐⭐ — Provides a practical and effective solution for utilizing unlabeled prior data to accelerate online RL exploration, offering high reference value in the HRL and exploration domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[CVPR 2025\] SkillMimic: Learning Basketball Interaction Skills from Demonstrations](../../CVPR2025/reinforcement_learning/skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)
- [\[ICML 2025\] KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies](kea_keeping_exploration_alive_by_proactively_coordinating_exploration_strategies.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)

</div>

<!-- RELATED:END -->
