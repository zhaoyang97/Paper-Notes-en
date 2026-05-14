---
title: >-
  [Paper Note] Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization
description: >-
  [NeurIPS 2025][Reinforcement Learning][multimodal policy] This paper proposes the Diversity-regularized Actor Critic (DrAC) algorithm, which unifies intractable multimodal policies (amortized actor and diffusion actor) u…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "multimodal policy"
  - "reparameterization"
  - "diversity regularization"
  - "diffusion policy"
  - "actor-critic"
date: 2026-05-08
content_hash: 94fa6138d4b51274
---

# Learning Intractable Multimodal Policies with Reparameterization and Diversity Regularization

**Conference**: NeurIPS 2025
**arXiv**: [2511.01374](https://arxiv.org/abs/2511.01374)
**Code**: [GitHub](https://github.com/PneuC/DrAC)
**Area**: Reinforcement Learning
**Keywords**: multimodal policy, reparameterization, diversity regularization, diffusion policy, actor-critic

## TL;DR

This paper proposes the Diversity-regularized Actor Critic (DrAC) algorithm, which unifies intractable multimodal policies (amortized actor and diffusion actor) under a stochastic-mapping formulation, enables direct policy gradient optimization via reparameterization without requiring probability density evaluation, and introduces a distance-based diversity regularization as an alternative to entropy regularization. DrAC demonstrates significant advantages on diversity-critical tasks such as multi-goal navigation and generative RL.

## Background & Motivation

Real-world decision-making is often multimodal—given the same state, there may exist multiple equally good yet fundamentally different action choices, such as multiple viable paths in maze navigation, diverse strategies in competitive games, or diversity requirements in generative tasks. Nevertheless, mainstream deep RL algorithms (e.g., SAC, TD3, DDPG) almost exclusively adopt deterministic or unimodal Gaussian policies, which are incapable of representing complex multimodal decision distributions.

**The core challenge of learning multimodal policies lies in intractability:**

**Amortized actor** (as used in SQL): concatenates state and latent variable and feeds them into a network to directly output actions, but the policy probability $\pi_\theta(a|s)$ has no closed form.

**Diffusion actor** (as in DACER): generates actions iteratively via a diffusion process, offering strong expressiveness but similarly intractable densities.

Intractability implies:
- The entropy $\mathcal{H}(\pi(\cdot|s))$ cannot be computed directly, making the maximum-entropy RL framework inapplicable.
- Existing approaches either sacrifice expressiveness by using tractable but weaker multimodal models, or rely on variational inference techniques such as SVGD with suboptimal performance or high computational cost.
- DACER controls diversity through noise scaling rather than directly optimizing policy parameters via gradients.

**Core Idea**: Intractable probability densities do not preclude trainability. As long as a policy can be expressed as a "deterministic mapping + fixed latent distribution," reparameterization can bypass density computation and enable direct policy gradient optimization.

## Method

### Overall Architecture

DrAC builds upon the actor-critic framework and introduces three core innovations:
- A unified stochastic-mapping actor formulation for multimodal policies
- Reparameterization-based policy gradients that require no probability density
- Distance-based diversity regularization as a replacement for entropy regularization

### Key Designs

1. **Stochastic-Mapping Actor Unified Framework**: The policy is defined as $\pi_\theta = \{f_\theta, p_z\}$—a composition of a parameterized mapping $f_\theta: \mathcal{S} \times \mathcal{Z} \to \mathcal{A}$ and a fixed latent distribution $p_z$. Actions are sampled as $a \leftarrow f_\theta(s, z),\ z \sim p_z$.

    - **Amortized actor**: $f_\theta^{Amort}(s, z) \equiv g_\theta(s \oplus z)$, directly concatenating state and latent variable as network input.
    - **Diffusion actor**: $f_\theta^{Diffus}(s, z) \equiv x_0$, where $x_0$ is obtained by iteratively denoising pure noise $z_T$ over $T$ reverse diffusion steps.

   This unified perspective reveals the shared nature of both actor types and lays the foundation for unified optimization.

2. **Policy Gradient via Reparameterization (PGRT)**: The paper proves that for any stochastic-mapping actor, the policy gradient can be estimated via the reparameterization trick:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{s \sim d^\pi, z \sim p_z}[\nabla_a Q(s, f_\theta(s, z)) \nabla_\theta f_\theta(s, z)]$$

This directly backpropagates the action gradient of the Q-function through $f_\theta$, entirely bypassing the computation of $\pi_\theta(a|s)$. This approach is both more efficient and more effective than SVGD.

3. **Distance-Based Diversity Regularization**: Traditional entropy regularization requires probability densities and is thus inapplicable to intractable actors. This paper proposes using the logarithmic geometric mean of pairwise distances as a diversity measure:

$$D^\pi(s) = \mathbb{E}_{x,y \sim \pi(\cdot|s)}[\log \delta(x, y)]$$

where $\delta$ denotes the L2 distance. The key motivation for using the geometric mean (rather than the arithmetic mean) is that the arithmetic mean can overestimate diversity—when data forms several small but widely separated clusters, the average pairwise distance is large yet actual diversity is low. The geometric mean is sensitive to small values and avoids such overestimation. Operating in log scale also facilitates balancing with the reward signal.

### Loss & Training

- **Critic loss** (dual critics + target network, incorporating diversity regularization):

$$\mathcal{L}_\phi = \mathbb{E}_{s,a,r,s' \sim \mathcal{D}}[\text{MSE}(Q(s,a;\phi_i), r + \gamma(\tilde{V}(s';\hat{\phi}) + \alpha \tilde{D}_\theta(s')))]$$

- **Actor loss** (PGRT + diversity gradient):

$$\mathcal{L}_\theta = -\mathbb{E}_{s \sim \mathcal{D}, z \sim p_z}[Q(s, f_\theta(s,z); \phi) + \alpha \tilde{D}_\theta(s)]$$

- **Automatic coefficient tuning**: Following SAC's automatic temperature adjustment, a target diversity $\hat{D}$ is specified and the coefficient $\alpha$ is optimized automatically:

$$\mathcal{L}_\alpha = \mathbb{E}_{s \sim \mathcal{D}}[\alpha(\tilde{D}_\theta(s) - \hat{D})]$$

## Key Experimental Results

### Multi-Goal Maze Navigation

| Algorithm | Simple Maze Goals Reached | Medium Maze Goals Reached | Hard Maze Goals Reached | Obstacle Robustness |
|-----------|--------------------------|--------------------------|------------------------|---------------------|
| SAC (unimodal) | ~2 | ~2 | ~1 | Low |
| SQL (amortized) | ~4 | ~3 | ~2 | Medium |
| DACER (diffusion) | ~2 | ~2 | ~1 | Low |
| **DrAmort (ours)** | **~4** | **~4** | **~3** | **Highest** |
| DrDiffus (ours) | ~3 | ~3 | ~2 | Medium |

### Generative RL (Game Level Generation)

| Algorithm | MarioPuzzle Return | MarioPuzzle Diversity | MultiFacet Return | MultiFacet Diversity |
|-----------|-------------------|----------------------|------------------|---------------------|
| SAC | Moderate | Low | Moderate | Low |
| SQL | Low | Moderate | Low | **High** |
| DACER | Moderate | Low | Moderate | Moderate |
| **DrAmort** | **Highest** | **Highest** | **Highest** | High |

### MuJoCo Standard Benchmark (6 locomotion tasks)

| Algorithm | Best Tasks | Overall Performance |
|-----------|-----------|---------------------|
| SAC | 2/6 | Baseline |
| SQL | 0/6 | Worst |
| DACER | 1/6 | On par with SAC |
| **DrAmort** | **3/6** | **Best or on par** |
| DrDiffus | 0/6 | On par with DACER |

### Key Findings

1. **Amortized actor prevails**: Across all experiments, the amortized-actor-based DrAmort demonstrates the strongest multimodal expressiveness and best overall performance. The diffusion actor underperforms expectations in multimodal representation and is substantially slower in both inference and training.
2. **Diversity yields robustness**: In out-of-distribution tests (goal removal, obstacle insertion), the high-diversity policy (DrAmort) exhibits the best few-shot robustness, confirming the practical value of multimodal policies.
3. **PGRT outperforms SVGD**: SQL uses SVGD as the policy gradient estimator, while DrAmort uses reparameterization; the latter consistently outperforms for amortized actors, demonstrating that reparameterization is a superior gradient estimation method.
4. **Distance regularization outperforms noise scaling**: DACER controls diversity through additional noise scaling and fails to learn multimodal behavior in multi-goal mazes, whereas DrDiffus with distance regularization succeeds.

## Highlights & Insights

- The paper unifies the seemingly disparate amortized and diffusion actors under a single theoretical framework, revealing that both are fundamentally "deterministic mapping + stochastic source" structures, providing a principled basis for unified optimization.
- Replacing the arithmetic mean with the geometric mean for diversity measurement is a seemingly minor choice that proves consequential in experiments.
- The paper rehabilitates the amortized actor—under a well-designed training algorithm, this simple model exhibits surprisingly strong capability in multimodal RL.

## Limitations & Future Work

- Diffusion actors may require deeper networks, more diffusion steps, and more careful hyperparameter tuning to fully realize their potential.
- Distance-based diversity measures with alternative distance functions and aggregation strategies remain to be explored.
- Validation is currently limited to continuous action spaces; adaptation to discrete or hybrid action spaces warrants further investigation.
- Temperature scheduling strategies can be further refined.

## Related Work & Insights

- Comparisons with SVGD-based methods such as SQL and S2AC validate the superiority of reparameterization for training intractable actors.
- The comparison with DACER reveals a fundamental distinction between "controlling diversity via noise scaling" and "optimizing diversity via gradients."
- The work provides a practical new tool for the quality-diversity RL community.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified framework and diversity regularization are innovative, though each component has precedents
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multi-goal navigation, generative RL, and MuJoCo benchmarks
- Writing Quality: ⭐⭐⭐⭐ Well-organized with intuitive illustrations
- Value: ⭐⭐⭐⭐ Provides a practical and efficient algorithmic framework for multimodal RL

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/reinforcement_learning/offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)

</div>

<!-- RELATED:END -->
