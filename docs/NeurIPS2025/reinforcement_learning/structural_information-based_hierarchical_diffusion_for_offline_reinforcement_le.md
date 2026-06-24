---
title: >-
  [Paper Note] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline Reinforcement Learning] This paper proposes SIHD, a framework that leverages structural information (structural entropy) extracted from historical trajectories to adaptively construct multi-scale diffusion hierarchies, replaces local reward prediction with structural information gain as the conditional guidance signal, and introduces structural entropy regularization to encourage exploration of sparse states in offline data. SIHD…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Offline Reinforcement Learning"
  - "Diffusion Models"
  - "Hierarchical Planning"
  - "Structural Entropy"
  - "Long-Horizon Decision Making"
date: 2026-05-08
content_hash: 42b82429e4c4ca91
---

# Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.21942](https://arxiv.org/abs/2509.21942)  
**Code**: [GitHub](https://github.com/SELGroup/SIHD.git)  
**Area**: Reinforcement Learning
**Keywords**: Offline Reinforcement Learning, Diffusion Models, Hierarchical Planning, Structural Entropy, Long-Horizon Decision Making

## TL;DR

This paper proposes SIHD, a framework that leverages structural information (structural entropy) extracted from historical trajectories to adaptively construct multi-scale diffusion hierarchies, replaces local reward prediction with structural information gain as the conditional guidance signal, and introduces structural entropy regularization to encourage exploration of sparse states in offline data. SIHD achieves up to 12.6% improvement in decision-making performance on the D4RL benchmark.

## Background & Motivation

### State of the Field

Offline RL trains policies from pre-collected data, eliminating the need for online interaction. Diffusion models have been incorporated into offline RL due to their powerful distribution modeling capabilities: (1) as policy generators (Diffusion Policy) for producing multimodal behaviors directly, and (2) as trajectory generators (Diffuser) that condition on rewards to generate high-return sequences. The latter formulates offline decision-making as a conditional generation problem.

### Limitations of Prior Work

In long-horizon tasks, diffusion models face two key challenges:

**Variance accumulation**: Value estimation variance grows exponentially with trajectory length.

**Computational cost**: Iterative denoising steps scale linearly with sequence length.

To address these, hierarchical diffusion methods (HDMI, Hierarchical Diffuser) decompose decision-making into high-level goal planning and low-level action generation. However, existing methods suffer from **two rigid limitations**:
- **Fixed two-level structure**: Only a subgoal layer and an action layer exist, unable to adapt to the temporal structural complexity of different tasks.
- **Uniform predefined temporal scale**: Trajectories are segmented at fixed intervals, ignoring heterogeneous state transition patterns across different regions.

### Starting Point

The core insight is that the **state topological structure** embedded in historical trajectories can guide the construction of diffusion hierarchies. By building a similarity graph over the state space and optimizing structural entropy, the method automatically discovers multi-level state community partitions, enabling adaptive trajectory segmentation and multi-scale hierarchy construction. Furthermore, structural information gain can replace unreliable local reward prediction as the guidance signal for conditional diffusion.

## Method

### Overall Architecture

SIHD consists of three modules:
1. **Hierarchy Construction Module**: Extracts an encoding tree from the topological structure of offline states to adaptively build a multi-scale diffusion hierarchy.
2. **Conditional Diffusion Module**: Applies a shared diffusion model at each level, conditioned on structural information gain.
3. **Regularized Exploration Module**: Encourages exploration of sparse states via structural entropy regularization while constraining exploration within low-level communities to avoid out-of-distribution errors.

### Key Designs

1. **Multi-Scale Diffusion Hierarchy Construction**:

    - All states $s \in \mathcal{S}$ are extracted from the offline dataset, and a k-nearest-neighbor state graph $\mathcal{G}_s$ is constructed based on feature similarity (cosine similarity).
    - The optimal $k$ is selected by maximizing the one-dimensional structural entropy $\mathcal{H}^1(\mathcal{G}_s)$.
    - The HCSE optimization algorithm is applied to solve for the optimal encoding tree $\mathcal{T}_s^*$ of height $\mathcal{K}$, minimizing the $\mathcal{K}$-dimensional structural entropy:

    $$\mathcal{H}^{\mathcal{K}}(\mathcal{G}) = \min_{h_\mathcal{T} \leq \mathcal{K}} \mathcal{H}^{\mathcal{T}}(\mathcal{G}), \quad \mathcal{H}^{\mathcal{T}}(\mathcal{G}) = -\sum_{\alpha \in \mathcal{T}, \alpha \neq \lambda}\left[\frac{g_\alpha}{\operatorname{vol}(\lambda)} \cdot \log\frac{\operatorname{vol}(\alpha)}{\operatorname{vol}(\alpha^-)}\right]$$

    - The encoding tree naturally defines multi-level community partitions: each node $\alpha$ corresponds to a state community $\mathcal{V}_\alpha$.
    - Trajectories are adaptively segmented according to community partitions: consecutive states within the same community form a segment, and the terminal state of each segment serves as a subgoal.
    - Nodes at different levels of the encoding tree correspond to trajectory segments at different temporal scales.

2. **Conditional Diffusion Model (Guided by Structural Information Gain)**:

   The top-level diffuser conditions on cumulative reward: $y(\tau_g^{\mathcal{K},1}) = \exp(\sum_{t=0}^T \mathcal{R}(s_t, a_t))$

   **Key innovation**: Intermediate and lower-level diffusers do not use reward but instead use **structural information gain** as the conditioning signal:

    $$y(\tau_g^{h,i}) = \mathcal{H}^{\mathcal{T}_s^*}(\mathcal{G}_s; \alpha) = -\frac{g_\alpha}{\operatorname{vol}(\lambda)} \cdot \log\frac{\operatorname{vol}(\alpha)}{\operatorname{vol}(\alpha^-)}$$

   This quantifies the additional information required to infer that a transition belongs to a lower-level segment, given that it is known to belong to a higher-level segment. Classifier-free guidance is used to integrate the conditioning signal: $\hat{\epsilon} = \epsilon_{\theta_h}(\tau_{g,k}^{h,i}, (1-\omega)y(\tau_{g,k}^{h,i}) + \omega\emptyset, k)$

   Theorem 4.1 proves that conditional generation can be decomposed into a hierarchical diffusion process:

    $$p(\tau_0|y(\tau_0)) \propto p(\tau_g^{\mathcal{K},1})y(\tau_g^{\mathcal{K},1}) \cdot \prod_{h=1}^{\mathcal{K}-1}\prod_{i=1}^{l_g^h} p(\tau_g^{h,i})y(\tau_g^{h,i})$$

3. **Structural Entropy Regularization**:

   A fully weighted graph $\mathcal{G}_s'$ is constructed from state transition probabilities estimated by the lowest-level diffusion model, and its structural entropy under the encoding tree is computed. Theorem 4.2 establishes a variational lower bound relating structural entropy to Shannon entropy:

    $$\mathcal{H}(S) - \sum_{h=1}^{\mathcal{K}-1}[\eta_h \cdot \mathcal{H}(\mathcal{U}_h)] \leq \mathcal{H}^{\mathcal{T}_s^*}(\mathcal{G}_s') \leq \mathcal{H}(S)$$

   The training objective includes a regularization term that simultaneously **maximizes** $\mathcal{H}(S)$ (encouraging coverage of sparse states) and **minimizes** $\mathcal{H}(\mathcal{U}_h)$ (preserving hierarchical structure and constraining exploration scope):

    $$\mathcal{L}(\theta_h) = \mathbb{E}\sum_{i=1}^{l_g^h}\left[\|\hat{\epsilon} - \epsilon\|^2 - \eta\mathbb{I}_{h=1}\left[\mathcal{H}(S) - \sum_{j=1}^{\mathcal{K}-1}[\eta_j \cdot \mathcal{H}(\mathcal{U}_j)]\right]\right]$$

### Loss & Training

Each level's diffuser shares parameters and employs the standard noise-prediction MSE loss augmented with structural entropy regularization (active only at the lowest level $h=1$). The encoding tree is precomputed and stored in a dictionary-structured community partition to avoid redundant computation during training.

## Key Experimental Results

### Gym-MuJoCo Standard Tasks

| Environment + Dataset | Diffuser | HDMI | HD | **SIHD** | Gain |
|----------------------|----------|------|-----|---------|------|
| HalfCheetah-Expert | 88.9 | 92.1 | 92.5 | **94.4** | 2.1% |
| Hopper-Medium | 74.3 | 76.4 | 99.3 | **103.1** | 3.8% |
| Walker2D-Replay | 70.6 | 80.7 | 84.1 | **89.7** | 6.7% |
| **Average Gain (vs HD)** | — | — | — | **3.2%** | — |

### Long-Horizon Navigation Tasks (Maze2D + AntMaze)

| Environment | Diffuser | HDMI | HD | **SIHD** | Gain |
|-------------|----------|------|-----|---------|------|
| Maze2D-U | 113.9 | 120.1 | 128.4 | **144.6** | **12.6%** |
| Maze2D-Medium | 121.5 | 121.8 | 135.6 | **148.5** | 9.5% |
| Maze2D-Large | 123.0 | 128.6 | 155.8 | **161.7** | 3.8% |
| AntMaze-U | 76.0 | — | 94.0 | **96.5** | 2.7% |
| AntMaze-Medium | 31.9 | — | 88.7 | **92.2** | 3.9% |
| AntMaze-Large | — | — | 83.6 | **89.4** | 6.9% |

### Ablation Study

| Variant | Hopper-Med | Maze2D-Med | AntMaze-Med | Notes |
|---------|-----------|------------|-------------|-------|
| SIHD (Full) | **103.1** | **148.5** | **92.2** | All modules |
| SIHD-DH (w/o multi-scale hierarchy) | ~93 | ~125 | ~80 | **Largest drop**, confirming hierarchy is central |
| SIHD-CG (w/o conditional guidance) | ~98 | ~140 | ~87 | Structural information guidance is beneficial |
| SIHD-ER (w/o regularization) | ~100 | ~142 | ~85 | Regularization is more critical under sparse reward |
| SIHD-FT (fixed-interval segmentation) | — | 146.8 | — | Structural entropy outperforms fixed segmentation |

### Computational Efficiency

| Method | Maze2D Training Time | Maze2D Planning Time |
|--------|---------------------|---------------------|
| Diffuser | 48.3s | 4.7s |
| HD | 5.8s | 1.9s |
| SIHD | 6.0s | 1.6s |

### Key Findings

- SIHD's advantage is more pronounced on medium- and low-quality datasets (Expert 1.6% vs. Medium 3.8% vs. Replay 3.9%), indicating that structural entropy regularization effectively reduces dependence on high-quality data.
- Gains are larger on long-horizon tasks (Maze2D average 8.3%), as a deeper diffusion hierarchy ($\mathcal{K}=4$) better supports long-term decision-making.
- Removing the multi-scale hierarchy (SIHD-DH) yields the largest performance drop, especially on long-horizon tasks, confirming that adaptive hierarchy construction is the core contribution.
- Even when the total parameter count is held equal to HD, increasing the number of hierarchy levels consistently improves performance—performance gains stem from structure, not capacity.
- Computational efficiency is comparable to HD and over 80% faster than Diffuser.

## Highlights & Insights

- **Data-driven hierarchy construction**: Reasonable hierarchical decompositions are discovered automatically from state topological structure rather than relying on manually designed temporal scales.
- **Structural information gain as a surrogate for reward prediction**: This approach avoids the unreliability of predicting rewards over local sub-trajectories, providing more stable guidance via topological information content.
- **Dual constraints in regularization**: The framework elegantly balances exploration and conservatism by simultaneously encouraging coverage of sparse states (maximizing Shannon entropy) and confining exploration within low-level communities (minimizing community entropy).
- **Theoretical completeness**: Theorem 4.1 guarantees the factorizability of hierarchical decomposition, and Theorem 4.2 establishes the variational bound for the regularization term.

## Limitations & Future Work

- Structural entropy optimization incurs non-trivial computational overhead on large-scale offline datasets (partially mitigated by precomputation).
- The subgoal constraint adopts a simple terminal-state substitution strategy, which may lack fine-grained control.
- The number of encoding tree levels $\mathcal{K}$ requires task-specific tuning (sensitivity analysis indicates $\mathcal{K}=3$ or $4$ is generally optimal).
- The regularization coefficient $\eta$ is sensitive to data quality (smaller $\eta$ for expert data, larger $\eta$ for medium-quality data).
- Evaluation is limited to the D4RL benchmark and has not been extended to more complex real-world tasks.

## Related Work & Insights

- Compared to HDMI/HD: those methods employ a fixed two-level structure with a single temporal scale, whereas SIHD adaptively constructs multiple levels at multiple scales.
- Structural entropy (Li & Pan 2016) was originally developed for network science and graph learning; this work is the first to apply it to hierarchical decision-making in RL.
- Evans & Şimşek (2023) demonstrate the advantage of multi-level policy hierarchies in compositional long-horizon tasks, providing motivation for SIHD.
- Broader implication: the structural information principle may find applications in other domains requiring hierarchical decomposition, such as hierarchical modeling in natural language processing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of structural information theory to hierarchical diffusion-based RL, with innovations across all three design components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive D4RL benchmarks, ablation studies, efficiency analysis, parameter sensitivity analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and complete theoretical derivations, though the notation system is relatively complex.
- Value: ⭐⭐⭐⭐ Provides significant and consistent improvements on long-horizon, sparse-reward tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[ICLR 2026\] Beyond Penalization: Diffusion-based Out-of-Distribution Detection and Selective Regularization in Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/beyond_penalization_diffusion-based_out-of-distribution_detection_and_selective_.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](../../ICML2025/reinforcement_learning/divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)

</div>

<!-- RELATED:END -->
