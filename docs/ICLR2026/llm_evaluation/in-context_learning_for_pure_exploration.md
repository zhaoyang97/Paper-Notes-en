---
title: >-
  [Paper Note] In-Context Learning for Pure Exploration
description: >-
  [ICLR 2026][LLM Evaluation][In-Context Learning] This paper proposes ICPE (In-Context Pure Exploration), an in-context learning framework that combines supervised learning and reinforcement learning. Using a Transformer…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "In-Context Learning"
  - "Pure Exploration"
  - "Hypothesis Testing"
  - "Best Arm Identification"
  - "Transformer"
date: 2026-05-08
content_hash: a17507e342cb2d50
---

# In-Context Learning for Pure Exploration

**Conference**: ICLR 2026
**arXiv**: [2506.01876](https://arxiv.org/abs/2506.01876)  
**Code**: Available (attached to paper)  
**Area**: LLM Evaluation
**Keywords**: In-Context Learning, Pure Exploration, Hypothesis Testing, Best Arm Identification, Transformer

## TL;DR

This paper proposes ICPE (In-Context Pure Exploration), an in-context learning framework that combines supervised learning and reinforcement learning. Using a Transformer trained directly from experience, ICPE learns exploration policies for active sequential hypothesis testing and pure exploration problems, achieving near-optimal instance-adaptive algorithmic performance without explicit modeling of the information structure.

## Background & Motivation

In active sequential hypothesis testing (also known as pure exploration), an agent must actively control the data collection process to efficiently identify the correct hypothesis. This problem arises broadly in medical diagnosis, image recognition, recommendation systems, and related domains. Existing approaches face three key challenges:

**Difficulty encoding inductive biases**: Designing adaptive exploration strategies requires deep understanding of the problem structure, which is particularly challenging when the underlying information structure is unknown.

**Limitations of RL methods**: When the relevant information structure is not adequately represented, conventional RL methods tend to perform poorly.

**Limitations of BAI methods**: Classical approaches such as Best Arm Identification are theoretically elegant but typically rely on explicit modeling assumptions, and in complex environments (e.g., MDPs) the associated optimization problems become non-convex.

Core problem: Can an agent **autonomously** discover and exploit latent structure from experience to enhance exploration efficiency?

## Method

### Overall Architecture

ICPE adopts a dual-network architecture:
- **Inference Network $I$**: Trained via supervised learning to infer the true hypothesis from the current data.
- **Exploration Network $\pi$**: Trained via RL to select actions that maximize the accuracy of the inference network.

Both networks use a Transformer architecture and take the data trajectory $\mathcal{D}_t = (x_1, a_1, \ldots, x_t)$ as sequential input.

### Key Designs

1. **Problem formulated as an MDP**:

    - State $s_t = (\mathcal{D}_t, \emptyset_{t:N})$, comprising the historical data trajectory and padding tokens.
    - Action space $\mathcal{A}$ (including a stop action for the fixed-confidence setting).
    - Design motivation: Reformulates the pure exploration problem into a form amenable to RL.

2. **Fixed-Confidence Setting**:

    - Objective: Minimize the stopping time $\tau$ subject to $\mathbb{P}(\hat{H}_\tau = H^*) \geq 1 - \delta$.
    - Solved via the dual problem: $\min_{\lambda \geq 0} \max_{I, \pi} V_\lambda(\pi, I)$.
    - Reward design: $r_\lambda(z) = -1 + d \cdot \lambda \log I_{\bar{\phi}}(H^* | s')$, where $d$ is a termination indicator.
    - Includes a dedicated stop action whose Q-value can be back-propagated from any state.

3. **Fixed-Horizon Setting**:

    - Objective: Maximize the probability of correct identification within a given budget of $N$ steps.
    - Reward is given only at the final step: $r_N = h(\hat{H}_N; M)$.
    - No stop action is included.

4. **Multi-timescale Optimization**:

    - Slowest timescale: Updates the dual variable $\lambda$.
    - Intermediate timescale: Supervised learning to optimize the inference network $I_\phi$ (cross-entropy loss).
    - Fastest timescale: DQN with Replay Buffer to optimize the policy network $Q_\theta$.
    - Target networks $Q_{\bar{\theta}}$ and $I_{\bar{\phi}}$ are used to maintain training stability.

### Loss & Training

- Inference network: Cross-entropy loss $-\log I_\phi(H^* | s_\tau)$.
- Policy network: TD loss + stopping action loss.
- Transformer architecture: 3 layers, 2 attention heads, hidden dimension 256, GELU activation, GPT-2 configuration.
- Training uses Adam optimizer with learning rates from $10^{-4}$ to $10^{-6}$.

## Key Experimental Results

### Main Results

**1. Deterministic Bandit (Fixed Horizon)**

| K (# actions) | ICPE Accuracy | DQN | Uniform | I-DPT |
|--------------|--------------|-----|---------|-------|
| 4–20 | ≈1.0 | Gradually decreases | Rapidly decreases | Moderate |

- ICPE spontaneously learns the optimal strategy of selecting each action exactly once.

**2. Stochastic Bandit (Fixed Confidence, $\delta=0.1$)**

| K | ICPE Avg. Stopping Time | TaS | TTPS | Uniform |
|---|------------------------|-----|------|---------|
| 4–14 | Lowest | Moderate | Moderate | Highest |

- ICPE achieves sample complexity close to the theoretical lower bound.

**3. Magic Action Bandit (Latent Information Structure)**

| $\sigma_m$ | ICPE | I-IDS | Theoretical Lower Bound |
|-----------|------|-------|------------------------|
| 0.0–1.0 | Near lower bound | Significantly higher | — |

- ICPE outperforms I-IDS across all noise levels.

**4. MNIST Pixel Sampling**

| Method | Accuracy | Avg. # Regions Sampled |
|--------|----------|------------------------|
| ICPE | Highest | Fewer |
| Deep CMAB | Moderate | More |
| Uniform | Lowest | Same |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|------------|-------|
| Fixed confidence vs. fixed horizon | Fixed confidence is superior | The stop action introduces a curriculum learning effect |
| ICPE policy vs. approximate TaS | Difference in total variation | ICPE leverages prior information |
| Class-specific sampling | ICPE shows the most variation | Chi-squared tests confirm significantly different strategies across digits |

### Key Findings

1. **ICPE spontaneously discovers optimal strategies**: In deterministic bandits it learns to select each action exactly once; in binary search tasks it learns an $O(\log_2 K)$ search strategy.
2. **Greatest advantage in environments with latent structure**: In the magic action environment, ICPE discovers and exploits information chains, whereas greedy information-gain methods such as IDS cannot.
3. **The stop action is critical in the fixed-confidence setting**: It functions as a form of curriculum learning, enabling the agent to adapt to problem difficulty.
4. **Policies exhibit substantial contextual adaptation**: In the MNIST task, sampling strategies differ significantly across digit classes.

## Highlights & Insights

- **Elegance of the dual-network design**: The inference network $I$ provides reward signals to the exploration network $\pi$, forming a virtuous cycle — $I$ improves → reward signals become more accurate → $\pi$ learns better exploration strategies → data becomes more informative → $I$ improves further.
- **Algorithm discovery capability**: In binary search tasks, ICPE automatically discovers a probabilistic analogue of binary search, with stopping times matching $\log_2 K$ exactly.
- **Theoretical proof of IDS sub-optimality** (Theorem B.1): In structured environments with a magic action, greedy information-gain policies (IDS) are sub-optimal because they cannot perform long-horizon planning.
- **Connection to cognitive science**: The dual-network architecture of ICPE is analogous to cognitive maps (exploration network) combined with goal-directed evaluation (inference network).

## Limitations & Future Work

1. **Finite hypothesis space $\mathcal{H}$**: The current work assumes $\mathcal{H}$ is a finite set; extension to continuous settings (active regression) is needed.
2. **Dependence on a prior distribution $\mathcal{P}(\mathcal{M})$**: The task distribution must be known and stationary.
3. **Oracle assumption**: A perfect verifier is required during training, which may not be available in practice.
4. **Transformer horizon constraint**: Limited by a fixed maximum context length $N$.
5. **Scalability**: Validation is currently limited to small-scale problems; scaling to larger settings requires improvements in architecture and training.
6. Integration with LLMs using linguistic priors to assist exploration is a promising future direction.

## Related Work & Insights

- **Relation to RL²**: Similarly represents policies via the hidden states of RNNs/Transformers, but targets identification rather than cumulative reward.
- **Distinction from ICEE**: ICEE addresses the exploration–exploitation trade-off (returning conditional learning), while ICPE focuses purely on identification objectives.
- **Connection to Track-and-Stop**: ICPE's learned policy resembles TaS in certain settings but surpasses it by exploiting prior information.
- Insight: In-context learning capability + sequence modeling = an automated algorithm design platform.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Introduces ICL to pure exploration; dual-network design is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Progresses systematically from simple bandits to MNIST to MDPs)
- Writing Quality: ⭐⭐⭐⭐ (Good integration of theory and experiments, though the paper is lengthy)
- Value: ⭐⭐⭐⭐⭐ (Provides a general deep learning framework for active hypothesis testing)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/llm_evaluation/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ACL 2026\] CUB: Benchmarking Context Utilisation Techniques for Language Models](../../ACL2026/llm_evaluation/cub_benchmarking_context_utilisation_techniques_for_language_models.md)
- [\[ICLR 2026\] Log Probability Tracking of LLM APIs](log_probability_tracking_of_llm_apis.md)

</div>

<!-- RELATED:END -->
