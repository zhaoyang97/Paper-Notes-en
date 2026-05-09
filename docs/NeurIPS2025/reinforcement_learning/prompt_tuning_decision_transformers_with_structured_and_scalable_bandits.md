---
title: >-
  [Paper Note] Prompt Tuning Decision Transformers with Structured and Scalable Bandits
description: >-
  [NeurIPS 2025][Reinforcement Learning][Decision Transformer] This paper proposes a structured prompt tuning method based on multi-armed bandits. By decomposing prompts into independent segments and leveraging a pretrained PDT as a feature extractor, the method reduces prompt search complexity from combinatorial explosion to linear scale, significantly improving inference performance of a frozen PDT backbone in multi-task offline RL.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Decision Transformer
  - prompt tuning
  - Multi-Armed Bandit
  - offline multi-task RL
  - few-shot generalization
date: 2026-05-08
content_hash: 2dac067d418ca06e
---

# Prompt Tuning Decision Transformers with Structured and Scalable Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2502.04979](https://arxiv.org/abs/2502.04979)
**Code**: Available (appendix materials)
**Area**: Reinforcement Learning
**Keywords**: Decision Transformer, prompt tuning, Multi-Armed Bandit, offline multi-task RL, few-shot generalization

## TL;DR

This paper proposes a structured prompt tuning method based on multi-armed bandits. By decomposing prompts into independent segments and leveraging a pretrained PDT as a feature extractor, the method reduces prompt search complexity from combinatorial explosion to linear scale, significantly improving inference performance of a frozen PDT backbone in multi-task offline RL.

## Background & Motivation

### State of the Field

Prompting Decision Transformer (PDT) extends offline RL to multi-task settings by prepending trajectory prompts to input sequences to distinguish different tasks, enabling inference-time adaptation to new tasks without additional training. This paradigm mirrors prompting in LLMs—modifying model behavior through input manipulation rather than weight updates.

### Limitations of Prior Work

The core issue with PDT lies in its **overly simplistic prompt sampling strategy**: it uniformly samples trajectory segments from an expert demonstration set to compose prompts, completely ignoring the varying informativeness of different prompts. Even in fully observable MDPs, different prompt segments contribute unequally to task identification. Sampling low-information prompts weakens PDT's ability to distinguish tasks, leading to performance degradation.

### Root Cause

Existing PDT prompt tuning methods suffer from three critical shortcomings:

**Limited expressiveness**: Yuan et al. (2024) replace trajectory prompts with goal conditioning, reducing representational capacity.

**Narrow applicability**: The generative methods of Hu et al. (2023, 2024) are inapplicable in discrete settings and do not respect causal relationships among prompt tokens.

**Poor scalability**: All existing methods treat prompts as flat, unstructured inputs and operate directly on MDP modalities, causing complexity to grow combinatorially with prompt size and state/action space dimensions.

### Starting Point

The core idea is to **exploit the structural nature of prompts**: a prompt consists of $J$ trajectory segments, each corresponding to a distinct position. Maintaining an independent reward model per position reduces the search space from $O(|P|^J)$ to $O(J \cdot |P|)$. The pretrained PDT itself is further leveraged as a segment feature extractor, addressing the scalability challenge of reward modeling in high-dimensional spaces.

## Method

### Overall Architecture

The method operates at inference time: given a pretrained frozen PDT model $\pi^*(\mathbf{x};\theta)$, a small demonstration set $\mathcal{P}_i$ for the target task, and a simulator $\mathcal{M}_i$, a multi-armed bandit (MAB) framework iteratively selects the optimal trajectory prompt from $\mathcal{P}_i$ to maximize downstream task performance.

At each iteration, the bandit selects a prompt $\rho_k$, the PDT executes a rollout using that prompt to obtain cumulative return $G_i^k$, and this return serves as the reward signal to update the reward model.

### Key Designs

1. **Structured Bandit Architecture**: The $J$ segment positions of a prompt are decoupled into $J$ independent contextual MAB problems. Each position $j$ maintains an independent reward model $\phi_j$ that predicts the expected PDT performance when placing segment $\tilde{\tau}$ at position $j$. During prompt selection, each model generates predictions over all candidate segments, and the highest-scoring segment at each position is selected, forming a prediction matrix $\mathbf{Y} \in \mathbb{R}^{J \times |\mathcal{P}_i|}$. Supported exploration strategies include $\epsilon$-greedy, UCB, and Thompson Sampling.

2. **PDT as Feature Extractor**: Learning a reward model directly on unencoded MDP modalities yields input dimensionality of $H \times (|\mathcal{S}| + |\mathcal{A}| + 1)$, scaling linearly with state/action space size. This paper instead proposes using the hidden representations of prompt tokens produced by the pretrained PDT as segment embeddings $\Psi: \tilde{\tau} \to \mathbb{R}^d$, yielding compact fixed-dimensional representations that enable efficient deployment in high-dimensional environments such as pixel-based observation spaces.

3. **Regret Analysis**: Assuming prompt reward decomposes as the sum of independent segment contributions plus a bounded interaction term:

$$G(\rho) = \frac{1}{J}\sum_{j=1}^{J}\phi_j(\tilde{\tau}_j) + h(\tilde{\tau}_1, \ldots, \tilde{\tau}_J), \quad |h| \leq \varepsilon$$

Under this assumption, the cumulative regret upper bound is:

$$\text{Regret}(K) \leq \frac{1}{J}\sum_{j=1}^{J}\text{Regret}_j(K) + 2K\varepsilon$$

When standard bandit algorithms are applied per slot, total regret is $\mathcal{O}(\sqrt{K\log|P|} + K\varepsilon)$, preserving sublinear regret.

### Loss & Training

The reward model $\phi_j$ at each position is trained independently using accumulated data pairs $\langle \tilde{\tau}_j^k, G_i^k \rangle$, optimizing the MSE loss:
$$\mathcal{L}(\phi_j) = \text{MSE}(\hat{y}_j, y)$$
where $\hat{y}_j = \phi_j(X_j)$ is the predicted reward and $y$ is the actual rollout return.

## Key Experimental Results

### Main Results (In-Distribution)

| Environment | Config | PDT (no tuning) | Hill-climbing | ZORankSGD | TS (Ours) | TS$^\Psi$ (Ours) |
|-------------|--------|-----------------|--------------|-----------|-----------|-----------------|
| Half Cheetah | J=2, H=20 | -42.68 | -29.93 | -34.77 | **-26.28** | -27.62 |
| Ant | J=1, H=5 | 694.43 | 738.56 | 735.47 | **835.38** | 800.95 |
| Pick-place | J=1, H=5 | 551.58 | 555.79 | 554.26 | **556.11** | 556.87 |

### OOD Generalization

| Environment | Config | PDT (no tuning) | PDT (fine-tuned) | TS (Ours) | Gain |
|-------------|--------|-----------------|-----------------|-----------|------|
| Half Cheetah | J=2, H=20 | -40.95 | -39.30 | **-26.28** | 35.8% |
| Ant | J=1, H=5 | 363.90 | 306.29 | **466.11** | +28.1% |
| Pick-place | J=2, H=2 | 524.37 | 488.17 | **553.34** | +5.5% |

### Sparse 2D Environment

| Method | J=1 | J=2 | J=4 |
|--------|-----|-----|-----|
| Optimal policy | 10.0 | 10.0 | 10.0 |
| PDT (no tuning) | 0.0±2.1 | 6.3±0.8 | 8.3±0.6 |
| TS (Ours) | **9.9±0.0** | **9.9±0.0** | **9.8±0.1** |
| Hill-climbing | 5.8±3.8 | 7.9±1.6 | 6.2±4.0 |

### Key Findings

- Bandit-based tuning consistently and substantially improves frozen PDT performance across all environments, even surpassing the single-task CQL oracle in the Ant environment.
- Methods using PDT-encoded features (denoted $\Psi$) achieve performance comparable to their unencoded counterparts, confirming that the PDT provides compact and effective representations.
- Fine-tuning the PDT backbone can actually degrade performance, particularly in OOD settings.
- Attention analysis reveals that the PDT primarily attends to the single most informative segment, supporting the segment independence assumption.

## Highlights & Insights

- **Structured decomposition is the key innovation**: decomposing a combinatorial optimization problem into linearly-scaled subproblems yields both theoretical guarantees and practical gains.
- **PDT itself is the optimal feature extractor**: no separate encoder training is required; the pretrained model's representations are directly reused.
- The method operates entirely at inference time without updating Transformer weights, incurring minimal deployment overhead.
- Even when only 10% of demonstration data consists of expert trajectories, prompt tuning recovers near-optimal performance.

## Limitations & Future Work

- The method requires an environment simulator for online rollouts to obtain bandit rewards, making it unsuitable for purely offline settings.
- As the number of demonstrations increases, the search space still grows combinatorially—this could be mitigated by learning a sampler to pre-select high-potential segments.
- The segment independence assumption may not hold perfectly in all tasks.
- Integration with meta-learning approaches such as In-Context RL remains unexplored.

## Related Work & Insights

- Connection to LLM prompt tuning: InstructZero uses Bayesian optimization to explore soft prompts, and INSTINCT replaces Gaussian processes with neural networks—this paper similarly shifts optimization from continuous to discrete prompt spaces.
- Prompt Diffuser synthesizes prompts via conditional generative modeling but is inapplicable in discrete settings.
- The proposed method generalizes to other scenarios requiring inference-time selection of input templates.

## Rating

- Novelty: ⭐⭐⭐⭐ — The structured bandit decomposition of prompt space is an elegant and effective idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple environments, in-distribution/OOD settings, data quality ablations, regret analysis, and attention analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, theoretical derivations are rigorous, and the appendix is thorough.
- Value: ⭐⭐⭐⭐ — Provides a practical inference-time adaptation solution with low computational overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Greedy Algorithm for Structured Bandits: A Sharp Characterization of Asymptotic Success / Failure](greedy_algorithm_for_structured_bandits_a_sharp_characterization_of_asymptotic_s.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[NeurIPS 2025\] Parameter Efficient Fine-tuning via Explained Variance Adaptation](parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)
- [\[NeurIPS 2025\] Emergent World Beliefs: Exploring Transformers in Stochastic Games](emergent_world_beliefs_exploring_transformers_in_stochastic_games.md)

</div>

<!-- RELATED:END -->
