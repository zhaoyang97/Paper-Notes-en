---
title: >-
  [Paper Note] Learning to Clean: Reinforcement Learning for Noisy Label Correction
description: >-
  [NeurIPS 2025][Reinforcement Learning][Noisy Labels] This paper formulates noisy label correction as a Markov Decision Process under the reinforcement learning framework…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Noisy Labels"
  - "Label Correction"
  - "Policy Gradient"
  - "Actor-Critic"
  - "k-Nearest Neighbors"
date: 2026-05-08
content_hash: 02a8a5e4cb82c8b0
---

# Learning to Clean: Reinforcement Learning for Noisy Label Correction

**Conference**: NeurIPS 2025
**arXiv**: [2511.19808](https://arxiv.org/abs/2511.19808)
**Code**: Unavailable
**Area**: Reinforcement Learning
**Keywords**: Noisy Labels, Label Correction, Policy Gradient, Actor-Critic, k-Nearest Neighbors

## TL;DR

This paper formulates noisy label correction as a Markov Decision Process under the reinforcement learning framework, proposing RLNLC. A policy function built upon a k-nearest-neighbor embedding space determines which labels should be corrected, guided by a label consistency reward and a cross-subset alignment reward. RLNLC achieves state-of-the-art performance across multiple benchmark datasets under both instance-dependent and symmetric noise settings.

## Background & Motivation

When trained on datasets with noisy labels, deep neural networks first learn features from clean data and then gradually overfit to noisy samples, causing severe degradation in generalization. Existing approaches to noisy labels fall into three main categories:

**Sample selection methods** (Co-teaching, MentorNet): These exploit training dynamics (loss values, prediction confidence) to filter reliable samples, but are inherently passive—they cannot actively correct labels.

**Label correction methods** (PLC, SSR): These adjust labels using model predictions, but typically perform single-step greedy corrections without considering **long-term consequences**—a correction that appears reasonable in the short term may lead to error accumulation over time.

**Semi-supervised learning methods** (DivideMix, LongReMix): These treat noisy samples as unlabeled data, but rely on fixed partitioning mechanisms that struggle to adapt dynamically to evolving data characteristics.

Core insight: **Noisy label correction is naturally suited to the sequential decision-making framework of RL**—corrections constitute a sequence of actions, each modifying the data state, requiring maximization of long-term cumulative reward (final label quality), and necessitating exploration of diverse correction strategies to avoid local optima.

## Method

### Overall Architecture

RLNLC defines the problem as an MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, \mathcal{R}, \gamma)$:
- **State**: The current dataset with its labels $\boldsymbol{s}^t = \{(\mathbf{x}_i, \hat{\mathbf{y}}_i^t)\}_{i=1}^N$
- **Action**: A binary correction decision for each sample $a_i \in \{0, 1\}$
- **Transition**: Deterministically replaces selected labels with k-nearest-neighbor predicted labels
- **Reward**: A composite function evaluating label quality after correction

An Actor-Critic method is used to learn the optimal policy. After training, the policy is deployed to iteratively correct labels, and the prediction model is then fine-tuned on the cleaned labels.

### Key Designs

1. **k-Nearest-Neighbor-Based Policy Function**: The policy function operates in the embedding space of a feature extraction network $f_\theta$. For each sample $\mathbf{x}_i$, an attention-weighted aggregation over its k-nearest neighbors produces a predicted label $\bar{\mathbf{y}}_i = \sum_{j \in \mathcal{N}(\mathbf{x}_i)} \alpha_{ij} \hat{\mathbf{y}}_j^t$, where attention weights are computed via cosine similarity with temperature parameter $\tau$. The correction probability is defined as:

    $\pi_\theta(\boldsymbol{s}^t)_i = \frac{\sum_{j=1}^C \mathbb{1}(\bar{\mathbf{y}}_{ij} > \bar{\mathbf{y}}_{i\hat{y}_i}) \cdot \bar{\mathbf{y}}_{ij}}{\sum_{j=1}^C \mathbb{1}(\bar{\mathbf{y}}_{ij} \geq \bar{\mathbf{y}}_{i\hat{y}_i}) \cdot \bar{\mathbf{y}}_{ij}}$

   This design is elegant: it quantifies the degree of disagreement between the k-nearest-neighbor prediction and the current label—if the current label already corresponds to the highest-probability class among the neighbors, the correction probability is zero; greater deviation yields a higher correction probability.

2. **Dual Reward Function**:

    - **Label Consistency Reward (LCR)**: Evaluates the global statistical smoothness of labels after correction. Using an independent, fixed backbone $f_\omega$ (decoupled from the policy network), it computes the negative mean KL divergence between each sample's label and its k-nearest-neighbor labels: $\mathcal{R}_{\text{LCR}} = -\mathbb{E}_{i \in [1:N]}[\text{KL}(\hat{\mathbf{y}}_i^{t+1}, \sum_j \alpha_{ij} \hat{\mathbf{y}}_j^{t+1})]$
    - **Noisy Label Alignment Reward (NLA)**: The data is partitioned into a "clean subset" ($a_i=0$, labels unchanged) and a "noisy subset" ($a_i=1$, labels corrected). The reward is the negative mean KL divergence between the corrected label of each sample in the noisy subset and the labels of its k-nearest neighbors in the clean subset.
    - Composite reward: $\mathcal{R} = \exp(\mathcal{R}_{\text{LCR}} + \lambda \mathcal{R}_{\text{NLA}})$, where the exponential function maps the unbounded negative KL divergences to $(0, 1]$, ensuring a bounded reward signal.

3. **Efficient State Encoding for the Critic**: Due to the deterministic transition mechanism, the next state $\boldsymbol{s}^{t+1}$ replaces the pair $(s^t, a^t)$ as the Critic input. To reduce dimensionality (which scales with dataset size $N$), a binning strategy is adopted: each sample is assigned to one of $N_b$ bins (where $N_b \ll N$) based on its label consistency reward $r(\mathbf{x}_i, \hat{\mathbf{y}}_i^{t+1})$, and the proportion of samples in each bin forms a vector of length $N_b$ as the Critic input.

### Loss & Training

- **Actor update**: $\theta \leftarrow \theta + \beta_\theta \nabla_\theta \log \pi_\theta(\boldsymbol{a}^t | \boldsymbol{s}^t) Q(\boldsymbol{s}^t, \boldsymbol{a}^t)$
- **Critic update**: Uses the SARSA TD error $\delta^{t-1} = \mathcal{R}(\boldsymbol{s}^{t-1}, \boldsymbol{a}^{t-1}) + \gamma Q(\boldsymbol{s}^t, \boldsymbol{a}^t) - Q_\phi(\boldsymbol{s}^{t-1}, \boldsymbol{a}^{t-1})$
- **Initial state randomization**: At the start of each training epoch, a small number of labels in $\boldsymbol{s}_0^0$ are randomly perturbed to generate a disturbed initial state $\boldsymbol{s}^0$, enhancing exploration
- **Train → Deploy → Fine-tune**: The policy network is first trained for 500 epochs, then deployed to execute $T'=25$ steps of label correction, followed by fine-tuning the prediction model on the corrected labels for 100 epochs

## Key Experimental Results

### Main Results

**CIFAR10-IDN / CIFAR100-IDN Instance-Dependent Noise (Test Accuracy %)**

| Method | CIFAR10 20% | CIFAR10 40% | CIFAR10 50% | CIFAR100 20% | CIFAR100 40% | CIFAR100 50% |
|--------|-------------|-------------|-------------|--------------|--------------|--------------|
| CE | 75.8 | 62.5 | 39.4 | 30.4 | 21.5 | 14.4 |
| DivideMix | 94.8 | 94.5 | 93.0 | 77.1 | 70.8 | 58.6 |
| SSR | 96.5 | 96.3 | 94.1 | 78.8 | 77.0 | 72.8 |
| **RLNLC** | **97.3** | **96.9** | **95.8** | **80.5** | **78.5** | **74.7** |

**Real-World Noisy Datasets**

| Method | Animal-10N | Food-101N |
|--------|-----------|----------|
| SURE | 89.0 | - |
| LongReMix | - | 87.3 |
| **RLNLC** | **90.2** | **89.2** |

### Ablation Study

**CIFAR100-IDN Ablation Study (Test Accuracy %)**

| Configuration | 20% | 30% | 40% | 45% | 50% | Description |
|---------------|-----|-----|-----|-----|-----|-------------|
| RLNLC (full) | **80.5** | **80.1** | **78.5** | **77.2** | **74.7** | All components |
| w/o $\mathcal{R}_{\text{NLA}}$ | 78.4 | 77.9 | 76.2 | 76.3 | 72.0 | Remove cross-subset alignment reward |
| w/o $\mathcal{R}_{\text{LCR}}$ | 79.3 | 78.5 | 76.1 | 76.1 | 73.9 | Remove label consistency reward |
| w/o randomized $s^0$ | 79.9 | 79.5 | 77.8 | 76.8 | 73.1 | No initial state perturbation |
| $f_\omega \leftarrow f_\theta$ | 78.4 | 76.9 | 75.2 | 74.3 | 73.8 | Shared feature network for policy and reward |

### Key Findings

- **RLNLC achieves state-of-the-art performance across all noise types and ratios**: Under 50% instance-dependent noise on CIFAR10-IDN, it outperforms SSR by 1.7%; on CIFAR100-IDN, it consistently surpasses SSR by 1.5–2.2% across all five noise ratios.
- **The advantage is more pronounced under extreme symmetric noise**: At 90% symmetric noise on CIFAR100, RLNLC achieves 44.2%, far exceeding DivideMix at 31% (+13.2%), indicating that RL's exploration capability is particularly critical in high-noise regimes.
- **Label correction accuracy improves steadily with deployment steps**: On CIFAR10-IDN under low noise, correction accuracy exceeds 90% within $T'=5$ steps; under high noise, 90% accuracy is reached within $T'=10$ steps.
- **Decoupling the policy and reward networks is critical**: The $f_\omega \leftarrow f_\theta$ variant suffers substantial performance degradation (0.9% drop at 50% noise), as sharing features introduces circular bias between reward evaluation and policy optimization.

## Highlights & Insights

- **Unique value of the RL perspective**: Modeling noisy label correction as a sequential decision-making problem is highly natural—label corrections are actions, the label state evolves, and non-myopic long-term optimization is required. Compared to single-step greedy methods, RL can discover superior correction trajectories through exploration.
- The policy function elegantly translates k-nearest-neighbor prediction inconsistency into correction probabilities, yielding both interpretability (greater inconsistency implies higher likelihood of noise) and differentiability (through the feature network $f_\theta$).
- The binning encoding scheme for the Critic is concise and efficient—compressing the $N$-dimensional input to $N_b$ dimensions (default 100) while preserving global information about the state distribution.

## Limitations & Future Work

- Computational cost is relatively high: the pipeline requires feature network pretraining, 500 epochs of RL policy learning, 25 deployment steps, and 100 epochs of fine-tuning, resulting in considerable overall complexity.
- k-nearest-neighbor computation may become a bottleneck on large-scale datasets, as feature distances must be recomputed at each step.
- Validation is limited to classification tasks; performance on other noisy-label settings such as detection and segmentation remains untested.
- The action space is binary (correct / do not correct), without considering finer-grained actions specifying which class to correct to—the current approach relies on automatic substitution via k-nearest-neighbor predictions.
- Comparisons with other RL algorithms (e.g., PPO, SAC) are absent, leaving the choice of Actor-Critic insufficiently justified.

## Related Work & Insights

- The key distinction from semi-supervised methods such as DivideMix is that DivideMix performs a one-time partition via GMM fitting on the loss distribution, whereas RLNLC iteratively refines labels through multi-step sequential decision-making, enabling adaptive behavior.
- The approach shares conceptual overlap with RLHF—RLHF uses RL to optimize human preferences, while RLNLC uses RL to optimize label quality; both reformulate a discriminative problem as an RL decision problem.
- The initial state randomization technique (analogous to domain randomization) is potentially transferable to other RL-for-ML frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Framing noisy label correction as an RL problem is a novel perspective; the policy function and reward designs are distinctive
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, multiple noise types and ratios, comprehensive ablations and hyperparameter sensitivity analyses
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are detailed and the method is presented clearly
- Value: ⭐⭐⭐⭐ — Opens a new application direction for RL in data quality improvement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
