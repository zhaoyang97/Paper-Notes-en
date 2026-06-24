---
title: >-
  [Paper Note] CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries
description: >-
  [ICML 2025][Self-Supervised Learning][Preference-based Reinforcement Learning] Proposes CLARIFY, a method that constructs a trajectory embedding space integrating preference information via contrastive learning and utilizes rejection sampling to select clearer, more distinguishable preference queries, thereby improving annotation efficiency and policy performance of offline PbRL under non-ideal feedback.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Preference-based Reinforcement Learning"
  - "Contrastive Learning"
  - "Ambiguous Queries"
  - "Trajectory Embedding"
  - "Offline RL"
date: 2026-05-08
content_hash: f8f52cb643f99842
---

# CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries

**Conference**: ICML 2025  
**arXiv**: [2506.00388](https://arxiv.org/abs/2506.00388)  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: Preference-based Reinforcement Learning, Contrastive Learning, Ambiguous Queries, Trajectory Embedding, Offline RL

## TL;DR

Proposes CLARIFY, a method that constructs a trajectory embedding space integrating preference information via contrastive learning and utilizes rejection sampling to select clearer, more distinguishable preference queries, thereby improving annotation efficiency and policy performance of offline PbRL under non-ideal feedback.

## Background & Motivation

Preference-based Reinforcement Learning (PbRL) eliminates the complexity of explicit reward engineering by querying human preferences over pairs of trajectory segments to infer a reward function. However, when two trajectory segments are highly similar, humans struggle to provide clear preference judgments, leading to the **ambiguous queries** problem. This issue not only compromises annotation efficiency but also restricts the application of PbRL in real-world scenarios.

The key challenge of existing methods is that most PbRL approaches (e.g., PEBBLE, PT, OPRL) either ignore the existence of ambiguous queries or address them only in online settings (e.g., Mu et al., 2024), which cannot be directly transferred to offline scenarios. In offline PbRL, data is fixed and interaction with the environment is prohibited. Thus, maximizing the selection of "clearly distinguishable" query pairs within a limited preference budget remains a critical bottleneck.

Key Insight of this work: Leverage contrastive learning to encode preference information into a trajectory embedding space, making "clearly distinguishable" segments far apart and "ambiguous" segments close together. Based on this embedding space, rejection sampling is used to select more unambiguous queries, thereby improving labeling efficiency. Core Idea: **Model preference structures using contrastive learning, distinguish query clarity via distance in the embedding space, and filter for high-quality queries using rejection sampling.**

## Method

### Overall Architecture

CLARIFY consists of two phases:
1. **Representation Learning Phase**: A trajectory encoder $z = f_\phi(\tau)$ is trained via contrastive learning to map trajectories to a fixed-dimensional embedding space while incorporating preference information (clear/ambiguous labels).
2. **Query Selection Phase**: Based on the learned embedding space, query pairs with larger embedding distances (i.e., clearer and more distinguishable) are selected via rejection sampling and handed to humans for annotation.

The detailed workflow is: first randomly sample a batch of queries to pre-train the encoder and reward model $\to$ select new queries based on the embedding space $\to$ update the preference dataset and reward model $\to$ retrain the embeddings $\to$ finally train the policy using offline RL algorithms like IQL.

### Key Designs

1. **Ambiguity Loss $\mathcal{L}_{\text{amb}}$**: The core mechanism is to maximize the embedding distance between clearly distinguishable segment pairs while minimizing the distance between ambiguous segment pairs. For "clear" queries labeled as $p \in \{0, 1\}$ in the preference dataset, the embeddings of the two segments are pulled apart; for "ambiguous" queries labeled as $p = \text{no\_cop}$, the embeddings are pulled closer together. The design motivation is to directly achieve the embedding space goal of "clear-far, ambiguous-close". However, using this loss in isolation leads to overfitting and representation collapse (where ambiguous segments map to the exact same point).

2. **Quadrilateral Loss $\mathcal{L}_{\text{quad}}$**: To address the issue of using $\mathcal{L}_{\text{amb}}$ alone, a quadrilateral loss is introduced to model preference relationships. For two sets of clear queries $(\sigma_+, \sigma_-)$ and $(\sigma_+', \sigma_-')$, it encourages the distance between "good" segments $(z_+, z_+')$ and between "bad" segments $(z_-, z_-')$ to be smaller than cross-set distances (e.g., $(z_+, z_-')$). The key formula minimizes: $-\mathbb{E}[\ell(z^+, z^{-\prime}) + \ell(z^{+\prime}, z^-) - \ell(z^+, z^{+\prime}) - \ell(z^-, z^{-\prime})]$. By pairing queries, the training data scales from $O(n)$ to $O(n^2)$, mitigating overfitting on small samples while serving as a regularizer to prevent representation collapse.

3. **Rejection Sampling Query Selection**: Calculates the embedding distance $d_{\text{emb}}$ of query pairs, estimates the density functions of clear and ambiguous queries ($\rho_{\text{clr}}$ and $\rho_{\text{amb}}$), and constructs a weighted density $\rho(d) = 0.5(\rho_1 + \rho_2)$, where $\rho_1$ is difference-based and $\rho_2$ is ratio-based. The final sampling distribution $q(d) = p(d) \cdot \rho(d)$ increases the probability of selecting clear queries. The design motivation is to avoid selecting only the queries with the largest distances (which would degrade diversity) but rather to increase the proportion of clear queries while maintaining diversity.

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{amb}}\mathcal{L}_{\text{amb}} + \lambda_{\text{quad}}\mathcal{L}_{\text{quad}} + \lambda_{\text{norm}}\mathcal{L}_{\text{norm}}$$

where $\mathcal{L}_{\text{recon}}$ is the reconstruction loss based on the Bi-directional Decision Transformer, and $\mathcal{L}_{\text{norm}}$ constrains the L2 norm of the embeddings to be close to 1 to stabilize training. The continuous distribution of embedding distances is discretized into $n_{\text{bin}}$ bins to handle rejection sampling.

## Key Experimental Results

### Main Results

Comparing CLARIFY with baselines including MR, OPRL, PT, OPPO, and LiRE across a total of 9 tasks in Metaworld and DMControl:

| Task | Metric | CLARIFY | Prev. SOTA (OPRL/LiRE) | Gain |
|------|------|---------|----------------------|------|
| dial-turn (ε=0.5) | Success Rate | 77.50 ± 7.37 | 57.33 ± 25.02 (OPRL) | +20.17 |
| drawer-open (ε=0.5) | Success Rate | 83.50 ± 7.40 | 72.67 ± 2.87 (OPRL) | +10.83 |
| handle-pull-side (ε=0.5) | Success Rate | 95.00 ± 1.22 | 89.75 ± 6.07 (PT) | +5.25 |
| walker-walk (ε=0.5) | Return | 796.34 ± 12.87 | 789.18 ± 28.77 (LiRE) | +7.16 |
| cheetah-run (ε=0.5) | Return | 617.31 ± 14.43 | 553.61 ± 43.16 (LiRE) | +63.70 |
| dial-turn (ε=0.7) | Success Rate | 79.40 ± 3.83 | 63.40 ± 9.46 (OPRL) | +16.00 |
| walker-walk (ε=0.7) | Return | 816.54 ± 11.08 | 795.02 ± 22.80 (LiRE) | +21.52 |

### Ablation Study

| Configuration | dial-turn | sweep-into | Description |
|------|-----------|------------|------|
| w/o $\mathcal{L}_{\text{amb}}$, w/o $\mathcal{L}_{\text{quad}}$ | 63.20 ± 4.79 | 40.00 ± 11.29 | Equivalent to OPRL |
| w/ $\mathcal{L}_{\text{amb}}$, w/o $\mathcal{L}_{\text{quad}}$ | 69.00 ± 11.20 | 52.80 ± 17.01 | Unstable, prone to overfitting |
| w/o $\mathcal{L}_{\text{amb}}$, w/ $\mathcal{L}_{\text{quad}}$ | 71.25 ± 8.81 | 62.20 ± 4.92 | Slower convergence |
| Both present (CLARIFY) | **77.50 ± 3.01** | **68.00 ± 3.19** | Best and most stable |

### Key Findings

- **Query Clarity**: At skip rate ε=0.5, CLARIFY achieves a clear query ratio of 76.33% in dial-turn, which is significantly higher than MR (46.95%), OPRL (31.67%), and PT (43.90%).
- **Human Experiment Validation**: In real-human labeling experiments on walker-walk, CLARIFY achieves a return of 420.75 vs. OPRL's 265.91, query clarity of 63.33% vs. 53.33%, and labeling accuracy of 87.08% vs. 66.67%.
- **Query Efficiency**: Even with only 100 queries, CLARIFY significantly outperforms MR (dial-turn: 59.50 vs. 49.50).
- Direct density-based selection (i.e., choosing only the clearest queries) performs poorly due to a lack of diversity; the rejection sampling approach successfully balances clarity and diversity.

## Highlights & Insights

- Formalizes the long-neglected "ambiguous query" problem in PbRL and provides a systematic solution.
- The design of the quadrilateral loss is highly elegant: it leverages query pairing to expand the sample size from $O(n)$ to $O(n^2)$ while modeling the global structure of preferences.
- Strong theoretical guarantees are provided (margin separation in Proposition 5.1 and convex separability in Proposition 5.2).
- t-SNE visualizations of the embedding space intuitively demonstrate the effectiveness of the proposed method.
- Consistency between real-human experiments and simulation results enhances the credibility of the approach.

## Limitations & Future Work

- Validation is currently restricted to offline PbRL; extension to online scenarios has not been explored.
- Embedding training relies on the BDT architecture, and its adaptability to different tasks warrants further investigation.
- Discretization for density estimation in rejection sampling introduces an additional hyperparameter $n_{\text{bin}}$.
- The scale of the human experiments is relatively small (only 20 or 100 feedbacks per round), leaving performance in large-scale human feedback scenarios unknown.
- Applying this method to filter preference data in LLM alignment (RLHF) represents a promising direction.

## Related Work & Insights

- Most closely related to Mu et al. (2024), but the latter only addresses ambiguous queries in online settings.
- The application of contrastive learning in RL (image representation learning, temporal distance learning) provides a methodological foundation.
- LiRE (Choi et al., 2024) enhances feedback through list-wise comparisons, offering a complementary direction.
- Insight for RLHF: A similar "ambiguous preference" problem in LLM alignment (where humans struggle to judge when two responses are of similar quality) could potentially benefit from this approach.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combined design of the quadrilateral loss and rejection sampling query selection is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 9 tasks, simulation + real-human experiments, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with well-reasoned theoretical analysis and experimental alignment.
- Value: ⭐⭐⭐⭐ — Resolves a practical pain point in PbRL, with promising prospects for RLHF applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)

</div>

<!-- RELATED:END -->
