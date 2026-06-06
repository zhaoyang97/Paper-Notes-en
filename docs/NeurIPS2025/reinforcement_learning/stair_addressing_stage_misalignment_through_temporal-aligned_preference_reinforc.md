---
title: >-
  [Paper Note] STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Preference-based Reinforcement Learning] This paper identifies and formalizes the "stage misalignment" problem in Preference-based Reinforcement Learning (PbRL)—wherein comparing be…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Preference-based Reinforcement Learning"
  - "Stage Alignment"
  - "Temporal Distance"
  - "Contrastive Learning"
  - "Multi-Stage Tasks"
date: 2026-05-08
content_hash: cb5ea030b1400aca
---

# STAIR: Addressing Stage Misalignment through Temporal-Aligned Preference Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.23802](https://arxiv.org/abs/2509.23802)  
**Code**: [GitHub](https://github.com/iiiiii11/STAIR)  
**Area**: Reinforcement Learning
**Keywords**: Preference-based Reinforcement Learning, Stage Alignment, Temporal Distance, Contrastive Learning, Multi-Stage Tasks

## TL;DR

This paper identifies and formalizes the "stage misalignment" problem in Preference-based Reinforcement Learning (PbRL)—wherein comparing behavior segments from different task stages produces uninformative feedback—and proposes STAIR, a method that learns temporal distances via contrastive learning to approximate stage discrepancy. By employing a quadrilateral distance metric for stage-aligned query selection, STAIR substantially outperforms existing PbRL methods on multi-stage tasks.

## Background & Motivation

### State of the Field

PbRL learns a reward function from human preferences over behavior segments, circumventing the need for manual reward engineering. The standard pipeline samples two segments $(\sigma_0, \sigma_1)$, collects human preference labels $y$, trains a reward model $\hat{r}_\psi$ via the Bradley-Terry model, and uses the learned reward to train a policy. Existing methods such as PEBBLE and SURF improve feedback efficiency through unsupervised pre-training and reward uncertainty guidance.

### Root Cause — Stage Misalignment

Many real-world tasks exhibit **multi-stage structure**—for example, a pick-and-place task comprises: ① navigate to object → ② grasp → ③ transport to goal. When PbRL methods randomly sample segment pairs, segments from different stages are frequently paired for comparison, such as contrasting "arm in transit" with "arm grasping." Drawing on the **event segmentation theory** in cognitive science, humans parse action sequences into discrete event units; comparisons that cross these boundaries increase cognitive load and yield **ambiguous feedback**.

### Theoretical Analysis

The paper rigorously analyzes the impact of stage misalignment via an abstract MDP formulation:
- **Proposition 2**: In the worst case, standard PbRL requires $\mathcal{O}(|\Omega||\Upsilon|\log(|\Omega||\Upsilon|))$ more queries than stage-aligned PbRL.
- **Proposition 3**: In the presence of stage reward bias (humans tend to prefer segments from later stages), the additional query complexity grows from linear to quadratic: $\mathcal{O}(|\Omega|^2|\Upsilon|\log|\Upsilon|)$.

A human study directly validates stage reward bias: on the MetaWorld window-open task, human annotations clearly show a preference for segments from later timesteps.

## Method

### Overall Architecture

STAIR consists of two core modules:
1. **Temporal Distance Learning**: A contrastive learning framework trains an energy function to quantify temporal distances between states.
2. **Stage-Aligned Query Selection**: The learned temporal distances define a stage discrepancy measure over segment pairs, prioritizing stage-aligned queries.

### Key Designs

1. **Temporal Distance Learning (Successor Distance)**: A temporal distance is defined via the discounted state occupancy measure:

$$d_{SD}^\pi(x, y) = \log\left(\frac{p_\gamma^\pi(s_+=y|s_0=y)}{p_\gamma^\pi(s_+=y|s_0=x)}\right)$$

where $p_\gamma^\pi(s_+=y|s_0=x) = (1-\gamma)\sum_{k=0}^{\infty}\gamma^k p^\pi(s_k=y|s_0=x)$. This constitutes a quasimetric satisfying non-negativity and the triangle inequality.

An energy function $f_\theta(x,y)$ is optimized via a symmetric InfoNCE loss:

$$\mathcal{L}_\theta = \sum_{i=1}^{B}\left[\log\frac{\exp(f_\theta(x_i, y_i))}{\sum_j \exp(f_\theta(x_i, y_j))} + \log\frac{\exp(f_\theta(x_i, y_i))}{\sum_j \exp(f_\theta(x_j, y_i))}\right]$$

The function is parameterized as $f_\theta(x,y) = c_{\theta_c}(y) - d_{\theta_d}(x,y)$, such that at optimality $d_{\theta_d^*}(x,y)$ recovers the successor distance. **Key advantage**: the temporal distance is learned on on-policy data and automatically adapts as the policy evolves.

2. **Quadrilateral Distance**: The state-level temporal distance is extended to a segment-level stage discrepancy measure:

$$d_{\text{stage}}(\sigma_0, \sigma_1) = \frac{1}{4}\left(d_{SD}^\pi(s_0^0, s_0^1) + d_{SD}^\pi(s_{H-1}^0, s_{H-1}^1) + d_{SD}^\pi(s_0^0, s_{H-1}^1) + d_{SD}^\pi(s_{H-1}^0, s_0^1)\right)$$

The edge terms (start-to-start and end-to-end distances) measure alignment of initial and terminal states, while the diagonal terms (cross start-to-end distances) penalize segments with long temporal spans, encouraging each segment to remain within a single stage. Proposition 5 proves that the quadrilateral distance is strictly monotonically increasing with the degree of stage misalignment.

3. **Stage-Aligned Query Selection**: Candidate queries are scored by jointly considering stage alignment and reward model uncertainty:

$$I(\sigma_0, \sigma_1) = (c_{\text{stage}} - d_{\text{stage}}(\sigma_0, \sigma_1))(c_{\text{state}} + d_{\text{state}}(\sigma_0, \sigma_1))$$

The first term favors stage alignment (small $d_{\text{stage}}$); the second favors high uncertainty (large $d_{\text{state}}$, based on the ensemble variance of the reward model). The top-$M$ scoring queries are selected for human annotation.

### Loss & Training

- Cross-entropy loss for the reward model under the Bradley-Terry model
- Symmetric InfoNCE loss for temporal distance learning
- Policy trained with SAC
- Temporal distance updated on-policy at frequency $K_{SD}$

## Key Experimental Results

### Multi-Stage Tasks (MetaWorld)

| Task | STAIR | PEBBLE | RUNE | MRN | RIME | QPA |
|------|-------|--------|------|-----|------|-----|
| door-open (5k) | **~100%** | ~86% | ~50% | ~60% | ~75% | ~80% |
| sweep-into (10k) | **~93%** | ~60% | ~30% | ~50% | ~55% | ~65% |
| window-open (2k) | **~95%** | ~70% | ~40% | ~55% | ~65% | ~70% |
| window-close (2k) | **~90%** | ~70% | ~50% | ~55% | ~65% | ~70% |
| door-unlock (5k) | **~100%** | ~85% | ~50% | ~75% | ~80% | ~85% |
| faucet-open (3k) | **~97%** | ~85% | ~65% | ~70% | ~75% | ~80% |

### Single-Stage Tasks (DMControl)

| Task | STAIR | PEBBLE | RUNE | QPA |
|------|-------|--------|------|-----|
| walker-run | **Best** | Moderate | Worst | Competitive |
| quadruped-walk | Competitive | Moderate | Best | Good |
| quadruped-run | **Competitive** | Moderate | Best | Good |

### Feedback Efficiency (door-open)

| Total Queries $N_{\text{total}}$ | STAIR | PEBBLE |
|----------------------------------|-------|--------|
| 500 | 52.01±23.18 | 20.00±17.88 |
| 2000 | 77.77±11.67 | 28.79±17.02 |
| 5000 | **100.00±0.00** | 85.57±12.77 |
| 10000 | **99.93±0.06** | 92.53±6.53 |

### Ablation Study

| Variant | door-open | Notes |
|---------|-----------|-------|
| STAIR (full) | **100%** | Quadrilateral distance + temporal distance |
| Timestep+ISR | ~90% | Timestep in place of temporal distance |
| STAIR(ISR) | ~95% | Temporal distance + ISR (1D metric) |
| $K_{SD}=5$ (low-frequency update) | ~95% | Reduced on-policy update frequency |
| $K_{SD}=50$ | ~85% | Severely lags behind policy evolution |

### Human Study

| Metric | STAIR | PEBBLE |
|--------|-------|--------|
| Proportion of queries judged same-stage by humans | **~70–80%** | ~30–40% |

### Key Findings

- STAIR consistently outperforms all baselines across multi-stage tasks, approaching 100% success rate on most benchmarks.
- Competitive performance is maintained even on single-stage tasks, potentially attributable to an implicit curriculum learning effect.
- The quadrilateral distance substantially outperforms the one-dimensional ISR metric, as it captures more complex segment relationships in a two-dimensional space.
- The on-policy update frequency of the temporal distance is critical to performance.
- The hyperparameters $c_{\text{stage}}$ and $c_{\text{state}}$ demonstrate good robustness.

## Highlights & Insights

- **Precise problem identification**: Stage misalignment is an overlooked yet consequential issue in PbRL, supported by both theoretical analysis and human experiments.
- **Elegant design**: The chain from temporal distance → quadrilateral distance → query selection is natural and well-motivated at each step.
- **No task prior required**: The method does not rely on predefined stage boundaries; stage structure is automatically discovered through temporal distances.
- **Alignment with human cognition**: Queries selected by STAIR are judged as "same-stage" by humans at twice the rate of those selected by PEBBLE, validating alignment with human cognitive processes.
- The strong performance on single-stage tasks suggests an **implicit curriculum learning** mechanism.

## Limitations & Future Work

- The quadrilateral distance evaluates only pairwise segment discrepancy, making it difficult to generalize to other preference formats such as rankings or ratings.
- Temporal distance learning relies on sufficient on-policy data and may be inaccurate in early training.
- Experiments primarily use scripted (oracle) teachers; noise tolerance under real human annotation remains to be more thoroughly validated.
- Automatic determination of the number of stages is not explicitly addressed.
- The possibility of incorporating stage information into the reward model architecture itself is not explored.

## Related Work & Insights

- **Temporal distance learning**: The successor distance (Myers et al., 2024) provides quasimetric guarantees; this paper is the first to apply it to query selection in PbRL.
- **Event segmentation theory**: Human event segmentation from cognitive science provides a psychological foundation for the stage misalignment problem.
- **Curriculum learning**: STAIR's success on single-stage tasks suggests that adaptive query selection can naturally induce curriculum effects.
- **Open question**: Do analogous "context misalignment" problems exist in other RL settings that involve comparisons, such as RLHF for LLMs?

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Precisely identifies and addresses an important yet overlooked problem in PbRL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across multi-domain experiments, ablations, human studies, noise robustness, and feedback efficiency.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem-driven narrative with tightly interlocking components: theoretical analysis → human validation → method design → empirical evaluation.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to real-world PbRL deployments; the method is both principled and practically effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preference-based Reinforcement Learning beyond Pairwise Comparisons: Benefits of Multiple Options](preference-based_reinforcement_learning_beyond_pairwise_comparisons_benefits_of_.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](temporal-difference_variational_continual_learning.md)
- [\[NeurIPS 2025\] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](incremental_sequence_classification_with_temporal_consistency.md)

</div>

<!-- RELATED:END -->
