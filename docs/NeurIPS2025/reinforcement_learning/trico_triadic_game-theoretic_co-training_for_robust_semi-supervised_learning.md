---
title: >-
  [Paper Note] TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Semi-supervised learning] This paper proposes TRiCo, a framework that reformulates semi-supervised learning as a three-player Stackelberg game among a teacher…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Semi-supervised learning"
  - "game theory"
  - "co-training"
  - "meta-learning"
  - "adversarial perturbation"
date: 2026-05-08
content_hash: c53dfc4395c8c9a9
---

# TRiCo: Triadic Game-Theoretic Co-Training for Robust Semi-Supervised Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.21526](https://arxiv.org/abs/2509.21526)
**Code**: Unavailable
**Area**: Reinforcement Learning
**Keywords**: Semi-supervised learning, game theory, co-training, meta-learning, adversarial perturbation

## TL;DR

This paper proposes TRiCo, a framework that reformulates semi-supervised learning as a three-player Stackelberg game among a teacher, two student classifiers, and an adversarial generator. It replaces confidence-based thresholding with mutual information for pseudo-label selection and employs a meta-learning teacher to adaptively regulate training dynamics, achieving state-of-the-art performance under low-label regimes.

## Background & Motivation

Semi-supervised learning (SSL) leverages large quantities of unlabeled data to reduce annotation costs. **Co-training** is a representative paradigm that mitigates confirmation bias by having two models exchange pseudo-labels across complementary views. However, conventional co-training exhibits three fundamental limitations in realistic settings:

**Unreliable pseudo-label selection**: Traditional methods apply fixed confidence thresholds to filter pseudo-labels, yet softmax confidence is poorly calibrated in the early stages of training and under distribution shift. Overconfident but incorrect pseudo-labels propagate across views, leading to semantic collapse.

**Static and symmetric view interaction**: Co-training assumes symmetric model capacity and fixed interaction protocols between views. In practice, model capacity, representation quality, and learning speed are naturally heterogeneous; the absence of an adaptive regulation mechanism results in interaction stagnation or degraded generalization.

**Lack of hard example mining**: Pseudo-labels are inherently biased toward high-confidence, easy samples, causing models to overfit to these regions while neglecting uncertain regions near decision boundaries—precisely the regions most critical for robustness.

The core idea of this paper is to introduce a **third-party role—the teacher**—upgrading the bilateral interaction to a tripartite game that forms a closed loop of "teacher-guided regulation, generator-induced challenges, and supervised student collaboration."

## Method

### Overall Architecture

TRiCo comprises three interacting components:

- **Two student classifiers** $f_1$, $f_2$: trained on complementary representations extracted by two frozen visual encoders (DINOv2 and MAE), using lightweight MLP heads.
- **Non-parametric adversarial generator** $G$: applies perturbations in the embedding space to expose decision boundary weaknesses.
- **Meta-learning teacher** $\pi_T$: adaptively controls the pseudo-label selection threshold and loss weights.

Their interaction is formalized as a **Stackelberg game**: the teacher acts as the leader (optimizing generalization objectives), while the students and generator act as followers.

### Key Designs

1. **Mutual information-based pseudo-label selection**: Rather than relying on confidence-based heuristics, the framework employs **mutual information (MI)** to quantify epistemic uncertainty. For each input, $K$ stochastic forward passes with dropout are performed to estimate the MI of the predictive distribution:

$$\text{MI}(x^{(i)}) = H[\bar{p}^{(i)}(y)] - \frac{1}{K}\sum_{k=1}^{K} H[p_{\theta_k}^{(i)}(y)]$$

Only samples with $\text{MI} > \tau_{\text{MI}}$ are accepted for cross-view supervision. MI more faithfully captures epistemic uncertainty than confidence, particularly in the early stages of training and on ambiguous samples. The cross-view unsupervised loss is:

$$\mathcal{L}_{\text{unsup}} = \mathbb{E}_{x_u}[\ell(f_1(x_u^{(1)}), \hat{y}^{(2)}) + \ell(f_2(x_u^{(2)}), \hat{y}^{(1)})]$$

2. **Entropy-driven adversarial generator**: Adversarial perturbations are constructed in the embedding space by maximizing predictive entropy plus MI:

$$\delta^{(i)*} = \arg\max_{\|\delta\|_\infty \leq \epsilon} [\mathcal{H}(f_i(x^{(i)}+\delta)) + \gamma \cdot \text{MI}(f_i(x^{(i)}+\delta))]$$

Computed via FGSM/PGD-style gradient ascent without training a generator model. The adversarial loss encourages confident predictions even in high-uncertainty regions: $\mathcal{L}_{\text{adv}} = \mathbb{E}[\mathcal{H}(f_1(x_g^{(1)})) + \mathcal{H}(f_2(x_g^{(2)}))]$.

3. **Meta-learning teacher**: Teacher parameters $\theta_T$ include $\tau_{\text{MI}}$, $\lambda_u$, and $\lambda_{\text{adv}}$ (constrained to $[0,1]$ via sigmoid). The core idea is that "a good pseudo-label strategy should lead to better student generalization on the validation set." The teacher is updated via single-step gradient unrolling:

$$\theta_T \leftarrow \theta_T - \eta_T \cdot \nabla_{\theta_T} \mathcal{L}_{\text{sup}}(f_{\theta_S - \eta \nabla_{\theta_S} \mathcal{L}_{\text{unsup}}^{\theta_T}})$$

The teacher optimizes its strategy by observing how its own decisions affect student generalization, transitioning from a static filter to an active policy learner.

### Loss & Training

The total loss is the sum of three terms: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \lambda_u \mathcal{L}_{\text{unsup}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$. Students are optimized with SGD; the teacher is updated via meta-gradient descent. The existence of a Nash equilibrium for the tripartite game is theoretically established (Theorem 1). ViT-B/16 (DINOv2 + MAE) serves as the frozen encoder, with two-layer MLPs as student heads. Training uses SGD with cosine annealing, batch size 64, for 512 epochs.

## Key Experimental Results

### Main Results

| Dataset | Setting | TRiCo | Meta Pseudo Label | FlexMatch | FixMatch |
|--------|------|-------|-------------------|-----------|----------|
| CIFAR-10 | 4k labels | **96.3** | 95.1 | 94.9 | 94.3 |
| SVHN | 1k labels | **94.2** | 93.5 | 92.7 | 92.1 |
| STL-10 | full labeled | **92.4** | 90.6 | 90.1 | 89.5 |
| ImageNet | 1% labels | **81.2** | 55.0 | 53.5 | 52.6 |
| ImageNet | 10% labels | **85.9** | 71.8 | 70.2 | 68.7 |
| ImageNet | 25% labels | **88.3** | 76.4 | 75.3 | 74.9 |

TRiCo achieves 88.3% on ImageNet with 25% labels, approaching the performance of fully supervised large models.

### Ablation Study

| Component | Accuracy | PGD Robustness | Notes |
|------|--------|----------|------|
| TRiCo (full) | **95.9** | **82.1** | All components combined |
| MI selection → confidence 0.70 | 95.0 | 77.7 | MI selection outperforms confidence |
| Fixed teacher parameters | 94.7 | 79.0 | Meta-learning regulation is necessary |
| Without generator | 94.2 | 78.9 | Adversarial training contributes ~1.7% |
| Random noise substitution | 70.5 | 66.4 | Entropy-guided perturbation far superior to random |
| 2-View only (no teacher) | 94.1 | 78.4 | Teacher contributes ~1.8% |

### Key Findings

- MI-based selection consistently outperforms all confidence threshold settings, with greater stability in the early training phase.
- Teacher parameters ($\tau_{\text{MI}}, \lambda_u, \lambda_{\text{adv}}$) exhibit smooth adaptive evolution during training—conservative early on and progressively relaxed thereafter.
- TRiCo's advantage further widens in few-shot settings (1/5/10-shot); e.g., CIFAR-100 1-shot: 23.8 vs. MCT's 21.2.
- t-SNE visualizations reveal more compact intra-class clusters and clearer inter-class separation in TRiCo's feature space.

## Highlights & Insights

- Elevating SSL from "bilateral collaboration" to a "tripartite game" represents a structural innovation—the teacher functioning as a "regulator" affords greater flexibility than serving as a "label generator" (as in Mean Teacher).
- Replacing confidence with mutual information for pseudo-label selection is theoretically more principled from an information-theoretic perspective: confidence measures a single model's "surface certainty," whereas MI measures consistency across multiple model samples.
- The combination of frozen pretrained encoders with lightweight MLP students constitutes a pragmatic architectural choice for SSL in the foundation model era.
- Formalizing the interaction as a Stackelberg game endows the method with theoretical elegance, including a proof of Nash equilibrium existence.

## Limitations & Future Work

- The framework relies on two **specific** frozen pretrained encoders (DINOv2 + MAE); the sensitivity of performance to encoder choice warrants more systematic investigation.
- Monte Carlo dropout estimation of mutual information (with $K=5$ forward passes) incurs additional computational overhead.
- The meta-learning inner-outer loop requires second-order gradient computation at each step (mitigated by a first-order approximation); scalability to large-scale tasks remains to be verified.
- Experiments are conducted primarily on image classification; extension to dense prediction tasks such as detection and segmentation has not been explored.

## Related Work & Insights

- Direct comparison with Meta Co-Training (MCT) demonstrates that TRiCo comprehensively surpasses it through the introduction of an adversarial generator and MI-based selection.
- Game-theoretic perspectives in SSL remain relatively underexplored, motivating the adoption of broader game-theoretic tools in learning paradigms.
- The teacher's meta-learning strategy is transferable to other scenarios requiring adaptive regulation of training hyperparameters, such as curriculum learning and data re-weighting.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — The combination of tripartite game, MI-based selection, and meta-learning teacher is novel, though each individual component is not entirely new
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers CIFAR/SVHN/STL/ImageNet, few-shot, OOD settings, and detailed ablations
- **Writing Quality**: ⭐⭐⭐⭐☆ — Well-structured, though the density of equations is high and the theoretical analysis could be presented more intuitively
- **Value**: ⭐⭐⭐⭐☆ — Significant gains in low-label SSL; the frozen encoder design is well-suited for practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[NeurIPS 2025\] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play](volleybots_a_testbed_for_multi-drone_volleyball_game_combining_motion_control_an.md)
- [\[ICLR 2026\] Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](../../ICLR2026/reinforcement_learning/co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agent-Based Conversational AI Defense Against LLM Jailbreaking](../../ICLR2026/reinforcement_learning/toward_a_dynamic_stackelberg_game-theoretic_framework_for_agent-based_conversat.md)

</div>

<!-- RELATED:END -->
