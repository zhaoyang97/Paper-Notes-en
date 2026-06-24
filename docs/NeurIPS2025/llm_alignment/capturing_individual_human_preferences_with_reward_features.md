---
title: >-
  [Paper Note] Capturing Individual Human Preferences with Reward Features
description: >-
  [NeurIPS 2025][LLM Alignment][Reward modeling] This paper proposes the Reward Feature Model (RFM), which learns shared reward features $\phi_\theta(x,y)$ such that each user obtains a personalized reward $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$ via a linear weight vector $\mathbf{w}_h$. The work provides the first PAC generalization bound for multi-annotator preference learning, proving that increasing the number of annotators $m$ is more effective than increasing pe…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "Reward modeling"
  - "personalized preferences"
  - "feature decomposition"
  - "multi-annotator learning"
  - "fast adaptation"
date: 2026-05-08
content_hash: c4ae3cf58685ac78
---

# Capturing Individual Human Preferences with Reward Features

**Conference**: NeurIPS 2025
**arXiv**: [2503.17338](https://arxiv.org/abs/2503.17338)  
**Code**: None  
**Area**: Alignment / RLHF
**Keywords**: Reward modeling, personalized preferences, feature decomposition, multi-annotator learning, fast adaptation

## TL;DR
This paper proposes the Reward Feature Model (RFM), which learns shared reward features $\phi_\theta(x,y)$ such that each user obtains a personalized reward $r_h = \langle \phi_\theta, \mathbf{w}_h \rangle$ via a linear weight vector $\mathbf{w}_h$. The work provides the first PAC generalization bound for multi-annotator preference learning, proving that increasing the number of annotators $m$ is more effective than increasing per-annotator sample count $n$, and that as few as 30 samples suffice for fast adaptation to new users.

## Background & Motivation

**Background**: RLHF trains a single reward function $r_\theta(x,y)$, implicitly assuming that all users share the same preference or that an average preference is a sufficient proxy.

**Limitations of Prior Work**: When user preferences are highly divergent, a single reward model is extremely inefficient — a response preferred by 51% of users may be entirely unsatisfactory to the remaining 49%. Existing personalization methods based on nonlinear MLP adaptation suffer from severe overfitting under low-data regimes.

**Key Challenge**: How can limited data be used to simultaneously learn a generalizable reward representation and enable rapid adaptation to individual user preferences? Theoretically: should one increase the number of annotators or the number of annotations per person?

**Goal**: Design a provably effective personalized reward modeling framework and provide theoretical guidance for data collection.

**Key Insight**: Assuming each user's preference can be decomposed as a linear combination of shared features, the adaptation phase reduces to convex optimization — inherently suited to low-data settings. PAC learning theory is applied to rigorously analyze how error scales with $m$ and $n$.

**Core Idea**: A shared neural network learns reward features while linear weights handle personalization. Both theory and experiments demonstrate that "diverse annotators outperform deep annotation."

## Method

### Overall Architecture
RFM decomposes the reward function as $r_{\theta, \mathbf{w}_h}(x,y) = \langle \phi_\theta(x,y), \mathbf{w}_h \rangle$. During training, shared features $\theta$ and all training users' weights $\{\mathbf{w}_h\}$ are jointly optimized. During adaptation, $\phi_\theta$ is frozen and only $\mathbf{w}_h$ is optimized for new users (convex optimization, equivalent to logistic regression).

### Key Designs

1. **Shared Reward Feature Network $\phi_\theta$**:

    - **Function**: Maps (prompt, response) pairs to a $d$-dimensional feature space.
    - **Mechanism**: Built upon Gemma 1.1 2B, with the final layer replaced by a $d$-dimensional output. All users share the backbone.
    - **Design Motivation**: The condition $e \gg d$ (backbone parameters $\gg$ feature dimensionality) ensures sufficient generality; small $d$ limits the number of adaptation parameters.

2. **Linear Personalization Weights $\mathbf{w}_h$**:

    - **Function**: A $d$-dimensional vector per user that linearly combines features to produce a personalized reward.
    - **Mechanism**: With $\phi_\theta$ fixed, $\arg\min_\mathbf{w} \sum_i \ell(\langle \Delta\phi_i, \mathbf{w} \rangle, z_i)$ is standard logistic regression (convex optimization).
    - **Design Motivation**: Convexity guarantees a global optimum, low data requirements, and known generalization bounds; weights are interpretable (each dimension corresponds to one evaluation criterion).

3. **PAC Generalization Bound (Theoretical Core)**:

    - **Function**: Analyzes generalization error in the multi-annotator setting.
    - **Mechanism**: Proposition 1 shows that the error comprises two terms — within-annotator noise ($\propto 1/n$, reducible by more samples) and cross-annotator disagreement ($\propto 1/m$, reducible only by more annotators).
    - **Design Motivation**: Provides theoretical guidance for data collection — given a fixed total budget $k = mn$, the optimal strategy is $n = 1,\ m = k$ (one sample per person, maximize annotator diversity).

### Loss & Training
- Bradley-Terry log-likelihood is minimized, jointly optimizing $\theta$ and all $\{\mathbf{w}_h\}$.
- Based on Gemma 1.1 2B; feature dimensionality $d \in \{8, 32, 128\}$.
- Adaptation phase: rapid personalization with only 30–50 preference pairs.

## Key Experimental Results

### Main Results (UltraFeedback Synthetic Users)

| Setting | Baseline (no personalization) | RFM ($d=32$) | Notes |
|---------|-------------------------------|--------------|-------|
| $m=20$, $p=0.5$ | 55.2% | 58.5% | Few annotators |
| $m=60$, $p=0.5$ | 59.1% | 63.8% | Standard setting |
| $m=256$, $p=0.5$ | 66.3% | 73.1% | Many annotators → large gain |
| $m=60$, $p=0.9$ | 78.2% | 78.8% | Homogeneous users → small gap |

### Ablation Study

| Method | Accuracy | Notes |
|--------|----------|-------|
| Baseline (no adaptation) | 55.2% | Single RM |
| Linear fine-tuning baseline | 55.8% | Non-user-aware features → ineffective |
| Park et al. nonlinear MLP | 38.5% | Severe overfitting at $\hat{n}=10$ |
| **RFM ($d=32$)** | **71.3%** | Convex optimization suited for low data |
| Gemini 1.5 Pro zero-shot | 51.2% | Large model without examples performs poorly |
| GPT-4o ($\hat{n}=10$) | 52.8% | LLM in-context preference learning is weak |

### Key Findings
- RFM performance increases significantly with the number of annotators $m$, consistent with the theoretical $O(1/\sqrt{m})$ rate.
- Adaptation accuracy with only 30 pairs approaches that with 50 pairs, confirming the feasibility of rapid personalization.
- Nonlinear adaptation methods collapse under low data (overfitting); RFM's convexity is the key advantage.
- LLMs' in-context preference learning is substantially weaker than a small RFM — consistent with the intuition that convex models are robust under limited data.
- Experiments on real reward models (leave-one-out cross-validation over 8 state-of-the-art RMs) further validate RFM's effectiveness.

## Highlights & Insights
- **First PAC bound for multi-annotator learning**: Clearly quantifies the trade-off between annotator count and sample count, providing a theoretical basis for crowdsourcing annotation strategies — "annotator diversity matters more than annotation depth."
- **Elegant linear decomposition**: Training is complex (learning high-dimensional features) while adaptation is minimal (convex optimization), perfectly matching the "abundant offline, limited online" paradigm of LLM serving.
- **Interpretability and composability**: The weight vector $\mathbf{w}_h$ directly reflects per-user preference strength and can be integrated into the successor features framework to enable multi-objective RL without retraining.
- Small convex models (RFM) substantially outperform LLM in-context learning under low data, challenging the assumption that large models are universally superior.

## Limitations & Future Work
- The linearity assumption limits expressiveness for nonlinear preferences (e.g., interaction effects between preference dimensions).
- Ecological validity of the synthetic user experiments: can real user preferences truly be linearly decomposed?
- Validation is limited to a 2B-parameter model; whether larger backbones yield better features remains unexplored.
- The adaptation phase requires users to provide paired preference annotations, which may be costly to collect in production settings.
- No end-to-end comparison with personalized RLHF training (comparisons are limited to the reward modeling stage).

## Related Work & Insights
- **vs. Standard RLHF (Ouyang et al.)**: A single reward model cannot handle user disagreement; RFM is the minimal extension — adding only linear weights.
- **vs. Nonlinear personalization (Park et al.)**: MLP adaptation collapses under few samples; RFM's convexity ensures robustness.
- **vs. VPL (Poddar et al.)**: VPL also performs personalization but provides no theoretical guidance for data collection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The linear decomposition idea is not new, but the PAC bound for multi-annotator preference learning is presented for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive synthetic and real RM experiments, though end-to-end RLHF validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations and well-designed experiments (controlled synthetic user setup).
- **Value**: ⭐⭐⭐⭐ — Directly actionable for personalized RLHF; the "diverse annotators" conclusion has practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](../../ICLR2026/llm_alignment/learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[ICML 2026\] Large Language Models Should Learn Personalized Rather Than Aggregated Human Preferences](../../ICML2026/llm_alignment/large_language_models_should_learn_personalized_rather_than_aggregated_human_pre.md)
- [\[NeurIPS 2025\] Can DPO Learn Diverse Human Values? A Theoretical Scaling Law](can_dpo_learn_diverse_human_values_a_theoretical_scaling_law.md)

</div>

<!-- RELATED:END -->
