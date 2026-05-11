---
title: >-
  [Paper Note] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization
description: >-
  [NeurIPS 2025][Reinforcement Learning][contextual MDP] DALI, a self-supervised context encoder, is introduced into the DreamerV3 architecture to infer latent environment parameters (e.g., gravity…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "contextual MDP"
  - "world model"
  - "zero-shot generalization"
  - "DreamerV3"
  - "latent context"
date: 2026-05-08
content_hash: 48bf2a51d7402bc8
---

# Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization

**Conference**: NeurIPS 2025  
**arXiv**: [2508.20294](https://arxiv.org/abs/2508.20294)  
**Code**: [github.com/frankroeder/DALI](https://github.com/frankroeder/DALI)  
**Area**: Reinforcement Learning  
**Keywords**: contextual MDP, world model, zero-shot generalization, DreamerV3, latent context

## TL;DR
DALI, a self-supervised context encoder, is introduced into the DreamerV3 architecture to infer latent environment parameters (e.g., gravity, friction) from interaction history. It achieves zero-shot generalization on cMDP benchmarks without retraining, outperforming ground-truth context-aware baselines by up to 96.4% on extrapolation tasks.

## Background & Motivation

**Background**: Contextual Markov Decision Processes (cMDPs) model environment variations via latent parameters (gravity, friction, actuator strength, etc.). Most existing methods rely on explicit context variable inputs, which are effective in controlled settings but difficult to scale.

**Limitations of Prior Work**: (a) Explicit context annotations are costly or infeasible to obtain in practice; (b) DreamerV3's RSSM compresses all information into a fixed-size GRU hidden state $h_t$, creating an information bottleneck where context signals may be lost amid competition from dynamic states and noise; (c) RSSM requires full episode-length $T$ interactions to reliably identify context, leading to slow adaptation.

**Key Challenge**: DreamerV3's recurrent state simultaneously handles dynamics modeling and context inference, and its limited capacity causes mutual interference between the two objectives.

**Goal**: Design a decoupled context inference module that efficiently extracts context representations from short interaction windows, enabling the world model and policy to generalize zero-shot to unseen contexts.

**Key Insight**: Forward dynamics prediction is used as a self-supervised signal — if a context representation can accurately predict the next observation, it necessarily encodes the key parameters governing the dynamics.

**Core Idea**: A dedicated context encoder is trained via self-supervised forward dynamics alignment, decoupling context inference from dynamics modeling and endowing DreamerV3 with zero-shot contextual generalization.

## Method

### Overall Architecture
DALI augments DreamerV3 with a Transformer-based context encoder $g_\varphi$. Given a history of observations and actions $(o_{t-K:t}, a_{t-K:t-1})$ of length $K$, the encoder produces a context representation $\mathfrak{z}_t \in \mathbb{R}^8$. This representation is injected into both the world model and the actor-critic, enabling context-awareness in imagined rollouts and policy learning.

### Key Designs

1. **Forward Dynamics Alignment**:

    - Function: Trains the context encoder so that its representations support accurate dynamics prediction.
    - Mechanism: Jointly trains the encoder $g_\varphi$ and predictor $f_\varphi^w$ by minimizing the forward dynamics loss $L_{\text{FD}}(\varphi) = \mathbb{E}\|o_{t+1} - f_\varphi^w(o_t, a_t, \mathfrak{z}_t)\|_2^2$.
    - Design Motivation: Forward dynamics are the most direct function of context — different gravity or friction values directly produce different state transitions. By predicting the next observation, the encoder is forced to distill context factors affecting dynamics into $\mathfrak{z}_t$.

2. **Cross-Modal Regularization**:

    - Function: Bidirectionally aligns the context representation $\mathfrak{z}_t$ with the RSSM posterior state $z_t$.
    - Mechanism: $L_{\text{cross}}(\varphi) = \mathbb{E}\|z_t - W_z\mathfrak{z}_t\|_2^2 + \mathbb{E}\|\mathfrak{z}_t - W_\mathfrak{z}z_t\|_2^2$, where $W_z, W_\mathfrak{z}$ are linear projections.
    - Design Motivation: (a) Aligning with $z_t$ rather than the full state $s_t = \{h_t, z_t\}$ avoids encoding redundant trajectory-specific information from $h_t$; (b) bidirectional constraints prevent degenerate solutions (e.g., $\mathfrak{z}_t$ collapsing to a constant); (c) the total loss is $L_{\text{total}} = L_{\text{FD}} + \lambda_{\text{cross}}L_{\text{cross}}$.

3. **Shallow vs. Deep Integration**:

    - **Shallow Integration**: $\mathfrak{z}_t$ is concatenated only to the world model encoder input $z_t \sim q_\theta(z_t|h_t, o_t, \mathfrak{z}_t)$, leaving all other components unchanged. Context propagates indirectly to $h_t$ through the recurrence.
    - **Deep Integration**: $\mathfrak{z}_t$ is injected into all components — the sequence model $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}, \mathfrak{z}_t)$, reward/continue predictors, and the actor-critic.
    - Experiments show that shallow integration performs better, acting as implicit regularization and preventing overfitting to noisy $\mathfrak{z}_t$.

4. **Gradient Stopping Strategy**:

    - Function: Decouples context learning from world model recurrent updates.
    - Mechanism: Gradients through $h_\tau$ and $z_\tau$ are stopped in the recurrent dynamics, and gradients through $h_\tau$ are stopped in the encoder; only $\mathfrak{z}_\tau$ retains gradients in $L_{\text{FD}}$ and $L_{\text{cross}}$ for updating $\varphi$.
    - Design Motivation: Prevents the training signal from the context encoder from interfering with world model learning.

### Loss & Training
The total loss comprises three components: the original DreamerV3 world model loss, the actor-critic loss, and the context encoder loss ($L_{\text{FD}} + \lambda_{\text{cross}}L_{\text{cross}}$). The DreamerV3 small variant is used, with a Transformer-based context encoder and window $K=50$.

## Key Experimental Results

### Main Results: DMC Ball-in-Cup Zero-Shot Generalization IQM

| Method | Interpolation (Feature) | Extrapolation (Feature) | Extrapolation (Pixel) | Mixed (Feature) |
|--------|------------------------|------------------------|-----------------------|-----------------|
| Dreamer-DR | 0.93 | 0.198 | 0.139 | 0.452 |
| cRSSM-S (ground-truth) | 0.93 | 0.227 | 0.187 | 0.564 |
| cRSSM-D (ground-truth) | 0.94 | 0.278 | 0.242 | 0.670 |
| **DALI-S-χ** | **0.949** | **0.372** | **0.273** | **0.683** |

| Method | Ball-in-Cup Extrapolation Gain (Feature) | Ball-in-Cup Extrapolation Gain (Pixel) |
|--------|------------------------------------------|----------------------------------------|
| vs Dreamer-DR | +87.9% | +96.4% |
| vs cRSSM-S | +63.9% | +45.9% |
| vs cRSSM-D | +33.8% | +12.8% |

### Walker Walk Zero-Shot Generalization IQM

| Method | Interpolation (Feature) | Extrapolation (Feature) | Extrapolation (Pixel) |
|--------|------------------------|------------------------|-----------------------|
| Dreamer-DR | 0.96 | 0.751 | 0.734 |
| cRSSM-S | 0.94 | 0.702 | 0.777 |
| cRSSM-D | 0.95 | 0.749 | 0.755 |
| **DALI-S** | **0.971** | **0.781** | 0.758 |

### Key Findings
- **Inferred context > ground-truth context**: DALI surpasses cRSSM baselines that use ground-truth context variables on extrapolation tasks, suggesting that ground-truth context may cause overfitting to the training distribution.
- **Task-dependent role of cross-modal regularization**: Ball-in-Cup (nonlinear pendulum dynamics) benefits from $L_{\text{cross}}$ (DALI-S-χ performs better), whereas Walker (linear torque scaling) does not (DALI-S performs better), with the forward dynamics loss alone being sufficient.
- **Counterfactual consistency**: Perturbing specific latent dimensions (e.g., $\mathfrak{z}_6$) yields physically consistent counterfactual trajectories — higher gravity leads to faster swinging, shorter string length leads to smaller amplitude.
- **Sample complexity gain**: Theoretical analysis shows DALI requires only $\mathcal{O}(K/\delta^2)$ transitions to infer context, compared to $\mathcal{O}(T/\delta^2)$ for DreamerV3, yielding a gain of $\mathcal{O}(T/K)$.

## Highlights & Insights
- **Advantages of decoupled design**: Separating context inference from the world model's recurrent state allows each module to focus on its own objective. This design principle generalizes to any model-based RL method that must handle latent variables.
- **Self-supervision outperforms supervision**: The finding that inferred context representations generalize better than ground-truth context during extrapolation is profound — the learned representations encode dynamics-relevant information rather than the absolute position in parameter space.
- **Shallow integration as regularization**: The simpler design (injecting $\mathfrak{z}_t$ only at the encoder input) proves more robust than deep integration, suggesting that excessive reliance on context signals may be harmful in OOD settings.

## Limitations & Future Work
- **Limited environment scope**: Validation is restricted to only two DMC tasks (Ball-in-Cup and Walker Walk), with no evaluation on high-dimensional observations or long-horizon tasks.
- **$\beta$-mixing assumption**: Theoretical results depend on the $\beta$-mixing assumption, which may not hold under slowly mixing dynamics (e.g., highly correlated trajectories).
- **Dependence on exploratory policies**: Context inference requires sufficiently exploratory policies to generate distinguishable trajectories, which may fail in sparse-reward or high-dimensional settings.
- **Fixed context dimensionality**: The current 8-dimensional context representation may be insufficient for more complex context spaces.

## Related Work & Insights
- **vs DreamerV3 + Domain Randomization**: DALI's core advantage lies in explicit context inference, whereas DR can only implicitly accumulate context information through the GRU, resulting in poor extrapolation.
- **vs cRSSM (Prasanna et al. 2024)**: cRSSM directly inputs ground-truth context, which is effective within the training distribution but overfits during extrapolation. DALI learns more generalizable representations.
- **vs Meta-RL (MAML, RL²)**: Meta-RL methods require fine-tuning on new tasks, whereas DALI achieves zero-shot generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a self-supervised context encoder with DreamerV3 is elegant and effective; theoretical analysis strengthens the contribution.
- Experimental Thoroughness: ⭐⭐⭐ Limited to two DMC tasks, but covers diverse generalization settings and counterfactual analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with good integration of theory and experiments; ablations are comprehensive.
- Value: ⭐⭐⭐⭐ Demonstrates the potential of self-supervised context inference in model-based RL; the finding that inferred context outperforms ground-truth is particularly insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)

</div>

<!-- RELATED:END -->
