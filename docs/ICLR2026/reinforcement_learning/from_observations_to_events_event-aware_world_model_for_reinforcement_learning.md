---
title: >-
  [Paper Note] From Observations to Events: Event-Aware World Model for Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][World Model] This paper proposes the Event-Aware World Model (EAWM), a general framework that automatically generates events from raw observations and learns event-aware representation…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "World Model"
  - "Event-Aware"
  - "Model-Based Reinforcement Learning"
  - "Representation Learning"
  - "Sample Efficiency"
date: 2026-05-08
content_hash: 9dbb576cdfda1426
---

# From Observations to Events: Event-Aware World Model for Reinforcement Learning

**Conference**: ICLR 2026  
**arXiv**: [2601.19336](https://arxiv.org/abs/2601.19336)  
**Code**: [https://github.com/MarquisDarwin/EAWM](https://github.com/MarquisDarwin/EAWM)  
**Area**: Reinforcement Learning  
**Keywords**: World Model, Event-Aware, Model-Based Reinforcement Learning, Representation Learning, Sample Efficiency

## TL;DR

This paper proposes the Event-Aware World Model (EAWM), a general framework that automatically generates events from raw observations and learns event-aware representations without manual annotations, improving existing MBRL baselines by 10%–45% and achieving new state-of-the-art results on Atari 100K, Craftax 1M, DeepMind Control 500K, and DMC-GB2 500K.

## Background & Motivation

Model-based reinforcement learning (MBRL) improves sample efficiency by learning world models, yet existing methods generalize poorly across structurally similar scenarios and are susceptible to spurious variations such as texture and color changes.

From a cognitive science perspective, humans segment continuous sensory streams into discrete **events** and rely on these key events for decision-making. Neurobiological research reveals that neurons in the superior colliculus respond specifically to changes in visual scenes—such as approaching predators or moving targets—enabling rapid behavioral responses. This finding suggests that animals often **respond directly to dynamic events** rather than relying on static observations.

Nevertheless, existing world models place excessive emphasis on observation prediction and tend to fail when confronted with novel objects or dynamics absent from training data. The authors argue that the human brain predicts **events** rather than future observations, motivating the construction of world models that capture motion-related features.

## Method

### Overall Architecture

EAWM is a **general framework** that can be stacked on top of different world model architectures. The paper presents a unified formulation of world models comprising five core components:

1. **Sequence Model**: $\mathbf{h}_t, \mathbf{y}_t = \mathbf{F}_\theta(\mathbf{h}_{t-1}, \mathbf{Z}_{t-1}, \mathbf{A}_{t-1})$
2. **Representation Model**: $\mathbf{z}_t \sim q_\theta(\mathbf{z}_t | \mathbf{o}_t, \mathbf{h}_t)$
3. **Dynamics Predictor**: $\hat{\mathbf{z}}_t \sim p_\theta(\hat{\mathbf{z}}_t | \mathbf{y}_t)$
4. **Reward Predictor**
5. **Continuation Predictor**

Building upon this foundation, EAWM adds an **observation predictor** and an **event predictor** to form event-aware representation learning. Crucially, the event predictor can leverage the outputs of the world model, making the framework theoretically compatible with any world model architecture. The paper instantiates two variants: EADream (based on DreamerV3) and EASimulus (based on Simulus).

### Key Designs

1. **Automated Event Generator**: Generates events from raw observations without manual annotation.

    - **Visual inputs**: An Adaptive Gaussian Mixture Model (AGMM) defines events as statistically significant deviations from a learned multimodal distribution, rather than raw pixel differences. Mahalanobis distance is used to determine whether a new observation constitutes an event, effectively filtering noise and slow luminance changes.
    - **Ordinal data** (e.g., joint angles, velocities): An event is triggered when the normalized difference exceeds a threshold.
    - **Nominal data** (e.g., categorical inputs): A change in category constitutes an event.
    - This modular design enables flexible adaptation to diverse observation modalities.

2. **Generic Event Segmentor (GES)**: Automatically detects **event boundaries**—the start and end points of meaningful observation segments.

    - Core idea: When events are overly dense (e.g., during sudden vibrations), excessive attention to transient fluctuations distracts from key events.
    - GES compares the event frequency ratio $\alpha_t^{(m)}$ against a threshold $\alpha_{\text{thr}}^{(m)}$ to determine whether an event boundary is present.
    - No additional trainable parameters are introduced; GES is implemented as a deterministic function of events.
    - At event boundaries, event prediction is suppressed and the world model redirects attention from events back to raw observations.

3. **Event Predictor**: Dedicated decoder networks are designed for each modality, with stop-gradient applied to prevent gradient flow into the targets.

    - Cross-entropy loss is used for ordinal data.
    - Focal loss is used for visual and nominal data.
    - Through information bottleneck optimization, the representation space is implicitly constrained to focus on meaningful spatiotemporal transitions.

### Loss & Training

The total loss is:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{WM}}(\theta) + \beta_o \mathcal{L}_o(\theta) + \beta_e \mathcal{L}_e(\theta)$$

where $\mathcal{L}_{\text{WM}}$ is the base world model loss, $\mathcal{L}_o$ is the event-aware observation prediction loss (GES guides the model to increase observation prediction weight at event occurrences), and $\mathcal{L}_e$ is the event prediction loss.

Behavior learning is conducted entirely on imagined trajectories; the event and observation predictors are not involved during policy inference, introducing no additional overhead at deployment.

## Key Experimental Results

### Main Results

| Benchmark | Metric | EASimulus | Simulus | Gain |
|-----------|--------|-----------|---------|------|
| Atari 100K | Mean HNS | 1.818 | 1.609 | +13% |
| Atari 100K | IQM | 1.004 | 0.913 | +10% |

| Benchmark | Metric | EADream | DreamerV3 | Gain |
|-----------|--------|---------|-----------|------|
| DMC 500K | Mean Score | 723.8 | 623.3 | +16% |
| DMC-GB2 500K | Mean Score | 606 | 400 | +45% |
| Craftax 1M | Score | 7.23% | 6.57% | +10% |

**Highlights**:
- EASimulus is the first MBRL method to achieve **superhuman-level** IQM (1.004) on Atari 100K.
- EADream on DMC-GB2 even surpasses SADA, a method specifically designed for that benchmark.
- Breakout improves by 55%; Acrobot Swingup improves by 115%.

### Ablation Study

| Configuration | Mean HNS (Atari 6 games) | Note |
|---------------|--------------------------|------|
| Full EAWM | Best | All components working together |
| w/o Event Predictor | −~0.4 | Event prediction is critical for motion-dependent tasks |
| w/o GES | Slow growth, high variance | GES stabilizes training, especially for continuous control |
| w/o Observation Prediction | DMC mean 737.2→519.5 | Event and observation prediction must be coupled |
| DreamerV3+RSSM-OP | Mean 1.951 vs EADream 2.501 | Performance gains stem primarily from joint modeling |

### Key Findings

- Event prediction is intrinsically **simpler** than observation prediction, as it abstracts away redundant information.
- In unseen visual environments, observation prediction fails severely, while event prediction maintains strong generalization.
- On Atari 1M, EADream achieves a Mean HNS of 5.273 (vs. DreamerV3's 2.213), indicating that EAWM's advantage grows with more data.
- Additional computational overhead is approximately 11%–13%.

## Highlights & Insights

1. **Cognitive-science-driven design**: Grounding the framework in the event segmentation theory of human cognition and introducing event prediction into world models is a novel perspective with solid neuroscientific support (superior colliculus neurons responding to dynamic changes).
2. **Strong generality**: EAWM is not a standalone architecture but a plug-in framework compatible with arbitrary world models, demonstrated to be effective on both DreamerV3 and Simulus.
3. **Multi-modal event handling**: A unified approach handles visual, ordinal, and nominal data modalities without benchmark-specific threshold tuning.
4. **Unsupervised event generation**: No manual annotation is required; AGMM automatically detects statistically significant changes, making the framework directly applicable to diverse environments.
5. **DMC-GB2 generalization**: GES achieves strong visual generalization by focusing on task-relevant motion features while ignoring background distractors.

## Limitations & Future Work

1. GES is deliberately kept simple; future work could model the GES function with a neural network to achieve both efficiency and expressiveness simultaneously.
2. Although EADream and EASimulus are trained with fixed hyperparameters across domains, developing a unified multi-task model remains a challenge.
3. Combining EAWM with large-scale pretrained vision-language models is a promising direction.
4. Application to model-free RL has not been explored.

## Related Work & Insights

- **DreamerV3/Simulus**: The two base world models underlying EAWM, representing the RSSM-based and token-based families, respectively.
- **DyMoDreamer**: The most closely related prior work, which augments DreamerV3 by encoding inter-frame difference masks; the proposed method is more general—predicting future events rather than merely using them as inputs.
- **HarmonyDream**: Improves DreamerV3 via harmonized losses but does not introduce the concept of events.
- **TD-MPC2**: Performs well in state-space settings but struggles with image-based inputs.

The broader implication for event-driven RL: discretizing continuous perceptual streams into meaningful event segments may be a key direction for improving the robustness and generalization of RL agents.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[AAAI 2026\] Object-Centric World Models for Causality-Aware Reinforcement Learning](../../AAAI2026/reinforcement_learning/object-centric_world_models_for_causality-aware_reinforcement_learning.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)

</div>

<!-- RELATED:END -->
