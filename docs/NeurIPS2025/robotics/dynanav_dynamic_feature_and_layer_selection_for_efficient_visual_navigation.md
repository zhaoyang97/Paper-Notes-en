---
title: >-
  [Paper Note] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation
description: >-
  [NeurIPS 2025][Robotics][visual navigation] DynaNav is proposed to dynamically adjust feature and layer usage according to scene complexity via a trainable hard feature selector and a Bayesian optimization-based early-exit mechanism, achieving a 2.26× FLOPs reduction and 42.3% inference time decrease in visual navigation while maintaining or improving navigation performance.
tags:
  - NeurIPS 2025
  - Robotics
  - visual navigation
  - dynamic inference
  - early exit
  - feature selection
  - efficient deployment
date: 2026-05-08
content_hash: d09c59a50d0651c5
---

# DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation

**Conference**: NeurIPS 2025
**arXiv**: [2509.21930](https://arxiv.org/abs/2509.21930)
**Code**: To be confirmed
**Area**: Robotics
**Keywords**: visual navigation, dynamic inference, early exit, feature selection, efficient deployment

## TL;DR

DynaNav is proposed to dynamically adjust feature and layer usage according to scene complexity via a trainable hard feature selector and a Bayesian optimization-based early-exit mechanism, achieving a 2.26× FLOPs reduction and 42.3% inference time decrease in visual navigation while maintaining or improving navigation performance.

## Background & Motivation

Visual navigation is a core capability in robotics and embodied AI. Foundation models such as ViNT and NoMaD have demonstrated strong cross-platform and cross-environment generalization, yet they rely on deep Transformer decoders with substantial computational overhead that makes deployment on edge devices challenging. Moreover, these models operate as black boxes, lacking interpretability.

Inspired by the human visual system—where the brain does not activate all neurons for every visual task but dynamically allocates resources according to task complexity—the authors pose two key research questions:

1. Does every navigation scene require activating all Transformer layers?
2. Which features are most important during decoding, and can the regions or pixels most critical to navigation be identified?

## Core Problem

How to substantially reduce the computational overhead of visual navigation foundation models without sacrificing navigation performance, while simultaneously improving model interpretability?

## Method

### Overall Architecture

DynaNav builds upon an EfficientNet-B0 encoder and Transformer decoder architecture, introducing two dynamic mechanisms:

1. **Dynamic Feature Selector**: generates a sparse mask before features enter the Transformer decoder.
2. **Dynamic Layer Inference**: terminates computation early based on scene complexity via an early-exit strategy.

### Feature Extraction

Two EfficientNet-B0 instances are employed:
- One processes a sequence of consecutive observation frames $\mathbf{o}_{t-p:t}$, extracting $\psi(\mathbf{o}_i) \in \mathbb{R}^{H \times W \times C}$.
- The other applies an early-fusion strategy to process the concatenation of the current observation and goal image $\phi([\mathbf{o}_t; \mathbf{o}_s])$.

### Dynamic Hard Feature Selector

This is one of the core contributions. The selector $f(\cdot)$ is a Gumbel-Softmax-based classification network that:

1. Projects encoded features via an MLP into $\mathbb{R}^{H \times W \times C \times 2}$ space.
2. Applies pixel-wise Gumbel-Softmax per channel for each pixel to compute selection probabilities.
3. Generates a binary mask $\mathbf{m}_i \in \mathbb{R}^{H \times W}$ that filters out pixels irrelevant to navigation prediction.

The temperature parameter $\tau$ controls the "hardness" of selection. During training, the selector progressively learns to filter redundant features, and the resulting saliency maps intuitively visualize the regions attended to by the model, enhancing interpretability.

### Dynamic Transformer Layer Inference (Early Exit)

**Feature-Aware Early Exit Strategy**:

- At each intermediate decoder layer, an action consistency condition is used to determine whether to exit early: $|h(\mathbf{x}_i) - h(\mathbf{x}_{i-1})|_2 \leq \eta_i$.
- A more aggressive strategy bypasses the entire Transformer decoder when the L2 difference between the goal state and current observation falls below a threshold.
- The feature selector is integrated with early exit, using the number of masked features to assist in determining the exit condition.

**Adaptive Threshold Optimization**: Bayesian Optimization is used to determine the optimal early-exit thresholds $\eta = \{\eta_1, \eta_2, \dots, \eta_N\}$. The optimization objective is to maximize the cosine similarity between predicted and ground-truth actions while satisfying three constraints:

- Inference time constraint: average inference time $\leq \mathcal{T}_{\max}$
- GPU memory constraint: peak memory $\leq G_{\max}$
- FLOPs constraint: average computation $\leq F_{\max}$

### Loss & Training

The training loss is a joint likelihood over action prediction and waypoint distance prediction:

$$\mathcal{L} = \mathbb{E}[\log p(\mathbf{a}_t^{\text{gt}} | \mathbf{a}_t) + \lambda \log p(\mathbf{w}_t^{\text{gt}} | \mathbf{w}_t)]$$

During training, intermediate-layer early exits are triggered randomly to improve robustness.

## Key Experimental Results

### Benchmark Datasets (Four Real-World Datasets)

Comparisons against ViNT and NoMaD on Recon, Go-Stanford, SACSoN, and SCAND:

| Metric | DynaNav vs. ViNT |
|--------|-----------------|
| FLOPs | ~58% reduction (avg. 1.93 vs. 4.37 × 10⁹) |
| Inference Time | 42.3% reduction (0.228s vs. 0.379s/traj) |
| Memory | 32.8% reduction (13.35 vs. 19.07 Gb) |
| Action Similarity | +0.83% (avg.) |
| Waypoint Similarity | +0.28% (avg.) |

NoMaD achieves slightly higher accuracy than DynaNav (~0.2%), but requires approximately 4× the FLOPs.

### CARLA Simulation

Evaluated on Town02 (easy), Town03 (medium), and Town10 (hard):

- DynaNav achieves comparable success rates to ViNT (0.727 vs. 0.724 / 0.664 vs. 0.659 / 0.588 vs. 0.589).
- FLOPs are reduced by more than 2×.
- As environment complexity increases, DynaNav's FLOPs automatically increase (1.58 → 1.70 → 1.93 × 10⁹), validating the rationality of dynamic inference.

### Ablation Study

- Feature selector alone: performance improves, efficiency slightly improves.
- Dynamic decoder alone: efficiency improves significantly, accuracy slightly decreases.
- Both combined: optimal efficiency and accuracy simultaneously.
- Bayesian optimization is critical for threshold determination; early exit degrades without it.
- The feature selector increases the frequency of layer-skipping in early exit (2–4 layer skips become more frequent).

## Highlights & Insights

1. **First introduction of dynamic network mechanisms into visual navigation**: a pioneering work applying dynamic inference to the navigation domain.
2. **Simultaneous gains in efficiency and performance**: a 2.26× FLOPs reduction is achieved while performance improves, breaking the conventional efficiency–accuracy trade-off.
3. **Feature selection enhances interpretability**: visualized masks clearly show the model's attended regions, revealing that the model does not simply focus on the largest common objects between observation and goal.
4. **Synergy between feature selection and early exit**: sparse features stabilize early exit and increase layer-skipping frequency.
5. **Adaptive complexity awareness**: simpler indoor scenes automatically consume less computation, while more complex outdoor scenes are automatically allocated more resources.

## Limitations & Future Work

1. **Additional optimization overhead**: Bayesian optimization requires a post-training step, increasing manual effort and pipeline complexity.
2. **Threshold generalizability**: a unified threshold is used across all three CARLA scenes; adaptive thresholds for different environments are not explored.
3. **Encoder remains static**: the EfficientNet encoder is kept fixed; dynamic inference at the encoding stage is not explored.
4. **RGB-only input**: dynamic selection for multimodal sensor inputs such as depth or LiDAR is not considered.
5. **Future directions**: integrating Bayesian optimization with training in parallel to realize a truly end-to-end dynamic inference system.

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|----------------|-------------|
| ViNT | Large-scale cross-platform navigation foundation model | Static inference; all layers fully activated; high computational cost |
| NoMaD | Diffusion policy + goal masking; high accuracy | FLOPs ~4× those of DynaNav; cannot run in real-time simulation |
| GNM | Learns navigation policies from heterogeneous RGB datasets | Insufficient generalization; success rate significantly lower than DynaNav |
| DeeR-VLA | Dynamic inference for multimodal LLMs | Still activates multiple layers; limited computational savings |
| **DynaNav** | Dynamic feature selection + early exit + Bayesian optimization | Requires additional optimization step; limited threshold generalizability |

**Broader implications**:

1. The dynamic inference paradigm is generalizable to other embodied AI tasks (grasping, manipulation, etc.) for computation scaling with task difficulty.
2. The Gumbel-Softmax hard feature selection scheme is applicable to other vision tasks requiring interpretable sparse attention.
3. The Bayesian optimization approach for early-exit threshold determination is transferable to LLM inference acceleration.
4. The co-design of feature selection and early exit merits exploration in VLMs and multimodal models.

## Rating
- Novelty: 8/10 (first application of dynamic networks to visual navigation; creative co-design of feature selection and early exit)
- Experimental Thoroughness: 8/10 (four real-world datasets + CARLA simulation + detailed ablation; lacks broader simulation scenarios and real-robot deployment)
- Writing Quality: 7/10 (clear structure, but some sections are formula-heavy and the early-exit strategy description could be more intuitive)
- Value: 8/10 (directly practical for edge-deployed navigation models; opens a new direction for efficiency optimization of navigation foundation models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STRNet: Visual Navigation with Spatio-Temporal Representation through Dynamic Graph Aggregation](../../CVPR2026/robotics/strnet_visual_navigation_with_spatio-temporal_representation_through_dynamic_gra.md)
- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](../../CVPR2026/robotics/force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)
- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)

</div>

<!-- RELATED:END -->
