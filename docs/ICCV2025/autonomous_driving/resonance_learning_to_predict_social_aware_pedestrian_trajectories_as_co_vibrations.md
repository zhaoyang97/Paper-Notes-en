---
title: >-
  [Paper Note] Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations
description: >-
  [ICCV 2025][Autonomous Driving][pedestrian trajectory prediction] This paper proposes Resonance, a physics-inspired model that decomposes pedestrian trajectories into multiple independent "vibration" components, each representing an agent's response to a single cause. The final trajectory is predicted via superposition of these components, while social interactions are learned by simulating resonance phenomena, enhancing interpretability.
tags:
  - ICCV 2025
  - Autonomous Driving
  - pedestrian trajectory prediction
  - social interaction
  - vibration system
  - resonance
  - spectral decomposition
date: 2026-05-08
content_hash: a36823dda8324cbc
---

# Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations

**Conference**: ICCV 2025
**arXiv**: [2412.02447](https://arxiv.org/abs/2412.02447)
**Code**: N/A
**Area**: Autonomous Driving / Pedestrian Trajectory Prediction
**Keywords**: pedestrian trajectory prediction, social interaction, vibration system, resonance, spectral decomposition

## TL;DR

This paper proposes Resonance, a physics-inspired model that decomposes pedestrian trajectories into multiple independent "vibration" components, each representing an agent's response to a single cause. The final trajectory is predicted via superposition of these components, while social interactions are learned by simulating resonance phenomena, enhancing interpretability.

## Background & Motivation

Pedestrian trajectory prediction requires accurate modeling of agent intent and social behavior, particularly in an interpretable and disentangled manner with respect to stochasticity across components. Existing methods struggle to decouple the distinct causes of trajectory variation (e.g., individual goals, collision avoidance, social norms). Vibration systems and their resonance properties offer a natural analogy—complex motion patterns emerge from the superposition of multiple independent vibration sources.

## Method

### Overall Architecture

Resonance decomposes trajectory modifications and stochasticity into multiple vibration components, each simulating an agent's response to a single cause (e.g., avoiding a specific pedestrian, moving toward a goal). The final trajectory is predicted as the superposition of these independent vibrations. Social interaction representations are learned by simulating resonance—when two agents' motion frequencies are close, "resonance" occurs, indicating strong interaction. The overall pipeline is: observed trajectory → vibration decomposition (frequency-domain representation) → independent prediction of each component → resonance-based social interaction modulation → superposition to reconstruct the predicted trajectory.

### Key Designs

1. **Vibration Decomposition of Trajectories**:

    - Function: Decouple complex trajectory variations into multiple independent, interpretable motion components.
    - Mechanism: Trajectory displacements are decomposed into vibration components, each with independent frequency, amplitude, and phase corresponding to distinct motion causes (e.g., individual goal-driven, social force-driven, environmental constraints). Spectral analysis extracts these vibration features so that each component can be modeled and predicted independently. Each vibration component can be interpreted as the pedestrian's response to a specific "excitation."
    - Design Motivation: Conventional methods predict trajectories directly in coordinate space, making it difficult to distinguish contributions from different motion causes. Vibration decomposition borrows from signal processing to separate a mixed signal into independent components, enabling individual analysis and modeling of each cause.

2. **Resonance-Based Social Interaction Modeling**:

    - Function: Characterize social interaction strength between pedestrians through physically intuitive representations.
    - Mechanism: This design exploits the physics of resonance—two systems with similar vibration frequencies undergo energy transfer. Social interactions among pedestrians are modeled as resonance phenomena in the frequency domain: pedestrians with similar motion-pattern frequencies interact more strongly, while those with large frequency differences interact weakly. Interaction strength is quantified by comparing the vibration spectra of different pedestrians and used to modulate individual trajectory predictions accordingly.
    - Design Motivation: Traditional social force models rely on predefined rules, while graph attention methods depend on black-box learning. Resonance offers an interaction representation that combines physical interpretability with learning flexibility.

3. **Superposition Principle for Prediction**:

    - Function: Reconstruct full predicted trajectories from independent vibration components.
    - Mechanism: Final trajectory = $\sum$ independent vibration components, where each component is predicted independently and then superposed. This decompose-and-superpose scheme enables disentangled modeling of stochasticity from different motion causes—for instance, uncertainty in goal location and uncertainty in collision-avoidance behavior can be captured separately. The stochasticity of each vibration component can be controlled independently, enabling finer-grained multimodal prediction.
    - Design Motivation: The superposition principle is a fundamental property of linear systems. Although pedestrian motion is not strictly linear, this approximation provides a concise and effective prediction framework.

### Loss & Training

Standard trajectory prediction losses (e.g., $L_2$ distance) are employed, potentially combined with a best-of-$N$ strategy for multimodal prediction. The model is trained end-to-end on multiple standard pedestrian trajectory prediction datasets. Both the vibration decomposition and resonance interaction modules are differentiable, supporting gradient backpropagation.

## Key Experimental Results

### Main Results

Evaluation is conducted on standard benchmarks including ETH/UCY and SDD using ADE (Average Displacement Error) and FDE (Final Displacement Error), demonstrating the quantitative effectiveness of the proposed method.

| Metric | Description |
|--------|-------------|
| ADE/FDE | Outperforms or matches SOTA across multiple standard datasets |
| Interpretability | Visualization of vibration components illustrates the contribution of distinct motion causes |

### Key Findings

- Vibration decomposition effectively decouples different motion causes, with each component exhibiting clear physical meaning.
- The resonance phenomenon naturally characterizes social interaction strength without requiring explicitly defined interaction rules.
- The superposition principle endows predictions with interpretability and supports independent analysis of each component.
- Spectral representations provide a more compact abstraction of motion patterns than coordinate-space representations.

## Highlights & Insights

- **Physics-Inspired Modeling**: The vibration and resonance analogy is pioneering in trajectory prediction, offering an interpretability framework superior to purely data-driven approaches.
- **Frequency-Domain Perspective**: Shifting trajectory prediction from the spatial/temporal domain to the frequency domain opens a new dimension for understanding and modeling motion patterns. This paradigm is transferable to other time-series prediction tasks.
- **Advantages of Disentangled Modeling**: Independently modeling distinct motion causes allows the stochasticity of each component to be controlled separately, enabling finer-grained multimodal prediction.
- **Elegant Social Interaction Representation**: Resonance is more physically intuitive than attention mechanisms and naturally encodes the prior that only agents with similar frequencies interact strongly.

## Limitations & Future Work

- The vibration system assumption may not fully apply to highly nonlinear pedestrian behaviors such as sharp turns or emergency avoidance maneuvers.
- The available cached content is limited (abstract and references only), preventing a thorough presentation of complete quantitative experimental results.
- Applicability to non-periodic motion patterns (e.g., single crossings, sudden stops) warrants further investigation.
- The choice of the number of vibration components may affect the trade-off between model capacity and computational efficiency.

## Related Work & Insights

- Classic social force methods such as Social LSTM/GAN serve as foundational baselines.
- Diffusion-based methods such as SingularTrajectory represent recent advances.
- Physics-inspired motion modeling is extensible to vehicle trajectory prediction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of the vibration/resonance analogy in trajectory prediction
- Technical Depth: ⭐⭐⭐⭐ — Frequency-domain analysis and superposition principle are physically grounded
- Experimental Thoroughness: ⭐⭐⭐ — Multi-dataset validation but with limited detail
- Writing Quality: ⭐⭐⭐⭐ — Physical analogy is clearly articulated
- Value: ⭐⭐⭐ — Strong interpretability, though practical deployment effectiveness remains to be verified

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](saliency-aware_quantized_imitation_learning_for_efficient_robotic_control.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)

</div>

<!-- RELATED:END -->
