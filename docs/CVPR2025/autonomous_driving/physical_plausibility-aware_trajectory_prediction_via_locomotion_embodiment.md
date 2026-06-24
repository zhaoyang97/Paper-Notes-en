---
title: >-
  [Paper Note] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment
description: >-
  [CVPR 2025][Autonomous Driving][Trajectory Prediction] This paper proposes the Locomotion Embodiment framework, which utilizes humanoid locomotion generation in a physical simulator to evaluate the physical plausibility of trajectories. It constructs a differentiable LocoVal function to replace the non-differentiable physical simulator during trajectory prediction network training, and filters implausible trajectories at inference time.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Trajectory Prediction"
  - "Physical Plausibility"
  - "Locomotion Generation"
  - "Physical Simulator"
  - "Differentiable Proxy"
date: 2026-05-08
content_hash: 9eccd01a92214f9a
---

# Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment

**Conference**: CVPR 2025  
**arXiv**: [2503.17267](https://arxiv.org/abs/2503.17267)  
**Code**: [GitHub](https://github.com/ImIntheMiddle/EmLoco)  
**Area**: Autonomous Driving  
**Keywords**: Trajectory Prediction, Physical Plausibility, Locomotion Generation, Physical Simulator, Differentiable Proxy

## TL;DR

This paper proposes the Locomotion Embodiment framework, which utilizes humanoid locomotion generation in a physical simulator to evaluate the physical plausibility of trajectories. It constructs a differentiable LocoVal function to replace the non-differentiable physical simulator during trajectory prediction network training, and filters implausible trajectories at inference time.

## Background & Motivation

Human Trajectory Prediction (HTP) aims to predict future pedestrian movement paths, which is crucial for autonomous driving, robotics, and security systems. Although existing methods utilize human pose information, they suffer from the following limitations:

- **Insufficient Utilization of Pose Information**: Existing methods only model pose implicitly as an auxiliary input, and the predicted trajectories may be physically inconsistent with the observed human poses.
- **Limitations of Data-Driven Approaches**: Large-scale paired pose-trajectory datasets are required to cover all possible combinations, but such datasets do not exist.
- **Incipient Observation Scenarios**: When only a few past frames are available (e.g., a pedestrian suddenly appearing from behind an obstacle), methods relying on long-term observations fail.
- **Diversity-Plausibility Dilemma in Stochastic Prediction**: The minMSE loss only supervises the best prediction head, leaving other heads underfitted; whereas standard MSE loss forces all predictions to converge to a single ground truth, sacrificing diversity.

## Method

### Overall Architecture

The framework comprises two training stages and an inference stage: (1) Training the LocoVal function — using the PACER locomotion generator in a physical simulator to obtain trajectory plausibility scores as ground truth to train a differentiable MLP proxy function; (2) Training the HTP network — jointly training using the EmLoco loss and standard MSE loss; (3) Inference — filtering out physically implausible trajectories using the LocoVal filter.

### Key Design 1: LocoVal Function — A Differentiable Physical Plausibility Proxy

**Function**: To replace the non-differentiable physical simulator and evaluate the physical plausibility of trajectories during training and inference.

**Mechanism**: A humanoid is controlled to walk along a given trajectory using the PACER locomotion generator in a physical simulator (IsaacGym). When a trajectory is physically implausible (e.g., sharp turns), the humanoid fails to follow it, resulting in a low cumulative reward $\Omega$. An MLP network $\mathcal{V}$ is trained to estimate this reward from available image-level cues (future trajectory $\tau_s$, initial pose $\mathbf{j}_0$, root joint velocity $\mathbf{v}_{\text{root},0}$):

$$\mathcal{L}_\mathcal{V} = \text{MSE}(\mathcal{V}(\tau_s, \mathbf{h}'_0), \mathcal{G}(\tau_s, \mathbf{h}_0))$$

During training, physically plausible and randomly generated implausible pairs are used to enable $V$ to distinguish plausible pose-trajectory combinations from implausible ones.

**Design Motivation**: (1) The physical simulator is non-differentiable and cannot propagate gradients directly; (2) The simulator requires complete physical states (joint angular velocities, etc.), which are unavailable in image scenarios. $\mathcal{V}$ estimates plausibility using only a few available cues, with negligible inference overhead.

### Key Design 2: EmLoco Loss — Jointly Supervising All Prediction Heads

**Function**: To concurrently train all prediction heads using the physical plausibility prior, avoiding underfitting issues caused by minMSE.

**Mechanism**: The EmLoco loss is defined as $\mathcal{L}_E = -\mathcal{V}(\hat{\tau}_f, \mathbf{h}'_0)$, encouraging the prediction of physically plausible trajectories. It is combined with the standard MSE loss as:

$$\mathcal{L}_\text{total} = \mathcal{L}_T + \alpha \mathcal{L}_E$$

For multi-head stochastic prediction, $\mathcal{L}_T$ employs $\min_k \text{MSE}(\hat{\tau}_f^k, \tau_f)$ (supervising only the best-performing head), while **$\mathcal{L}_E$ averages the prediction loss across all heads**, enabling concurrent optimization of all prediction heads.

**Design Motivation**: The MSE loss serves as a data term (fitting ground truth), while the EmLoco loss acts as a regularization term (incorporating a physical prior). Since EmLoco does not require a single ground-truth trajectory, it can enhance the plausibility of all heads while maintaining prediction diversity, resolving the dilemma between efficiency and diversity in stochastic HTP.

### Key Design 3: LocoVal Filter — Plug-and-Play Inference-Time Filtering

**Function**: To evaluate and filter out implausible trajectories within multi-head predictions during inference.

**Mechanism**: Plausibility scores are computed for all predicted trajectories, and those below a threshold $\lambda$ are filtered out: trajectories satisfying $\mathcal{V}(\hat{\tau}_f^k, \mathbf{h}'_0) \geq \lambda$ are retained; if all predictions fall below the threshold, the one with the highest score is kept.

**Design Motivation**: It can be used plug-and-play with any pre-trained stochastic HTP network without needing retraining.

### Loss & Training

$$\mathcal{L}_\text{total} = \underbrace{\min_k \text{MSE}(\hat{\tau}_f^k, \tau_f)}_{\text{minMSE (data term)}} + \alpha \underbrace{(-\frac{1}{K}\sum_k \mathcal{V}(\hat{\tau}_f^k, \mathbf{h}'_0))}_{\text{EmLoco (physics prior)}}$$

where $\alpha = 100$.

## Key Experimental Results

### Main Results: Standard Setting (9-Frame Observation, Deterministic Prediction)

| Method | JTA ADE↓ | JTA FDE↓ | JRDB ADE↓ | JRDB FDE↓ |
|------|---------|---------|----------|----------|
| EqMotion | 1.13 | 2.39 | 0.40 | 0.77 |
| Social-Trans | 1.11 | 2.26 | 0.40 | 0.76 |
| **Ours** | **0.97** | **1.91** | **0.37** | **0.72** |

### Multi-Head Stochastic Prediction (5 Heads / 20 Heads)

| Method | ADE (5/20) | FDE (5/20) | $\chi^2$ Vel. | $\chi^2$ Acc. |
|------|-----------|-----------|---------|---------|
| Social-Trans | 1.86/2.14 | 3.51/4.26 | 0.134/0.169 | 0.009/0.009 |
| **Ours** | **1.68/1.80** | **3.34/3.56** | **0.100/0.087** | **0.002/0.003** |

### LocoVal Filter Performance (Plug-and-Play, ETH/UCY)

| Method | ADE/FDE (w/o filter) | ADE/FDE (w/ filter) |
|------|----------------|----------------|
| EqMotion | 0.36/0.55 | **0.35/0.53** |

### Key Findings

- On the JTA dataset, deterministic ADE drops from 1.11 to **0.97** (-12.6%), and FDE drops from 2.26 to **1.91** (-15.5%).
- Joint optimization of all prediction heads by the EmLoco loss greatly improves $\chi^2$ distances (physical indicators such as velocity and acceleration).
- Performance gains are still achieved even when JRDB uses estimated instead of ground-truth 3D poses, demonstrating robustness to noisy inputs.
- The LocoVal filter can be applied in a plug-and-play manner to enhance pre-trained model performance.

## Highlights & Insights

1. **Crucially introduces humanoid locomotion control in physical simulators to trajectory prediction for the first time**, establishing an evaluation standard for pose-trajectory physical consistency.
2. **Elegant Design of EmLoco Loss**: Operating as a regularization term, it does not require ground-truth labels and can thus simultaneously optimize all prediction heads.
3. **The physical simulator is solely utilized during the pre-training stage**, incurring zero extra cost during inference.
4. **Model-Agnostic Methodology**: Applicable to both deterministic and stochastic HTP methods.

## Limitations & Future Work

- The LocoVal function is trained on AMASS locomotion data, which may not generalize well to extreme motions (e.g., running, jumping).
- Currently, only single-agent physical plausibility is considered; physical constraints of multi-agent interactions are not yet modeled.
- Bird's-eye view (BEV) scenarios (e.g., ETH/UCY) cannot utilize 3D pose, meaning only the LocoVal filter can be applied.
- Future work could explore incorporating environmental physical constraints (e.g., terrains, obstacles) into the plausibility assessment.

## Related Work & Insights

- **Social-Transmotion**: Transformer-based social trajectory prediction supporting pose inputs.
- **PACER**: A reinforcement learning-based physical humanoid locomotion controller.
- **EqMotion**: Equivariant network-based stochastic trajectory prediction method.
- **MATRIX**: Proposes $\chi^2$ distance to evaluate the physical plausibility of trajectories.

## Rating

⭐⭐⭐⭐ — Highly innovative, linking physical simulation with trajectory prediction for the first time. Elegant formulation (differentiable proxy replacing non-differentiable simulator) and comprehensive experiments (various datasets, settings, and ablations). The EmLoco loss improvement on multi-head training has clear theoretical explanation and experimental verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](../../ICCV2025/autonomous_driving/world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](../../ICCV2025/autonomous_driving/donut_a_decoder-only_model_for_trajectory_prediction.md)

</div>

<!-- RELATED:END -->
