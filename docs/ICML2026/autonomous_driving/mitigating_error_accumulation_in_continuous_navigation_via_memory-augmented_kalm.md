---
title: >-
  [Paper Note] Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering
description: >-
  [ICML 2026][Autonomous Driving][VLN] Ours reformulates step-by-step prediction in continuous UAV VLN as a closed-loop "Recursive Bayesian Estimation = GRU Prior + Memory Likelihood + Learnable Kalman Gain." By fine-tuning on only 10% of the data, it improves the Success Rate (SR) of L1-Full on TravelUAV from 17.6% to 25.9% and stabilizes position drift—wh
tags:
  - ICML 2026
  - Autonomous Driving
  - VLN
date: 2026-05-08
content_hash: a8c96acfd6074e2b
---
# Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering

**Conference**: ICML 2026  
**arXiv**: [2602.11183](https://arxiv.org/abs/2602.11183)  
**Code**: https://github.com/yinntag/Neuro-Kalman (Available)  
**Area**: Embodied AI / UAV Vision-Language Navigation / State Estimation  
**Keywords**: Kalman Filter, Memory Retrieval, State Drift, VLN, Bayesian Estimation

## TL;DR
Ours reformulates step-by-step prediction in continuous UAV VLN as a closed-loop "Recursive Bayesian Estimation = GRU Prior + Memory Likelihood + Learnable Kalman Gain." By fine-tuning on only 10% of the data, it improves the Success Rate (SR) of L1-Full on TravelUAV from 17.6% to 25.9% and stabilizes position drift—which previously accumulated indefinitely—to within 30–40 meters after 100 steps.

## Background & Motivation
**Background**: Current continuous UAV VLN systems (TravelUAV, OpenVLN, NavFoM, etc.) largely follow a dead-reckoning paradigm—predicting the next waypoint directly from current multi-view images and global instructions, then plugging the new position back in for the next step to roll out a complete trajectory.

**Limitations of Prior Work**: The primary issue with this open-loop rollout is that **errors accumulate exponentially over time**. Any deviation at one step pollutes the "internal position belief" for the next. Since global language instructions are planned from the starting position, once the internal belief drifts from the true coordinates, subsequent waypoint grounding with language instructions becomes misaligned. Ours terms this "state drift," observed empirically as L2 position error diverging linearly until failure after 100 steps.

**Key Challenge**: Existing methods focus heavily on "improving prior estimation accuracy" (larger MLLMs, more pre-training data) but **lack an explicit error-correction mechanism**. Once a prediction is output, it is fully trusted without an update step to adjust the prior using observations. This corresponds to the degenerate case of "prediction only, no update" in Bayesian filtering.

**Goal**: (1) Explicitly model navigation as a Bayes filter $P(\mathbf{z}_t|o_{1:t}, w_{1:t-1}) \propto P(o_t|\mathbf{z}_t) P(\mathbf{z}_t|\mathbf{z}_{t-1}, w_{t-1})$; (2) Online correction of the current belief using historical observations without updating model weights; (3) Outperforming 100% data baselines using only 10% of the training data.

**Key Insight**: Ours identifies a widely overlooked mathematical equivalence—**attention-based memory retrieval is essentially a Kernel Density Estimation (KDE) of the likelihood $P(o_t|\mathbf{z}_t)$ via Nadaraya-Watson kernel regression.** This implies that connecting a memory bank with softmax attention provides a likelihood estimator for free, without needing to learn an explicit probabilistic model.

**Core Idea**: A three-stage architecture—"GRU Prior + Retrieved Historical Anchor Likelihood + Learnable Kalman Gain"—is used to integrate the Kalman filter prediction-update cycle directly into the VLN latent space, allowing the model to pull the current belief back to the true manifold using historical observations at each step.

## Method

### Overall Architecture
NeuroKalman addresses the issue of belief state drift in continuous VLN by rewriting waypoint prediction from "one-shot open-loop extrapolation" into a recursive Bayesian filtering loop of "prior extrapolation + historical observation correction." Inputs include multi-view images $v_t$, current 3D coordinates $p_t$, and global instructions $l$. The model operates on a $d$-dimensional latent belief state $\mathbf{z}_t$: first, a GRU extrapolates a prior $\tilde{\mathbf{z}}_t$ without looking at images; second, an MLLM provides a measurement $\mathbf{r}_t$ and confidence $\sigma_t$ by combining current vision with historical anchors retrieved from a memory bank; finally, a learnable Kalman gain fuses these into a posterior $\mathbf{z}_t$, which decodes the waypoint $w_t$ and passes $\mathbf{z}_t$ to the next step.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    I["Multi-view Images + 3D Coordinates + Global Instructions"] --> G["GRU Prior Channel<br/>Blind extrapolation of prior z̃t from previous posterior and action"]
    I --> R["Memory Retrieval = KDE Likelihood<br/>MLLM + historical anchors provide measurement rt and confidence σt"]
    G --> K["Learnable Kalman Gain<br/>Gating network adaptively fuses prior and measurement based on innovation"]
    R --> K
    K --> Z["Posterior belief zt → Decode waypoint wt"]
    Z -->|High-confidence posterior writeback| R
    Z -->|zt passed to next step| G
```

### Key Designs

**1. GRU Prior Channel: Isolating Dead-Reckoning as Pure Kinematic Evidence**

The root of error accumulation is the entanglement of prior and measurement, preventing independent correction. This channel is intentionally "blind"—consuming only the previous posterior and action without current visual input: $\mathbf{h}_t = \mathrm{GRU}([\mathbf{z}_{t-1}, \mathbf{w}_{t-1}], \mathbf{h}_{t-1})$, $\tilde{\mathbf{z}}_t = \mathrm{MLP}_{prior}(\mathbf{h}_t)$. This ensures the prior is a pure kinematic extrapolation, leaving visual information entirely for the update channel as independent evidence. This separation is crucial because if vision pollutes the prior, "prediction" and "measurement" are no longer independent, breaking the optimality premise of Kalman fusion. Additionally, the temporal recursion of the GRU provides smoothness, helping filter high-frequency noise in measurements.

**2. Memory Retrieval = KDE Likelihood: Interpreting Attention as Non-parametric Bayesian Estimation**

Defining an explicit probability model for the likelihood $P(o_t|\mathbf{z}_t)$ in high-dimensional visual space is difficult. Ours bypasses this using KDE with samples. Starting from Nadaraya-Watson kernel regression, the retrieval from a memory bank $\mathcal{M} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^{N}$ is written as $\hat{\mathbf{z}}_{evi} = \sum_i \mathcal{K}(\mathbf{f}_t, \mathbf{f}_i) \mathbf{f}_i / \sum_j \mathcal{K}(\mathbf{f}_t, \mathbf{f}_j)$. By choosing $\mathcal{K}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top \mathbf{y}/\sqrt{d})$, this formula **exactly reduces to softmax attention**. Thus, attention is not just an engineering trick but a discrete implementation of KDE for likelihood. This allows the likelihood estimator to be integrated into the MLLM pipeline for free. A post-correction write-back strategy is used: only visual features corresponding to posteriors with $\sigma_t > 0.5$ are saved, ensuring the anchor bank only contains samples already corrected by Kalman filtering and reported as high-confidence.

**3. Learnable Kalman Gain: Replacing Explicit Covariance with Gating Networks**

Classical Kalman filters require explicit estimation of process noise $\mathbf{Q}$ and measurement noise $\mathbf{R}$ to calculate gain. In deep latent space, these covariances are hard to define. Ours learns the gain directly. The "innovation" $\mathbf{r}_t - \tilde{\mathbf{z}}_t$ and the MLP projection of confidence $\phi(\sigma_t)$ are concatenated and passed through a gating network to obtain a dimension-wise gain $\mathbf{K}_t = \mathrm{Sigmoid}(\mathbf{W}_g [(\mathbf{r}_t - \tilde{\mathbf{z}}_t); \phi(\sigma_t)] + \mathbf{b}_g)$. The Bayesian update is completed as $\mathbf{z}_t = \tilde{\mathbf{z}}_t + \mathbf{K}_t \odot (\mathbf{r}_t - \tilde{\mathbf{z}}_t)$, which is algebraically equivalent to $\mathbf{z}_{post} = \mathbf{z}_{prior} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{H}\mathbf{z}_{prior})$ when $\mathbf{H} = \mathbf{I}$. Learnable gain proves vital: fixed gains lead to failure ($\mathbf{K}_t = 0.1$ causes catastrophic drift; $\mathbf{K}_t = 0.9$ introduces high-frequency noise). Adaptive gain allows the model to switch between "smoothness" and "correction" per dimension and per step.

### Loss & Training
The EVA-CLIP visual backbone and Vicuna-7B language backbone are frozen; gradients are computed only for the visual projector, waypoint predictor, and LoRA layers. In addition to the main waypoint loss, an $L_1$ supervision is applied to both the prior $\tilde{\mathbf{z}}_t$ and measurement $\mathbf{r}_t$ (coefficient 0.2) to force both channels to predict waypoints independently. Training uses Adam, lr=$5\mathrm{e}{-5}$, batch=16, 4×A6000. All experiments involve pre-training on 100% data followed by fine-tuning on a fixed 10% subset of training trajectories.

## Key Experimental Results

### Main Results

Evaluated on the TravelUAV "UAV-Need-Help" benchmark: 12,149 human-operated trajectories, 20 training scenes + 2 Unseen-Map scenes, 89 object categories; Metrics: NE↓ (meters), SR↑, OSR↑, SPL↑. Difficulty levels: Easy (<250 m) / Hard (≥250 m), and instruction intensity levels L1/L2/L3.

| Split | Method | NE↓ | SR↑ | OSR↑ | SPL↑ |
|---|---|---|---|---|---|
| L1 Test-Seen Full | TravelUAV (100% data) | 106.28 | 16.10 | 44.26 | 14.30 |
| L1 Test-Seen Full | TravelUAV-FT (10% data) | 99.79 | 17.56 | 41.89 | 14.71 |
| L1 Test-Seen Full | OpenVLN | 125.97 | 14.39 | 28.03 | 12.94 |
| L1 Test-Seen Full | **Ours (10% data)** | **71.56** | **25.86** | **58.73** | **22.43** |
| L1 Test-Seen Hard | TravelUAV-FT | 143.85 | 13.70 | 36.85 | 12.15 |
| L1 Test-Seen Hard | **Ours** | **105.07** | **20.11** | **53.90** | **18.21** |
| L1 Unseen-Object | NavFoM | 108.04 | 29.83 | 47.99 | 27.20 |
| L1 Unseen-Object | **Ours** | **71.01** | **32.48** | **60.82** | **28.50** |
| L1 Unseen-Map | TravelUAV-FT | 117.84 | 4.68 | 19.03 | 3.17 |
| L1 Unseen-Map | **Ours** | **100.32** | **8.34** | **34.15** | **7.12** |

Notably, on Test-Seen-Hard, Ours with 10% data (SR 20.1%) significantly outperforms the 100% data TravelUAV baseline (SR 12.8%), reducing NE from 152m to 105m.

### Ablation Study

| Configuration | NE↓ | SR↑ | Note |
|---|---|---|---|
| $\mathbf{K}_t = 0.1$ (Prior heavy) | 217.09 | 0.00 | No correction, drifts completely |
| $\mathbf{K}_t = 0.5$ (Fixed mean) | 83.14 | 24.12 | Better than baseline, but lacks adaptive gain |
| $\mathbf{K}_t = 0.9$ (Measurement heavy) | 100.96 | 18.05 | Loss of smoothness, retrieval noise enters |
| **Learnable $\mathbf{K}_t$** | **71.56** | **25.86** | Adaptive weighting |
| Memory length $M = 5$ | 84.39 | 21.23 | Insufficient historical anchors |
| **$M = 10$** | **71.56** | **25.86** | Sweet spot |
| $M = 15$ | 77.17 | 23.77 | Outdated anchors introduce noise |
| Threshold $\sigma_t = 0.3$ | 82.45 | 20.50 | Low bar, noisy anchors pollute memory |
| **$\sigma_t = 0.5$** | **71.56** | **25.86** | Optimal |
| TravelUAV + Post-hoc Classical KF | 96.67 | 18.17 | Geometric smoothing helps slightly, but latent correction is key |

### Key Findings
- **The gap between learned and fixed gain is up to 25 SR points**: Fixed $\mathbf{K}_t = 0.1$ drops everything to zero, proving open-loop dead-reckoning is disastrous without correction. Adaptive uncertainty awareness is essential.
- **Memory length follows a U-shaped curve**: $M = 10$ is optimal, suggesting that in trajectories of 100–200 steps, ~10 high-quality anchors cover the local manifold. Excess anchors from dozens of steps ago become outdated noise.
- **Post-hoc classical Kalman filtering in output space only pushes SR from 16.1% to 18.2%**, far below Ours (25.9%). This proves correction must occur in latent semantic space rather than just applying geometric smoothing to $(x, y, z)$ coordinates.
- **Drift Curve**: While TravelUAV's L2 error diverges linearly after 100 steps, Ours **stops growing** after an initial rise to ~30–40 meters, visualizing the effect of the Kalman closed loop.

## Highlights & Insights
- **The "attention = KDE likelihood" mathematical equivalence** is a unifying insight. It clarifies that retrieval-augmented methods are not just engineering tricks but discretized implementations of non-parametric Bayesian likelihood estimation.
- **Post-correction write-back strategy**: By only accepting corrected, high-confidence samples into memory, Ours avoids the vicious cycle where noisy data is retrieved and further pollutes the system.
- **10% data outperforming 100% data**: Dead-reckoning models rely on memorizing transitions (prone to overfitting with less data), whereas Ours encodes "long-term consistency" as an explicit inductive bias. Structural priors here beat brute-force scaling.

## Limitations & Future Work
- GRU as a prior may suffer from information decay over extremely long horizons. However, the Bayesian correction framework is architecture-agnostic and could use Transformers or Mamba.
- Self-identified limitations: (1) Hard-coded $M=10$ and $\sigma_t > 0.5$ might not be optimal across all scenarios; (2) All experiments are in AirSim; real-world visual noise and kinematic mismatch are not yet validated; (3) The "KDE-attention equivalence" is a theoretical highlight, but the implementation is standard attention, potentially overstating the "incremental" engineering contribution.
- Future directions: Learning the memory write-back threshold $\sigma_t$ and using contrastive loss to further disentangle prior and measurement channels.

## Related Work & Insights
- **vs TravelUAV / OpenVLN**: These use step-by-step waypoint regression without explicit correction, leading to inevitable drift over long horizons. Ours adds a Bayesian correction loop around their backbones.
- **vs MapNet / SkyVLN**: These use memory as a "passive buffer" for feature concatenation. Ours treats memory as "probabilistic evidence" for Bayesian fusion, shifting from "passive aggregation" to "active correction."
- **vs Deep Bayesian Filtering (KalmanNet)**: While they learn transitions well, defining likelihood in high-dimensional vision is difficult. Ours solves the likelihood problem using KDE-attention, providing a high-dimensional visual version of this line of work.

## Rating
- Novelty: ⭐⭐⭐⭐ The "attention = KDE likelihood" equivalence and the integration of a full prediction-update loop in VLN latent space provide a clean, theoretically grounded perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple splits, drift visualization, and extensive ablations, though limited to the TravelUAV benchmark.
- Writing Quality: ⭐⭐⭐⭐ The derivation from Bayesian framework to KDE equivalence to implementation is very clear.
- Value: ⭐⭐⭐⭐ Significant gains in long-horizon and low-data VLN scenarios; the method is transferable to other sequential decision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding](../../ICLR2026/autonomous_driving/marc_memory-augmented_rl_token_compression_for_efficient_video_un.md)
- [\[ICML 2026\] Plug-and-Play Label Map Diffusion for Universal Goal-Oriented Navigation](plug-and-play_label_map_diffusion_for_universal_goal-oriented_navigation.md)
- [\[NeurIPS 2025\] Continuous Simplicial Neural Networks](../../NeurIPS2025/autonomous_driving/continuous_simplicial_neural_networks.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](../../CVPR2026/autonomous_driving/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](../../ICCV2025/autonomous_driving/occupancy_learning_with_spatiotemporal_memory.md)

</div>

<!-- RELATED:END -->
