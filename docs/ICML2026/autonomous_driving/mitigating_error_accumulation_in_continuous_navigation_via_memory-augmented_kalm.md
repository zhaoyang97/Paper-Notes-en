---
title: >-
  [Paper Note] Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering
description: >-
  [ICML 2026][Autonomous Driving][Kalman Filter] The step-by-step prediction of continuous UAV VLN is reformulated as a closed loop of "Recursive Bayesian Estimation = GRU Prior + Memory Bank Likelihood + Learnable Kalman…
tags:
  - "ICML 2026"
  - "Autonomous Driving"
  - "Kalman Filter"
  - "Memory Bank Retrieval"
  - "State Drift"
  - "VLN"
  - "Bayesian Estimation"
date: 2026-05-08
content_hash: a27fc04e8449032d
---

# Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering

**Conference**: ICML 2026  
**arXiv**: [2602.11183](https://arxiv.org/abs/2602.11183)  
**Code**: https://github.com/yinntag/Neuro-Kalman (Available)  
**Area**: Embodied AI / UAV Vision-Language Navigation / State Estimation  
**Keywords**: Kalman Filter, Memory Bank Retrieval, State Drift, VLN, Bayesian Estimation

## TL;DR
The step-by-step prediction of continuous UAV VLN is reformulated as a closed loop of "Recursive Bayesian Estimation = GRU Prior + Memory Bank Likelihood + Learnable Kalman Gain." By fine-tuning on TravelUAV with only 10% of the data, the L1-Full Success Rate (SR) was increased from 17.6% to 25.9%, while flattening the position drift—which typically accumulates after 100 steps—to approximately 30–40 meters.

## Background & Motivation
**Background**: Current continuous UAV VLN systems (TravelUAV, OpenVLN, NavFoM, etc.) predominantly follow a dead-reckoning paradigm. They take the current multi-view image frame and global instructions to directly predict the next waypoint, then plug this new position back into the model for the next step, rolling out a complete trajectory.

**Limitations of Prior Work**: The primary issue with such open-loop rollouts is that **errors accumulate like compound interest over time**. Any deviation in a single step contaminates the "internal position belief" for the next. Since global linguistic instructions are planned based on the initial position, once the internal belief drifts away from the true coordinates, subsequent waypoint grounding with linguistic instructions becomes misaligned. The paper refers to this as "state drift," observing that the position L2 error diverges linearly after 100 steps until the agent crashes.

**Key Challenge**: Existing methods focus entirely on "how to make the prior estimation more accurate" (e.g., larger MLLMs, more pre-training data) but **lack any explicit error correction mechanism**. Predictions are trusted immediately upon output without an "update step" to inversely correct the prior using observations. This corresponds to a degenerate case of Bayesian filtering where there is "only prediction, no update."

**Goal**: (1) Explicitly model navigation as a Bayes filter $P(\mathbf{z}_t|o_{1:t}, w_{1:t-1}) \propto P(o_t|\mathbf{z}_t) P(\mathbf{z}_t|\mathbf{z}_{t-1}, w_{t-1})$; (2) Correct the current belief online using historical observations without updating model weights; (3) Outperform 100%-data baselines using only 10% training data.

**Key Insight**: The authors identify a widely overlooked mathematical equivalence—**attention-based memory retrieval is essentially a kernel density estimation (KDE) of the likelihood $P(o_t|\mathbf{z}_t)$ via Nadaraya-Watson kernel regression**. This implies that connecting a memory bank to softmax attention provides a likelihood estimator for free, without needing to learn an explicit probabilistic model.

**Core Idea**: By using a three-stage architecture composed of a "GRU prior + retrieved historical anchor likelihood + learnable Kalman gain," the prediction-update loop of Kalman filtering is ported directly into the latent space of VLN. This allows the model to pull the current belief back to the true manifold using historical observations at every step.

## Method

### Overall Architecture
The input to NeuroKalman consists of multi-view images $v_t$ at each step, the current 3D coordinates $p_t$, and a global instruction $l$. The output is the next waypoint $w_t$. The model maintains a $d$-dimensional latent belief state $\mathbf{z}_t$. Each time step follows three blocks:

1.  **Prediction Block**: A GRU calculates the prior $\tilde{\mathbf{z}}_t$ based on the previous posterior $\mathbf{z}_{t-1}$, the previous waypoint $\mathbf{w}_{t-1}$, and a hidden state $\mathbf{h}_{t-1}$ (dead-reckoning, without viewing the current image).
2.  **Update Block**: An MLLM (EVA-CLIP vision + Vicuna-7B) processes the "current vision + historical anchors retrieved from the memory bank + instructions + position," outputting a measurement representation $\mathbf{r}_t$ and a confidence score $\sigma_t \in [0,1]$.
3.  **Kalman Correction**: A learnable Kalman gain $\mathbf{K}_t$ fuses $\tilde{\mathbf{z}}_t$ and $\mathbf{r}_t$ into a posterior $\mathbf{z}_t$. Simultaneously, the visual representation decoded from $\mathbf{z}_t$ is selectively written back to the memory bank as an anchor for future retrieval if $\sigma_t > 0.5$.

Finally, $\mathbf{z}_t$ is fed into a waypoint prediction head to predict $w_t$, and $\mathbf{z}_t$ becomes $\mathbf{z}_{t-1}$ for the next iteration, forming a closed loop.

### Key Designs

1.  **GRU Prior as Dead-reckoning Channel**:
    *   **Function**: Extrapolates the current state using only "previous belief + previous action" without current observations, serving as the Kalman prior.
    *   **Mechanism**: $\mathbf{h}_t = \mathrm{GRU}([\mathbf{z}_{t-1}, \mathbf{w}_{t-1}], \mathbf{h}_{t-1})$, $\tilde{\mathbf{z}}_t = \mathrm{MLP}_{prior}(\mathbf{h}_t)$. This channel intentionally "runs blind" to ensure it purely reflects kinematic priors, leaving visual information as independent evidence for the update channel.
    *   **Design Motivation**: If visual information were mixed into the prior, the "measurement" and "prediction" steps would no longer be independent, breaking the optimality of Kalman fusion. The GRU ensures temporal smoothness, helping the update channel filter high-frequency noise.

2.  **Memory Retrieval = KDE-based Likelihood**:
    *   **Function**: Retrieves historical visual anchors from a memory bank $\mathcal{M} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^{N}$ to serve as a non-parametric estimation of the likelihood $P(o_t|\mathbf{z}_t)$.
    *   **Mechanism**: Derived from Nadaraya-Watson kernel regression, the retrieval is formulated as $\hat{\mathbf{z}}_{evi} = \sum_i \mathcal{K}(\mathbf{f}_t, \mathbf{f}_i) \mathbf{f}_i / \sum_j \mathcal{K}(\mathbf{f}_t, \mathbf{f}_j)$. By setting the kernel $\mathcal{K}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top \mathbf{y}/\sqrt{d})$, it **automatically degenerates into softmax attention**. Thus, attention is not merely an "engineering trick" but an exact implementation of KDE for likelihood. A post-correction strategy is used for writing: only visual features corresponding to posteriors with $\sigma_t > 0.5$ are saved to prevent noisy samples from contaminating the evidence bank.
    *   **Design Motivation**: Modeling explicit probability distributions in high-dimensional visual space is nearly impossible, whereas KDE only requires samples. By equating it with attention, likelihood estimation is integrated into the MLLM pipeline for free. The memory bank provides invariant "evaluated anchors" suitable for test-time correction without gradient updates.

3.  **Learnable Kalman Gain as Uncertainty Modulator**:
    *   **Function**: Dynamically decides whether to trust the prior or the measurement at each step, replacing the explicit covariance noise models ($\mathbf{Q}, \mathbf{R}$) required in classical Kalman filters.
    *   **Mechanism**: $\mathbf{K}_t = \mathrm{Sigmoid}(\mathbf{W}_g [(\mathbf{r}_t - \tilde{\mathbf{z}}_t); \phi(\sigma_t)] + \mathbf{b}_g)$, which concatenates the "innovation (residual)" and the "MLP projection of measurement confidence" through a gating network to yield per-dimension gains. The Bayesian update is performed as $\mathbf{z}_t = \tilde{\mathbf{z}}_t + \mathbf{K}_t \odot (\mathbf{r}_t - \tilde{\mathbf{z}}_t)$. This is algebraically equivalent to the classical Kalman form $\mathbf{z}_{post} = \mathbf{z}_{prior} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{H}\mathbf{z}_{prior})$ with $\mathbf{H} = \mathbf{I}$.
    *   **Design Motivation**: Fixed $\mathbf{K}_t$ weights failed in ablations—$\mathbf{K}_t = 0.1$ (predominantly trusting the prior) led to catastrophic drift (SR=0%), while $\mathbf{K}_t = 0.9$ (predominantly trusting the measurement) lost temporal smoothness (SR=18%). Learnable gains automatically switch between "smoothness-oriented" and "correction-oriented" based on current innovation, ensuring stability across noise regimes.

### Loss & Training
The EVA-CLIP vision backbone and Vicuna-7B language backbone are frozen. Gradients are calculated only for the visual projector, waypoint predictor, and LoRA layers. In addition to the main waypoint loss, an $L_1$ supervision is applied to both the prior $\tilde{\mathbf{z}}_t$ and measurement $\mathbf{r}_t$ (coefficient 0.2) to force both channels to predict waypoints independently, preventing free-riding. Training used Adam, lr=$5\mathrm{e}{-5}$, batch size 16, on 4×A6000 GPUs. All experiments were pre-trained on 100% data and then fine-tuned on a fixed 10% subset of training trajectories.

## Key Experimental Results

### Main Results
The UAV-Need-Help benchmark on TravelUAV includes 12,149 human trajectories across 20 training scenarios and 2 Unseen-Map scenarios, with 89 object categories. Metrics used are NE↓ (meters), SR↑, OSR↑, and SPL↑. Difficulty is categorized by distance (Easy < 250m / Hard $\ge$ 250m) and instruction assistance (L1/L2/L3).

| Split | Method | NE↓ | SR↑ | OSR↑ | SPL↑ |
|---|---|---|---|---|---|
| L1 Test-Seen Full | TravelUAV (100% data) | 106.28 | 16.10 | 44.26 | 14.30 |
| L1 Test-Seen Full | TravelUAV-FT (10% data) | 99.79 | 17.56 | 41.89 | 14.71 |
| L1 Test-Seen Full | OpenVLN | 125.97 | 14.39 | 28.03 | 12.94 |
| L1 Test-Seen Full | **NeuroKalman (10% data)** | **71.56** | **25.86** | **58.73** | **22.43** |
| L1 Test-Seen Hard | TravelUAV-FT | 143.85 | 13.70 | 36.85 | 12.15 |
| L1 Test-Seen Hard | **NeuroKalman** | **105.07** | **20.11** | **53.90** | **18.21** |
| L1 Unseen-Object | NavFoM | 108.04 | 29.83 | 47.99 | 27.20 |
| L1 Unseen-Object | **NeuroKalman** | **71.01** | **32.48** | **60.82** | **28.50** |
| L1 Unseen-Map | TravelUAV-FT | 117.84 | 4.68 | 19.03 | 3.17 |
| L1 Unseen-Map | **NeuroKalman** | **100.32** | **8.34** | **34.15** | **7.12** |

Notably, in the Test-Seen-Hard split, NeuroKalman fine-tuned on 10% data (SR 20.1%) outperformed TravelUAV trained on 100% data (SR 12.8%), while NE dropped from 152 to 105.

### Ablation Study

| Configuration | NE↓ | SR↑ | Description |
|---|---|---|---|
| $\mathbf{K}_t = 0.1$ (Prior bias) | 217.09 | 0.00 | No correction; catastrophic drift prevents navigation |
| $\mathbf{K}_t = 0.5$ (Fixed mean) | 83.14 | 24.12 | Better than baseline, but weaker than adaptive gain |
| $\mathbf{K}_t = 0.9$ (Measurement bias) | 100.96 | 18.05 | Loss of temporal smoothness; retrieval noise interferes |
| **Learnable $\mathbf{K}_t$** | **71.56** | **25.86** | Adaptive weight adjustment |
| Memory length $M = 5$ | 84.39 | 21.23 | Insufficient historical anchors |
| **$M = 10$** | **71.56** | **25.86** | Sweet spot |
| $M = 15$ | 77.17 | 23.77 | Outdated anchors introduce noise |
| Write threshold $\sigma_t = 0.3$ | 82.45 | 20.50 | Low threshold; noisy anchors contaminate memory |
| **$\sigma_t = 0.5$** | **71.56** | **25.86** | Optimal |
| TravelUAV + Post-hoc Classical KF | 96.67 | 18.17 | Geometric smoothing in output space has limited utility |

### Key Findings
- **The gap between learned and fixed gains is as high as 25 SR points**. $\mathbf{K}_t = 0.1$ results in zero success, proving that open-loop dead-reckoning without correction is disastrous. Conversely, blindly trusting measurements is also insufficient; the key is per-step, per-dimension uncertainty awareness.
- **Memory length follows a U-shaped curve**, with $M = 10$ being optimal. This suggests that for UAV trajectories of 100–200 steps, approximately 10 high-quality historical anchors are enough to cover the local manifold. More anchors introduce outdated visual information from dozens of steps back, interfering with attention.
- **Post-hoc application of a classical Kalman Filter (constant velocity model) in output space only improved SR from 16.1% to 18.2%**, far below NeuroKalman's 25.9%. This demonstrates that error correction must occur in the latent semantic space rather than through geometric smoothing of (x, y, z) coordinates.
- **Drift Curves**: Position L2 error for TravelUAV diverges linearly after 100 steps. In contrast, NeuroKalman's error **stops growing** after reaching ~30–40 meters, visually confirming the effect of the Kalman closed loop.

## Highlights & Insights
- **The "attention = KDE likelihood" equivalence** is a major unifying insight. Previously, retrieval augmentation was viewed as an engineering trick; this paper identifies it as a discretization of non-parametric Bayesian likelihood estimation. This provides a probabilistic foundation for retrieval-augmented methods that can be migrated to RAG-LLMs, world models, and TTA.
- **The post-correction write strategy** is highly effective: memory only accepts samples that have been corrected by Kalman and reported with high confidence. This ensures the memory bank stores only "verified anchors," avoiding the vicious cycle where noisy caches lead to increasingly noisy retrievals.
- **The explanation for the 10% data outperforming 100% data** is critical. Dead-reckoning models rely on "memorizing all possible transitions" with large datasets, making them prone to overfitting when data is scarce. NeuroKalman embeds "long-term consistency" as an explicit inductive bias into the architecture, allowing this capability to exist without needing massive data volume. This is a clear case of structured priors defeating brute-force scaling.

## Limitations & Future Work
- The authors acknowledge that using a GRU as a prior may lead to information decay over extremely long horizons, though the primary contribution lies in the Bayesian correction framework itself. The GRU can be replaced by Transformers or Mamba.
- Self-identified limitations: (1) The memory length $M=10$ is hardcoded and might vary across tasks; (2) The fixed $\sigma_t > 0.5$ threshold might prevent saving anchors in early stages where model confidence is naturally low; (3) Experiments were confined to AirSim simulations, with no validation for real-world UAV visual noise or kinematic mismatches; (4) While the KDE equivalence is a theoretical highlight, the actual implementation is standard attention, potentially overstating the "incremental" theoretical contribution.
- Future directions: Making the write threshold $\sigma_t$ learnable; applying contrastive loss to explicitly decouple the "prior-only" and "measurement-only" channels to further ensure independence (a prerequisite for optimal Kalman fusion).

## Related Work & Insights
- **vs TravelUAV / OpenVLN**: These models use step-by-step waypoint regression without explicit correction, leading to inevitable drift over long horizons. This work adds a Bayesian correction loop over their existing backbones.
- **vs MapNet / SkyVLN / OpenFly**: These topological memory approaches treat memory as a "passive buffer" for feature concatenation. This work treats memory as "probabilistic evidence" for Bayesian fusion, shifting the paradigm from "passive aggregation" to "active correction."
- **vs FEEDTTA / FSTTA**: These TTA methods rely on online gradient updates to combat distribution shift, which can be unstable in VLN due to lack of reliable supervision. NeuroKalman corrects the belief state without weight updates, making it more robust.
- **vs KalmanNet / Backprop-KF**: These deep Bayesian filters excel at learning transitions but struggle to define likelihood in high-dimensional visual space. This work solves the likelihood problem using KDE-attention, providing a high-dimensional vision version of deep filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ The "attention = KDE likelihood" equivalence and porting the full Kalman prediction-update loop into VLN latent space offer a clean, theoretically grounded perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Inclusion of Seen/Unseen-Map/Unseen-Object splits, multiple instruction levels, drift visualization, and extensive ablations, although limited to a single benchmark (TravelUAV).
- Writing Quality: ⭐⭐⭐⭐ The derivation from Bayesian framework to KDE equivalence to architectural implementation is very clear.
- Value: ⭐⭐⭐⭐ Provides significant gains for real-world VLN pain points (long horizons, low data) and the method is generalizable to other sequential decision tasks involving priors and retrieval.

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
