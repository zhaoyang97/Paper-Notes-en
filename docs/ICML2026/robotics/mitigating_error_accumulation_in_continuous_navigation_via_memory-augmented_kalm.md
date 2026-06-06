---
title: >-
  [Paper Note] Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering
description: >-
  [ICML 2026][Robotics][Kalman Filtering] Reformulates step-by-step prediction in continuous UAV VLN as a "recursive Bayesian estimation = GRU prior + memory bank likelihood + learnable Kalman gain" closed loop. On TravelU…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Kalman Filtering"
  - "Memory Bank Retrieval"
  - "State Drift"
  - "VLN"
  - "Bayesian Estimation"
date: 2026-05-08
content_hash: 1d6fe556bc6e73b4
---

# Mitigating Error Accumulation in Continuous Navigation via Memory-Augmented Kalman Filtering

**Conference**: ICML 2026  
**arXiv**: [2602.11183](https://arxiv.org/abs/2602.11183)  
**Code**: https://github.com/yinntag/Neuro-Kalman (available)  
**Area**: Embodied Intelligence / UAV Vision-Language Navigation / State Estimation  
**Keywords**: Kalman Filtering, Memory Bank Retrieval, State Drift, VLN, Bayesian Estimation

## TL;DR
Reformulates step-by-step prediction in continuous UAV VLN as a "recursive Bayesian estimation = GRU prior + memory bank likelihood + learnable Kalman gain" closed loop. On TravelUAV, fine-tuning with only 10% of the data boosts L1-Full SR from 17.6% to 25.9%, while position drift after 100 steps is flattened to 30–40 meters.

## Background & Motivation
**Background**: Current continuous UAV VLN systems (TravelUAV, OpenVLN, NavFoM, etc.) mostly follow a dead-reckoning paradigm—using the current multi-view image and global instruction to directly predict the next waypoint, then plugging the new position back for the next step, rolling out the full trajectory.

**Limitations of Prior Work**: The main issue with this open-loop rollout is **error compounds over time**. Any deviation at one step contaminates the "internal position belief" for the next, and since the global language instruction is planned from the initial position, once the internal belief drifts from the true coordinates, subsequent waypoints become misaligned with the instruction grounding. The paper refers to this as "state drift"; empirically, after >100 steps, position L2 error diverges linearly until collision.

**Key Challenge**: Existing methods focus solely on "improving prior estimation" (larger MLLMs, more pretraining data), but **lack any explicit error correction mechanism**—once a prediction is made, it is trusted without an update step to correct the prior using observations. This mirrors the degenerate case in Bayesian filtering where there is "only prediction, no update".

**Goal**: (1) Explicitly model navigation as a Bayes filter $P(\mathbf{z}_t|o_{1:t}, w_{1:t-1}) \propto P(o_t|\mathbf{z}_t) P(\mathbf{z}_t|\mathbf{z}_{t-1}, w_{t-1})$; (2) Use historical observations to online-correct the current belief without updating model weights; (3) Outperform baselines trained on 100% data using only 10% training data.

**Key Insight**: The authors note a widely overlooked mathematical equivalence—**attention-based memory retrieval is essentially Nadaraya-Watson kernel regression for likelihood $P(o_t|\mathbf{z}_t)$**. This means that attaching a memory bank with softmax attention provides a likelihood estimator for free, without needing to learn an explicit probabilistic model.

**Core Idea**: Employ a three-stage architecture of "GRU prior + retrieved historical anchor likelihood + learnable Kalman gain", directly transplanting the Kalman filter's prediction-update loop into the VLN latent space, allowing the model to pull the current belief back to the true manifold at each step using historical observations.

## Method

### Overall Architecture
NeuroKalman takes as input the multi-view image $v_t$, current 3D position $p_t$, and global instruction $l$ at each step, and outputs the next waypoint $w_t$. It maintains a $d$-dimensional latent belief state $\mathbf{z}_t$. Each time step consists of three blocks:

1. **Prediction Block**—The GRU computes the prior $\tilde{\mathbf{z}}_t$ (dead-reckoning, ignoring current image) using the previous posterior $\mathbf{z}_{t-1}$, previous waypoint $\mathbf{w}_{t-1}$, and hidden state $\mathbf{h}_{t-1}$.
2. **Update Block**—The MLLM (EVA-CLIP vision + Vicuna-7B) processes "current vision + historical anchors retrieved from the memory bank + instruction + position", outputting measurement representation $\mathbf{r}_t$ and confidence $\sigma_t \in [0,1]$.
3. **Kalman Correction**—A learnable Kalman gain $\mathbf{K}_t$ fuses $\tilde{\mathbf{z}}_t$ and $\mathbf{r}_t$ into the posterior $\mathbf{z}_t$. Visual representations decoded from $\mathbf{z}_t$ are selectively written back to the memory bank as anchors for future retrieval if $\sigma_t > 0.5$.

Finally, $\mathbf{z}_t$ is fed to the waypoint prediction head to predict $w_t$, and also becomes $\mathbf{z}_{t-1}$ for the next step, forming a closed loop.

### Key Designs

1. **GRU Prior as Dead-Reckoning Channel**:

    - **Function**: Extrapolates the current state using only "previous belief + previous action", serving as the Kalman filter prior, without current observation.
    - **Mechanism**: $\mathbf{h}_t = \mathrm{GRU}([\mathbf{z}_{t-1}, \mathbf{w}_{t-1}], \mathbf{h}_{t-1})$, $\tilde{\mathbf{z}}_t = \mathrm{MLP}_{prior}(\mathbf{h}_t)$. This channel intentionally "runs blind", not using vision, ensuring it purely reflects kinematic priors, leaving visual information for the update channel as independent evidence.
    - **Design Motivation**: Mixing vision into the prior would break the independence between "measurement" and "prediction", undermining optimal Kalman fusion; the GRU ensures temporal smoothness, helping the update channel filter out high-frequency noise.

2. **Memory Retrieval = KDE-based Likelihood**:

    - **Function**: Retrieves historical visual anchors from the memory bank $\mathcal{M} = \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^{N}$ as a non-parametric estimate of the likelihood $P(o_t|\mathbf{z}_t)$.
    - **Mechanism**: Based on Nadaraya-Watson kernel regression, retrieval is written as $\hat{\mathbf{z}}_{evi} = \sum_i \mathcal{K}(\mathbf{f}_t, \mathbf{f}_i) \mathbf{f}_i / \sum_j \mathcal{K}(\mathbf{f}_t, \mathbf{f}_j)$, with kernel $\mathcal{K}(\mathbf{x}, \mathbf{y}) = \exp(\mathbf{x}^\top \mathbf{y}/\sqrt{d})$, which **degenerates to softmax attention**—thus, attention is not just an "engineering trick" but an exact implementation of KDE for likelihood. Memory writing uses a post-correction strategy: only posterior visual features with $\sigma_t > 0.5$ are written, preventing noisy samples from contaminating the evidence bank.
    - **Design Motivation**: Explicit probabilistic models are nearly impossible in high-dimensional visual space, but KDE only requires samples; equating it with attention allows likelihood estimation to be seamlessly integrated into the MLLM pipeline, and the memory bank, as a fixed set of "evaluated anchors", requires no gradient updates, making it naturally suited for test-time correction.

3. **Learnable Kalman Gain as Uncertainty Modulator**:

    - **Function**: Dynamically determines whether to trust the prior or measurement at each step, replacing the explicit covariance noise models $\mathbf{Q}, \mathbf{R}$ in classical Kalman filters.
    - **Mechanism**: $\mathbf{K}_t = \mathrm{Sigmoid}(\mathbf{W}_g [(\mathbf{r}_t - \tilde{\mathbf{z}}_t); \phi(\sigma_t)] + \mathbf{b}_g)$, concatenating "innovation (residual)" and "MLP projection of measurement confidence" through a gating network to obtain per-dimension gain; then $\mathbf{z}_t = \tilde{\mathbf{z}}_t + \mathbf{K}_t \odot (\mathbf{r}_t - \tilde{\mathbf{z}}_t)$ completes the Bayesian update. Algebraically, this is equivalent to the classical Kalman update $\mathbf{z}_{post} = \mathbf{z}_{prior} + \mathbf{K}_t(\mathbf{y}_t - \mathbf{H}\mathbf{z}_{prior})$ ($\mathbf{H} = \mathbf{I}$).
    - **Design Motivation**: Fixing $\mathbf{K}_t$ leads to failure in ablation—$\mathbf{K}_t = 0.1$ (trusting only the prior) causes catastrophic drift (SR=0%), $\mathbf{K}_t = 0.9$ (trusting only measurement) loses temporal smoothness (SR=18%); learnable gain can automatically balance "smoothing" and "correction" based on current innovation, remaining robust across noise regimes.

### Loss & Training
EVA-CLIP vision backbone and Vicuna-7B language backbone are frozen; only the visual projector, waypoint predictor, and LoRA layers are trained. In addition to the main waypoint loss, an extra $L_1$ supervision (weight 0.2) is applied to both the prior $\tilde{\mathbf{z}}_t$ and measurement $\mathbf{r}_t$, forcing both channels to independently predict waypoints and preventing free-riding. Adam optimizer, lr=$5\mathrm{e}{-5}$, batch=16, 4×A6000. All experiments pretrain on 100% data, then fine-tune on a fixed 10% subset of training trajectories.

## Key Experimental Results

### Main Results

TravelUAV's UAV-Need-Help benchmark: 12,149 human-operated trajectories, 20 training scenes + 2 Unseen-Map scenes, 89 object categories; metrics: NE↓ (meters), SR↑, OSR↑, SPL↑; difficulty split by distance <250 m / ≥250 m (Easy/Hard), and by instruction assistance level L1/L2/L3.

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

The most striking comparison is Test-Seen-Hard: with only 10% data fine-tuning, NeuroKalman achieves SR (20.1%) surpassing TravelUAV trained on 100% data (12.8%), and NE drops from 152 to 105.

### Ablation Study

| Configuration | NE↓ | SR↑ | Notes |
|---|---|---|---|
| $\mathbf{K}_t = 0.1$ (prior only) | 217.09 | 0.00 | No correction, full drift, navigation fails |
| $\mathbf{K}_t = 0.5$ (fixed equal weight) | 83.14 | 24.12 | Better than baseline, but not as strong as adaptive gain |
| $\mathbf{K}_t = 0.9$ (measurement only) | 100.96 | 18.05 | Loses temporal smoothness, retrieval noise backfires |
| **Learnable $\mathbf{K}_t$** | **71.56** | **25.86** | Adaptive weighting |
| Memory length $M = 5$ | 84.39 | 21.23 | Not enough historical anchors |
| **$M = 10$** | **71.56** | **25.86** | Sweet spot |
| $M = 15$ | 77.17 | 23.77 | Outdated anchors introduce noise |
| Write threshold $\sigma_t = 0.3$ | 82.45 | 20.50 | Low threshold, noisy anchors pollute memory |
| **$\sigma_t = 0.5$** | **71.56** | **25.86** | Best |
| TravelUAV + post-hoc classical KF | 96.67 | 18.17 | Geometric smoothing in output space only marginally helps; correction must be in latent space |

### Key Findings
- **Learnable vs fixed gain yields up to 25 SR points difference**—fixed $\mathbf{K}_t = 0.1$ collapses to zero, showing open-loop dead-reckoning without correction is disastrous, while blindly trusting measurement is also suboptimal; per-step, per-dimension uncertainty awareness is key.
- **Memory length shows a clear U-shaped curve**, with $M = 10$ optimal; suggests that for 100–200 step UAV trajectories, about 10 high-quality historical anchors suffice to cover the local manifold, while more introduces outdated visual noise that distracts attention.
- **Post-hoc classical Kalman smoothing in output space only raises SR from 16.1% to 18.2%**, far less than NeuroKalman's 25.9%; this is the strongest control—correction must be in latent semantic space, not just geometric smoothing in (x, y, z).
- **Drift curve**: TravelUAV's position $L_2$ error diverges linearly after 100 steps until failure; NeuroKalman rises to ~30–40 meters early on and then **stops growing**, visually demonstrating the effect of the Kalman closed loop.

## Highlights & Insights
- The mathematical equivalence of **"attention = KDE likelihood"** is a true unifying insight—previously, retrieval augmentation was seen as an engineering trick; this work clarifies it as a discretized nonparametric Bayesian likelihood estimator, providing a probabilistic interpretation for retrieval-augmented methods, which can be transferred to any "prediction + retrieval" architecture (RAG-LLM, world models, TTA).
- The **post-correction memory writing strategy** is clever: memory only accepts samples already corrected by Kalman and self-reported as high-confidence, ensuring the memory bank contains only "verified anchors" and avoiding the vicious cycle of "the dirtier the cache, the more it is retrieved, and vice versa" seen in traditional retrieval caches.
- The phenomenon of **10% data outperforming 100% data** is crucial—dead-reckoning models rely on "memorizing all possible transitions" with abundant data, but overfit with less data; NeuroKalman explicitly encodes "long-range consistency" as an inductive bias in the architecture, obviating the need for large data to emerge this ability, exemplifying how structured priors can outperform brute-force scaling.

## Limitations & Future Work
- The authors acknowledge that using GRU for the prior leads to information decay over very long horizons, but their main contribution is the Bayesian correction framework itself; GRU can be replaced with Transformer/Mamba, etc.
- Additional limitations: (1) Memory bank size $M=10$ is tuned manually; optimal value may vary across scenarios/tasks, with no adaptive $M$ mechanism; (2) Post-correction writing uses a fixed $\sigma_t > 0.5$ threshold, which may prevent any anchors from being stored early in trajectories when model confidence is low; (3) All experiments are in AirSim simulation—robustness to real UAV visual noise and kinematic mismatch is untested; (4) While the KDE equivalence is a theoretical highlight, the actual architecture is standard attention, so the incremental engineering contribution may be somewhat overstated.
- Possible improvements: Make the memory write threshold $\sigma_t$ learnable; use contrastive loss to explicitly separate the "prior-only" and "measurement-only" channels, further ensuring their independence (a prerequisite for optimal Kalman fusion).

## Related Work & Insights
- **vs TravelUAV / OpenVLN**: Both use step-by-step waypoint regression with no explicit error correction, leading to inevitable drift over long horizons; this work adds a Bayesian correction loop atop their backbones.
- **vs MapNet / SkyVLN / OpenFly (topological map memory)**: These treat memory as a "passive buffer" for feature concatenation, whereas this work treats memory as "probabilistic evidence" for Bayesian fusion—a paradigm shift from "passive aggregation" to "active correction".
- **vs FEEDTTA / FSTTA (TTA methods)**: These rely on online gradient updates to model weights to counter distribution shift, but lack reliable supervision in VLN and may reinforce existing errors; NeuroKalman leaves weights untouched, correcting only the belief state, which is safer.
- **vs KalmanNet / Backprop-KF (Deep Bayesian Filtering)**: These can learn transitions well, but likelihood is hard to define in high-dimensional vision; this work uses KDE-attention to fully delegate the likelihood problem to retrieval, providing a high-dimensional visual version for this line of research.

## Rating
- Novelty: ⭐⭐⭐⭐ The equivalence of "attention = KDE likelihood" and transplanting the full Kalman prediction-update into VLN latent space offer a clean, theoretically grounded new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Seen/Unseen-Map/Unseen-Object splits, L1/L2/L3 levels, drift visualization, and multiple ablations, though only on the TravelUAV benchmark.
- Writing Quality: ⭐⭐⭐⭐ The derivation from Bayesian framework → KDE equivalence → architectural implementation is very clear and coherent.
- Value: ⭐⭐⭐⭐ Delivers significant gains on both long-horizon and low-data VLN challenges, and the method can be easily transferred to other "prior + retrieval" sequential decision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[ICCV 2025\] NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments](../../ICCV2025/robotics/navmorph_a_self-evolving_world_model_for_vision-and-language_navigation_in_conti.md)
- [\[ICLR 2026\] Token Taxes: Mitigating AGI's Economic Risks](../../ICLR2026/robotics/token_taxes_mitigating_agis_economic_risks.md)

</div>

<!-- RELATED:END -->
