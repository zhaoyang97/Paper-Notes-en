---
title: >-
  [Paper Note] SPARC: OOD Generalization for Controlling 100 Unseen Vehicles with a Single Policy
description: >-
  [AAAI 2026][Autonomous Driving][OOD Generalization] This paper proposes SPARC (Single-Phase Adaptation for Robust Control), which unifies the two-phase context encoding and history-based adaptation of RMA into a single-p…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "OOD Generalization"
  - "Contextual Reinforcement Learning"
  - "Single-Phase Adaptation"
  - "Gran Turismo"
  - "SPARC"
date: 2026-05-08
content_hash: 5bcab6f3c0b3c8af
---

# SPARC: OOD Generalization for Controlling 100 Unseen Vehicles with a Single Policy

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.09737](https://arxiv.org/abs/2511.09737)  
**Code**: [https://github.com/bramgrooten/sparc](https://github.com/bramgrooten/sparc)  
**Area**: Reinforcement Learning / Policy Generalization / Autonomous Driving
**Keywords**: OOD Generalization, Contextual Reinforcement Learning, Single-Phase Adaptation, Gran Turismo, SPARC

## TL;DR

This paper proposes SPARC (Single-Phase Adaptation for Robust Control), which unifies the two-phase context encoding and history-based adaptation of RMA into a single-phase training procedure. Using a single policy in the high-fidelity Gran Turismo 7 racing simulator, SPARC achieves state-of-the-art OOD generalization performance across 100+ unseen vehicles.

## Background & Motivation

Deep reinforcement learning has achieved notable success in robotics, nuclear fusion control, and racing simulators, yet generalizing to unseen environments (OOD contexts) remains a central challenge. Environmental conditions such as friction coefficients, wind speed, and vehicle dynamics may change unpredictably at deployment, leading to catastrophic failures.

Rapid Motor Adaptation (RMA) is a representative method in this line of work, employing a two-phase training procedure:
1. Train an expert policy with a context encoder $\psi(c) = z$ using privileged context information $c$ via reinforcement learning (PPO/QR-SAC).
2. Freeze the expert policy and train a history-based adaptation module $\phi(h) = \hat{z}$ by regressing the encoder output $z$ via MSE.
3. At deployment, only the adapter policy $\pi_{ad}$ is used (no privileged information $c$ required).

However, the two-phase approach has notable drawbacks: complex implementation, the need for careful checkpoint selection at the end of phase one, no straightforward support for continual learning, and multi-dimensional evaluation required for intermediate model selection.

The core insight of SPARC is: **merging the two phases into a single simultaneous training stage, where the context encoding $z$ is a non-stationary (moving) target, yet the adapter is capable of handling this learning dynamic**.

## Method

### Overall Architecture

Two policy networks are trained simultaneously:
- **Expert policy $\pi_{ex}$**: receives observation $o$ plus privileged context $c$; contains a context encoder $\psi(c) = z$; trained via QR-SAC (32 quantiles).
- **Adapter policy $\pi_{ad}$**: receives observation $o$ plus history $h$ ($H = 50$ obs-action pairs); contains a history adapter $\phi(h) = \hat{z}$; $\phi$ is trained via MSE supervision to regress the output $z$ of $\psi$.

Both networks share the same backend decision network architecture. $\pi_{ad}$ periodically copies weights from $\pi_{ex}$ (excluding the $\phi$ module). At test time, only $\pi_{ad}$ is deployed (no privileged information $c$ required). Evaluation uses the BIAI ratio (RL agent lap time / built-in AI lap time; lower is better).

### Key Designs

1. **Single-phase training**: Unlike RMA, SPARC simultaneously updates $\pi_{ex}$ (RL objective) and $\phi$ (MSE objective $L_\phi = \mathbb{E}[(z - \hat{z})^2]$) within the same training loop. Although $z = \psi(c)$ is a non-stationary target for $\phi$, experiments show that the adapter can track this moving target. Key advantages include: no need to select an optimal phase-one checkpoint, support for indefinite continual training, and natural compatibility with distributed asynchronous training systems.
2. **Adapter-collected experience**: $\pi_{ad}$ (rather than $\pi_{ex}$) acts in the environment to collect experience. This brings $\pi_{ad}$'s learning closer to an on-policy setting and allows it to correct its own inference inaccuracies before deployment. Ablation studies confirm this choice is superior across most OOD settings. This design leads to greater training-deployment distribution consistency, as $\pi_{ad}$ faces the state distribution induced by its own imperfect inference during training.
3. **Network architecture**: The history adapter $\phi$ uses a 1D CNN (kernel = 8, 5, 5; stride = 4, 1, 1) followed by FC layers to process $H$ steps of (observation, action) pairs. The output dimensions of $\psi$ and $\phi$ are identical; their outputs are concatenated with the observation embedding $\ell$ and fed into the decision layers (2048-dim FC × 2 → 2-dim control output: throttle/brake + steering angle). The critic network shares the expert policy architecture and has access to context $c$.

### Loss & Training

- **RL loss** ($\pi_{ex}$): QR-SAC (Quantile Regression Soft Actor-Critic), 32 quantiles.
- **Adapter loss**: $L_\phi = \mathbb{E}[(z - \hat{z})^2]$, MSE regression of the expert's context encoding.
- Training configuration: 9M steps on Gran Turismo (12M steps on Nürburgring), 3M steps on MuJoCo. Distributed asynchronous training with up to 20 PlayStations collecting experience simultaneously, with an A100 GPU for computation. A single GT7 training run takes approximately 6 days.

## Key Experimental Results

### Main Results: Gran Turismo 7 Track Performance

| Track | Metric | SPARC | RMA | History Input | Only Obs | Oracle |
|-------|--------|-------|-----|--------------|----------|--------|
| Grand Valley | OOD BIAI ratio↓ | **1.049** | 1.056 | 1.083 | 1.064 | 1.135 |
| Grand Valley | OOD Success Rate↑ | **98.1%** | 97.1% | 92.6% | 95.2% | 90.9% |
| Nürburgring | OOD BIAI ratio↓ | **1.120** | 1.300 | 1.120 | 1.175 | 1.118 |
| Nürburgring | OOD Success Rate↑ | **89.0%** | 78.0% | 86.7% | 81.9% | 89.6% |
| Catalunya | OOD BIAI ratio↓ | 0.963 | 0.967 | **0.955** | 0.956 | 1.135 |
| Catalunya | OOD Success Rate↑ | **100%** | **100%** | 99.3% | **100%** | 85.3% |

### Ablation Study

| Configuration | GV OOD ratio↓ | Nür OOD ratio↓ | Note |
|---------------|--------------|----------------|------|
| SPARC ($\pi_{ad}$ collects) | **1.049** | 1.120 | Default |
| SPARC ($\pi_{ex}$ collects) | 1.069 | **1.099** | Better on some tracks but less consistent overall |
| History $H=50$ | **1.049** | — | Optimal history length |
| History $H=10$ | 1.067 | — | Insufficient information to infer context |
| History $H=100$ | 1.055 | — | Overly long history dilutes relevant signals |
| MuJoCo HalfCheetah | SPARC: 10018 | — | vs. RMA 9034 (+10.9%) |
| Power & Mass experiment | **0.991** | — | Surpasses Oracle (0.996) |

### Key Findings

- SPARC **surpasses Oracle** in the Power & Mass experiment (0.991 vs. 0.996), suggesting that history-based inference is more robust than explicit context encoding.
- Oracle (with privileged information) performs worse under OOD conditions (Grand Valley: 1.135), indicating overfitting to the training vehicle distribution.
- After a physics engine update, SPARC demonstrates minimal performance degradation in zero-shot transfer, confirming that it learns transferable context representations.
- Having the adapter policy collect experience aligns the training and deployment state distributions more closely, forming tighter training-deployment consistency.
- The large-scale experimental design spanning 3 tracks × 100+ OOD vehicles is exceptionally rare in the context-adaptive RL literature.

## Highlights & Insights

- **Elegant simplicity**: eliminates two-phase training, intermediate checkpoint selection, and separate training pipelines—a single loop handles everything.
- **Surpassing Oracle with privileged information** demonstrates the value of history-based reasoning—in-context adaptation can be more robust than explicit context encoding.
- Robustness after the GT7 physics engine update (zero-shot transfer) indicates that the model learns an abstract representation of vehicle dynamics.
- The insight of using $\pi_{ad}$ to collect experience—closer to on-policy learning—benefits the quality of the ultimately deployed policy.
- Naturally compatible with asynchronous distributed computing and continual learning, making it deployment-friendly.

## Limitations & Future Work

- Validation is limited to simulators; sim-to-real transfer on physical robots has not been tested.
- The Gran Turismo codebase is based on a proprietary platform and is not open-source; only the MuJoCo component is reproducible.
- The history length $H = 50$ is a global hyperparameter and is not adaptively adjusted based on environment complexity.
- Training computational cost is high (approximately 6 days per GT7 run, requiring 20 PlayStations).
- No direct comparison with meta-RL methods (e.g., MAML).

## Related Work & Insights

- vs. RMA: single-phase vs. two-phase; SPARC achieves superior OOD performance and naturally supports continual learning.
- vs. Domain Randomization (Only Obs): explicit use of contextual priors yields systematic improvements.
- vs. Oracle: SPARC's ability to surpass Oracle suggests that context inference can be more robust than directly accessing context information.
- The single-phase adaptation paradigm is generalizable to robot locomotion, UAV control, industrial robotics, and other domains requiring context-adaptive RL.

## Rating

⭐⭐⭐⭐⭐ (5/5)
The method is elegant and effective, demonstrating impressive OOD generalization in the challenging Gran Turismo 7 environment. The experiments are exceptionally comprehensive—3 tracks × 500 vehicles + MuJoCo + physics engine transfer + detailed ablations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](../../CVPR2026/autonomous_driving/open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[ICCV 2025\] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters](../../ICCV2025/autonomous_driving/rescue_crowd_evacuation_simulation_via_controlling_sdm-united_characters.md)
- [\[ICCV 2025\] Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks](../../ICCV2025/autonomous_driving/wavelet_policy_lifting_scheme_for_policy_learning_in_long-horizon_tasks.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)
- [\[CVPR 2026\] HorizonForge: Driving Scene Editing with Any Trajectories and Any Vehicles](../../CVPR2026/autonomous_driving/horizonforge_driving_scene_editing_with_any_trajectories_and_any_vehicles.md)

</div>

<!-- RELATED:END -->
