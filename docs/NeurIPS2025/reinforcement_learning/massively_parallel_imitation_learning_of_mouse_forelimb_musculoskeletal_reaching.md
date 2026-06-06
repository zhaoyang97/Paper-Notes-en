---
title: >-
  [Paper Note] Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics
description: >-
  [NeurIPS 2025][Reinforcement Learning][musculoskeletal simulation] This work presents MIMIC-MJX, a massively parallel imitation learning pipeline for mouse forelimb musculoskeletal simulation. Leveraging JAX-accelerated…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "musculoskeletal simulation"
  - "mouse forelimb"
  - "PPO"
  - "MuJoCo-MJX"
  - "EMG prediction"
  - "imitation learning"
  - "Takens theorem"
date: 2026-05-08
content_hash: 1835ac85ef394d49
---

# Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics

**Conference**: NeurIPS 2025
**arXiv**: [2511.21848](https://arxiv.org/abs/2511.21848)  
**Code**: [track-mjx](https://github.com/talmolab/track-mjx) + [stac-mjx](https://github.com/talmolab/stac-mjx)  
**Area**: Computational Neuroscience / Biomechanics / Imitation Learning
**Keywords**: musculoskeletal simulation, mouse forelimb, PPO, MuJoCo-MJX, EMG prediction, imitation learning, Takens theorem

## TL;DR
This work presents MIMIC-MJX, a massively parallel imitation learning pipeline for mouse forelimb musculoskeletal simulation. Leveraging JAX-accelerated PPO at 1.2 million steps/second across thousands of parallel environments, the pipeline trains physically-informed imitation learning policies. The study demonstrates that control cost regularization enables simulated muscle activity to better predict real EMG signals, and employs a Takens-theorem-based nonlinear dynamical systems approach to predict muscle activation from joint kinematics.

## Background & Motivation
**Background**: Understanding brain–body interaction—specifically the sensorimotor transformations underlying embodied control—is a central goal of neuroscience. Motor control research has historically relied on kinematic observations alone to infer neural mechanisms, largely neglecting the underlying musculoskeletal dynamics and physical constraints.

**Limitations of Prior Work**:
- Dynamical parameters are difficult to measure directly in experiment, particularly in small animals such as mice.
- Existing simulation platforms are computationally prohibitive, precluding large-scale parameter searches and high-throughput experimentation.
- High-fidelity musculoskeletal models of the mouse forelimb are scarce, and available models lack systematic validation against EMG data.

**Key Challenge**: A comprehensive understanding of embodied control requires simultaneously modeling behavioral dynamics, biomechanics, and neural circuit architecture; however, conventional simulation speeds render such integrative modeling infeasible.

**Key Insight**: Recent work (Aldarondo et al., 2024) demonstrated that physically constrained imitation learning can reproduce experimentally observed motor behaviors and predict real neural activity. Replicating this paradigm for the mouse forelimb, combined with GPU-accelerated high-throughput training, would enable systematic investigation of how physical constraints shape control strategies.

**Core Idea**: JAX + MuJoCo-MJX massively parallel imitation learning + physics-informed regularization = efficient and biologically plausible mouse forelimb musculoskeletal simulation.

## Method

### Overall Architecture
The full pipeline comprises three stages: (1) **3D pose estimation** — SLEAP-Anipose reconstructs keypoints from multi-camera video; (2) **model registration** — STAC-MJX registers motion capture data to a MuJoCo musculoskeletal model to generate reference kinematics; (3) **imitation learning** — TRACK-MJX uses PPO to reproduce reference motions within physical simulation.

### Key Designs
1. **Musculoskeletal Model**:
    - Constructed from light-sheet microscopy data; 4 degrees of freedom (3 shoulder DOFs: elevation/rotation/extension + 1 elbow DOF: flexion-extension).
    - 9 Hill-type muscle actuators: triceps brachii (long/lateral heads), biceps brachii (long head), brachialis, pectoralis major (clavicular portion), latissimus dorsi, deltoid (anterior/middle/posterior).
    - Muscle parameters tuned so that generated forces fall within the physiological range of real mouse forelimb muscles (0.2–1.2 N).

2. **STAC-MJX Registration**:
    - Transforms 3D motion capture data into the MuJoCo model coordinate frame.
    - Bayesian inverse kinematics solves for joint angle sequences (tolerance 1e-20, max iterations 600).
    - Registration error < 1 mm; produces target reference trajectories for imitation learning.

3. **TRACK-MJX Physics-Informed Imitation Learning**:
    - PPO with an encoder–decoder architecture separated by a multivariate Gaussian information bottleneck (KL regularization).
    - The encoder receives STAC-registered reference trajectories; the decoder outputs muscle control signals.
    - Both encoder and decoder are 3-layer MLPs with 512 units per layer.
    - The bottleneck latent space represents "movement intent" — a compressed encoding of the trajectory.
    - Reward function: $r_t = \lambda_{joint} r_t^{joint} - \lambda_{ctrl} c_t^{ctrl} - \lambda_{energy} c_t^{energy}$
        - Joint reward: $r_t^{joint} = \exp(-\alpha_{joint} \sum_i (q_{t,i} - \hat{q}_{t,i})^2)$, encouraging matching of reference joint angles.
        - Control cost: penalizes the squared sum of action magnitudes to promote smoother actuation.
        - Energy cost: penalizes the product of joint velocity and actuator force $c_t^{energy} = \sum_j |v_{t,j}| \cdot |f_{t,j}^{act}|$, promoting energy efficiency.

4. **Massively Parallel Training**:
    - JAX + MuJoCo-MJX enables GPU acceleration; 4,096 parallel environments on dual A40 GPUs achieve **1.2 million steps/second**.
    - 2,048 parallel environments on a single A40 achieve 600,000 steps/second.
    - Convergence within 40M steps, completing in approximately 30 seconds per training run.

5. **Takens Theorem Nonlinear Prediction**:
    - Reconstructs dynamical attractors via Takens delay embedding: $\Phi(x_t) = (x(t), x(t+\tau), x(t+2\tau), \dots, x(t+(m-1)\tau))$
    - Exploits the homeomorphism between shadow manifolds for cross-variable prediction: the manifold $\mathcal{M}_x$ reconstructed from joint angles maps to the muscle activation manifold $\mathcal{M}_y$.
    - Simplex projection: identifies $k = E+1$ nearest neighbors forming a simplex on the shadow manifold and interpolates to predict future states.
    - Optimal parameters: delay $\tau = -1$, embedding dimension $E = 3$ (actions) / $E = 2$ (joints), prediction horizon $T_p = 5$.

### Experimental Data Collection
- Head-fixed mice (n = 1, 46 trials) performed a goal-directed water-reaching task.
- 3-camera SLEAP pose estimation → SLEAP-Anipose triangulation → 3D keypoints.
- Intramuscular EMG electrodes recorded biceps and triceps activity (30 kHz; bandpass 20–1000 Hz; 50 Hz low-pass envelope extraction).
- Each trial was segmented to a 300 ms reach epoch and downsampled to 200 Hz.

## Key Experimental Results

### Core Metrics

| Metric | Value |
|--------|-------|
| Training speed | **1.2M steps/sec** (dual A40, 4096 envs) |
| STAC registration error | < 1 mm (mean across all keypoints) |
| Imitation tracking error | < 1 mm |
| Convergence time | ~40M steps |
| Simulated biceps → joint prediction | Simplex $\rho$ = **0.802** |
| Simulated triceps → joint prediction | Simplex $\rho$ = **0.789** |
| Reference joints + simulated actions → triceps EMG | Simplex $\rho$ = **0.7** |
| Reference joints + simulated actions → biceps EMG | Simplex $\rho$ = 0.328 |

### Control Cost Parameter Search

| $\lambda_{ctrl}$ | Joint Reward | EMG Fit | High-Frequency Activity |
|------------------|--------------|---------|------------------------|
| 0 | Highest | Poor | High |
| 0.1 | Slightly reduced | Improved | Reduced |
| 0.15–0.2 | Moderate | **Optimal** | Low |
| 0.3–0.4 | Significantly reduced | Over-regularized | Lowest |

### Latent Space Analysis

| Layer | PC1/PC2/PC3 Variance Explained | Characteristics |
|-------|-------------------------------|-----------------|
| Intent bottleneck | 45.2%/32.1%/20.7% (98%) | Highly compressed |
| Decoder layer 1 | 29.5%/20.3%/12.0% | Expanded mixing |
| Decoder layer 2 | 25.9%/17.3%/11.6% | Feature transformation |
| Decoder layer 3 | 55.2%/16.4%/6.5% (78%) | Re-compressed → muscle synergies |

### Key Findings
- **Control cost as a critical biological prior**: Without regularization, simulated muscles exhibit unnatural high-frequency oscillations; introducing control cost yields biceps activation patterns substantially closer to real EMG.
- **An optimal regularization regime exists**: At $\lambda_{ctrl} \in [0.15, 0.2]$, joint reward and EMG MAE are simultaneously acceptable; excessive values degrade joint tracking.
- **Energy cost offers limited benefit**: The current formulation ($|v| \cdot |f^{act}|$) does not meaningfully improve EMG fit; penalizing mechanical work may be more effective.
- **Latent space structure is interpretable**: The encoder-compression → decoder-expansion → final re-compression pattern is consistent with a motor control hierarchy of "movement intent → muscle synergies."
- **Triceps EMG is more predictable than biceps**: Likely because the shoulder joint is fixed in the current model, requiring the biceps to compensate with larger, less structured activity.

## Highlights & Insights
- **Computational efficiency breakthrough**: 1.2M steps/second reduces parameter searches from days to minutes, making high-throughput experimentation tractable.
- **Physical constraints as biological priors**: Control cost is not merely an engineering optimization trick but serves as a critical bridge between simulation and real EMG—a finding with broad implications for musculoskeletal simulation research.
- **Novel application of Takens theorem**: Using nonlinear dynamical systems methods to predict muscle activation from kinematics avoids the ill-posedness of direct inverse dynamics, and enables cross-domain validation of biological plausibility.
- **Information bottleneck design of the encoder–decoder**: The latent space corresponds to "movement intent," providing both functional interpretability (abstraction at the level of motor planning) and utility for downstream analysis.

## Limitations & Future Work
- **Severely limited data**: Only 1 mouse, a single target location, and 46 trials; statistical power is weak and generalizability cannot be assessed.
- **Fixed shoulder joint**: This simplification causes biceps compensation, resulting in poor EMG prediction accuracy ($\rho = 0.328$); relaxing this constraint may yield substantial improvements.
- **Simplified muscle model**: Only 9 actuators are included; the real mouse forelimb has more muscles, and the wrist joint is absent.
- **Energy cost formulation requires refinement**: $|v| \cdot |f^{act}|$ does not correspond to true metabolic cost; physiologically grounded energy terms (e.g., ATP consumption models) may be more effective.
- **Causal validity of Simplex prediction is questionable**: High trial-to-trial repeatability with a single target location may produce spurious predictability; multi-target and perturbation experiments are necessary for proper validation.
- **No comparison with real neural data**: Aldarondo et al. (2024) demonstrated that a virtual rodent can predict real brain activity; the present work validates only EMG prediction and does not address cortical or spinal neural activity.

## Related Work & Insights
- **vs. Aldarondo et al. (2024)**: The Virtual Rodent uses a whole-body musculoskeletal model to predict neural activity; the present work focuses on the forelimb with greater anatomical detail but has not yet achieved neural predictive validity.
- **vs. OpenSim**: Traditional biomechanics platforms are CPU-bound; MIMIC-MJX's GPU acceleration provides orders-of-magnitude speedup.
- **vs. DeepMimic (Peng et al.)**: DeepMimic applies physics-based imitation learning to humanoid characters; the present work extends this paradigm to real animal musculoskeletal systems with EMG validation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of massively parallel musculoskeletal simulation with Takens reconstruction constitutes a novel tool–method pairing.
- **Experimental Thoroughness**: ⭐⭐⭐ — Methodological validation is clear, but the dataset is extremely limited (1 mouse); larger-scale validation is required.
- **Writing Quality**: ⭐⭐⭐⭐ — Interdisciplinary integration is handled well and the pipeline is described clearly, though some sections of the discussion are verbose.
- **Value**: ⭐⭐⭐⭐ — The MIMIC-MJX platform represents a significant tooling contribution to the computational neuroscience and biomechanics communities, with broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](quantifying_generalisation_in_imitation_learning.md)
- [\[NeurIPS 2025\] PARCO: Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](parco_parallel_autoregressive_models_for_multi-agent_combinatorial_optimization.md)
- [\[NeurIPS 2025\] Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)

</div>

<!-- RELATED:END -->
