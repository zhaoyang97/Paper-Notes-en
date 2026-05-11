---
title: >-
  [Paper Note] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][End-to-end planning] MeanFuser is an end-to-end autonomous driving framework that replaces discrete trajectory vocabulary with Gaussian mixture noise for continuous multi-modal trajectory…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "End-to-end planning"
  - "MeanFlow"
  - "Gaussian mixture noise"
  - "one-step sampling"
  - "adaptive trajectory reconstruction"
date: 2026-05-08
content_hash: 139bb20bfda06143
---

# MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2602.20060](https://arxiv.org/abs/2602.20060)
**Code**: [https://github.com/wjl2244/MeanFuser](https://github.com/wjl2244/MeanFuser)
**Area**: Autonomous Driving
**Keywords**: End-to-end planning, MeanFlow, Gaussian mixture noise, one-step sampling, adaptive trajectory reconstruction

## TL;DR
MeanFuser is an end-to-end autonomous driving framework that replaces discrete trajectory vocabulary with Gaussian mixture noise for continuous multi-modal trajectory modeling, leverages the MeanFlow Identity for error-free one-step sampling, and introduces an Adaptive Reconstruction Module (ARM) that implicitly decides between selecting an existing proposal and reconstructing a new trajectory. On NAVSIM, using only RGB input with a ResNet-34 backbone, it achieves 89.0 PDMS at 59 FPS.

## Background & Motivation

**Background**: End-to-end autonomous driving learns directly from sensor inputs to planning trajectories. Methods such as TransFuser, UniAD, and VAD perform well with unimodal trajectory prediction but fail to capture the inherently multi-modal nature of driving behavior. VADv2 and Hydra-MDP introduce trajectory vocabulary to predict probability distributions, but a fixed vocabulary involves a trade-off between efficiency and robustness. DiffusionDrive and GoalFlow bring generative models into trajectory planning; however, the former requires multi-step sampling and the latter relies on discrete anchors.

**Limitations of Prior Work**: (1) **Inherent limitations of discrete anchor vocabularies** — a vocabulary must be large enough to cover the trajectory distribution at test time, yet large vocabularies slow down inference. When test scenarios fall outside the predefined anchor distribution, all proposals deviate from the optimal trajectory. (2) **Computational overhead of multi-step sampling** — flow matching requires multiple ODE solver steps (e.g., GoalFlow needs 5 steps) to achieve optimal performance, and the ODE solver introduces numerical errors that curve the sampling path. (3) **Mode collapse from standard Gaussian noise** — vanilla methods sample from a standard Gaussian, leading to insufficient trajectory diversity.

**Key Challenge**: How can multi-modal driving behavior be effectively modeled without relying on a fixed discrete vocabulary, while maintaining high inference efficiency?

**Goal**: (1) Eliminate dependence on discrete trajectory vocabularies; (2) achieve high-quality one-step sampling; (3) handle cases where all sampled proposals are suboptimal.

**Key Insight**: Introduce the MeanFlow Identity into end-to-end planning. MeanFlow directly models the mean velocity field between the noise distribution and the trajectory distribution rather than the instantaneous velocity field, enabling exact single-step sampling without numerical error. A Gaussian mixture model is used as the prior distribution, with each component capturing one driving mode.

**Core Idea**: Replace anchors with Gaussian mixture noise, replace multi-step ODE solving with MeanFlow, and replace score-based selection with an adaptive reconstruction module — three complementary designs for fast and robust multi-modal trajectory planning.

## Method

### Overall Architecture
MeanFuser consists of three components: (1) **Scene context encoder**: an image encoder extracts BEV features and a vehicle state encoder captures ego information, with auxiliary map decoding supervision; (2) **Multi-modal trajectory sampling**: samples from an 8-component Gaussian mixture noise distribution and generates multiple trajectory proposals in a single step via a lightweight MeanFlow network; (3) **Adaptive Reconstruction Module (ARM)**: fuses all proposals with BEV features via cross-attention to output the final planned trajectory. Training uses a standard flow matching loss, an ARM reconstruction loss, and a map loss.

### Key Designs

1. **Gaussian Mixture Noise (GMN)**:

    - **Function**: Replaces discrete trajectory vocabulary with a continuous distribution; each Gaussian component captures one driving mode.
    - **Mechanism**: All expert trajectories in the training set are normalized — stepwise differences $\Delta\tau_j$ are computed and normalized by global mean and maximum values, then all normalized trajectories are clustered into $K=8$ groups via K-means. The mean and standard deviation of each group parameterize a Gaussian component: $p_0 = \sum_{k=1}^K \pi_k \mathcal{N}(\mu_k, \sigma_k^2 \cdot I)$. At inference, one noise sample is drawn from each component, generating 8 multi-modal trajectories in parallel. During training, the Gaussian component closest to the ground truth is selected to compute the loss.
    - **Design Motivation**: Standard Gaussian sampling leads to mode collapse, while discrete anchors cannot cover the continuous space. GMN combines the advantages of trajectory priors (cluster centers encode typical driving modes) and continuity (variance within each Gaussian allows intra-mode variation). A notable byproduct is that different components naturally correspond to different driving styles (conservative at 3.45 m/s to aggressive at 9.11 m/s), providing a zero-cost interface for personalized driving.

2. **MeanFlow Identity Adapted for End-to-End Planning**:

    - **Function**: Enables exact one-step sampling, eliminating numerical errors from the ODE solver.
    - **Mechanism**: Conventional flow matching learns the instantaneous velocity field $v_\theta(z_t, t)$; even with a linear probability path, the learned field does not guarantee straight-line sampling trajectories, necessitating multi-step ODE solving. MeanFlow directly learns the **mean velocity field** over a time interval: $u(z_t, r, t) = \frac{1}{t-r}\int_r^t v(z_\tau,\tau)d\tau$. The training target is derived via the MeanFlow Identity: $u_{\text{tgt}} = v(z_t,t) - (t-r)(v(z_t,t)\partial_z u_\theta + \partial_t u_\theta)$, with stop-gradient applied. Inference is completed in a single step: $x_1 = x_0 + 1 \cdot u_\theta(x_0, 0, 1)$. The Jacobian-vector product is computed efficiently using `torch.autograd.functional.jvp` during training.
    - **Design Motivation**: GoalFlow requires 5 sampling steps to reach optimal performance, and DiffusionDrive's diffusion process also requires iteration. MeanFlow's one-step sampling brings planning module inference speed to 434 FPS (compared to GoalFlow's 11 FPS, a 39.45× speedup) with no numerical error.

3. **Adaptive Reconstruction Module (ARM)**:

    - **Function**: Implicitly reconstructs a superior trajectory when all sampled proposals are suboptimal.
    - **Mechanism**: All candidate trajectories $\{\hat{\tau}_k\}_{k=1}^K$ are encoded and fused with BEV scene features $c_{\text{bev}}$ via cross-attention; the result is passed through a projector to produce the final trajectory $\hat{\tau}$. The attention weights implicitly learn whether to "select or reconstruct" — if one proposal is sufficiently good, attention concentrates on it (equivalent to selection); if none are good enough, attention distributes across multiple proposals to synthesize a better trajectory. Training uses only expert trajectory L1 supervision: $\mathcal{L}_\tau = \|\tau - \hat{\tau}\|_1$.
    - **Design Motivation**: Hydra-MDP and WoTE score candidates using benchmark sub-metrics (e.g., sub-scores of PDM Score), which depends on benchmark-specific rules and cannot handle cases where all proposals are poor. ARM requires no benchmark rules, is supervised solely by expert trajectories, and can reconstruct rather than merely select.

### Loss & Training
$\mathcal{L} = \lambda_1 \mathcal{L}_\tau + \lambda_2 \mathcal{L}_{\text{flow}} + \lambda_3 \mathcal{L}_{\text{map}}$, where the flow loss and ARM reconstruction loss both use L1, supplemented by map decoding semantic supervision to accelerate convergence. The AdamW optimizer is used with weight decay 0.1, cosine annealing learning rate $2\times10^{-4}$, and 3 epochs of warmup. The hidden dimension is 128 (only 54.6M parameters total), and 1 trajectory is sampled from each of the 8 GMN components, yielding 8 trajectories in total.

## Key Experimental Results

### Main Results

| Method | Input | PDMS↑(v1) | EPDMS↑(v2) | Plan FPS↑ | FPS↑ |
|--------|-------|-----------|------------|-----------|------|
| TransFuser | C&L | 84.0 | 76.7 | 3934 | 63 |
| GoalFlow | C&L | 85.7 | - | 11 | 10 |
| Hydra-MDP | C&L | 86.5 | 81.4 | 25 | 20 |
| DiffusionDrive | C&L | 88.1 | 88.3 | 75 | 39 |
| WoTE | C&L | 88.3 | - | - | - |
| **MeanFuser** | **C only** | **89.0** | **89.5** | **434** | **59** |

Note: MeanFuser surpasses all Camera+LiDAR (C&L) methods using only RGB camera input (no LiDAR). Its parameter count of 54.6M is the smallest among all compared methods.

### Ablation Study

| Configuration | PDMS↑ | N_proposals | P_{L2>0.5}↓ | N_{DAC=0}↓ |
|---------------|-------|-------------|-------------|------------|
| DiffusionDrive | 88.1 | 20 | 20.0% | 84 |
| TransFuser (base) | 84.0 | - | - | - |
| + vanilla MeanFlow (ℳ₀) | 87.3 (+3.3) | 16 | 40.6% | 143 |
| + GMN (ℳ₁) | 88.2 (+0.9) | 16 | 18.5% | 58 |
| + ARM (ℳ₂ = MeanFuser) | **89.0 (+0.8)** | 17 | 16.9% | 48 |
| + simple averaging (ℳ₃) | 71.2 (−17.8) | 17 | 18.0% | 57 |

### Key Findings
- **MeanFlow itself contributes the most (+3.3 PDMS)**: Replacing the MLP with a conditioned MeanFlow decoder yields a substantial improvement, validating the effectiveness of flow-based trajectory distribution modeling.
- **GMN substantially reduces DAC=0 cases**: ℳ₀ has 143 scenarios where all proposals leave the drivable area; adding GMN reduces this to 58 (even fewer than DiffusionDrive's 84), demonstrating that GMN's coverage far exceeds that of standard Gaussian or discrete anchors.
- **Simple averaging of proposals causes catastrophic degradation (−17.8 PDMS)**: This confirms that sampled trajectories genuinely capture distinct modes rather than collapsing to a single mode, and that ARM's implicit select/reconstruct mechanism is essential.
- **ARM further reduces DAC=0 from 58 to 48**: This demonstrates that ARM can reconstruct superior trajectories when all proposals are suboptimal.
- **Camera-only outperforms multi-modal methods**: MeanFuser without LiDAR surpasses all Camera+LiDAR methods, suggesting that perception information is not the bottleneck — planning strategy is.
- **Different Gaussian components naturally correspond to different driving styles**: Speeds range from 3.45 m/s (conservative) to 9.11 m/s (aggressive), providing a zero-cost control interface for personalized driving.

## Highlights & Insights
- **The GMN design is particularly elegant** — K-means clustering of training trajectories followed by Gaussian mixture fitting retains the "mode prior" advantage of anchor-based methods (each cluster center encodes a typical driving mode) while overcoming the fatal limitation of discrete anchors (the variance of each Gaussian allows continuous exploration within the mode). This idea is transferable to any scenario requiring multi-modal action generation, such as robotic manipulation.
- **The first application of the MeanFlow Identity to end-to-end planning** eliminates two major pain points of flow matching: slow multi-step sampling and numerical errors. Planning FPS increases from GoalFlow's 11 to 434 (39× speedup), making flow-based methods competitive with direct MLP regression in real-time performance for the first time.
- **The ARM "reconstruct if selection fails" design** addresses a long-overlooked problem: what if all candidates are poor? A conventional selector can only pick the least-bad option, whereas ARM synthesizes a new trajectory by integrating information from all proposals. This design is transferable to any multi-candidate selection scenario.

## Limitations & Future Work
- The number of Gaussian components $K=8$ and mixing coefficients $\pi_k=1$ are predefined; adaptively determining $K$ and predicting scene-conditioned mixing coefficients could further improve performance.
- ARM performs selection/reconstruction implicitly via cross-attention, which lacks interpretability — it is unclear whether the model "selected a proposal" or "reconstructed a new trajectory."
- Evaluation is conducted only on NAVSIM (non-reactive simulation); performance in more realistic reactive simulation (e.g., nuPlan) and on real vehicles remains to be validated.
- Trajectory planning covers only 4 seconds (8 waypoints); applicability to long-horizon planning scenarios requires further investigation.
- MeanFlow training requires JVP computation; a detailed comparison of training cost against standard flow matching is not provided.

## Related Work & Insights
- **vs. GoalFlow**: GoalFlow uses flow matching with goal-point guidance but requires 5 sampling steps and relies on discrete goal point prediction. MeanFuser uses one-step MeanFlow sampling with a continuous GMN prior, achieving +3.3 PDMS and 39.45× higher Plan FPS.
- **vs. DiffusionDrive**: DiffusionDrive uses diffusion with clustered trajectory prototypes and iterative refinement. MeanFuser uses MeanFlow with GMN for one-step generation, achieving +0.9 PDMS and 1.55× higher speed.
- **vs. Hydra-MDP**: Hydra-MDP uses a discrete trajectory vocabulary with probability prediction. MeanFuser replaces this with a continuous GMN, achieving better performance (+2.5 PDMS) and higher speed (2.65×).
- **vs. WoTE**: WoTE introduces a world model to evaluate candidate trajectories (relying on benchmark sub-metrics). MeanFuser replaces this evaluation with ARM, eliminating dependence on benchmark-specific rules.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce MeanFlow into end-to-end planning, combined with GMN as a continuous prior and ARM for reconstruction; all three designs are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation on NAVSIMv1/v2 with thorough ablations, but more complex benchmarks such as nuPlan are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Technical details are clear, preliminaries are well presented, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a key efficiency bottleneck in flow-based planning; GMN and ARM designs are broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](../../AAAI2026/autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[CVPR 2026\] CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](../../AAAI2026/autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](../../AAAI2026/autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)

</div>

<!-- RELATED:END -->
