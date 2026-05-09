---
title: >-
  [Paper Note] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving
description: >-
  [ICLR 2026][Autonomous Driving][Diffusion bridge model] BridgeDrive proposes replacing truncated diffusion with a diffusion bridge to achieve anchor-guided trajectory planning for autonomous driving, ensuring theoretical symmetry between the forward and reverse processes. On the Bench2Drive closed-loop benchmark, it achieves success rates of 74.99% (PDM-Lite) and 89.25% (LEAD), surpassing the previous SOTA by 7.72% and 2.45%, respectively.
tags:
  - ICLR 2026
  - Autonomous Driving
  - Diffusion bridge model
  - anchor trajectory guidance
  - closed-loop planning
  - geometric path waypoints
  - Bench2Drive
date: 2026-05-08
content_hash: 85334c1f5c457750
---

# BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving

**Conference**: ICLR 2026  
**arXiv**: [2509.23589](https://arxiv.org/abs/2509.23589)  
**Code**: [https://github.com/shuliu-ethz/BridgeDrive](https://github.com/shuliu-ethz/BridgeDrive)  
**Area**: Autonomous Driving  
**Keywords**: Diffusion bridge model, anchor trajectory guidance, closed-loop planning, geometric path waypoints, Bench2Drive  

## TL;DR
BridgeDrive proposes replacing truncated diffusion with a diffusion bridge to achieve anchor-guided trajectory planning for autonomous driving, ensuring theoretical symmetry between the forward and reverse processes. On the Bench2Drive closed-loop benchmark, it achieves success rates of 74.99% (PDM-Lite) and 89.25% (LEAD), surpassing the previous SOTA by 7.72% and 2.45%, respectively.

## Background & Motivation

**Background**: Diffusion models have emerged as a powerful paradigm for autonomous driving planning due to their ability to model multimodal behavior distributions. DiffusionDrive introduced anchor trajectories (K-means cluster centers representing typical human driving behaviors) to guide the diffusion process, achieving state-of-the-art performance.

**Limitations of Prior Work**: DiffusionDrive employs a truncated diffusion schedule, initiating denoising from a noisy version of the anchor rather than from pure Gaussian noise. This introduces a theoretical asymmetry between the forward process (adding noise to anchors) and the denoising process (recovering the true trajectory)—the denoiser is trained to regress from noisy anchors to true trajectories rather than to invert the forward diffusion process.

**Key Challenge**: This asymmetry deviates from the core principles of diffusion models, potentially leading to unpredictable behavior and degraded performance. The central question is how to preserve the benefits of anchor guidance while maintaining theoretical consistency with the diffusion framework.

**Goal**: (a) Design a theoretically consistent anchor-guided diffusion framework; (b) adopt a trajectory representation better suited to diffusion models; (c) achieve real-time closed-loop deployment.

**Key Insight**: The planning task is formulated as a diffusion bridge—learning a diffusion process that explicitly connects anchor trajectories to refined planning trajectories, guaranteeing perfect symmetry between the forward and reverse processes.

**Core Idea**: Replace truncated diffusion with a diffusion bridge, making anchor guidance an intrinsic component of the diffusion model rather than an external workaround.

## Method

### Overall Architecture
The input consists of sensor data (LiDAR + front-facing camera + target point), from which a perception module extracts BEV features and conditioning information $z$. Planning proceeds in two steps: (1) a classifier $h_\phi$ selects the best anchor $y$ from a predefined anchor set $\mathcal{Y}$; (2) a denoiser $x_\theta$ progressively transforms the anchor $x_T = y$ into a refined planning trajectory $x_0$ via the diffusion bridge PF-ODE.

### Key Designs

1. **Diffusion Bridge Formulation**:

    - **Function**: The planning task is formulated as a diffusion bridge from anchor $x_T = y$ to ground-truth trajectory $x_0 = x$, guaranteeing symmetry between the forward and reverse processes.
    - **Mechanism**: A Doob h-transform is employed to construct a conditional diffusion process, defining a Gaussian transition kernel $q(x_t|x_0, x_T) = \mathcal{N}(x_t | a_t x_T + b_t x_0, c_t^2 I)$, where $a_t, b_t, c_t$ are determined by the noise schedule. Key properties: at $t=0$, $a_0=c_0=0, b_0=1$ ensures recovery of the true trajectory; at $t=T$, the process terminates exactly at the anchor. The denoiser is trained with a standard mean squared error objective: $\min_\theta \mathbb{E}[w(t)\|x_\theta(x_t, t, x_T, z) - x_0\|^2]$, and training is simulation-free.
    - **Design Motivation**: In DiffusionDrive's truncated diffusion, the noisy anchor $y_t = \alpha_t y + \sigma_t \epsilon$ deviates from the true trajectory $x$ for all $t$, making it impossible to recover $x$ through denoising—a violation of the invertibility principle of diffusion models. The diffusion bridge resolves this by establishing an exact probabilistic path between the two endpoints.
    - **Difference from DiffusionDrive**: During training, BridgeDrive's noise sample $x_t = a_t y + b_t x + c_t \epsilon$ depends jointly on the true trajectory and the anchor (Line 7, Algorithm 1), whereas DiffusionDrive's $y_t = \alpha_t y + \sigma_t \epsilon$ depends only on the anchor (Line 7, Algorithm 2).

2. **Geometric Path Waypoints**:

    - **Function**: Equally spaced geometric path coordinates combined with a scalar speed $(x^{\text{geo}}, v) \in \mathbb{R}^{N \times 2} \times \mathbb{R}$ replace equally spaced temporal waypoints $x^{\text{temp}} \in \mathbb{R}^{N \times 2}$.
    - **Mechanism**: Geometric path waypoints decouple path shape from speed—overtaking maneuvers at different speeds need only learn similar geometric patterns with different speed scalars, whereas temporal waypoints must stretch inter-point spacing to accommodate varying speeds, making generalization more difficult.
    - **Design Motivation**: Temporal waypoints introduce ambiguity in speed encoding and are more prone to violating route topology constraints. Ablation studies show that geometric path waypoints yield substantial gains across all diffusion methods (BridgeDrive: +15.09% SR).

3. **Anchor Classifier**:

    - **Function**: During training, the nearest-neighbor anchor is used as the ground-truth label; during inference, the classifier $h_\phi(z, \mathcal{Y})$ predicts the optimal anchor.
    - **Mechanism**: The classifier interacts with all anchors and BEV features via cross-attention, outputting a probability for each anchor. It is executed only once before denoising iterations, introducing no additional iterative overhead.
    - **Design Motivation**: The true trajectory is unavailable at inference time, precluding direct nearest-neighbor computation and necessitating a learned predictor. Incorrect anchor selection leads to catastrophic failures (Fig. 1, red trajectories), making classifier accuracy critical.

### Loss & Training
- Denoiser loss: weighted MSE $\mathbb{E}[w(t)\|x_\theta(x_t, t, x_T, z) - x_0\|^2]$
- Classifier loss: cross-entropy with the nearest anchor to the ground-truth trajectory as the label
- Inference uses a DDIM first-order ODE solver, generating trajectories with a small number of function evaluations

## Key Experimental Results

### Main Results
Bench2Drive closed-loop evaluation (CARLA Leaderboard 2.0, 220 routes):

| Method | Dataset | VLA | Diffusion | DS | SR(%) |
|--------|---------|-----|-----------|-----|-------|
| DriveTransformer | Think2Drive | ✘ | ✘ | 63.46 | 35.01 |
| ORION | Think2Drive | ✓ | ✘ | 77.74 | 54.62 |
| DiffusionDrive-geo | PDM-Lite | ✘ | ✓ | 80.79 | 58.18 |
| SimLingo | PDM-Lite | ✓ | ✘ | 85.07 | 67.27 |
| TransFuser++ | PDM-Lite | ✘ | ✘ | 84.21 | 67.27 |
| **BridgeDrive** | PDM-Lite | ✘ | ✓ | **87.99** | **74.99** |
| **BridgeDrive** | LEAD | ✘ | ✓ | **96.34** | **89.25** |

### Ablation Study

| Configuration | DS | SR(%) | Note |
|--------------|-----|-------|------|
| DiffusionDrive-temp | 77.68 | 52.72 | Truncated diffusion + temporal waypoints |
| DiffusionDrive-geo | 80.79 | 58.18 | Truncated diffusion + geometric waypoints (+5.46%) |
| Full Diffusion-geo | 83.85 | 67.27 | Full diffusion + geometric waypoints |
| **BridgeDrive-geo** | **87.99** | **74.99** | Diffusion bridge + geometric waypoints (+15.09% vs. temp) |

### Key Findings
- Geometric path waypoints outperform temporal waypoints across all diffusion methods, with the largest gain observed in BridgeDrive (+15.09% SR)
- Full diffusion outperforms truncated diffusion (demonstrating the importance of theoretical consistency), and the diffusion bridge further outperforms full diffusion (demonstrating the value of anchor guidance)
- BridgeDrive yields the most significant improvement in the Merging scenario (+11.17), as anchor guidance provides a strong prior in ambiguous situations
- Inference speed satisfies real-time deployment requirements; the DDIM first-order solver is sufficient

## Highlights & Insights
- **Theory-driven methodological improvement**: Rather than stacking modules, the paper identifies a theoretical flaw in DiffusionDrive from the mathematical foundations of diffusion models and provides a principled solution via the diffusion bridge formulation. This theory-to-practice reasoning strategy is noteworthy.
- **The choice of representation for decoupling path shape and speed is critical**: A seemingly simple representational change (geometric vs. temporal waypoints) yields a 15% SR improvement, highlighting the importance of inductive bias selection in diffusion-based planning.
- **Anchors as boundary conditions of the diffusion bridge** rather than external guidance signals represent a more elegant integration paradigm, transferable to other conditional generation tasks.

## Limitations & Future Work
- Performance on Comfortness and Give Way metrics is suboptimal; the model tends to brake frequently, prioritizing safety at the cost of ride comfort
- Integration with VLAs (Vision-Language-Action models) has not been explored; the authors explicitly identify this as a future direction
- Anchor classifier accuracy is a performance bottleneck—incorrect anchor selection leads to catastrophic failures (Fig. 1)
- Improvements on open-loop evaluation (NAVSIM) are less pronounced than on closed-loop, suggesting the method's primary advantage lies in handling feedback loops and interaction

## Related Work & Insights
- **vs. DiffusionDrive (Liao et al., 2025)**: The key distinction is theoretical consistency—DiffusionDrive's truncated diffusion introduces asymmetry between the forward and reverse processes, which BridgeDrive eliminates via the diffusion bridge, improving SR from 58.18% to 74.99%
- **vs. SimLingo (Renz et al., 2025)**: SimLingo relies on a VLA large language model, yet BridgeDrive, as a pure diffusion approach without VLA, surpasses it, suggesting that the potential of diffusion models in planning tasks is underestimated
- **vs. ORION (Fu et al., 2025)**: ORION is augmented with VLA and VQA; its diffusion variant (46.54% SR) performs worse, underscoring that the correct use of diffusion models is paramount

## Rating
- Novelty: ⭐⭐⭐⭐ The application of diffusion bridges to autonomous driving planning is novel, though the core technique (diffusion bridge) builds on prior work
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Closed-loop and open-loop evaluation, detailed ablations, multi-dataset validation, and three-seed repeated experiments
- Writing Quality: ⭐⭐⭐⭐⭐ The analysis of DiffusionDrive's theoretical flaw is exceptionally clear, and the side-by-side algorithm pseudocode comparison is immediately illuminating
- Value: ⭐⭐⭐⭐⭐ Achieves substantial gains on the most challenging closed-loop benchmark with a concise method and efficient inference

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](../../AAAI2026/autonomous_driving/diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[AAAI 2026\] ReflexDiffusion: Reflexion-Enhanced Trajectory Planning for High Lateral Acceleration in Autonomous Driving](../../AAAI2026/autonomous_driving/reflexdiffusion_reflection-enhanced_trajectory_planning_for_.md)
- [\[CVPR 2026\] ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](../../CVPR2026/autonomous_driving/colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)

<!-- RELATED:END -->
