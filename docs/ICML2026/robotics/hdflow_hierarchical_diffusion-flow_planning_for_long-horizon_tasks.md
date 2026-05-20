---
title: >-
  [Paper Note] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks
description: >-
  [ICML 2026][Robotics][Hierarchical Planning] HDFlow uses a diffusion model to generate sparse strategic subgoals and a rectified flow to generate dense trajectories…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Hierarchical Planning"
  - "Diffusion Model"
  - "Rectified Flow"
  - "Energy Model"
  - "Manifold Projection"
date: 2026-05-08
content_hash: f0181c1669075b78
---

# HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks

**Conference**: ICML 2026  
**arXiv**: [2605.04525](https://arxiv.org/abs/2605.04525)  
**Code**: https://hdflow-page.github.io/ (project page)  
**Area**: Robotics / Long-horizon Planning / Generative Planning  
**Keywords**: Hierarchical Planning, Diffusion Model, Rectified Flow, Energy Model, Manifold Projection

## TL;DR
HDFlow uses a diffusion model to generate sparse strategic subgoals and a rectified flow to generate dense trajectories, further incorporating energy guidance and manifold projection. This constructs a two-layer planner with a division of labor between slow and fast modules, boosting the success rate of long-horizon, sparse-reward tasks such as furniture assembly by 20–30 percentage points.

## Background & Motivation

**Background**: Long-horizon robotic manipulation (e.g., furniture assembly, maze navigation) currently follows two main approaches: imitation learning to directly clone expert trajectories, or using diffusion models to treat planning as a "conditional generation" problem, sampling entire trajectories from noise. The latter is represented by Diffuser, Decision Diffuser, and hierarchical stacking of diffusion models such as SHD and HDMI.

**Limitations of Prior Work**: Pure diffusion planners require multi-step denoising at each step, resulting in slow inference and difficulty in real-time control. In long-horizon tasks, it is easy to generate "apparently reasonable but dead-end" plans, as standard conditional diffusion lacks explicit mechanisms to evaluate the long-term feasibility of subgoal sequences. Using diffusion at all hierarchy levels (high + low) further amplifies the speed bottleneck.

**Key Challenge**: High-level planning requires **exploration**—the ability to generate diverse strategic subgoal candidates; low-level execution requires **speed and determinism**—converting subgoals into smooth, dense trajectories. A single generative paradigm (all diffusion or all flow) cannot optimally achieve both.

**Goal**: (1) Use the most suitable generative model for each level; (2) Add guidance signals to the high level to "identify dead ends"; (3) Prevent guidance signals from pushing samples off the feasible manifold.

**Key Insight**: Treat diffusion and rectified flow as complementary tools—diffusion is suitable for high-diversity exploration, while rectified flow can generate trajectories in one or two steps via ODE solvers, offering speed. An energy-based model (EBM) is trained as a "long-term feasibility evaluator," assigning low energy to successful trajectories and high energy to failed ones.

**Core Idea**: The high level uses an "EBM-guided + manifold projection" diffusion planner to generate sparse subgoals in latent space, while the low level uses rectified flow to quickly connect dense trajectories, assuming a contrastively trained world model organizes the latent space so that embeddings of goal-near states are close.

## Method

### Overall Architecture
Two-stage training: **Stage 1** trains the world model (RSSM + DINOv2 encoder) with joint objectives of observation reconstruction, KL, contrastive learning, and inverse dynamics loss, ensuring the latent space is both predictive and reflects "distance to goal." The encoder is then frozen. **Stage 2** trains the hierarchical planner in the frozen latent space: the high-level diffusion model $\epsilon_\theta$ learns to conditionally generate $K$ sparse latent subgoals $z = (z_1, ..., z_K)$ from $(z_0, z_G)$; the low-level rectified flow $v_\theta$ learns to generate $H$-step dense latent trajectories between adjacent subgoals. During MPC inference, the high level replans periodically, the low level expands the first subgoal into a dense trajectory, and the inverse dynamics model maps it to actions.

### Key Designs

1. **Contrastive World Model**:

    - **Function**: Compresses high-dimensional, multimodal observations into latent states, structuring the latent space so that "closer to goal → closer embeddings."
    - **Mechanism**: Adds an InfoNCE contrastive loss $\mathcal{L}_{contrastive}$ to the standard RSSM reconstruction + KL objective $\mathcal{L}_{WM}$, treating intermediate latent states from successful trajectories and their final goal $z_G$ as positives, and pushing away intermediate states from failed trajectories; an additional inverse dynamics MSE loss encourages the model to encode "adjacent state pairs" in a way that makes actions predictable.
    - **Design Motivation**: Standard world models only ensure "accurate prediction," not "planning-friendliness." The contrastive term effectively carves out a "direction toward the goal" in latent space, enabling effective guidance by downstream high-level diffusion and energy models.

2. **Manifold-aware EBM-guided Diffusion (High Level)**:

    - **Function**: Incorporates explicit "long-term feasibility" signals into conditional diffusion, while preventing guidance from pushing samples into infeasible regions.
    - **Mechanism**: Trains an energy model with contrastive loss $\mathcal{L}_{EBM} = \log(1 + \exp(E_\phi(z_{pos}) - E_\phi(z_{neg})))$, assigning low energy to successful subgoal sequences. Sampling proceeds in two steps: first, EBM-guided sampling $z_{\ell-1}^{temp} \sim \mathcal{N}(\mu_\theta(z_\ell) + w_{ebm}\Sigma^\ell g, \Sigma^\ell)$, where $g = \nabla_{z_\ell} E_\phi$; then, $z_{\ell-1}^{temp}$ is projected onto the local manifold—using the Tweedie formula for denoised estimate $\hat z^{0|\ell-1}$, retrieving $k$ nearest neighbors, performing rank-$r$ PCA to obtain projection basis $U$, and finally $\mathcal{P}(z) = \mu + UU^T(z - \mu)$.
    - **Design Motivation**: The authors theoretically show that the lower bound of energy guidance error is proportional to $\sqrt{d}/\sqrt{1-\bar\alpha_\ell}$; in high-dimensional latent space, approximate EBM inevitably pushes samples off the feasible manifold. The projection step pulls the deviation back, imposing a hard constraint between "high quality" and "feasibility."

3. **Rectified Flow Low-level Trajectory Planner**:

    - **Function**: Quickly generates a dense $H$-step latent trajectory from the previous subgoal $z_{k-1}$ to the next subgoal $z_k$.
    - **Mechanism**: Treats "moving from $z_{k-1}$ to $z_k$ in latent space" as optimal transport, where the optimal solution is a straight-line trajectory—well-suited for rectified flow. The training objective is flow-matching $\mathcal{L}_{LL} = \mathbb{E}[\| v_\theta((1-u)\tau_0 + u\tau_1, u, c_k) - (\tau_1 - \tau_0)\|^2]$; inference directly solves the ODE to generate the entire trajectory in one or two steps.
    - **Design Motivation**: The low level does not require diversity, only "speed and accuracy." Rectified flow is an order of magnitude faster than diffusion, directly addressing the real-time bottleneck of pure diffusion hierarchical planners.

### Loss & Training
Two stages: The first stage jointly optimizes $\mathcal{L}_{WM\text{-}total} = \lambda_{WM}\mathcal{L}_{WM} + \lambda_{IDM}\mathcal{L}_{IDM} + \lambda_{contrastive}\mathcal{L}_{contrastive}$; the second stage freezes the world model and jointly trains the planner $\mathcal{L}_{planner} = \lambda_{HL}\mathcal{L}_{HL} + \lambda_{LL}\mathcal{L}_{LL} + \lambda_{EBM}\mathcal{L}_{EBM} + \lambda_{proj}\mathcal{L}_{projection}$. Here, $\mathcal{L}_{projection}$ encourages high-level generated subgoals to stay close to the learned latent manifold. The high level uses 100 denoising steps and CFG scale 2.0; the low level uses DiT with 4 layers, 8 heads, and latent dimension 512.

## Key Experimental Results

### Main Results

| Baseline / Task | Difficulty | SHD (Prev. SOTA) | HDFlow | Gain |
|-----------------|-----------|------------------|--------|------|
| FurnitureBench one_leg | Low/Med/High | 71/31/15 | **92/71/39** | +21~+24 |
| FurnitureBench lamp | Low/Med/High | 43/22/16 | **68/49/34** | +18~+27 |
| FurnitureBench round_table | Low/Med/High | 41/21/12 | **61/43/27** | +20~+22 |
| OGBench antmaze-giant-v0 | — | 19 | **48** | +13 (vs 35 DV) |
| OGBench humanoidmaze-giant-v0 | — | 7 | **25** | +9 |
| RLBench Insert Peg | — | 65.6 (3D Actor) | **93.3** | +27.7 |

On 18 RLBench tasks, HDFlow achieves the best results on 7 tasks and is significantly better on average than RVT-2, 3D Diffuser Actor, and other specialized visual manipulation models.

### Ablation Study

| Configuration | lamp Success Rate (%) | Inference Time (ms/step) |
|---------------|----------------------|--------------------------|
| Full HDFlow | **68** | **88** |
| w/o Manifold Projection | 57 (one_leg 84) | — |
| w/o Manifold-aware EBM | 33 (one_leg 61) | — |
| w/o Contrastive World Model | 27 (one_leg 58) | — |
| FD (Flat Diffusion) | 24 | 197 |
| HF (Hierarchical Rectified Flow) | 24 | 53 |
| HD (Hierarchical Diffusion) | 43 | 142 |

### Key Findings
- The contrastively trained world model is the main contributor: removing it causes the largest drop, indicating that both EBM and diffusion rely on the latent space having a "distance structure."
- "Hierarchical + hybrid paradigm" clearly outperforms single paradigms: HD (all diffusion) 43% vs HDFlow 68%, HF (all flow) 24% vs HDFlow 68%, proving that high/low-level tasks are indeed different in nature.
- Inference time drops from 142 ms (HD) to 88 ms, more than twice as fast as single-layer diffusion FD (197 ms), confirming that the rectified flow low level significantly improves speed.
- After fine-tuning with 50 demonstrations on a real Franka robot, success is still achieved, partially validating sim-to-real transfer.

## Highlights & Insights
- **"Right tool for the right job" hierarchical philosophy**: Assigning diffusion's exploration to the high level and rectified flow's speed to the low level is a highly instructive division of labor, generalizable to any task requiring "strategize first, then execute quickly" (e.g., VLA, document analysis).
- **Manifold-aware EBM guidance**: Theoretical proof that high-dimensional guidance inevitably deviates from the manifold, then using local PCA projection to pull it back—this trick is almost plug-and-play for any guided diffusion task (image editing, molecule design), not just robotics.
- **EBM as a "long-horizon evaluator"**: Rather than fitting a reward function, it directly scores the "entire plan," avoiding the sparse reward problem; contrastive training uses positive/negative samples from successful vs failed demos, with very low annotation cost.

## Limitations & Future Work
- Requires both successful and failed demonstrations to train the EBM and contrastive world model; the authors acknowledge that "data collection cost is not low," especially as failed demonstrations are hard to systematically collect on real robots.
- The high level still requires 100 denoising steps per replan; although the low level is fast, overall latency is still higher than pure imitation learning—distillation to fewer sampling steps could be considered.
- The subgoal interval $H$ is a task-dependent hyperparameter without an adaptive mechanism; long tasks may require segment refinement.
- Multimodal conditions (e.g., language instructions) are not yet integrated; currently only "image goal conditioning" is supported, so deployment to general household robots is still one step away.

## Related Work & Insights
- **vs SHD / HDMI**: Also hierarchical diffusion, but using diffusion at all levels slows down the low level; HDFlow replaces the low level with rectified flow and adds EBM guidance, leading to clear improvements in speed and success rate.
- **vs Diffuser / DD**: Single-layer diffusion lacks explicit hierarchy, so subgoal errors in long-horizon tasks can compound; HDFlow's high/low-level division and replanning mechanism naturally provide fault tolerance.
- **vs Manifold Preserving Guided Diffusion (He et al., 2024)**: This work transfers the manifold projection idea from image generation to robotic planning and combines it with EBM guidance, representing an interesting cross-domain migration.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "diffusion + rectified flow + EBM + manifold projection" is new, though each component has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks + real robot + detailed ablation + inference time comparison, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Motivation and reasoning are clear, the theoretical part (Appendix A) is carefully derived, but some formulas in the main text are abrupt.
- Value: ⭐⭐⭐⭐ Represents a solid SOTA advance for long-horizon robotic planning, and the cross-domain tricks are worth learning from.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](../../NeurIPS2025/robotics/rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[ICLR 2026\] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](../../ICLR2026/robotics/robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)

</div>

<!-- RELATED:END -->
