---
title: >-
  [Paper Note] HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks
description: >-
  [ICML 2026][Robotics][Hierarchical Planning] HDFlow employs diffusion models to generate sparse strategic subgoals and rectified flow to generate dense trajectories. By integrating energy guidance and manifold projection…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Hierarchical Planning"
  - "Diffusion Models"
  - "Rectified Flow"
  - "Energy-Based Models"
  - "Manifold Projection"
date: 2026-05-08
content_hash: 1ec17206c45fdfb0
---

# HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.04525](https://arxiv.org/abs/2605.04525)  
**Code**: https://hdflow-page.github.io/ (Project Page)  
**Area**: Robotics / Long-horizon Planning / Generative Planning  
**Keywords**: Hierarchical Planning, Diffusion Models, Rectified Flow, Energy-Based Models, Manifold Projection

## TL;DR
HDFlow employs diffusion models to generate sparse strategic subgoals and rectified flow to generate dense trajectories. By integrating energy guidance and manifold projection, it constructs a bi-level planner with a "slow-fast" division of labor, increasing success rates by 20–30 percentage points in long-horizon sparse-reward tasks such as furniture assembly.

## Background & Motivation

**Background**: Current mainstream approaches for long-horizon robotic manipulation (e.g., furniture assembly, maze navigation) follow two paths: imitation learning to clone expert trajectories directly, or using diffusion models to treat planning as a "conditional generation" problem by sampling entire trajectories from noise. Representative works include Diffuser, Decision Diffuser, and hierarchical diffusion variants like SHD and HDMI.

**Limitations of Prior Work**: Pure diffusion planners require multiple denoising steps at every execution step, resulting in slow inference speeds that hinder real-time control. Additionally, long-horizon tasks frequently lead to sequences that "look reasonable but hit a dead end," as standard conditional diffusion lacks an explicit mechanism to evaluate the long-term feasibility of subgoal sequences. Using diffusion at every level of the hierarchy (high-level + low-level) further amplifies this speed bottleneck.

**Key Challenge**: High-level planning requires **exploratory capacity**—the ability to generate diverse strategic subgoal candidates. Low-level execution requires **speed and determinism**—converting subgoals into smooth, dense trajectories. A single generative paradigm (either entirely diffusion-based or entirely flow-based) cannot optimize both simultaneously.

**Goal**: (1) Utilize the most suitable generative model for the high and low levels respectively; (2) Introduce guidance signals to the high level capable of "identifying dead ends"; (3) Prevent guidance signals from pushing samples off the feasible manifold.

**Key Insight**: Diffusion and rectified flow are viewed as complementary tools—diffusion is suitable for high-diversity exploration, while rectified flow can generate trajectories in one or two steps via an ODE solver, offering high speed. Additionally, an Energy-Based Model (EBM) is trained as a "long-term feasibility evaluator," assigning low energy to successful trajectories and high energy to failed ones.

**Core Idea**: The high level uses an "EBM-guided + manifold projection" diffusion planner to produce sparse subgoals in latent space. The low level uses rectified flow to quickly string together dense trajectories. This is underpinned by a world model trained via contrastive learning that organizes the latent space such that "goal-proximal state embeddings are close."

## Method

### Overall Architecture
Two-stage training: **Stage 1** trains the world model (RSSM + DINOv2 encoder) using a joint loss of observation reconstruction + KL + contrastive learning + inverse dynamics to ensure the latent space is both predictive and reflective of "distance to goal"; the encoder is then frozen. **Stage 2** trains the hierarchical planners in the frozen latent space: the high-level diffusion model $\epsilon_\theta$ learns to generate $K$ sparse latent subgoals $z = (z_1, ..., z_K)$ conditioned on $(z_0, z_G)$; the low-level rectified flow $v_\theta$ learns to generate an $H$-step dense latent trajectory between two adjacent subgoals. During MPC inference, the high level replans at intervals, while the low level unfolds the first subgoal into a dense trajectory, which is mapped to actions via the inverse dynamics model.

### Key Designs

1.  **Contrastive World Model**:
    - **Function**: Compresses high-dimensional multimodal observations into latent states and embeds a distance structure where "proximity to goal → similar embeddings."
    - **Mechanism**: In addition to standard RSSM reconstruction + KL objectives $\mathcal{L}_{WM}$, an InfoNCE contrastive loss $\mathcal{L}_{contrastive}$ is added. This treats intermediate latent states of successful trajectories and their final goals $z_G$ as positive pairs, while pushing away states from failed trajectories. An inverse dynamics MSE loss is also included to force the model to encode "adjacent state pairs" in an action-predictable manner.
    - **Design Motivation**: Standard world models only guarantee "predictive accuracy," not "planning-friendliness." The contrastive term effectively carves out a "direction toward the goal" in the latent space, enabling effective guidance for downstream high-level diffusion and energy models.

2.  **Manifold-aware EBM Guided Diffusion (High-level)**:
    - **Function**: Incorporates an explicit "long-term feasibility" signal into conditional diffusion and prevents guidance from pushing samples into infeasible regions.
    - **Mechanism**: An EBM is trained using a contrastive loss $\mathcal{L}_{EBM} = \log(1 + \exp(E_\phi(z_{pos}) - E_\phi(z_{neg})))$ to assign low energy to successful subgoal sequences. Sampling involves a two-step process: first, EBM-guided sampling $z_{\ell-1}^{temp} \sim \mathcal{N}(\mu_\theta(z_\ell) + w_{ebm}\Sigma^\ell g, \Sigma^\ell)$, where $g = \nabla_{z_\ell} E_\phi$; then, projecting $z_{\ell-1}^{temp}$ onto the local manifold—denoising via the Tweedie formula to get $\hat z^{0|\ell-1}$, retrieving $k$ nearest neighbors to perform rank-$r$ PCA for a projection basis $U$, and finally $\mathcal{P}(z) = \mu + UU^T(z - \mu)$.
    - **Design Motivation**: The authors theoretically prove that the error lower bound of energy guidance is proportional to $\sqrt{d}/\sqrt{1-\bar\alpha_\ell}$. In high-dimensional latent space, approximate EBMs inevitably push samples off the feasible manifold; the projection step corrects the deviation introduced by guidance, acting as a hard constraint between "high quality" and "feasibility."

3.  **Rectified Flow Low-level Trajectory Planner**:
    - **Function**: Rapidly generates an $H$-step dense latent trajectory from the previous subgoal $z_{k-1}$ to the next subgoal $z_k$.
    - **Mechanism**: Views the transition from $z_{k-1}$ to $z_k$ in latent space as optimal transport. The optimal solution is a trajectory that is as straight as possible, perfectly matching the rectified flow paradigm. The training objective is flow-matching $\mathcal{L}_{LL} = \mathbb{E}[\| v_\theta((1-u)\tau_0 + u\tau_1, u, c_k) - (\tau_1 - \tau_0)\|^2]$, allowing for single-step or few-step trajectory generation during inference.
    - **Design Motivation**: The low level does not require diversity, only "speed and accuracy." Rectified flow is an order of magnitude faster than diffusion, resolving the real-time bottleneck of pure-diffusion hierarchical planners.

### Loss & Training
Two stages: The first stage jointly optimizes $\mathcal{L}_{WM\text{-}total} = \lambda_{WM}\mathcal{L}_{WM} + \lambda_{IDM}\mathcal{L}_{IDM} + \lambda_{contrastive}\mathcal{L}_{contrastive}$. The second stage freezes the world model and jointly trains the planners: $\mathcal{L}_{planner} = \lambda_{HL}\mathcal{L}_{HL} + \lambda_{LL}\mathcal{L}_{LL} + \lambda_{EBM}\mathcal{L}_{EBM} + \lambda_{proj}\mathcal{L}_{projection}$. Here, $\mathcal{L}_{projection}$ ensures high-level subgoals stay close to the learned latent manifold. The high level uses 100 denoising steps with a CFG scale of 2.0; the low level uses a 4-layer 8-head DiT with a hidden dimension of 512.

## Key Experimental Results

### Main Results

| Benchmark / Task | Difficulty | SHD (Prev. SOTA) | HDFlow (Ours) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| FurnitureBench one_leg | Low/Med/High | 71/31/15 | **92/71/39** | +21~+24 |
| FurnitureBench lamp | Low/Med/High | 43/22/16 | **68/49/34** | +18~+27 |
| FurnitureBench round_table | Low/Med/High | 41/21/12 | **61/43/27** | +20~+22 |
| OGBench antmaze-giant-v0 | — | 19 | **48** | +13 (vs 35 DV) |
| OGBench humanoidmaze-giant-v0 | — | 7 | **25** | +9 |
| RLBench Insert Peg | — | 65.6 (3D Actor) | **93.3** | +27.7 |

On 18 RLBench tasks, HDFlow achieves the best performance in 7 tasks and significantly outperforms specialized visual manipulation models such as RVT-2 and 3D Diffuser Actor on average.

### Ablation Study

| Configuration | lamp Success Rate (%) | Inference Time (ms/step) |
| :--- | :--- | :--- |
| Full HDFlow | **68** | **88** |
| w/o Manifold Projection | 57 (one_leg 84) | — |
| w/o Manifold-aware EBM | 33 (one_leg 61) | — |
| w/o Contrastive WM | 27 (one_leg 58) | — |
| FD (Flat Diffusion) | 24 | 197 |
| HF (Hierarchical Flow) | 24 | 53 |
| HD (Hierarchical Diffusion) | 43 | 142 |

### Key Findings
- **Contrastive World Model is critical**: Performance drops most significantly when it is removed, indicating that EBM and diffusion rely on the "distance structure" of the latent space to function.
- **"Hierarchical + Mixed Paradigm" is superior**: HD (All Diffusion) 43% vs. HDFlow 68%, HF (All Flow) 24% vs. HDFlow 68%, proving that high-level and low-level task properties are fundamentally different.
- **Inference time reduction**: Reduced from 142 ms (HD) to 88 ms, more than twice as fast as single-level diffusion (FD at 197 ms), proving that the rectified flow low-level effectively boosts speed.
- **Real-robot transfer**: Success on a Franka robot after fine-tuning with 50 demos partially validates sim-to-real transferability.

## Highlights & Insights
- **Hierarchical philosophy of "using the right tool"**: Placing diffusion's exploratory power at the high level and rectified flow's speed at the low level is a highly valuable "division of labor" strategy applicable to any task requiring "strategic thinking followed by fast execution" (VLA, document analysis, etc.).
- **Manifold-aware EBM Guidance**: Theoretically proving that high-dimensional guidance inevitably deviates from the manifold and using PCA local projection to pull it back is a trick that can be plug-and-played into almost any guided diffusion task (image editing, molecular design), beyond robotics.
- **EBM as a "long-range evaluator"**: Instead of fitting a reward function, it scores the "entire plan" directly, avoiding reward sparsity issues. By using contrastive training with success vs. failure demos, the annotation cost is extremely low.

## Limitations & Future Work
- Training EBM and contrastive world models requires both successful and failed demonstrations; the author acknowledges that "data collection costs are not low," especially since failed demonstrations are difficult to collect systematically on real robots.
- High-level replanning still requires 100 denoising steps. Although the low level is fast, overall latency remains high compared to pure imitation learning; distillation into few-step sampling could be considered.
- The subgoal interval $H$ is a task-dependent hyperparameter with no adaptive mechanism; long tasks might require multi-stage refinement.
- Multimodal conditioning (language instructions) has not yet been integrated; it currently only supports "image goal conditioning," leaving a gap for deployment in general-purpose household robots.

## Related Work & Insights
- **vs SHD / HDMI**: Both use hierarchical diffusion, but using diffusion for everything slows down the low level; HDFlow leads significantly in speed and success rate by replacing the low level with rectified flow and adding EBM guidance.
- **vs Diffuser / DD**: Single-level diffusion lacks an explicit hierarchy; subgoal errors can amplify in long-horizon tasks. HDFlow’s high-low division + replanning mechanism is naturally fault-tolerant.
- **vs Manifold Preserving Guided Diffusion (He et al., 2024)**: This paper migrates the manifold projection idea from image generation to robotic planning and combines it with EBM guidance, representing an interesting cross-domain transfer.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "Diffusion + Rectified Flow + EBM + Manifold Projection" quartet is new, although each component has precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three benchmarks + real robot + detailed ablations + inference time comparisons; very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logical motivation; the theoretical part (Appendix A) is rigorously derived, though some formulas in the main text are abrupt.
- **Value**: ⭐⭐⭐⭐ A solid SOTA advancement for long-horizon robotic planning; the cross-domain tricks are worth learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](../../NeurIPS2025/robotics/rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](../../NeurIPS2025/robotics/robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)

</div>

<!-- RELATED:END -->
