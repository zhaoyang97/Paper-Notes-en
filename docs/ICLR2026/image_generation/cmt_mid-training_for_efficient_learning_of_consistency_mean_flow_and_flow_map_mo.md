---
title: >-
  [Paper Note] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models
description: >-
  [ICLR 2026][Image Generation][flow map] This paper proposes Consistency Mid-Training (CMT), which inserts a lightweight intermediate training stage between a pretrained diffusion model and flow map post-training. By training the model to map arbitrary points on ODE trajectories back to clean samples, CMT yields trajectory-aligned initialization, reducing training cost by up to 98% while achieving state-of-the-art two-step generation quality.
tags:
  - ICLR 2026
  - Image Generation
  - flow map
  - consistency model
  - mid-training
  - few-step generation
  - diffusion distillation
date: 2026-05-08
content_hash: 90583cb3488df71a
---

# CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models

**Conference**: ICLR 2026
**arXiv**: [2509.24526](https://arxiv.org/abs/2509.24526)
**Code**: [https://github.com/sony/cmt](https://github.com/sony/cmt)
**Area**: Diffusion Models / Few-Step Generation
**Keywords**: flow map, consistency model, mid-training, few-step generation, diffusion distillation

## TL;DR
This paper proposes Consistency Mid-Training (CMT), which inserts a lightweight intermediate training stage between a pretrained diffusion model and flow map post-training. By training the model to map arbitrary points on ODE trajectories back to clean samples, CMT yields trajectory-aligned initialization, reducing training cost by up to 98% while achieving state-of-the-art two-step generation quality.

## Background & Motivation

**State of the Field**: Diffusion models achieve high generation quality but suffer from slow inference due to multi-step ODE solving. Flow map models (e.g., Consistency Models, Mean Flow) learn solution mappings of the PF-ODE to enable few-step (1–2 step) generation, representing the dominant paradigm for accelerating diffusion models.

**Limitations of Prior Work**: Flow map model training is unstable, sensitive to hyperparameters, and computationally expensive. The root cause is the absence of true regression targets—existing methods rely on stop-gradient pseudo-targets that drift dynamically during training, producing biased and unstable optimization signals.

**Root Cause**: Although initializing from pretrained diffusion models is beneficial, diffusion models learn infinitesimal denoising steps whereas flow maps must learn large trajectory jumps. This "differential vs. integral" mismatch renders diffusion-based initialization fragile, still requiring extensive heuristics (time sampling schedules, loss reweighting, etc.), and training remains slow and unstable.

**Paper Goals**: (a) How to provide trajectory-aligned, high-quality initialization for flow map models? (b) How to eliminate the pseudo-target bias introduced by stop-gradient? (c) How to substantially reduce flow map training cost?

**Starting Point**: Inspired by the mid-training concept from the LLM literature, the paper inserts an intermediate stage between pretraining and post-training. The ODE solver of the pretrained model generates reference trajectories that provide deterministic, stop-gradient-free regression targets.

**Core Idea**: Fixed ODE trajectories from the pretrained model serve as supervision signals. Simple regression trains the model to "jump along the trajectory to the endpoint," providing trajectory-aware initialization for flow map post-training.

## Method

### Overall Architecture

CMT proposes a three-stage pipeline: **Pretraining** → **Mid-Training (CMT)** → **Post-Training (Flow Map)**.

- **Input**: A pretrained diffusion model $\mathbf{D}_\phi$ (or flow matching model)
- **Mid-Training**: Samples $\mathbf{x}_T$ from the prior $p_{\text{prior}}$, generates discrete trajectories $\{\hat{\mathbf{x}}_{t_i}\}_{i=0}^M$ via the pretrained model's ODE solver (e.g., DPM-Solver++ with 16 steps), and trains the model to map any trajectory point to the clean endpoint
- **Post-Training**: Initializes the flow map model (ECT/ECD/MF) with CMT weights and trains normally
- **Output**: A flow map model capable of high-quality image generation in 1–2 steps

### Key Designs

1. **CMT-CM Loss (for Consistency Models)**:

    - **Function**: Learns to directly map any intermediate trajectory point $\hat{\mathbf{x}}_{t_i}$ to the clean sample $\hat{\mathbf{x}}_{t_0}$
    - **Mechanism**: $\mathcal{L}_{\text{CMT-CM}}(\theta) = \mathbb{E}_i \mathbb{E}_{\mathbf{x}_T \sim p_{\text{prior}}} [d(\mathbf{f}_\theta(\hat{\mathbf{x}}_{t_i}, t_i), \hat{\mathbf{x}}_{t_0})]$, where $\hat{\mathbf{x}}_{t_0}$ is the deterministic "clean" sample produced by the ODE solver and $d$ is LPIPS or $\ell_2$ distance
    - **Design Motivation**: This is a discrete approximation of the oracle CM loss. Since solver-generated points approximate the true flow map ($\hat{\mathbf{x}}_{t_i} \approx \Psi_{T \to t_i}(\mathbf{x}_T)$), the loss reduces to a standard regression problem **requiring no stop-gradient, no custom time sampling, and no loss reweighting schedules**. Each $\mathbf{x}_T$ uniquely determines a trajectory, yet $\mathbf{x}_T$ can be sampled arbitrarily, avoiding overfitting.

2. **CMT-MF Loss (for Mean Flow)**:

    - **Function**: Learns the average drift between trajectory points
    - **Mechanism**: $\mathcal{L}_{\text{CMT-MF}}(\theta) = \mathbb{E}_{i>j} \mathbb{E}_{\mathbf{x}_T} [\|\mathbf{h}_\theta(\hat{\mathbf{x}}_{t_i}, t_i, t_j) - \frac{\hat{\mathbf{x}}_{t_i} - \hat{\mathbf{x}}_{t_j}}{t_i - t_j}\|_2^2]$
    - **Design Motivation**: Simplifies the complex MF training objective into regression on finite differences of trajectory points. Setting $t_j = 0$ recovers CMT-CM, making CMT-MF the more general formulation. It similarly eliminates stop-gradient and Jacobian-vector product (JVP) computations, substantially reducing cost.

3. **Flexible Teacher Sampler**:

    - **Function**: Trajectory generation in CMT is not restricted to a diffusion model ODE solver
    - **Mechanism**: In ImageNet 256 experiments, a small MF-B/4 model (8-step FID = 13.44) is used as the teacher to generate trajectories for training a larger MF-XL/2 model
    - **Design Motivation**: Demonstrates that CMT mid-training is architecture-agnostic—any ODE trajectory generator suffices. This implies that a small model can be trained quickly and then used to accelerate training of a larger model.

4. **Trajectory Reuse Mechanism**:

    - **Function**: Intermediate states from multi-step solvers such as DPM-Solver++ can be reused
    - **Mechanism**: A single $M$-step trajectory yields $M$ training pairs, each consisting of an intermediate point $\hat{\mathbf{x}}_{t_i}$ and the endpoint $\hat{\mathbf{x}}_{t_0}$
    - **Design Motivation**: Compared to using only endpoints (Slow CMT), CMT achieves approximately $3\times$ higher data efficiency and lower GPU time overhead.

### Loss & Training

- LPIPS perceptual loss (pixel space) or ELatentLPIPS (latent space) for CM-type experiments
- $\ell_2$ loss for MF-type experiments
- ODE solver: DPM-Solver++ with 16 steps or MF teacher with 8 steps
- Post-training eliminates numerous ad-hoc techniques ($\Delta t$ annealing, loss reweighting, custom time sampling, EMA variants, nonlinear learning rate schedules, etc.)

## Key Experimental Results

### Main Results

| Dataset | Metric | CMT (Ours) | Prev. SOTA | Gain |
|--------|------|-----------|-----------|------|
| CIFAR-10 32×32 | 2-step FID | **1.97** | 1.98 (IMM) | −0.01 |
| ImageNet 64×64 | 2-step FID | **1.32** (w/ ECD) | 1.25 (AYF) | +0.07 |
| ImageNet 64×64 | 2-step FID | **1.48** (w/ ECT) | 1.48 (sCT) | on par, 98% less training |
| ImageNet 512×512 | 2-step FID | **1.84** | 1.87 (AYF) | −0.03 |
| ImageNet 256×256 | 1-step FID | **3.34** | 3.43 (MF) | −0.09 |
| AFHQv2 64×64 | 2-step FID | **2.34** | 2.61 (ECT) | −0.27 |
| FFHQ 64×64 | 2-step FID | **2.75** | 4.02 (iCT) | −1.27 |

### Ablation Study

| Configuration | 1-step FID | 2-step FID | Note |
|------|-----------|-----------|------|
| Full model (CMT) | **2.74** | **1.97** | Complete model |
| Vanilla ECT (51.2M) | 3.54 | 2.12 | No mid-training |
| CMT_short (1.28M mid + 49.92M post) | 3.42 | 2.11 | Short mid-training |
| CMT_long (25.6M mid + 25.6M post) | 3.30 | 2.04 | Long mid-training |
| KD initialization | 3.54 | 2.19 | Knowledge distillation init, inferior to CMT |
| Slow CMT | 2.75 | 1.98 | Endpoint-only; comparable quality but ~3× slower |

### Key Findings
- Longer CMT mid-training yields better results, confirming the importance of trajectory-aligned initialization.
- Even a low-quality small teacher model (MF-B/4, 8-step FID = 13.44) proves effective—halving MF-XL/2 training time while achieving better FID.
- Theoretical analysis shows that the gradient bias of CMT initialization is $\mathcal{O}(\varepsilon + \Delta t^2)$, far smaller than that of diffusion or random initialization.
- CMT is also effective on MS-COCO text-to-image tasks, reducing training time by 47%.

## Highlights & Insights
- **Cross-domain transfer of mid-training**: The mid-training paradigm from the LLM literature is systematically introduced into visual generation, resolving the long-standing instability of flow map training in a remarkably simple manner. The key insight is identifying a naturally available, easily obtainable fixed regression target—ODE trajectories.
- **Engineering simplification**: CMT enables post-training to discard nearly all ad-hoc techniques ($\Delta t$ annealing, custom time sampling, loss reweighting schedules), substantially reducing hyperparameter tuning burden—a significant practical simplification.
- **Weak teachers suffice**: The paper demonstrates that the mid-training teacher need not be powerful; a small model is sufficient. This finding transfers to other distillation and initialization settings—a small model can be trained quickly to provide coarse trajectories that guide large-model training.

## Limitations & Future Work
- A pretrained diffusion model is still required as a foundation; the approach cannot be applied from scratch.
- The number of ODE solver steps (16) during mid-training is fixed; the effect of varying this number on final quality is unexplored.
- One-step FID on text-to-image tasks remains relatively large (15.12), possibly due to dataset limitations.
- Theoretical analysis relies on simplified assumptions (uniform weighting, $\ell_2$ distance); theoretical guarantees when using perceptual losses are not fully addressed.
- The effectiveness of CMT on more complex generation tasks such as video generation remains to be explored.

## Related Work & Insights
- **vs. ECT/ECD**: CMT adopts these as post-training methods and significantly improves performance at equal or lower cost by adding mid-training. The essential difference lies in the superior initialization provided by CMT.
- **vs. sCT/sCD**: Comparable performance, but CMT reduces training cost by 93–98% by eliminating expensive JVP computations.
- **vs. Knowledge Distillation**: KD learns only endpoint mappings, whereas CMT exploits intermediate trajectory information, achieving higher data efficiency.
- **vs. Mean Flow**: CMT can use a small MF model as the teacher, and the subsequent MF post-training is accelerated by 50%.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Mid-training is systematically proposed for visual generation for the first time, though the core technique (trajectory regression) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple datasets, resolutions, pixel and latent spaces, both CM and MF frameworks, and text-to-image tasks, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Unifies the perspectives of CM, CTM, and MF; theoretical analysis is clear; experimental organization is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — Reduces practical training cost by over 90% while achieving state-of-the-art performance; engineering value is exceptionally high.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[NeurIPS 2025\] How to Build a Consistency Model: Learning Flow Maps via Self-Distillation](../../NeurIPS2025/image_generation/how_to_build_a_consistency_model_learning_flow_maps_via_self-distillation.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](../../NeurIPS2025/image_generation/freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)

<!-- RELATED:END -->
