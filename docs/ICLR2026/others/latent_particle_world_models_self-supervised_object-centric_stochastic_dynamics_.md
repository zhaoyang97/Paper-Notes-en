---
title: >-
  [Paper Note] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics
description: >-
  [ICLR 2026][Object-centric] LPWM is the first self-supervised object-centric world model that scales to real-world multi-object datasets. Its core innovation is learning independent per-particle latent action distributions ($z_c^m$) for each particle, encoding all frames in parallel via a causal spatiotemporal Transformer, supporting diverse conditioning signals (actions, language, image goals, multi-view), achieving state-of-the-art video prediction, and demonstrating imitation learning capability (89% success rate on OGBench task3).
tags:
  - ICLR 2026
  - Object-centric
  - Latent particles
  - Self-supervised
  - World models
  - Stochastic dynamics
  - Latent actions
date: 2026-05-08
content_hash: ba26e10f44c99eac
---

# LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics

**Conference**: ICLR 2026
**arXiv**: [2603.04553](https://arxiv.org/abs/2603.04553)
**Code**: [Project Page](https://taldatech.github.io/lpwm-web)
**Area**: World Models / Object-Centric Representation / Video Prediction
**Keywords**: Object-centric, Latent particles, Self-supervised, World models, Stochastic dynamics, Latent actions

## TL;DR
LPWM is the first self-supervised object-centric world model that scales to real-world multi-object datasets. Its core innovation is learning independent per-particle latent action distributions ($z_c^m$) for each particle, encoding all frames in parallel via a causal spatiotemporal Transformer, supporting diverse conditioning signals (actions, language, image goals, multi-view), achieving state-of-the-art video prediction, and demonstrating imitation learning capability (89% success rate on OGBench task3).

## Background & Motivation
**Background**: Object-centric world models decompose scenes into independent object representations (slots/patches/particles), making them naturally suited for modeling multi-object interactions. The DLP (Deep Latent Particles) framework represents objects via keypoints and extended attributes (position, scale, depth, transparency, visual features).

**Limitations of Prior Work**:
- **Slot-based methods** (SlotFormer, etc.): suffer from inconsistent decomposition, blurry predictions, and convergence difficulties, and require two-stage training.
- **Patch-based methods** (G-SWM, etc.): rely on cross-frame post-hoc matching and fail to scale to complex data.
- **DDLP** (the current best particle-based method): depends on explicit particle tracking and sequential encoding → non-parallelizable and unable to model stochasticity.
- All object-centric methods are **limited to simple simulated environments** and cannot handle real-world multi-object videos.

**Key Challenge**: Object-centric representations offer natural advantages (interpretability, compositional generalization, sparse interaction modeling), but the key bottleneck for scaling to real-world complex scenes is handling the independent stochastic motion of multiple objects. Global latent actions cannot capture independent behaviors such as "object A moves left while object B remains stationary."

**Core Idea**: Learn independent latent action distributions $z_c^m$ for each latent particle — inferred from frame pairs via inverse dynamics during training, and sampled from a learned latent policy at inference time, conditioned into a causal spatiotemporal Transformer via AdaLN.

## Method

### Overall Architecture
Video frame sequence → Parallel particle encoder (extracting $M$ foreground particles $z_{fg}^m = [z_p, z_s, z_d, z_t, z_f]$ + background particle per frame) → Context Module (learning per-particle latent actions $z_c^m$) → Causal spatiotemporal Transformer for dynamics prediction → Particle decoder for next-frame reconstruction.

### Key Designs

1. **Latent Particle Representation (Tracking-Free)**

    - **Function**: Each frame is encoded into $M$ foreground particles, each with an attribute vector $z_{fg}^m = [z_p, z_s, z_d, z_t, z_f]$ (2D position, scale, depth, transparency, visual features).
    - **Key difference from DDLP**: **All $M$ encoded particles retain their identity** (based on patch origin), rather than tracking the trajectories of a small subset of particles. This enables parallel encoding across frames (no sequential dependency).
    - Positioned as a middle ground between patch-based and fully object-centric: particles can move within a bounded region around their origin but do not roam freely.
    - **Design Motivation**: Explicit tracking is the scalability bottleneck of DDLP — tracking failures cause compounding errors.

2. **Per-Particle Latent Actions (Context Module — Core Innovation)**

    - **Function**: Learns independent latent action distributions $q(z_c^m | o_t, o_{t+1})$ for each particle $m$.
    - **During training (inverse dynamics)**: Given two consecutive frames, infers each particle's latent action (analogous to an inverse model).
    - **During inference (latent policy)**: $\pi(z_c^m | o_{\leq t})$ — predicts the latent action distribution for each particle based solely on historical frames; stochastic prediction is achieved by sampling.
    - **Training objective**: KL divergence regularization $D_{KL}(q(z_c^m | o_t, o_{t+1}) \| \pi(z_c^m | o_{\leq t}))$.
    - **vs. global latent actions** (Genie, CADDY): Ablation experiments confirm that per-particle actions are critical — global actions cannot capture independent motion patterns across multiple objects.
    - **Design Motivation**: In multi-object scenes, object motions are independent (a ball moves left while a block stays still), requiring per-object stochasticity modeling.

3. **Causal Spatiotemporal Transformer Dynamics**

    - **Function**: Predicts particle attribute changes for the next frame.
    - **Mechanism**: Causal attention (attending only to historical frames) + spatial attention (inter-particle interactions within the same frame) + AdaLN conditioning (integrating latent actions $z_c^m$ into Transformer layers via Adaptive Layer Normalization).
    - **Design Motivation**: AdaLN more effectively incorporates conditioning signals than additive positional embeddings (validated by ablation).

4. **Multimodal Conditioning**

    - Action conditioning: External action signals are directly fused into the Transformer.
    - Language conditioning: Text encodings serve as additional conditioning.
    - Image goal conditioning: A goal frame encoding guides generation.
    - Multi-view: Multi-view particles can be jointly modeled for dynamics.
    - **Design Motivation**: A unified conditioning interface allows the same model to be applied to a variety of downstream tasks.

### Loss & Training
- End-to-end self-supervised training on raw video (no object labels or segmentation annotations required).
- Reconstruction loss + KL regularization (latent action distribution).
- Training resolution: 128×128; $M$ particles adjusted per dataset.

## Key Experimental Results

### Video Prediction (Main Results)

| Dataset | Condition Type | DVAE LPIPS↓ | LPWM LPIPS↓ | DVAE FVD↓ | LPWM FVD↓ |
|--------|---------|-------------|-------------|-----------|-----------|
| Sketchy-U | Latent action | 0.113 | **0.070** | 140.06 | **85.45** |
| BAIR-U | Latent action | 0.063 | **0.062** | 164.41 | **163.91** |
| Bridge-L | Language | — | — | 146.85 | **47.78** |
| Mario-U | Latent action | — | **Best** | — | **Best** |

LPWM surpasses all baselines on LPIPS and FVD across all stochastic dynamics datasets.

### Imitation Learning

| Environment / Task | GCIVL | HIQL | **LPWM** |
|-----------|-------|------|---------|
| PandaPush 1 Cube | 74±4 | — | **100±0** |
| PandaPush 3 Cubes | 62.1±4.4 | — | **89.4±2.5** |
| OGBench task1 | 84±4 | 80±6 | **100±0** |
| OGBench task3 | 16±8 | 61±11 | **89±9** |

LPWM significantly outperforms baselines across PandaPush and OGBench tasks. On OGBench task3 (involving 4 atomic behaviors), LPWM achieves 89% success vs. 16% for EC Diffuser.

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| Global vs. per-particle latent actions | Per-particle significantly superior | Core innovation validated |
| Latent action dimensionality | Best near effective particle dimension ($6+d_{obj}$) | Model is robust to dimensionality |
| AdaLN vs. additive positional embedding | AdaLN superior | Conditioning mechanism matters |

### Key Findings
- Per-particle latent actions are the decisive performance factor — global latent actions prevent independent motion modeling in multi-object scenes.
- LPWM's compact model matches large-scale video generation models on BAIR-64 FVD (89.4), demonstrating the efficiency advantage of object-centric inductive biases.
- Success on real-world datasets (Sketchy, BAIR, Bridge) challenges the prevailing assumption that object-centric methods are restricted to simulation.
- Imagined trajectories closely match actual execution (Figure 4), confirming that world model accuracy directly translates to decision-making capability.

## Highlights & Insights
- **Scalability breakthrough for object-centric representations**: Object-centric methods have long been considered viable only in simple simulated settings. LPWM demonstrates that the right design choices — per-particle latent actions, removing explicit tracking, and parallel encoding — enable scaling to real-world data, representing a conceptual advance.
- **Computational realization of the "what-where" visual pathway**: The particle representation naturally corresponds to the object decomposition in the visual system; latent actions correspond to motor prediction; the overall architecture aligns with neuroscientific findings on human visuospatial world models.
- **Unified multi-condition interface**: A single model supports unconditional, action, language, image goal, and multi-view conditioning, enabling natural transfer from video pretraining to robotic control.

## Limitations & Future Work
- Assumes limited camera motion and repetitive scenes (e.g., robot tabletop manipulation); not suitable for arbitrary video (e.g., street scenes, films).
- Long-horizon tasks involving more than 4 atomic behaviors (OGBench tasks 4/5) remain unsolved across all methods — long-term planning is still a challenge.
- The particle count $M$ is fixed and cannot dynamically adapt to scenes of varying complexity.
- Implicit tracking (particles moving near their origin) may fail in large-displacement scenarios.
- Integration with explicit reward models and reinforcement learning has not been explored.

## Related Work & Insights
- **vs. SlotFormer (slot-based)**: Two-stage training, inconsistent decomposition, and blurry predictions; LPWM uses end-to-end training and per-particle latent actions for greater flexibility.
- **vs. Genie/CADDY (global latent actions)**: Global actions cannot capture independent multi-object motion; PlaySlot adds slot-level latent actions but is constrained by the limitations of slot representations.
- **vs. DDLP (closest particle-based predecessor)**: DDLP requires explicit tracking and sequential encoding; LPWM removes tracking, enables parallel encoding, and introduces stochasticity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Per-particle latent actions represent a natural yet non-obvious key innovation; scaling object-centric world models to real-world data is a significant breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ 6+ datasets (synthetic + real) covering video prediction, imitation learning, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation and design logic are clearly presented, with thorough comparison to prior work.
- Value: ⭐⭐⭐⭐⭐ A pioneering contribution to object-centric world models, with code, data, and models fully open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICLR 2026\] Latent Fourier Transform](latent_fourier_transform.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[AAAI 2026\] LILAD: Learning In-context Lyapunov-stable Adaptive Dynamics Models](../../AAAI2026/others/lilad_learning_in-context_lyapunov-stable_adaptive_dynamics_models.md)

</div>

<!-- RELATED:END -->
