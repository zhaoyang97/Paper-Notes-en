---
title: >-
  [Paper Note] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation
description: >-
  [CVPR 2026][Human Understanding][Multi-human motion generation] This paper proposes Sketch2Colab, which distills a sketch-driven diffusion prior into a rectified flow student network, and combines energy guidance with continuous-time Markov chain (CTMC) discrete event planning to generate coordinated multi-human–object interaction 3D motions from storyboard sketches, achieving state-of-the-art constraint compliance and perceptual quality on CORE4D and InterHuman.
tags:
  - CVPR 2026
  - Human Understanding
  - Multi-human motion generation
  - sketch guidance
  - rectified flow distillation
  - human-object-human collaboration
  - CTMC discrete events
date: 2026-05-08
content_hash: c5d155c4b3fe0c3c
---

# Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation

**Conference**: CVPR 2026
**arXiv**: [2603.02190](https://arxiv.org/abs/2603.02190)
**Code**: Unavailable
**Area**: Human Understanding
**Keywords**: Multi-human motion generation, sketch guidance, rectified flow distillation, human-object-human collaboration, CTMC discrete events

## TL;DR

This paper proposes Sketch2Colab, which distills a sketch-driven diffusion prior into a rectified flow student network, and combines energy guidance with continuous-time Markov chain (CTMC) discrete event planning to generate coordinated multi-human–object interaction 3D motions from storyboard sketches, achieving state-of-the-art constraint compliance and perceptual quality on CORE4D and InterHuman.

## Background & Motivation

**State of the Field**: Diffusion models have demonstrated strong performance in single-person motion generation, with mature solutions for text-, trajectory-, and style-conditioned control. Collaborative multi-person–object scenarios (e.g., two people carrying a table together) remain an open challenge; representative work COLLAGE has made preliminary progress via LLM planning combined with latent diffusion.

**Limitations of Prior Work**: (1) Text as a control channel is too coarse—temporal ordering and spatial layout are difficult to express precisely; (2) achieving accurate constraint compliance in diffusion models under strong multi-entity constraints requires expensive posterior guidance or dedicated control modules, leading to slow and unstable sampling; (3) multi-entity interaction involves discrete events (contact, grasping, handover) that are difficult to model with purely continuous flows.

**Root Cause**: Sketches provide spatiotemporal control signals that are more precise than text, yet extending sketch-based constraints to multi-person–object scenarios faces: keyframe misalignment, multi-agent phase drift, and difficulty optimizing discrete contact/handover states.

**Paper Goals**: How to generate coordinated multi-human–object 3D motion sequences from storyboard sketches (keyframe poses, joint trajectories, and object paths)?

**Starting Point**: Diffusion-to-rectified-flow distillation for fast and stable sampling + energy guidance for precise constraint enforcement + CTMC for discrete contact event modeling.

**Core Idea**: Distill a diffusion teacher into a rectified flow student; use differentiable energy functions in dual spaces (raw motion space and latent space) to guide constraint compliance; use CTMC to schedule discrete interaction events that modulate the continuous flow.

## Method

### Overall Architecture

Input storyboard (keyframe poses, joint trajectories, object masks, optional text) → 2D/3D alignment encoder extracts control signals → rectified flow student network generates motions in VQ-VAE latent space → CTMC schedules contact/handover events to modulate sub-fields → dual-space energy guidance enforces constraint compliance → frozen decoder produces final multi-human–object 3D motions.

### Key Designs

1. **Diffusion-to-Rectified-Flow Distillation (PF-Distillation)**:

    - **Function**: Distills a pretrained sketch-driven diffusion teacher into a rectified flow student.
    - **Mechanism**: The teacher is trained with a VP noise schedule to obtain the probability flow velocity field $v_\theta^{PF}$. The student jointly minimizes the rectified flow objective $\mathcal{L}_{RF}$ and the PF distillation objective $\mathcal{L}_{distill}$. Condition injection adopts a lift-then-fuse scheme: ControlNet-style trajectory paths combined with a time-gated keyframe adapter.
    - **Design Motivation**: Directly training a multi-entity rectified flow model is computationally expensive and difficult; distilling a pretrained diffusion teacher inherits its learned motion distribution while enabling faster and more stable sampling.

2. **Dual-Space Energy Guidance (Dual-Space Conditioning)**:

    - **Function**: Simultaneously guides constraint compliance in raw motion space and latent space.
    - **Mechanism**: Defines differentiable energy functions $E_{key}$ (keyframe alignment), $E_\tau$ (trajectory tracking), $E_{int}$ (contact/spacing), and $E_{phys}$ (physical constraints such as foot sliding prevention and ground plane). A learned low-rank block-Toeplitz Jacobian approximation $\mathbf{B}_\rho$ maps gradients from raw space to latent space. A latent-space anchor loss $\mathcal{L}_{lat}$ maintains samples on the motion manifold.
    - **Design Motivation**: Constraints applied purely in raw space may leave the manifold, while purely latent-space constraints lack geometric precision—the dual-space approach is complementary.

3. **CTMC Discrete Event Planning**:

    - **Function**: Handles temporal scheduling of discrete events such as contact, grasping, and handover.
    - **Mechanism**: A continuous-time Markov chain is defined over contact states, with transition rates learned via physics-informed objectives. The resulting contact/handover schedule $\boldsymbol{\pi}_t$ modulates the sub-fields and contact weights of the continuous flow, coupled via importance weighting.
    - **Design Motivation**: Contact on/off, grasping, and handover are inherently discrete events; purely continuous flow optimization tends to produce ambiguous mode transitions and contact flickering—CTMC provides clean, correctly phased discrete scheduling.

### Loss & Training

The total training loss comprises $\mathcal{L}_{RF}$ (rectified flow) + $\mathcal{L}_{distill}$ (distillation) + $\mathcal{L}_{Lyap}$ (Lyapunov potential) + $\mathcal{L}_{CTMC}$ (discrete events) + $\mathcal{L}_{lat}$ (latent anchor) + individual energy terms. Classifier-free guidance is applied with 10% dropout and guidance scale $\omega \in [1.4, 1.8]$. SMPL-X with 22 joints and 6D rotation representation is used throughout.

## Key Experimental Results

### Main Results (CORE4D and InterHuman)

| Method | Constraint Compliance | FID↓ | Perceptual Quality | Inference Speed |
|---|---|---|---|---|
| COLLAGE | Baseline | — | Baseline | Slow (diffusion) |
| Sketch2Anim | Single-person only | — | — | Moderate |
| **Sketch2Colab** | **SOTA** | **Lowest** | **Highest** | **Fast (rectified flow)** |

### Ablation Study

| Configuration | Constraint Compliance | Note |
|---|---|---|
| Full model | Best | Complete pipeline |
| w/o CTMC | Degraded (contact ambiguity) | Discrete events lost |
| w/o dual-space | Degraded (geometric deviation) | Lack of precise guidance |
| w/o distillation | Significant degradation | Direct rectified flow training is unstable |

### Key Findings

- Diffusion-to-rectified-flow distillation substantially outperforms directly extending a diffusion baseline to multi-entity scenarios, avoiding keyframe misalignment and phase drift.
- CTMC contributes most to contact quality—without it, contacts exhibit flickering and temporal errors.
- Dual-space guidance is more effective than either raw-space-only or latent-space-only guidance.

## Highlights & Insights

- **Three-tier architecture of distillation + energy + CTMC**: Continuous dynamics are handled by rectified flow, precise constraints by energy guidance, and discrete events by CTMC—each component has a well-defined role while remaining tightly coupled, resulting in an elegant overall design.
- **Sketches are better suited for interaction control than text**: Sketches naturally encode spatiotemporal information (when, where, and in what pose), offering far greater precision than textual descriptions.
- **Jacobian approximation bridges the dual spaces**: Learning a low-rank block-Toeplitz Jacobian to map gradients from raw to latent space avoids costly automatic differentiation.

## Limitations & Future Work

- The method requires storyboard-level inputs—creating precise keyframe sketches still demands animator expertise.
- The discrete state space of CTMC is fixed (contact/non-contact), leaving richer interaction semantics (e.g., "carefully," "forcefully") unmodeled.
- Evaluation is conducted primarily on CORE4D and InterHuman; the complexity of real-world film and game production scenarios is considerably higher.

## Related Work & Insights

- **vs. COLLAGE**: COLLAGE relies on text + LLM planning + latent diffusion; Sketch2Colab uses sketches + rectified flow + CTMC, offering more precise control and faster inference.
- **vs. Sketch2Anim**: Sketch2Anim supports only single-person generation; Sketch2Colab extends to collaborative multi-human–object scenarios.
- **vs. MotionLab**: MotionLab also employs rectified flow but targets unified single-person generation and editing; Sketch2Colab focuses specifically on sketch-conditioned multi-entity control.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of distillation + energy guidance + CTMC is pioneering in motion generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two multi-human interaction benchmarks, CORE4D and InterHuman.
- Writing Quality: ⭐⭐⭐⭐ Method description is systematic but notation-dense.
- Value: ⭐⭐⭐⭐⭐ Significant advancement for multi-person collaborative motion generation in animation and game industries.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2026\] LaScA: Language-Conditioned Scalable Modelling of Affective Dynamics](lasca_language-conditioned_scalable_modelling_of_affective_dynamics.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)
- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](../../ICCV2025/human_understanding/dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[AAAI 2026\] PressTrack-HMR: Pressure-Based Top-Down Multi-Person Global Human Mesh Recovery](../../AAAI2026/human_understanding/presstrack-hmr_pressure-based_top-down_multi-person_global_human_mesh_recovery.md)

<!-- RELATED:END -->
