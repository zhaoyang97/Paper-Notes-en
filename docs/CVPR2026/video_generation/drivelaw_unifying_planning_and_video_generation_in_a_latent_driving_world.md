---
title: >-
  [Paper Note] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World
description: >-
  [CVPR 2026][Video Generation][World Model] This paper proposes DriveLaW, a driving world model that unifies video generation and motion planning through a shared latent space. The intermediate latent features of the vide…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "World Model"
  - "Autonomous Driving Planning"
  - "Latent Space"
  - "Diffusion Policy"
date: 2026-05-08
content_hash: e4ee00ed2cf07f94
---

# DriveLaW: Unifying Planning and Video Generation in a Latent Driving World

**Conference**: CVPR 2026
**arXiv**: [2512.23421](https://arxiv.org/abs/2512.23421)  
**Code**: [https://github.com/xiaomi-research/drivelaw](https://github.com/xiaomi-research/drivelaw)  
**Area**: Video Generation
**Keywords**: World Model, Autonomous Driving Planning, Video Generation, Latent Space, Diffusion Policy

## TL;DR

This paper proposes DriveLaW, a driving world model that unifies video generation and motion planning through a shared latent space. The intermediate latent features of the video generator are directly injected into a diffusion-based planner, achieving state-of-the-art performance simultaneously on the nuScenes video prediction benchmark and the NAVSIM planning benchmark.

## Background & Motivation

World models address long-tail challenges in real-world driving by learning the temporal evolution of driving scenes. However, existing approaches confine the role of the world model to three indirect paradigms: (1) as a **data generator**—synthesizing rare scene data or serving as a closed-loop simulation environment; (2) as a **supervisory signal**—predicting future visual or reachability signals to supervise planning; and (3) as a **parallel generator**—jointly generating video and trajectories within a unified architecture but still via a decoupled process.

**Key Challenge**: Even within "unified" architectures, the video generator and planner still operate as independent modules. Epona and DriveVLA-W0, for instance, train video generation and policy heads separately, without leveraging the generator's internal latent representations as planning states. Although the video generator acquires rich scene semantics, agent dynamics, and physical priors from large-scale data, this knowledge is "wasted" on rendering and never transferred to the planner.

**Core Idea**: The internal activations of a video generator encode rich, temporally coherent scene understanding—precisely the representations required for planning. DriveLaW repositions the generator from a "renderer" to a "feature extractor," using its denoised latent features directly as conditional inputs to the planner.

## Method

### Overall Architecture

DriveLaW consists of two core components connected in a chain: (1) **DriveLaW-Video**—a spatiotemporal video generator comprising a spatiotemporal VAE and a Video DiT (Diffusion Transformer), which takes historical observations and actions as input and outputs denoised video latent features; and (2) **DriveLaW-Act**—a lightweight Action DiT diffusion planner conditioned on the video latent features, generating future trajectories via flow matching. The two components are optimized through a three-stage progressive training strategy.

### Key Designs

1. **Chained Generation-Planning Architecture (Chained Design)**:

    - **Function**: Transfers representations from the video generator directly to the planner.
    - **Mechanism**: Unlike parallel designs (where video and trajectory are output independently), DriveLaW directly injects the denoised latent features $z$ from the Video DiT into the Action DiT as conditioning. These latent features encode compact representations of scene semantics, agent dynamics, and physical priors learned from large-scale video pretraining. The Action DiT follows a standard DiT architecture and is trained with a flow matching objective.
    - **Design Motivation**: The chained design offers three advantages over parallel designs: (a) it fully leverages representations learned from large-scale video pretraining; (b) it avoids gradient interference between video generation and planning during training; and (c) the cascade ensures consistency between generated visual details and planned trajectories.

2. **Noise Reinjection Mechanism**:

    - **Function**: Balances aggressive compression with visual fidelity.
    - **Mechanism**: Under the high compression ratio of the spatiotemporal VAE, early denoising stages may produce structural inconsistencies and blurriness, particularly in high-speed scenarios. Noise reinjection explores and selects the optimal generation path during early denoising—by reintroducing controlled noise into intermediate denoised results, the model is allowed to re-explore alternative generation paths.
    - **Design Motivation**: There is an inherent tension between high-fidelity video synthesis and stable real-time planning. A highly compressed VAE benefits planning efficiency but degrades visual quality; noise reinjection serves as a mediator between these two objectives.

3. **Three-Stage Progressive Training Strategy**:

    - **Function**: Coordinates the optimization of video generation and planning.
    - **Mechanism**: (a) *Stage 1—Learning Long-Horizon Motion*: trains the Video DiT to generate coarse-grained video, establishing an understanding of temporal dynamics; (b) *Stage 2—Refining Spatial Details*: fine-tunes video quality at higher resolution or with more refined denoising steps; (c) *Stage 3—Chained Planning*: freezes the Video DiT, chains its latent features to the Action DiT, and trains the planner.
    - **Design Motivation**: Direct end-to-end training leads to conflicting objectives between video generation and planning. The progressive strategy allows each component to be optimized within its own optimal learning window.

### Loss & Training

The Video DiT uses a standard diffusion loss (denoising objective), while the Action DiT uses a flow matching objective to generate trajectories. In Stage 3, the Video DiT parameters are frozen and only the Action DiT is trained.

## Key Experimental Results

### Main Results

**nuScenes Video Generation**

| Method | FID↓ | FVD↓ | Notes |
|--------|------|------|-------|
| Prev. SOTA | Baseline | Baseline | Various world models and video generators |
| **DriveLaW-Video** | **−33.3%** | **−1.8%** | Substantial margin |

**NAVSIM Planning Benchmark (PDMS)**

| Method | PDMS | Notes |
|--------|------|-------|
| Prev. SOTA (world model methods) | Baseline | Various world model + planning approaches |
| **DriveLaW-Act** | **New record** | No post-training (RL) or post-processing (scorers) required |

### Ablation Study

| Configuration | FID | PDMS | Notes |
|---------------|-----|------|-------|
| BEV features → planner | Higher | Lower | Conventional BEV representation |
| VLM features → planner | Medium | Medium | Vision-language model features |
| **Video latent features → planner** | **Lowest** | **Highest** | Video generator representations are optimal |
| Parallel design | Medium | Medium | Generation and planning as independent outputs |
| **Chained design** | **Lowest** | **Highest** | Latent features transferred to planner |

### Key Findings

- Video generator latent representations outperform BEV and VLM features as planning inputs, demonstrating the unique value of representations learned from large-scale video pretraining.
- The chained design outperforms the parallel design on both tasks, validating that representation transfer is superior to independent outputs.
- The noise reinjection mechanism significantly reduces blurriness and structural inconsistencies in high-speed scenarios.
- NAVSIM SOTA is achieved without RL post-training or scorer-based post-processing, indicating that the video prior alone is sufficiently powerful.

## Highlights & Insights

- **"Video generator as feature extractor"** represents a profound paradigm shift: repositioning a generative model from a terminal output module to a provider of intermediate representations, bridging the boundary between "generation" and "understanding."
- The **chained vs. parallel** comparison is compelling: even within a "unified" architecture, the direction and coupling of information flow are critical.
- The **three-stage training** strategy elegantly avoids multi-objective conflicts—the approach of optimizing components separately before cascaded fine-tuning is broadly generalizable.

## Limitations & Future Work

- The chained design ties planning latency to video generation speed, which may be insufficient for real-time deployment.
- Only single-view video generation is considered; multi-view consistency is not addressed.
- Validation is limited to nuScenes and NAVSIM; robustness in real-world closed-loop driving deployment remains untested.
- Errors in the video generator propagate directly to the planner (error cascading).

## Related Work & Insights

- **vs. Epona**: Generates video and trajectories in parallel; the decoupled design does not exploit the generator's internal representations.
- **vs. DriveVLA-W0**: Also employs a hybrid Transformer to generate both modalities, but retains parallel output streams.
- **vs. DiffusionDrive**: A pure diffusion-based planner with no world-understanding prior from video generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to use intermediate latent representations of a video generator as planning states; the chained design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-task SOTA, representation comparison ablations, and architecture design ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, though the three-stage training details could be elaborated further.
- Value: ⭐⭐⭐⭐⭐ — Introduces a new paradigm for autonomous driving world models; the Xiaomi EV affiliation suggests practical deployment relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](../../ICML2026/video_generation/olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)

</div>

<!-- RELATED:END -->
