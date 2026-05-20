---
title: >-
  [Paper Note] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation
description: >-
  [ICCV 2025][Autonomous Driving][traffic simulation] LangTraj is proposed as the first diffusion-based trajectory simulator that incorporates natural language as a training-time condition. It is accompanied by the InterDr…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "traffic simulation"
  - "language conditioning"
  - "diffusion model"
  - "trajectory generation"
date: 2026-05-08
content_hash: 9ed33cd8bcb10634
---

# LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation

**Conference**: ICCV 2025
**arXiv**: [2504.11521](https://arxiv.org/abs/2504.11521)  
**Code**: [Project Page](https://langtraj.github.io/)  
**Area**: Autonomous Driving
**Keywords**: traffic simulation, language conditioning, diffusion model, trajectory generation, autonomous driving

## TL;DR

LangTraj is proposed as the first diffusion-based trajectory simulator that incorporates natural language as a training-time condition. It is accompanied by the InterDrive dataset, containing 150K human-annotated interaction behaviors, enabling language-controllable multi-agent interaction simulation and safety-critical scenario generation.

## Background & Motivation

Traffic simulation is central to safety testing for autonomous driving. Existing approaches suffer from the following limitations:

**Hand-crafted scenarios do not scale**: Traditional structured testing relies on manually designed failure scenarios, which fail to cover long-tail distributions.

**Limited controllability of diffusion models**: Existing diffusion-based trajectory generation methods (e.g., CTG, SAFE-SIM) rely on post-training, domain-specific guidance functions that require expert knowledge to design and slow down inference.

**Absence of language conditioning**: Language is the most intuitive control interface, yet no prior method directly learns language-to-trajectory mappings during training.

**Scarcity of interaction behavior data**: Existing datasets (e.g., ProSim-Instruct) focus primarily on single-agent behaviors, lacking fine-grained annotations for multi-agent interactions.

Core insight: **Introducing language conditions at training time allows the model to learn semantics from the data distribution, rather than approximating them with heuristic guidance functions at inference time. Language conditioning and guidance functions are orthogonal and can be used in combination.**

## Method

### Overall Architecture

LangTraj consists of three main components: a Scene Encoder, a Language Encoder, and a Denoiser, jointly modeling the future trajectories of all agents via a diffusion model. A closed-loop training strategy is additionally proposed to improve closed-loop simulation stability.

### Key Designs

1. **Scene Encoder**: Adopts a query-centric + GNN approach, extracting features in each scene element's local coordinate frame (independent of global coordinates) and encoding relative spatiotemporal information via attention mechanisms. The output per-agent embedding is $\mathbf{z}_{enc}^i = E_{enc}(I, \mathbf{S}_{t-T_{hist}:t})$. **Design Motivation**: The local coordinate frame ensures symmetric encoding across agents.

2. **Language Encoder**: Rewrites agent roles in input sentences (e.g., "target agent," "other agent1") and extracts sentence embeddings $\mathbf{e}_{lang}$ via DistilBERT, which are then fused with scene embeddings:

    $\mathbf{z}_{lang}^i = E_L(\mathbf{e}_{lang}, \mathbf{z}_{enc}^i)$

   LoRA is used for end-to-end fine-tuning, balancing efficiency and effectiveness. **Design Motivation**: Role rewriting ensures language conditions are precisely aligned to specific agents.

3. **Denoiser**: Stacks Transformer blocks incorporating three types of attention:
    - Inter-agent query-centric attention: captures multi-agent interactions
    - Agent-Context cross-attention: aligns behavior with spatiotemporal context
    - Text-Cross attention: injects language conditions

4. **Closed-Loop Training Strategy**: One of the core innovations. Conventional diffusion models are trained in an open-loop fashion, leading to distribution shift and error accumulation during closed-loop inference. This work proposes incorporating model-generated samples into training:
    - At each step, GT trajectories are noised → one-step denoising generates $M$ candidates → the candidate closest to GT is selected for execution
    - L2 loss between the executed trajectory and GT is computed in the global coordinate frame
    - Teacher forcing is applied: a subset of agents still follow GT to stabilize training

5. **Dual-Mode Controllable Generation**:
    - **Classifier-free guidance**: interpolates conditional and unconditional predictions, $\hat{\tau}_0 = (1+w) \cdot g(\mathbf{e}_{lang}, \mathbf{z}_{enc}) - w \cdot g(\emptyset, \mathbf{z}_{enc})$
    - **Classifier-based guidance**: modifies the denoising step mean via gradients of an objective $J(\tau)$, supporting collision-inducing guidance and similar objectives

### Loss & Training

Open-loop pre-training followed by closed-loop fine-tuning. During closed-loop training, the loss is the L2 distance between the executed trajectory and GT in the global coordinate frame. Reducing the number of denoising steps from $K=100$ to $K=5$ does not degrade performance and substantially improves efficiency.

## Key Experimental Results

### Main Results (WOSAC Test Set)

| Method | Meta ↑ | Kinematic ↑ | Interactive ↑ | Map ↑ |
|--------|--------|-------------|---------------|-------|
| UniMM | 0.769 | 0.491 | 0.811 | 0.874 |
| VBD | 0.720 | 0.417 | 0.814 | 0.776 |
| **LangTraj** | **0.719** | **0.426** | **0.795** | **0.789** |
| ProSim | 0.718 | 0.401 | 0.778 | 0.822 |
| SceneDiffuser | 0.703 | 0.430 | 0.776 | 0.768 |

- Achieves **best performance** among diffusion-based models, competitive with VBD and SceneDiffuser.

### Language Controllability Evaluation

| Text Conditioning | Meta ↑ | Map ↑ | mADE ↓ |
|-------------------|--------|-------|--------|
| Unconditional | 0.72 | 0.80 | 2.65 |
| **Direct Conditioning (Ours)** | **0.72** | **0.79** | **2.29** |
| LLM-Based Guidance (CTG++) | 0.70 | 0.80 | 2.70 |

- Direct language conditioning reduces mADE by **13.6%**, while LLM-based guidance performs even worse than the unconditional baseline.

### Safety-Critical Scenario Generation

| Setting | Collision Rate ↑ | Kinematic ↑ | Map ↑ |
|---------|-----------------|-------------|-------|
| No text + No guidance | 0.04 | 0.42 | 0.81 |
| No text + Collision guidance | 0.41 | 0.39 | 0.70 |
| **Direct conditioning + Collision guidance** | **0.43** | **0.41** | **0.74** |
| LLM Guidance + Collision guidance | 0.33 | 0.37 | 0.72 |

### Ablation Study

| Setting | Meta ↑ | Map ↑ |
|---------|--------|-------|
| Open-loop ($K=100$) | 0.68 | 0.73 |
| Open-loop ($K=5$) | 0.68 | 0.72 |
| Closed-loop | 0.69 | 0.75 |
| **Closed-loop + Teacher Forcing** | **0.70** | **0.79** |

### Key Findings

- Direct language conditioning substantially outperforms LLM-based guidance, particularly for interactive behavior descriptions.
- Closed-loop training with teacher forcing significantly improves map adherence (0.72 → 0.79), preventing model drift.
- Reducing denoising steps from 100 to 5 does not degrade realism; 5 steps are sufficient.
- Language conditioning and collision guidance are orthogonal and complementary; their combination achieves a 43% collision rate alongside improved map metrics.

## Highlights & Insights

- **Training-time direct language conditioning vs. inference-time guidance functions**: The paper empirically demonstrates the clear superiority of direct conditioning, offering broad implications for the controllable generation field.
- **InterDrive Dataset**: 150K human-annotated interaction behaviors across 6 categories (merging, yielding, passing, etc.), filling the gap in interaction-level language annotations.
- **Closed-loop training** for diffusion models is elegantly implemented: single-step denoising with best-candidate selection avoids the computational overhead of double-loop schemes.

## Limitations & Future Work

- The dataset is predominantly drawn from Waymo; generalization to other domains (e.g., Chinese road scenarios) remains to be validated.
- DistilBERT has limited capacity as a language encoder; replacing it with a more powerful LLM may yield further improvements.
- Closed-loop training requires careful balancing between convergence speed and the teacher forcing ratio.
- Future work may explore multi-turn dialogue-based scenario editing.

## Related Work & Insights

- Comparison with ProSim suggests that diffusion models are better suited than autoregressive approaches for inference-time guidance and out-of-distribution sampling.
- The closed-loop training strategy is transferable to other diffusion-based sequential generation tasks (e.g., robotic manipulation planning).
- Language-trajectory alignment shares potential connections with methods in the text-to-motion domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First diffusion trajectory model with training-time language conditioning and closed-loop training.
- **Technical Depth**: ⭐⭐⭐⭐ — Closed-loop training strategy is cleverly designed; multiple controllable generation paradigms are effectively integrated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers WOSAC benchmark, language alignment, safety-critical scenarios, and ablations comprehensively.
- **Value**: ⭐⭐⭐⭐ — Supports flexible AV testing, though large-scale language-annotated data is required.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[CVPR 2026\] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](../../CVPR2026/autonomous_driving/sparseworld_tc_trajectory_conditioned_sparse_occupancy_world_model.md)
- [\[ICCV 2025\] Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)

</div>

<!-- RELATED:END -->
