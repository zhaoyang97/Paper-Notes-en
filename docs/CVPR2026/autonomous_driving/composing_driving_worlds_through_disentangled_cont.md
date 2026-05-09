---
title: >-
  [Paper Note] Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation
description: >-
  [CVPR 2026][Autonomous Driving][Driving World Model] This paper proposes CompoSIA, a compositional driving video simulator that injects three control factors — scene structure, object identity, and ego-vehicle action — through independent pathways into a Flow Matching DiT. It supports both individual and compositional editing, enabling systematic adversarial scenario synthesis. CompoSIA achieves a 17% FVD improvement on identity editing, 30%/47% reduction in rotation/translation error for action control, and an average 173% increase in collision rate for downstream planners.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Driving World Model
  - Disentangled Control
  - Adversarial Scenario Generation
  - Noise-Level Identity Injection
  - Flow Matching
date: 2026-05-08
content_hash: b497427bb1fe94d6
---

# Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12864](https://arxiv.org/abs/2603.12864)
**Code**: [GitHub](https://github.com/Yifever20002/CompoSIA)
**Area**: Autonomous Driving
**Keywords**: Driving World Model, Disentangled Control, Adversarial Scenario Generation, Noise-Level Identity Injection, Flow Matching

## TL;DR

This paper proposes CompoSIA, a compositional driving video simulator that injects three control factors — scene structure, object identity, and ego-vehicle action — through independent pathways into a Flow Matching DiT. It supports both individual and compositional editing, enabling systematic adversarial scenario synthesis. CompoSIA achieves a 17% FVD improvement on identity editing, 30%/47% reduction in rotation/translation error for action control, and an average 173% increase in collision rate for downstream planners.

## Background & Motivation

Autonomous driving systems face the core challenge of "long-tail" safety-critical scenarios: dangerous edge cases typically arise from uncommon combinations of common traffic elements (e.g., a truck suddenly changing lanes forcing hard braking), which are severely underrepresented in datasets such as nuScenes and Waymo. To deliberately construct such adversarial scenarios, generative models require fine-grained, independent control over scene layout (where elements are), object identity (what elements look like), and ego-vehicle behavior (how the ego vehicle moves).

However, existing methods suffer from fundamental limitations:
- **DriveEditor** supports structural and identity editing but cannot generate novel viewpoints or control actions
- **ReCamMaster** controls only camera actions, with no element-level structure or identity control
- **MagicDrive-V2** injects multi-condition signals through shared pathways, leading to signal coupling and degraded generation quality
- **Vista** exhibits insufficient action-following accuracy, with significantly elevated rotation and translation errors

The authors argue that the key to disentangled control lies in injecting signals from different factors at different levels of the diffusion process, rather than sharing a single pathway.

## Method

### Overall Architecture

CompoSIA is built upon Wan2.1-T2V-1.3B initialized as a Flow Matching DiT, with independent injection pathways designed for three types of control signals:

1. **Structure signal**: 3D bounding box sequences are projected to the image plane and VAE-encoded, then additively injected into latent tokens via zero-initialized projections.
2. **Identity signal**: A single reference image is aligned to the target region in noise space according to 2D bounding boxes and replaces the corresponding latent region, with hard binding enforced at high-noise timesteps.
3. **Action signal**: Frame-level camera trajectories are injected through a hierarchical dual-branch mechanism — local AdaLN residual modulation and global PRoPE attention.

The intermediate state is constructed as $z_{(t)} = \sigma_t z_{(0)} + (1-\sigma_t) \epsilon$, with the training objective being a v-prediction loss $\mathcal{L}_{CFM} = \mathbb{E}_{\epsilon} \| v_\Theta(z_{(t)}, t) - (z_{(0)} - \epsilon) \|_2^2$. The training modality ratio is 0.6:0.3:0.1 (action-only / structure+identity+action / unconditional).

### Key Designs

1. **Structure Condition (Spatiotemporal Layouts)**:
    - Function: Provides explicit constraints on element-level spatial positions and motion trajectories.
    - Mechanism: Each scene element is represented as a 3D bounding box sequence $\bm{b} \in \mathbb{R}^{F \times 7}$ (position, size, orientation), projected to the image plane via pinhole projection to obtain $\bm{b}_f$, processed through VAE encoding and a lightweight convolutional adapter to produce layout tokens $\bm{h}_{\bm{b}_f}$, and added to latent tokens via a zero-initialized projection: $\bm{h}_{(t)} \leftarrow \bm{h}_{(t)} + f_{\text{zero}}(\bm{h}_{\bm{b}_f})$.
    - Design Motivation: Projecting 3D boxes to 2D ensures alignment with the latent space; zero initialization prevents disruption of pre-trained priors during early training.

2. **Noise-Level Identity Injection**:
    - Function: Precisely injects element identity appearance from a single reference image without requiring pose alignment.
    - Mechanism: During training, one frame is randomly selected and its cropped reference image is aligned back to all frames by 2D bounding box to construct the identity cue $\bm{r}_f$. At high-noise timesteps ($t > T_{id}=0.2$), the target region latent is hard-replaced by the reference image latent: $z_{(t)} \leftarrow \bm{m} \odot z_{\bm{r}_f(t)} + (1-\bm{m}) \odot z_{(t)}$, with mask $\bm{m} = \bm{m}_{\bm{r}_f} \cdot \mathbb{I}(t > T_{id})$. During sampling, a stopping step $T_{id}=0.4$ is used to balance identity fidelity and generative freedom.
    - Design Motivation: Stronger identity constraints in the attention mechanism reduce motion expressiveness. Noise-level injection naturally blends identity information during the denoising recovery process while avoiding disruption of the denoising path at low-noise timesteps.

3. **Hierarchical Dual-Branch Action Control**:
    - Function: Precisely controls the ego-vehicle motion trajectory, balancing convergence speed and long-range accuracy.
    - Mechanism:
        - **Local Branch**: Extracts relative pose between adjacent frames $\bm{a} = (\Delta x, \Delta y, \Delta \text{yaw}) \in \mathbb{R}^{F \times 3}$, encodes it with sinusoidal frequency encoding, and produces 6-channel AdaLN parameters (3 each for self-attention and FFN: shift/scale/gate) via zero-initialized projections, implementing frame-level residual modulation.
        - **Global Branch**: Computes camera attention in a 1/8-dimensional subspace based on PRoPE (Projective Positional Encoding) and injects it into the main attention branch via zero convolution. $D^{proj}$ is derived from camera intrinsics and extrinsics.
    - Design Motivation: The local residual signal accelerates early training convergence but loses precise camera intrinsic information; the global PRoPE provides accurate long-range trajectory guidance but converges slowly. The two branches are complementary, and the 1/8 low-dimensional projection reduces computational overhead.

### Loss & Training

- **Loss**: v-prediction CFM loss
- **Training Configuration**: 16× A100 (80GB); learning rate $2 \times 10^{-4}$ for action projector and $1 \times 10^{-5}$ for other components; weight decay $5 \times 10^{-2}$; approximately 20,000 steps.
- **VAE Fine-tuning**: Temporal downsampling removed (stride 1 replacing the original 4×); fine-tuned for 7 days on 100 hours of internally collected data.
- **Training Data**: nuScenes 700 multi-view 20s videos + 100 hours of internal multi-view autonomous driving data at 10 Hz.
- **Mixed Resolution**: $33 \times 256 \times 512$ and $33 \times 480 \times 960$.
- **First Frame Handling**: Background regions are replaced with clean latents to anchor scene identity; foreground regions are filled with reference images; intermediate regions are treated as inpainting targets.
- **Condition Decoupling**: Structural conditions can leak action information (e.g., surrounding vehicles moving backward implying ego-vehicle moving forward), so structural conditions are always paired with action conditions during training.

## Key Experimental Results

### Main Results

**Video Generation Quality and Condition Alignment (Tab. 2)**:

| Task | Method | FVD ↓ | VBench Score ↑ |
|---|---|---|---|
| Scene Following | MagicDrive-V2 | 152.80 | 77.23% |
| Scene Following | **CompoSIA** | **133.66** | **81.05%** |
| Identity Control | TTM | 231.17 | 75.16% |
| Identity Control | LoRA-Edit | 161.32 | 79.83% |
| Identity Control | DriveEditor | 179.57 | 79.13% |
| Identity Control | **CompoSIA** | **149.15** | 80.30% |
| Action Control | ReCamMaster | 190.52 | 74.29% |
| Action Control | Vista | 171.49 | 75.35% |
| Action Control | MagicDrive-V2 | 279.61 | 73.44% |
| Action Control | **CompoSIA** | **137.21** | **80.79%** |

**Action Control Accuracy (Tab. 3, TransErr ×1000)**:

| Method | RotErr ↓ (Following) | TransErr ↓ (Following) | RotErr ↓ (Editing) | TransErr ↓ (Editing) |
|---|---|---|---|---|
| ReCamMaster | 1.12 | 20.35 | 2.17 | 25.45 |
| Vista | 0.81 | 14.25 | 2.33 | 28.12 |
| MagicDrive-V2 | 0.76 | 13.66 | 2.21 | 22.86 |
| **CompoSIA** | **0.55** | **7.37** | **1.54** | **12.15** |

**Planning Robustness Evaluation (Tab. 5, Epona Open-Loop)**:

| Edit Type | L2 Avg ↓ | Collision Rate 1s | Collision Rate 2s | Collision Rate 3s | Collision Rate Avg | Change |
|---|---|---|---|---|---|---|
| Following GT | 1.42 | 0.04% | 0.24% | 0.76% | 0.35% | — |
| Following Generation | 1.65 | 0.08% | 0.36% | 1.32% | 0.59% | — |
| Editing Structure | — | 0.72% | 2.68% | 5.28% | 2.89% | +390% |
| Editing Identity | 2.19 | 0.12% | 0.48% | 1.64% | 0.75% | +27% |
| Editing Action | 2.32 | 0.16% | 0.76% | 2.64% | 1.19% | +102% |

### Ablation Study

**Action Condition Branch Ablation (Tab. 4)**:

| Configuration | RotErr ↓ | TransErr ↓ |
|---|---|---|
| w/o local residual modulation (r.m.) | 2.84 | 15.80 |
| w/o global PRoPE attention (p.a.) | 0.62 | 11.24 |
| **Full** | **0.55** | **7.37** |

Removing local residual modulation causes RotErr to spike from 0.55 to 2.84 (+416%), demonstrating the local branch's critical role in rotation control. Removing global PRoPE increases TransErr from 7.37 to 11.24 (+53%), validating the global branch's contribution to translation accuracy.

### Key Findings

- **Structure Ablation**: Removing the structure condition causes complete failure in surrounding vehicle motion and spatial alignment.
- **Action Ablation**: Retaining only the structure condition is insufficient to infer ego-vehicle motion, confirming that action signals do not leak from structural conditions.
- **Identity Stopping Step $T_{id}$**: $T_{id}=0.6$ yields high generative freedom but identity drift; $T_{id}=0.2$ enforces strong identity preservation but over-anchors to the reference; $T_{id}=0.4$ achieves the optimal trade-off without requiring per-case tuning.
- **Additional Benefit of Identity Injection**: In scenes with complex illumination changes (e.g., tunnel traversal), noise-level identity injection significantly reduces cross-frame identity drift.

## Highlights & Insights

- The paper models driving scene generation as a three-factor composition problem (structure–identity–action), with each signal injected independently at different levels of the diffusion process — representing genuine disentanglement rather than shared-pathway conditioning.
- Noise-level identity injection elegantly reformulates identity control as a diffusion recovery problem, circumventing the inherent conflict between identity and motion in the attention mechanism.
- The downstream planner stress test elevates the world model from a "data synthesizer" to a "controllable simulator": structure editing causes a 390% surge in collision rate, exposing hidden failure modes that standard benchmarks cannot reveal.
- The hierarchical dual-branch design embodies an elegant complementarity between fast local convergence and accurate global control.

## Limitations & Future Work

- Identity editing generalization is limited by training data (primarily driving scenarios); performance degrades on completely out-of-distribution categories (e.g., animals), necessitating more diverse video data.
- The identity editing pipeline requires manual specification of approximate 3D bounding box dimensions for the reference target (currently assisted by Gemini estimation), remaining a semi-automated process.
- Planning robustness is evaluated only on nuScenes, without validation on larger-scale datasets such as Waymo.
- Joint editing of multi-agent interactions (e.g., simultaneously controlling the coordinated behavior of multiple vehicles) remains unexplored.

## Related Work & Insights

- vs **DriveEditor**: Supports only structure and identity editing; cannot generate novel viewpoints or control actions; identity transfer deviates from the original reference.
- vs **MagicDrive-V2**: Shared-pathway injection of structure and action signals causes coupling; scene-following FVD 152.80 vs. 133.66; action-following TransErr 13.66 vs. 7.37.
- vs **ReCamMaster**: Controls only camera actions; FVD 190.52; no element-level structure or identity control.
- vs **Vista**: Inferior action-following accuracy (RotErr 0.81 vs. 0.55; TransErr 14.25 vs. 7.37).
- vs **TTM**: Training-free sampling strategies struggle to maintain identity control under precise motion constraints; FVD 231.17.
- The disentangled control design paradigm is generalizable to other multi-condition controllable generation tasks (e.g., indoor scenes, robot manipulation videos).
- The trade-off between noise-level injection and attention-level injection represents a key design choice in the controllable generation literature.

## Rating

- Novelty: ⭐⭐⭐⭐ Three-factor disentangled injection + noise-level identity control + hierarchical action modulation; systematic and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative comparison across three tasks + multi-dimensional ablations + downstream planner stress testing.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, precise technical exposition, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Practical significance for autonomous driving safety evaluation and scenario diversity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)
- [\[ICLR 2026\] Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](../../ICLR2026/autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](ttsg_text_to_traffic_scene_generation_from_natural_language.md)

</div>

<!-- RELATED:END -->
