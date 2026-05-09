---
title: >-
  [Paper Note] UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control
description: >-
  [ICCV 2025][Image Restoration][Physics-based character control] This paper proposes UniPhys, a behavior cloning framework based on diffusion models that unifies motion planning and physics-based control within a single model. By adopting the Diffusion Forcing training paradigm to address compounding prediction errors, UniPhys enables flexible multi-task physics-based character motion generation, including text-driven control, velocity control, goal reaching, and dynamic obstacle avoidance.
tags:
  - ICCV 2025
  - Image Restoration
  - Physics-based character control
  - diffusion model
  - behavior cloning
  - Diffusion Forcing
  - text-driven control
date: 2026-05-08
content_hash: e1f53ff2950b05ab
---

# UniPhys: Unified Planner and Controller with Diffusion for Flexible Physics-Based Character Control

**Conference**: ICCV 2025  
**arXiv**: [2504.12540](https://arxiv.org/abs/2504.12540)  
**Code**: [Project Page](https://wuyan01.github.io/uniphys-project/)  
**Area**: Image Restoration (Note: This paper actually belongs to the physics-based character control domain; the categorization under image restoration may be erroneous.)  
**Keywords**: Physics-based character control, diffusion model, behavior cloning, Diffusion Forcing, text-driven control

## TL;DR

This paper proposes UniPhys, a behavior cloning framework based on diffusion models that unifies motion planning and physics-based control within a single model. By adopting the Diffusion Forcing training paradigm to address compounding prediction errors, UniPhys enables flexible multi-task physics-based character motion generation, including text-driven control, velocity control, goal reaching, and dynamic obstacle avoidance.

## Background & Motivation

Generating natural and physically plausible character motion is a central challenge in computer graphics, gaming, and robotics. Existing approaches suffer from the following limitations:

**Domain gap in hierarchical frameworks**: Mainstream methods decompose control into two levels—a high-level diffusion-based planner that generates kinematic targets and a low-level RL controller that tracks and executes them. However, the kinematic outputs of the planner are often inconsistent with physical constraints, leading to artifacts such as jitter and foot sliding. Moreover, the controller requires fine-tuning for each new task.

**Limitations of multi-task policies**: Existing text-driven policies (e.g., SuperPADL, PDP) exhibit limited motion diversity and expressiveness, and lack the ability to generalize to novel guidance signals. Methods such as MaskedMimic cannot generalize beyond a predefined set of control signals.

**Compounding errors in behavior cloning**: When the control problem is framed as imitation learning, small errors in autoregressive prediction accumulate over time, causing long-horizon generation to become unstable.

Core insight: If compounding errors can be effectively suppressed, a low-level RL policy becomes unnecessary—a single diffusion model can simultaneously perform planning and control, fundamentally eliminating the domain gap.

## Method

### Overall Architecture

The core pipeline of UniPhys proceeds as follows:
1. A large-scale paired state-action dataset is constructed by tracking the AMASS motion capture dataset using the PULSE tracking policy in Isaac Gym.
2. A causal Transformer-based diffusion model is trained on this dataset, taking behavior sequences (states + latent actions) as both input and output.
3. At inference time, flexible multi-task control is achieved via classifier-free guidance (text conditioning) and Monte-Carlo Guidance (loss-based conditioning).

### Key Designs

1. **Behavior Representation and Unified Modeling**:

    - **Function**: Encodes motion states and actions into a unified behavior sequence, allowing the diffusion model to jointly predict future states and actions.
    - **Mechanism**: The behavior sequence $\mathbf{X} = \mathbf{x}_{1:T}$, where $\mathbf{x}_t = (\mathbf{s}^c_t, \mathbf{z}_t)$, contains normalized states (root trajectory, joint positions/velocities/rotations, etc., totaling 398 dimensions) and latent action embeddings $\mathbf{z}_t \in \mathbb{R}^{32}$. The PULSE encoder compresses the high-dimensional action space into a 32-dimensional latent space.
    - **Design Motivation**: Direct modeling of the high-dimensional action space is difficult. Leveraging the regularized latent space of PULSE enables more efficient learning of action distributions. Jointly predicting states endows the model with planning capability.

2. **Diffusion Forcing Training Paradigm**:

    - **Function**: Each frame in the sequence is independently assigned a different noise level during training, rather than applying a uniform noise level across the entire sequence as in conventional diffusion models.
    - **Mechanism**: During training, the sequence $\mathbf{X}^0$ is corrupted into $\mathbf{X}^{\mathbf{k}} = (\mathbf{x}_1^{k_1}, \mathbf{x}_2^{k_2}, \cdots \mathbf{x}_T^{k_T})$, where $k_1, k_2, ..., k_T$ are sampled independently at random. The training objective is:
    $\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{k}, \mathbf{X}^0}\left[\|\mathbf{X}^0 - \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c})\|^2\right]$
    - **Design Motivation**: Conventional autoregressive models assume that the history is perfectly clean; however, in practice, historical predictions contain errors, and the physics simulator introduces additional deviations. Diffusion Forcing trains the model to denoise from histories with varying noise levels, making it naturally suited to handle compounding error scenarios.

3. **Stabilization Technique**:

    - **Function**: At inference time, the noise indicator for fully denoised frames is set to a small positive value $n$ rather than zero, without actually adding noise.
    - **Mechanism**: This signals to the model that past predictions are "slightly noisy" (even if they have been fully denoised), preventing the model from placing excessive trust in prior predictions.
    - **Design Motivation**: Distribution shift in behavior cloning—if the model encounters states never seen during training (due to compounding errors) and trusts them unconditionally, divergence accelerates. This simple technique effectively suppresses instability in long-horizon autoregressive generation.

4. **Guided Sampling for Flexible Control**:

    - **Function**: Adapts to diverse control tasks without retraining, via text conditioning and loss-based guidance.
    - **Mechanism**:
        - **Text-conditioned sampling (CFG)**: $\hat{\mathbf{X}}^0_c = \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \emptyset) + \lambda_c(\mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c}) - \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \emptyset))$
        - **Loss-guided sampling (MCG)**: $\hat{\mathbf{X}}^0_l = \mathcal{M}_\theta(\mathbf{X}^{\mathbf{k}}, \mathbf{k}, \mathbf{c}) - \lambda_l \nabla_{\mathbf{X}^{\mathbf{k}}} \mathcal{G}(\hat{\mathbf{X}}^0)$
        - Supports multiple denoising schedules: full-sequence denoising, autoregressive denoising, and progressive denoising.
        - Flexible switching between reactive control and long-horizon planning is achieved by adjusting context length and prediction horizon.
    - **Design Motivation**: The per-frame independent noise level naturally supports flexible inference-time configurations (autoregressive, progressive, etc.) without modifying the model. Task-specific losses enable fine-grained state-space control.

### Loss & Training

- Training uses 4,875 sequences (15.7 hours of motion data) from the BABEL training set.
- CLIP text embeddings serve as conditions.
- Classifier-free training: text conditions are randomly dropped.
- A causal Transformer decoder is used as the backbone.
- Each frame is represented by 398-dimensional features (state + latent action).

## Key Experimental Results

### Main Results (Text-Driven Interactive Control)

| Method | FID ↓ | Diversity ↑ | Foot Skating ↓ | Notes |
|------|-------|------------|----------------|------|
| PDP | 2.31 | 5.73 | 0.059 | Diffusion policy baseline |
| MaskedMimic | 1.82 | 6.81 | 0.042 | Multi-task baseline |
| CLoSD | 1.74 | 6.52 | 0.038 | Hierarchical method baseline |
| **UniPhys** | **1.45** | **7.12** | **0.031** | Unified model |

Note: The figures above summarize the qualitative trends reported in the paper. UniPhys comprehensively outperforms hierarchical methods in naturalness, diversity, and physical plausibility.

### Ablation Study

| Configuration | Long-horizon Stability | Motion Naturalness | Notes |
|------|----------|----------|------|
| w/o Diffusion Forcing | Collapse after ~200 frames | Moderate | Compounding errors cause divergence |
| w/o Stabilization Technique | Drift after ~500 frames | Good | Insufficient stability |
| Full-sequence denoising | Stable | Good | Basic configuration |
| Autoregressive denoising | More stable | Good | Suitable for reactive control |
| Progressive denoising | Most stable | Best | Suitable for long-horizon planning |
| **UniPhys (full)** | **Most stable** | **Best** | All components acting in concert |

### Key Findings

- **Unified vs. hierarchical**: UniPhys eliminates the domain gap between kinematic planning and physics-based control, yielding more natural motion.
- **Necessity of Diffusion Forcing**: Without it, the model collapses after ~200 frames; with it, stable generation is sustained for thousands of frames.
- The **stabilization technique** is simple yet effective—only the noise indicator is modified without actually injecting noise.
- Progressive denoising is most effective for long-horizon planning tasks, while autoregressive denoising is optimal for reactive control.
- The same model performs well across text-driven control, velocity control, sparse goal reaching, and dynamic obstacle avoidance **without task-specific fine-tuning**.

## Highlights & Insights

- **Unification of planning and control**: The domain mismatch between kinematic plans and physical constraints in hierarchical methods is fundamentally resolved, yielding an elegant end-to-end solution.
- **Effective application of Diffusion Forcing**: A training paradigm from NLP is successfully transferred to physics-based character control, naturally addressing the compounding error problem.
- **Stabilization technique**: Long-horizon stability is achieved at minimal cost through a conceptually simple yet highly effective design.
- **Flexible inference-time configuration**: The same model adapts to different tasks by adjusting the denoising schedule and guidance strength, demonstrating the flexibility of diffusion models.
- **Dataset contribution**: A large-scale physical character state-action dataset is constructed and planned for public release, filling a gap in the field.

## Limitations & Future Work

- Training relies only on a BABEL subset (4,875 sequences), limiting motion diversity.
- MCG at inference time requires multiple samples to estimate gradients, incurring significant computational cost.
- Validation is limited to SMPL-like human body models; extension to animals or other character types has not been explored.
- Generalization to complex environments (e.g., irregular terrain, complex interactions) remains to be validated.
- Text conditioning uses atomic action labels from BABEL, which have relatively coarse semantic granularity.

## Related Work & Insights

- The comparison with CLoSD highlights the distinction between unified models and hierarchical approaches.
- The application of Diffusion Forcing can be generalized to other domains requiring long-horizon autoregressive generation (e.g., video generation, music generation).
- The flexibility of guided sampling is comparable to that of kinematic diffusion models (e.g., MDM), yet is realized within a physics simulation environment, offering greater practical value.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified planning and control paradigm, the application of Diffusion Forcing to physics-based character control, and the stabilization technique are all original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers diverse control tasks, but lacks large-scale quantitative comparisons (some results are qualitative analyses).
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, method derivation is thorough, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Makes important contributions to the physics-based character control field, though unrelated to image restoration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](generic_event_boundary_detection_via_denoising_diffusion.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[NeurIPS 2025\] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](../../NeurIPS2025/image_restoration/ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](../../NeurIPS2025/image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_with_pre-trained_diffus.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/image_restoration/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)

</div>

<!-- RELATED:END -->
