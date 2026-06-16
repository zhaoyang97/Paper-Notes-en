---
title: >-
  [Paper Note] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions
description: >-
  [ICCV 2025][3D Vision][3D human-object interaction] TriDi is proposed as the first unified diffusion model that jointly models the three-variable distribution of humans (H), objects (O)…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D human-object interaction"
  - "joint probability modeling"
  - "trilateral diffusion"
  - "contact maps"
  - "multimodal generation"
date: 2026-05-08
content_hash: 3d1b4e273dac88a2
---

# TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions

**Conference**: ICCV 2025
**arXiv**: [2412.06334](https://arxiv.org/abs/2412.06334)  
**Code**: [https://virtualhumans.mpi-inf.mpg.de/tridi/](https://virtualhumans.mpi-inf.mpg.de/tridi/)  
**Area**: 3D Vision / Human-Object Interaction
**Keywords**: 3D human-object interaction, joint probability modeling, trilateral diffusion, contact maps, multimodal generation

## TL;DR

TriDi is proposed as the first unified diffusion model that jointly models the three-variable distribution of humans (H), objects (O), and interactions (I). A single network covers 7 conditional generation modes, outperforming dedicated unidirectional baselines across all settings.

## Background & Motivation

**Background**: Modeling 3D human-object interactions (HOI) is critical for applications such as AR/VR and virtual human generation. Existing methods operate in a **unidirectional conditional** manner: some recover human pose from objects as $P(H|O)$, while others recover object pose from humans as $P(O|H)$, each requiring a dedicated model architecture, training pipeline, and design.

**Limitations of Prior Work**: Training specialized models for each conditional combination is (1) not scalable, (2) ignores the mutual dependencies among the three modalities, and (3) cannot support unconditional joint generation. Given a human and an object, multiple plausible interaction types exist (sitting, lifting, pushing, etc.), and a comprehensive model should simultaneously capture the relationships among all modalities.

**Key Insight**: This work shifts HOI modeling from the paradigm of "unidirectional conditional distributions" to "three-variable joint distributions." Inspired by UniDiffuser (bimodal diffusion), it extends the framework to three modalities, modeling $P(H, O, I)$ within a compact architecture that naturally yields $2^3 - 1 = 7$ operational modes.

**Core Idea**: (1) A Transformer-based trilateral diffusion process assigns independent timesteps to each modality and discovers fine-grained cross-modal relationships via token-level self-attention. (2) Textual descriptions and body contact maps are embedded into a shared latent space, balancing user controllability with spatial expressiveness.

## Method

### Overall Architecture

TriDi parameterizes the three modalities (H, O, I) as follows:
- **Human H** = (pose $\theta \in \mathbb{R}^{51\times3}$, shape $\beta \in \mathbb{R}^{10}$, global pose $g_H \in \mathbb{R}^9$), based on the SMPL+H model
- **Object O** = (global pose $g_O \in \mathbb{R}^9$), with geometry encoded via PointNeXt features and category one-hot encoding
- **Interaction I** = (latent variable $z_I \in \mathbb{R}^{128}$), a compact encoding that jointly embeds contact maps and textual descriptions

The model receives noisy three-modal tokens along with three independent timesteps $(t^H, t^O, t^I)$ and predicts the denoised original samples.

### Key Designs

1. **Contact-Text Interaction Representation**: Two encoders are trained — a contact map encoder $E_\phi$ and a text encoder $E_T$ — both mapping to a shared 128-dimensional latent space, along with a contact map decoder $D_\phi$. The loss includes: contact map autoencoding BCE, text-to-contact-map encoding BCE, and text–contact-map latent alignment L2. This enables users to guide generation using either text or contact maps.

2. **Trilateral Diffusion Formulation**: UniDiffuser is extended to three modalities, with each modality assigned an independent noise timestep $(t^H, t^O, t^I)$. The training objective is:

    $\min_\psi \mathbb{E}_p \mathbb{E}_t \mathbb{E}_q \| \text{TriDi}_\psi(H^{t^H}, O^{t^O}, I^{t^I}; t^H, t^O, t^I; C_O) - (H^0, O^0, I^0) \|_2$

    Any operational mode can be selected by adjusting the timesteps: $t=0$ denotes conditioning and $t=T$ denotes the generation target.

3. **Reconstruction-Based Guidance Mechanism**: During denoising, a contact distance supervision function $F$ is computed from the predicted contact maps over human–object distances, and gradient-based guidance is applied to correct predictions:

    $F(\hat{H}, \hat{O}, \hat{I}) = \sum_j |\hat{\phi}_{I_j} \hat{d}_j|$

    where $\hat{d}_j$ is the distance from each human vertex to the nearest point on the object. This guidance is applied during the final 200 steps with weight $\lambda=2.0$.

### Loss & Training

The total loss consists of 6 terms: parameter-space losses (human L1, object L1, interaction L2) and vertex-space losses (human vertex L2, object vertex L2, human–object distance L2), with weighting coefficients $(2, 1, 1, 6, 2, 4)$ respectively.

Training details: 15M parameters, batch size 1024, learning rate $1\times10^{-4}$ with cosine decay, AdamW optimizer, approximately 20 hours of training on an RTX 4090. A ZY-plane mirror augmentation is applied to address the right-hand bias in the training data, effectively increasing training diversity.

## Key Experimental Results

### Main Results

Distribution quality and geometric consistency are evaluated on the BEHAVE and GRAB datasets:

| Method | 1-NNA (→50) | COV↑ | MMD↓ | MPJPE↓ | MPJPE-PA↓ |
|--------|-------------|------|------|--------|-----------|
| GNet (BEHAVE, H,I\|O) | 80.01 | 40.71 | 1.789 | 35.6 | 14.6 |
| ObjPOP (BEHAVE, O,I\|H) | 81.36 | 35.02 | 0.329 | — | — |
| **TriDi (H,I\|O)** | **67.89** | **47.81** | **1.352** | **20.8** | **12.3** |
| **TriDi (O,I\|H)** | **63.72** | **51.71** | **0.166** | — | — |

TriDi achieves 1-NNA values closer to the ideal of 50 and improves COV by up to 47%, while also surpassing dedicated baselines in geometric consistency.

### Ablation Study

| Configuration | 1-NNA | COV | MPJPE | Acc_cont |
|---------------|-------|-----|-------|----------|
| w/o augmentation | worse | lower | — | — |
| w/o I modality | — | — | worse | lower |
| w/o guidance | — | — | worse geometry | lower |
| **Full model** | **best** | **best** | **best** | **best** |

Augmentation improves distribution quality (1-NNA); guidance and the interaction modality contribute significantly to geometric consistency; joint three-modal modeling outperforms bimodal variants.

### Key Findings

- Jointly trained TriDi matches or outperforms variants trained specifically for a single mode (s-TriDi-HI, s-TriDi-OI), demonstrating that joint modeling improves generalization.
- In a user study (N=40), TriDi outputs were preferred over baseline outputs approximately 89% of the time, approaching preference rates against ground truth (~52%).
- TriDi generalizes to unseen geometries (e.g., chairs and stools) and supports indirect interaction reconstruction from RGB images.

## Highlights & Insights

- **Unification**: A single 15M-parameter model covers 7 operational modes, encompassing all specialized scenarios addressed by prior work while enabling new use cases such as joint generation of H+O+I.
- **Contact-Text Representation**: Elegantly unifies precise but unintuitive contact maps with user-friendly but coarse textual descriptions within a shared latent space, balancing controllability and spatial precision.
- **Left-Right Symmetry Augmentation**: Simple yet effective; prior HOI methods have uniformly neglected the right-hand bias present in training data.

## Limitations & Future Work

- Due to skewed training data, the model performs well on high-frequency objects but exhibits limited generalization to functionally diverse unseen objects (e.g., wheelchairs, bicycles).
- The current method handles only single-frame static interactions; extension to dynamic sequences is an important future direction.
- The framework addresses single-person, single-object interactions; scaling to multi-person, multi-object scenarios is a natural next step.

## Related Work & Insights

- The approach is conceptually aligned with UniDiffuser [Bao 2023], extending bimodal to trilateral diffusion and demonstrating the scalability of multimodal joint diffusion.
- CG-HOI [Diller 2024] requires strong text conditioning and trains on one dataset at a time; TriDi is more general.
- The method has direct applicability to scene population, virtual reality content creation, and interaction reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ — Trilateral joint diffusion + unified Contact-Text representation
- Technical Depth: ⭐⭐⭐⭐ — Rigorous probabilistic modeling and guidance mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple datasets, ablations, user study, and downstream applications
- Practical Value: ⭐⭐⭐⭐ — AR/VR scene population and interaction reconstruction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-Driven Storytelling with Multiple Lifelike Humans in a 3D Scene](event-driven_storytelling_with_multiple_lifelike_humans_in_a_3d_scene.md)
- [\[ICCV 2025\] ETCH: Generalizing Body Fitting to Clothed Humans via Equivariant Tightness](etch_generalizing_body_fitting_to_clothed_humans_via_equivariant_tightness.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[ICCV 2025\] VolumetricSMPL: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions](volumetricsmpl_a_neural_volumetric_body_model_for_efficient_interactions_contact.md)

</div>

<!-- RELATED:END -->
