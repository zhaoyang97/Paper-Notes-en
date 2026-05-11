---
title: >-
  [Paper Note] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion
description: >-
  [ICCV 2025][Image Generation][human-object interaction reconstruction] ScoreHOI employs a score-based diffusion model as an optimizer, integrating DDIM inversion–forward sampling with physical constraints (contact…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "human-object interaction reconstruction"
  - "diffusion models"
  - "score-guided sampling"
  - "physical constraints"
  - "contact prediction"
date: 2026-05-08
content_hash: fd7d76c44afddae9
---

# ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion

**Conference**: ICCV 2025
**arXiv**: [2509.07920](https://arxiv.org/abs/2509.07920)
**Code**: [https://github.com/RammusLeo/ScoreHOI.git](https://github.com/RammusLeo/ScoreHOI.git)
**Area**: Image Generation
**Keywords**: human-object interaction reconstruction, diffusion models, score-guided sampling, physical constraints, contact prediction

## TL;DR
ScoreHOI employs a score-based diffusion model as an optimizer, integrating DDIM inversion–forward sampling with physical constraints (contact, penetration, ground contact) to guide the denoising process. Combined with a contact-driven iterative refinement strategy, it achieves physically plausible 3D reconstruction of human-object interactions from monocular images, improving contact F-Score by 9% on BEHAVE.

## Background & Motivation

**Background**: Jointly reconstructing 3D meshes of humans and interacting objects from monocular images is an important yet challenging task. Existing methods fall into two categories: (a) optimization-based methods (e.g., CHORE) that iteratively optimize physical constraints using Adam, but over-emphasize physics while neglecting image features, resulting in large reconstruction errors and slow inference; (b) regression-based methods (e.g., CONTHO) that predict outputs in a single forward pass, but suffer from poor robustness in single-step refinement, especially under heavy occlusion or depth ambiguity.

**Limitations of Prior Work**: Optimization-based methods lack prior knowledge of human-object interactions and are prone to local optima; regression-based methods are fast but lack iterative refinement capability, leading to instability in challenging scenarios.

**Key Challenge**: How can one simultaneously leverage data-driven priors and satisfy physical constraints? Traditional optimizers lack distributional priors over data, while regression networks lack controllable iterative optimization mechanisms.

**Goal**: (1) Incorporate prior knowledge of human-object interactions into the optimization process; (2) supervise the generation direction with physical constraints during sampling.

**Key Insight**: Diffusion models characterize the gradient field of the data distribution via score functions and support conditional guided sampling. This enables treating the diffusion model as an "optimizer with rich priors," injecting physical constraints into the denoising process to achieve guided refinement.

**Core Idea**: Replace the traditional optimizer with a score-based diffusion model; inject contact, penetration, and ground constraints as physical guidance during DDIM sampling; and iteratively update contact masks to improve physical plausibility.

## Method

### Overall Architecture
The inputs are a monocular RGB image $I$, human/object segmentation masks $S_h, S_o$, and an object template $P_o$. An **Affordance-Aware Regressor** first extracts image features $\mathcal{F}$ and coarsely estimates SMPL-H parameters $\theta, \beta$ and object pose $R_o, t_o$. These parameters are then passed to the **Contact-Driven Iterative Refinement** module: DDIM inversion maps the initial estimate to a noisy latent variable, and physical constraints are injected as guidance during forward DDIM sampling while contact masks are iteratively updated. A dual-branch Transformer further refines the human and object meshes.

### Key Designs

1. **Affordance-Aware Regressor**:

    - Function: Coarsely estimates human pose and object pose from the input image and object template.
    - Mechanism: A pretrained PointNeXt extracts affordance features of the object (i.e., priors on how the object can be used), which are injected into the image feature extraction process. This enables generalization to unseen object shapes via affordance.
    - Design Motivation: Conventional methods inject object information via category IDs, which cannot handle out-of-vocabulary objects. Affordance provides a category-agnostic universal prior.

2. **Score-Guided Physical Optimization**:

    - Function: Injects physical constraint guidance into the diffusion model's denoising sampling.
    - Mechanism: The optimization target is defined as $\bm{x} = \{\theta, \beta, R_o, t_o\} \in \mathbb{R}^{331}$. DDIM inversion maps the initial estimate $\bm{x}^{\text{init}}$ to the noise space $\bm{x}_\tau$. During DDIM sampling, the conditional score is modified as: $\nabla_{\bm{x}_t}\log p(\bm{x}_t|\bm{c},\mathcal{P}) = \nabla_{\bm{x}_t}\log p(\bm{x}_t|\bm{c}) + \nabla_{\bm{x}_t}\log p(\mathcal{P}|\bm{c},\hat{\bm{x}_0}(\bm{x}_t))$, where the second term approximates the gradient of the physical constraint loss via the denoised estimate $\hat{\bm{x}_0}$.
    - Modified noise prediction: $\epsilon'_\phi = \epsilon_\phi(\bm{x}_t, t, \bm{c}) + \rho\sqrt{1-\alpha_t}\nabla_{\bm{x}_t}L_\mathcal{P}$
    - Design Motivation: Directly computing physical constraint gradients in noise space is intractable; the Tweedie formula enables approximation via $\hat{\bm{x}_0}$, making constraints computable.

3. **Physical Constraint Loss Functions**:

    - Total loss: $L_\mathcal{P} = \lambda_{ho}L_{ho} + \lambda_{of}L_{of} + \lambda_{pt}L_{pt}$
    - **Human-object contact** $L_{ho}$: Euclidean distance between human and object vertices in the contact region should be zero.
    - **Object-floor contact** $L_{of}$: Height of contact vertices at the bottom of the object should be zero.
    - **Penetration avoidance** $L_{pt}$: Penalizes human vertices that penetrate the object using the object's SDF.

4. **Contact-Driven Iterative Refinement (CDIR)**:

    - Function: Iteratively updates contact masks to improve contact prediction accuracy.
    - Mechanism: At each iteration $n$: (1) sample human/object features $\mathcal{F}_h, \mathcal{F}_o$ from image features based on current parameters $\bm{x}_0^n$; (2) update contact masks $\mathbf{M}_h, \mathbf{M}_o, \mathbf{M}_f$; (3) perform DDIM inversion + guided sampling to obtain $\bm{x}_0^{n+1}$. The process runs for $N=10$ iterations.
    - Design Motivation: Single-pass contact mask prediction is error-prone, especially under heavy occlusion. Iterative updates allow contact prediction and pose optimization to mutually reinforce each other.

5. **IG-Adapter (Image-Geometry Adapter)**:

    - Function: Injects image observations and object geometry priors into the diffusion model.
    - Mechanism: Additional cross-attention blocks and linear fusion heads are introduced to fuse the image condition $\bm{c}_I$ (average-pooled from $\mathcal{F}$) and the geometry condition $\bm{c}_G$ (from pretrained PointNeXt).
    - Training objective: $L_{DM} = \mathbb{E}_{\bm{x}_0,\epsilon,t,\bm{c}_I,\bm{c}_G}\|\epsilon - \epsilon_\theta(\bm{x}_t, t, \bm{c}_I, \bm{c}_G)\|^2$

### Loss & Training
Training proceeds in two stages: (1) the image backbone, contact predictor, and vertex optimization module are trained for 50 epochs at LR $10^{-4}$; (2) the image backbone is frozen, and the diffusion model is trained for 30 epochs with the IMHD dataset augmenting generative capacity. Training uses 4 RTX 4090 GPUs for approximately 1.5 days.

## Key Experimental Results

### Main Results

| Dataset | Method | CD_human↓ | CD_object↓ | Contact_prec↑ | Contact_recall↑ | Contact_F-S↑ |
|--------|------|-----------|-----------|--------------|----------------|-------------|
| BEHAVE | PHOSA | 12.17 | 26.62 | 0.393 | 0.266 | 0.317 |
| BEHAVE | CHORE | 5.58 | 10.66 | 0.587 | 0.472 | 0.523 |
| BEHAVE | CONTHO | 4.99 | 8.42 | 0.628 | 0.496 | 0.554 |
| BEHAVE | **ScoreHOI** | **4.85** | **7.86** | **0.634** | **0.586** | **0.609** |
| InterCap | PHOSA | 11.20 | 20.57 | 0.228 | 0.159 | 0.187 |
| InterCap | CHORE | 7.01 | 12.81 | 0.339 | 0.253 | 0.290 |
| InterCap | CONTHO | 5.96 | 9.50 | 0.661 | 0.432 | 0.522 |
| InterCap | **ScoreHOI** | **5.56** | **8.75** | 0.627 | **0.590** | **0.578** |

On BEHAVE, ScoreHOI achieves a contact F-Score of 0.609, representing approximately **9%** improvement over CONTHO.

### Ablation Study

| Configuration | CD_human↓ | CD_object↓ | Contact_F-S↑ | Note |
|------|-----------|-----------|-------------|------|
| w/o diffusion | 5.03 | 8.48 | 0.588 | Remove diffusion module |
| w/o CDIR | 4.93 | 7.98 | 0.577 | Remove iterative refinement |
| No condition | 4.94 | 8.23 | 0.585 | No conditional guidance |
| w/o $\bm{c}_G$ | 4.87 | 7.99 | 0.591 | Remove geometry condition |
| w/o $\bm{c}_I$ | 4.88 | 8.03 | 0.597 | Remove image condition |
| No guidance | 4.93 | 8.01 | 0.570 | No physical guidance |
| w/o $L_{ho}$ | 4.87 | 7.95 | 0.574 | Remove human-object contact |
| w/o $L_{pt}$ | 4.87 | 7.93 | 0.592 | Remove penetration constraint |
| w/o $L_{of}$ | 4.89 | 7.95 | 0.602 | Remove ground contact |
| **Full model** | **4.85** | **7.86** | **0.609** | Complete model |

### Efficiency Comparison

| Method | CD_human↓ | CD_object↓ | FPS↑ |
|------|-----------|-----------|------|
| CHORE | 5.58 | 10.66 | 0.0035 |
| VisTracker | 5.24 | 7.89 | 0.0359 |
| ScoreHOI-Faster(N=2) | 4.87 | 7.95 | 2.008 |
| ScoreHOI | 4.85 | 7.86 | 0.290 |

### Key Findings
- **Diffusion model contributes significantly**: Removing the diffusion module reduces Contact_F-S from 0.609 to 0.588, demonstrating that the diffusion prior provides valuable distributional knowledge.
- **CDIR is a critical component**: Removing iterative refinement drops F-Score from 0.609 to 0.577, confirming that iterative contact mask updates are essential for contact quality.
- **Human-object contact constraint is most critical**: Removing $L_{ho}$ causes a substantial drop in recall, indicating that explicit contact guidance is necessary.
- **Clear efficiency advantage**: ScoreHOI is approximately 80× faster than CHORE while achieving superior performance. The fast variant (N=2) enables near-real-time inference at 2 FPS with only marginal performance loss.

## Highlights & Insights
- **Diffusion model as optimizer** represents an elegant perspective shift: rather than using the diffusion model for generation, the method exploits the learned data distribution prior to guide the optimization process. This paradigm is transferable to any reconstruction task requiring "optimization with priors."
- **DDIM inversion + guided sampling** allows starting from an arbitrary initial estimate, injecting task-specific constraints while preserving prior knowledge—better suited for refinement tasks than sampling from pure noise.
- **Contact-driven iterative refinement** embodies a predict-update loop design philosophy, forming a mutually reinforcing closed loop between contact mask prediction and pose parameter optimization.

## Limitations & Future Work
- Requires a predefined object canonical pose template; cannot handle previously unseen objects.
- Training data is limited (BEHAVE covers only 20 object categories), constraining generalization.
- Inference overhead from 10 iterations × DDIM sampling remains substantial (0.29 FPS).

## Related Work & Insights
- **vs. CONTHO**: CONTHO performs single-step refinement via regression-based cross-attention; ScoreHOI performs multi-step iterative refinement via diffusion models, achieving significantly superior contact quality.
- **vs. CHORE**: CHORE uses an Adam optimizer with physical constraints; ScoreHOI replaces Adam with a diffusion prior, achieving 80× speedup with higher accuracy.
- **vs. ScoreMDM/ScoreHMR**: These works apply diffusion models to HMR; ScoreHOI is the first to extend this paradigm to human-object interaction scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The use of a diffusion model as a prior-equipped optimizer is a novel framing, though score-guided sampling itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations (modules/conditions/guidance/hyperparameters) with multi-dataset validation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed mathematical derivations.
- Value: ⭐⭐⭐⭐ Offers concrete advances for HOI reconstruction; methodology is transferable to other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)

</div>

<!-- RELATED:END -->
