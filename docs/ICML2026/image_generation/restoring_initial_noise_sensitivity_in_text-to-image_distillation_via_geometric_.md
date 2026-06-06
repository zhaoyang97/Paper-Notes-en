---
title: >-
  [Paper Note] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment
description: >-
  [ICML 2026][Image Generation][Diffusion Distillation] This paper identifies that existing T2I diffusion distillation methods produce "pointwise output alignment…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diffusion Distillation"
  - "Initial Noise Sensitivity"
  - "Jacobian-Vector Product"
  - "Geometric Alignment"
  - "T2I"
date: 2026-05-08
content_hash: 929cea930b16f400
---

# Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment

**Conference**: ICML 2026  
**arXiv**: [2606.01651](https://arxiv.org/abs/2606.01651)  
**Code**: https://github.com/Hannah1102/GAD (Available)  
**Area**: Diffusion Models / Text-to-Image Distillation  
**Keywords**: Diffusion Distillation, Initial Noise Sensitivity, Jacobian-Vector Product, Geometric Alignment, T2I

## TL;DR
This paper identifies that existing T2I diffusion distillation methods produce "pointwise output alignment," causing a collapse in the student model's sensitivity to initial noise. It proposes GAD: using a finite difference approximation of the Jacobian-Vector Product (JVP) under paired perturbed inputs to force the student to match the teacher's directional response to noise perturbations, thereby restoring layout controllability and generation diversity without compromising fidelity.

## Background & Motivation

**Background**: Diffusion Models (DM) and Flow Matching have become mainstream for T2I, but require 20-100 NFEs. Consequently, distillation (output matching / distribution matching / score distillation) is widely used to compress multi-step trajectories into 1-4 step student models.

**Limitations of Prior Work**: Existing distillation methods focus primarily on "average output quality" (e.g., FID/CLIP) and treat the teacher as a static input-output mapping. Consequently, when changing the seed $\mathbf{z}$, the student's output remains nearly identical—losing "initial noise sensitivity." This undermines downstream tasks: training-free layout control (attention guidance injecting spatial constraints via $\mathbf{z}$), attribute control via optimal noise retrieval (e.g., NoiseQuery), and simple generation diversity from seed variations, all of which rely on the "teacher's differentiated response to $\mathbf{z}$."

**Key Challenge**: The standard distillation objective $\mathcal{L}_{\text{base}}=\mathbb{E}_{\mathbf{z}}[\mathcal{D}(\Phi_S(\mathbf{z}),\Phi_T(\mathbf{z}))]$ focuses on pointwise alignment—matching outputs for each $\mathbf{z}$ independently. Under multi-modal targets, MSE or reverse KL causes the student to converge to the conditional expectation (a smooth "average path"), erasing the teacher's local geometry (directional gradients, curvature) in the neighborhood of $\mathbf{z}$. Diagnostic experiments provide direct evidence: while pointwise MSE between student and teacher is low, the JVP MSE remains high (Teacher 0.000 vs TDM 0.0003, Tab. 1), and JVP cosine similarity is only 0.012—the "shape" is correct, but the "tangent vectors" are completely wrong.

**Goal**: To align the student's local "differential response" with the teacher's without introducing new architectures, relying on extra data, or compromising the base loss, thereby restoring noise sensitivity and downstream controllability.

**Key Insight**: Drawing from "relational knowledge" in classical Knowledge Distillation (KD) (Park et al. 2019, Tung & Mori 2019)—learning relative relationships between samples rather than just absolute outputs. In generative scenarios, this "relative relationship" is the directional response characterized by the Jacobian $\mathbf{J}_{\Phi_T}(\mathbf{z})$ of the teacher mapping $\Phi_T$.

**Core Idea**: Use the output difference of a pair $(\mathbf{z}, \mathbf{z}+h\mathbf{v})$ as a finite difference approximation of the JVP. Forcing the student response to equal the (stop-grad) teacher response serves as a pluggable regularization term added to any base distillation loss.

## Method

### Overall Architecture
GAD is a model-agnostic "additional loss term" that can be seamlessly integrated into three major distillation paradigms: output matching (LADD/ADD), distribution matching (DMD/TDM), and score identity distillation (SiD):

- **Input**: Gaussian noise $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ (initial noise for one-step distillation, intermediate latent $\mathbf{x}_t$ for few-step), random directional vector $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, perturbation magnitude $h$, and text condition $c$.
- **Teacher/Student Forward**: The teacher $\Phi_T$ (endpoint of multi-step sampling) and student $\Phi_S(\cdot;\theta)$ are each run twice at points $\mathbf{z}$ and $\mathbf{z}'=\mathbf{z}+h\mathbf{v}$.
- **Response Vector Construction**: $\Delta\Phi_S=\Phi_S(\mathbf{z}')-\Phi_S(\mathbf{z})$ and $\Delta\Phi_T=\Phi_T(\mathbf{z}')-\Phi_T(\mathbf{z})$ (with stop-gradient on the teacher side).
- **Total Loss**: $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{base}}+\lambda\mathcal{L}_{\text{GAD}}$, where $\mathcal{L}_{\text{GAD}}$ enforces $\Delta\Phi_S\approx\Delta\Phi_T$.
- **Output**: A student model with restored noise sensitivity, capable of inference in 1-4 steps.

### Key Designs

1. **JVP Alignment Objective**:
    - **Function**: Transforms the computationally intractable goal of "matching the teacher's Jacobian $\mathbf{J}_{\Phi_T}$" into "matching the teacher's directional derivative along a random direction $\mathbf{v}$."
    - **Mechanism**: The ideal loss is $\mathcal{L}_{\text{Jacobian}}=\mathbb{E}_{\mathbf{z}}[\|\mathbf{J}_{\Phi_S}(\mathbf{z})-\mathbf{J}_{\Phi_T}(\mathbf{z})\|_F^2]$, but Jacobian calculation causes memory explosion in $d\approx 10^5$ dimensions. Based on the Hutchinson trace estimator, the authors prove that matching JVP for random $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ is equivalent in expectation to matching the Frobenius norm: $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|\nabla_{\mathbf{z}}\Phi_S(\mathbf{z})\mathbf{v}-\nabla_{\mathbf{z}}\Phi_T(\mathbf{z})\mathbf{v}\|_2^2$.
    - **Design Motivation**: JVP acts as a "compressed sensing" of the Jacobian—requiring only one forward directional derivative with $O(d)$ memory instead of $O(d^2)$, while implicitly covering the entire Jacobian geometry in expectation.

2. **Finite Difference Approximation + Paired Forwarding**:
    - **Function**: Avoids dependence on forward-mode autodiff (which is memory-intensive or incompatible with black-box teachers like SDXL/PixArt) by replacing JVP with "two-point output difference."
    - **Mechanism**: $\nabla_{\mathbf{z}}\Phi(\mathbf{z})\cdot\mathbf{v}\approx[\Phi(\mathbf{z}+h\mathbf{v})-\Phi(\mathbf{z})]/h$. Substituting this back and absorbing $1/h^2$ into the weight $\lambda$ yields the practical objective: $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|(\Phi_S(\mathbf{z}')-\Phi_S(\mathbf{z}))-\text{sg}(\Phi_T(\mathbf{z}')-\Phi_T(\mathbf{z}))\|_2^2$. The stop-gradient on the teacher side locks the "reference tangent vector" for the student to align with.
    - **Design Motivation**: Each step requires only one additional forward pass for the student and teacher (4 total); it involves no reverse JVP or second-order graphs, making it "easy to implement in a few lines." It is backbone-agnostic (working for UNet, DiT, Flow-DiT).

3. **Unified Instantiation under Three Paradigms**:
    - **Function**: Integrates GAD as a plug-and-play regularizer into existing distillation frameworks.
    - **Mechanism**: 
        - (a) **Output matching** (LADD/ADD): $\Phi$ is the student's predicted $\hat{\mathbf{x}}_0=f_\theta(\mathbf{x}_t,t,c)$. Paired perturbation $\mathbf{x}_t'=\mathbf{x}_t+h\mathbf{v}$ yields $\mathcal{L}_{\text{GAD}}^{\text{out}}$.
        - (b) **Distribution / Score-based** (DMD/TDM/SiD): $\mathcal{L}_{\text{base}}$ propagates gradients via the difference between two score estimators $\epsilon_{\text{real}}$ (teacher) and $\epsilon_{\text{fake}}$ (auxiliary score net for student distribution). GAD matches the difference in these score fields under directional perturbation $\Delta\epsilon(\mathbf{x}_t,\mathbf{v})=\epsilon(\mathbf{x}_t+h\mathbf{v},t,c)-\epsilon(\mathbf{x}_t,t,c)$, yielding $\nabla_\theta\mathcal{L}_{\text{GAD}}^{\text{score}}=\mathbb{E}[\Delta\epsilon_{\text{fake}}-\Delta\epsilon_{\text{real}}]\partial\mathbf{x}_t/\partial\theta$.
    - **Design Motivation**: $\mathcal{L}_{\text{base}}$ handles first-order moment alignment (ensuring the student converges to high-density regions), while $\mathcal{L}_{\text{GAD}}$ handles local curvature/divergence alignment. They are orthogonal, explaining why GAD does not conflict with the original loss.

### Loss & Training
The total objective is $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{base}}+\lambda\mathcal{L}_{\text{GAD}}$. Each iteration requires sampling paired $(\mathbf{z},\mathbf{z}+h\mathbf{v})$ and performing teacher and student forward passes for each. $h$ and $\lambda$ details are in Appendix D; a key trick is using `torch.no_grad` and stop-grad for teacher forward passes. The training follows the base framework's timestep schedule, CFG, and optimizer with nearly zero migration cost.

## Key Experimental Results

### Main Results
GAD was integrated across 3 backbones (SD2 UNet / PixArt-α DiT / SANA Flow-DiT) and 3 distillation paradigms (LADD / TDM / SiD), evaluating 11 distilled baselines.

**Seed Identifiability (Tab. 2, SD2 Architecture)**: A classifier is trained to predict which seed an image originated from; higher values indicate stronger sensitivity.

| Model | Self-Identifiability ↑ | Teacher Alignment ↑ |
|------|------------------------|---------------------|
| SD2 Teacher (Multi-step) | 93.70% | - |
| SD-Turbo | 77.80% | 63.20% |
| SwiftBrush | 52.90% | 45.80% |
| TCD | 87.30% | 84.50% |
| LADD | 87.60% | 83.70% |
| LADD + GAD (Ours) | **92.40%** | **87.40%** |

**General Generation Quality (Tab. 3)**: GAD does not degrade CLIP/PickScore; instead, it often provides slight improvements. Notably, SiD+GAD improved the CLIP score on SANA from 32.75 to 34.40.

**Layout Control (Tab. 5, COCO 800 prompts + bbox)**:

| Model | AP ↑ | AP50 ↑ | CLIP ↑ |
|------|------|--------|--------|
| SD2 Teacher | 6.6 | 21.3 | 0.3333 |
| SD-Turbo | 3.0 | 8.4 | 0.3237 |
| LADD | 5.0 | 17.4 | 0.3187 |
| LADD + GAD | **5.8** | **20.6** | 0.3184 |

GAD restores 87% of the teacher's layout accuracy.

### Ablation Study
**Direct Geometric Alignment Metrics (Tab. 1, PixArt-α)**:

| Configuration | JVP Cos ↑ | Jac Norm ↑ | Spec KL ↓ | JVP MSE ↓ |
|------|-----------|-----------|-----------|-----------|
| Teacher | 1.000 | 1.000 | 0.000 | 0.000 |
| TDM | 0.012 | 0.98 | 0.008 | 0.0003 |
| TDM + GAD | 0.014 | 0.99 | 0.006 | 0.0002 |

**Trajectory Accumulation Error (Tab. 4, PixArt-α, 200 unseen prompts)**: GAD consistently achieves lower accumulation errors across four time intervals. The endpoint error at $t=0$ decreased from 0.491 to 0.427 ($−13\%$), indicating GAD keeps the student closer to the teacher's denoising trajectory on unseen prompts.

### Key Findings
- Pointwise MSE is already minimized by existing methods (Fig. 2a), leaving "geometric alignment" as the primary area for improvement. GAD significantly aligns JVP behavior with the teacher (Fig. 2b) without affecting pointwise MSE, proving that "low MSE ≠ correct dynamics."
- Restoring noise sensitivity simultaneously improves general quality: the authors attribute this to GAD forcing the student to be consistent in the local neighborhood of $\mathbf{z}$, acting as a smoothness regularizer that improves generalization on unseen prompts.
- Downstream zero-shot transfer (Tab. 6, NoiseQuery feeding the teacher's optimal $\mathbf{z}^*$ to the student) shows that baseline distilled models cannot use teacher-selected noise, whereas GAD enables students to benefit from teacher-side test-time enhancements.
- Diversity / Fidelity trade-off (Fig. 5): Baselines are restricted in the Vendi vs. CLIP space, while GAD pushes all three backbones toward the teacher's upper-right corner.

## Highlights & Insights
- Framing the "distilled noise sensitivity collapse" is valuable—it explicitly addresses an implicit degradation often hidden by FID/CLIP and introduces metrics like seed classification, JVP cos, and Spec KL to make the problem measurable.
- Approximating JVP via finite difference and paired forwarding reduces the computationally expensive "Jacobian matching" to virtually zero engineering cost (4 forward passes, no second-order derivatives), making it truly plug-and-play.
- Using stop-gradient to anchor the teacher's response and $\lambda$ to decouple "global fidelity" from "local curvature" addresses a neglected question: in knowledge distillation, "what to teach" is more important than "how accurately to teach."
- The approach is transferable: any scenario where a student is distilled via MSE/KL from a trajectory or distribution (e.g., speech synthesis, policy distillation, video distillation) could benefit from a JVP-style paired response alignment term.

## Limitations & Future Work
- Training cost: Each step requires two additional teacher and student forward passes, increasing training time by approximately 1.5x–2x. Wall-clock comparisons were not provided.
- Perturbation $h$ is a critical hyperparameter: too small and it is drowned by numerical noise; too large and the first-order approximation fails. Systematic sensitivity analysis for $h$ is missing.
- The Hutchinson estimator requires random directions $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. For latents of size 64×64×4, the variance of a single-direction unbiased estimate might be significant; variance reduction via multiple $\mathbf{v}$ samples was not discussed.
- Experimental scope: Layout control was only tested with YOLOv4 AP; verification with stronger detectors (ControlNet / Grounded-SAM) is omitted.
- Impact on "sister problems" like negative prompt adherence or trajectory invertibility—which are closely related to noise sensitivity—was not evaluated.

## Related Work & Insights
- **vs. Standard Distillation (ADD / LADD / DMD / TDM / SiD)**: These use pointwise alignment (MSE / reverse KL / Fisher divergence). GAD acts as an orthogonal regularizer rather than a replacement, demonstrating complementarity.
- **vs. Relational Knowledge Distillation (Park 2019 RKD / Tung 2019 SP)**: Classical KD emphasizes matching relationships between samples using pairwise distances/angles; GAD extends this to continuous input space using JVP to characterize relationships in infinitesimal neighborhoods.
- **vs. Diversity Enhancement (Diverse Distillation / Gandikota & Bau 2025)**: Those methods often regularize output entropy or explicit diffusion seeds. GAD restores diversity as a byproduct of aligning local response to noise, which is more elegant and unlocks layout/retrieval tasks.
- **vs. NoiseQuery (Wang et al. 2025)**: NoiseQuery assumes the $\mathbf{z}\to\mathbf{x}$ geometry of the teacher is preserved in the student, which often fails. GAD resolves this assumption mismatch.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematize the "noise sensitivity collapse" and provide a clean geometric alignment framework. JVP + finite difference is a novel application in T2I distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 3 backbones, 3 paradigms, 11 baselines, and 3 downstream tasks, plus direct geometric measurements.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative from motivation to diagnostic to formulation. Conceptual diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play with almost zero migration cost, no fidelity loss, and restored controllability. This is a significant contribution to the T2I distillation pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](implicit_preference_alignment_for_human_image_animation.md)

</div>

<!-- RELATED:END -->
