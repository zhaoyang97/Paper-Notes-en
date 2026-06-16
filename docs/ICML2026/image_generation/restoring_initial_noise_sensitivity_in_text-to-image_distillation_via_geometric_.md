---
title: >-
  [Paper Note] Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment
description: >-
  [ICML 2026][Image Generation][Jacobian-Vector Product] This paper points out that existing T2I diffusion distillation methods, which focus only on "pointwise output alignment," lead to the collapse of the student model's sensitivity to initial noise. It proposes GAD: using a finite-difference approximation of the Jacobian-Vector Product (JVP) under a pair of perturbed inpu
tags:
  - ICML 2026
  - Image Generation
  - Jacobian-Vector Product
  - T2I
date: 2026-05-08
content_hash: 531a0e044c4b81c0
---
# Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment

**Conference**: ICML 2026  
**arXiv**: [2606.01651](https://arxiv.org/abs/2606.01651)  
**Code**: https://github.com/Hannah1102/GAD (Available)  
**Area**: Diffusion Models / T2I Distillation  
**Keywords**: Diffusion Distillation, Initial Noise Sensitivity, Jacobian-Vector Product, Geometric Alignment, T2I

## TL;DR
This paper points out that existing T2I diffusion distillation methods, which focus only on "pointwise output alignment," lead to the collapse of the student model's sensitivity to initial noise. It proposes GAD: using a finite-difference approximation of the Jacobian-Vector Product (JVP) under a pair of perturbed inputs to force the student to match the teacher's directional response to noise perturbations, thereby restoring layout controllability and generation diversity without sacrificing fidelity.

## Background & Motivation

**Background**: DM and Flow Matching have become mainstream for T2I but require 20-100 NFEs. Therefore, distillation (output matching / distribution matching / score distillation) is widely used to compress multi-step trajectories into 1-4 step student models.

**Limitations of Prior Work**: Existing distillation methods only focus on "average output quality" such as FID/CLIP and treat the teacher as a static input-output mapping. Consequently, when changing the seed $\mathbf{z}$, the student's output remains nearly the same—losing "sensitivity to initial noise." This directly undermines a class of downstream tasks: training-free layout control (attention guidance injecting spatial constraints via $\mathbf{z}$), methods like NoiseQuery that rely on optimal noise retrieval for attribute control, and simple generation diversity achieved by changing seeds, all of which depend on the "teacher's differentiated response to $\mathbf{z}$."

**Key Challenge**: The standard distillation objective $\mathcal{L}_{\text{base}}=\mathbb{E}_{\mathbf{z}}[\mathcal{D}(\Phi_S(\mathbf{z}),\Phi_T(\mathbf{z}))]$ is a pointwise alignment—matching outputs independently for each $\mathbf{z}$. Under multimodal targets, MSE or Reverse KL causes the student to converge to the conditional expectation (a smooth "average path"), erasing the teacher's local geometry (directional gradients, curvature) in the neighborhood of $\mathbf{z}$. Diagnostic experiments provide direct evidence: the pointwise MSE between student and teacher is already very low, but the JVP MSE remains high (Teacher 0.000 vs TDM 0.0003, Tab. 1), with a JVP cosine similarity of only 0.012—the shape is correct, but the "tangent vectors" are entirely wrong.

**Goal**: To make the student model's "differential response" in the local neighborhood of $\mathbf{z}$ consistent with the teacher's, thereby restoring noise sensitivity and downstream controllability without introducing new architectures, relying on extra data, or undermining the base loss.

**Key Insight**: Borrowing ideas from "relational knowledge" in classic KD (Park et al. 2019, Tung & Mori 2019)—do not just learn absolute outputs, but learn the relative relationships between samples. In generation scenarios, this "relative relationship" is the directional response characterized by the Jacobian $\mathbf{J}_{\Phi_T}(\mathbf{z})$ of the teacher mapping $\Phi_T$.

**Core Idea**: Use the output difference of a pair $(\mathbf{z}, \mathbf{z}+h\mathbf{v})$ as a finite-difference approximation of the JVP, forcing the student response to equal the (stop-grad) teacher response. This "replication of the teacher's reaction to perturbations" is added as a plug-and-play regularization term to any base distillation loss.

## Method

### Overall Architecture
GAD addresses the implicit degradation where the student is no longer sensitive to the initial noise $\mathbf{z}$ after distillation. Its approach is to keep the original distillation objective untouched while adding an orthogonal regularization term: aligning the student's "differential response" to perturbations in the local neighborhood of $\mathbf{z}$ with the teacher's, thereby recovering the local geometry (directional derivatives, curvature) that was averaged away. The entire module is model-agnostic and can be directly attached to the three major distillation paradigms: output matching (LADD/ADD), distribution matching (DMD/TDM), and score identity distillation (SiD). The final student requires only 1-4 steps for inference. In addition to the original forward pass, each iteration runs both the teacher and student on noise $\mathbf{z}$ and a perturbed point $\mathbf{z}'=\mathbf{z}+h\mathbf{v}$ (where $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ is a random direction and $h$ is the perturbation magnitude), then constrains the output differences on both sides to be consistent.

### Key Designs

**1. JVP Alignment Objective: Compressing Uncomputable "Jacobian Matching" into Directional Derivatives**

To restore noise sensitivity, the most direct goal is to match the teacher's Jacobian, i.e., $\mathcal{L}_{\text{Jacobian}}=\mathbb{E}_{\mathbf{z}}[\|\mathbf{J}_{\Phi_S}(\mathbf{z})-\mathbf{J}_{\Phi_T}(\mathbf{z})\|_F^2]$. However, explicitly storing the Jacobian in a latent space of dimension $d\approx 10^5$ would lead to memory explosion. The authors bypass this using the Hutchinson trace estimator: matching the Jacobian-Vector Product (JVP) for a random direction $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ is equivalent to matching the entire Frobenius norm in expectation. Thus, the objective is rewritten as $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|\nabla_{\mathbf{z}}\Phi_S(\mathbf{z})\mathbf{v}-\nabla_{\mathbf{z}}\Phi_T(\mathbf{z})\mathbf{v}\|_2^2$. This acts as "compressed sensing" for the Jacobian—calculating only one directional derivative with $O(d)$ rather than $O(d^2)$ memory while implicitly covering the entire Jacobian geometry.

**2. Finite-Difference Approximation + Paired Forward: Replacing JVP with Two-Point Differences to Bypass Forward-Mode Autodiff**

Even with the JVP objective, directly calculating directional derivatives requires forward-mode autodiff, which is incompatible with black-box teachers like SDXL/PixArt and incurs high memory costs. The authors use first-order finite difference $\nabla_{\mathbf{z}}\Phi(\mathbf{z})\cdot\mathbf{v}\approx[\Phi(\mathbf{z}+h\mathbf{v})-\Phi(\mathbf{z})]/h$ to replace the directional derivative with a "two-point output difference." Absorbing the constant $1/h^2$ into the weight $\lambda$, the practical objective becomes $\mathcal{L}_{\text{GAD}}=\mathbb{E}_{\mathbf{z},\mathbf{v}}\|(\Phi_S(\mathbf{z}')-\Phi_S(\mathbf{z}))-\text{sg}(\Phi_T(\mathbf{z}')-\Phi_T(\mathbf{z}))\|_2^2$. The teacher's reaction to perturbation is locked as a "reference tangent vector" via stop-gradient, and the student is forced to align with it. The cost is only one extra forward pass for both student and teacher per step (4 total), with no backward JVP or second-order computation graphs. It is "runnable with a few lines of code" and universal across UNet, DiT, and Flow-DiT.

**3. Unified Instantiation Across Three Paradigms: Switching what $\Phi$ Represents**

When GAD is integrated as a regularization term into different distillation frameworks, only the aligned mapping $\Phi$ needs to be replaced. For output matching (LADD/ADD), $\Phi$ is the student's predicted $\hat{\mathbf{x}}_0=f_\theta(\mathbf{x}_t,t,c)$, and the paired perturbation is $\mathbf{x}_t'=\mathbf{x}_t+h\mathbf{v}$, resulting in $\mathcal{L}_{\text{GAD}}^{\text{out}}$ in the output space. For distribution/score-based paradigms (DMD/TDM/SiD), the $\mathcal{L}_{\text{base}}$ already backpropagates gradients based on the difference between the teacher score $\epsilon_{\text{real}}$ and the student's auxiliary distribution score $\epsilon_{\text{fake}}$. GAD then matches the differential of the two score fields relative to directional perturbations $\Delta\epsilon(\mathbf{x}_t, \mathbf{v})=\epsilon(\mathbf{x}_t+h\mathbf{v},t,c)-\epsilon(\mathbf{x}_t,t,c)$ at a higher order, with the gradient form $\nabla_\theta\mathcal{L}_{\text{GAD}}^{\text{score}}=\mathbb{E}[\Delta\epsilon_{\text{fake}}-\Delta\epsilon_{\text{real}}]\partial\mathbf{x}_t/\partial\theta$. The reason this does not conflict with the original loss is that $\mathcal{L}_{\text{base}}$ manages first-order moment alignment (student converging to high-density regions), while $\mathcal{L}_{\text{GAD}}$ manages local curvature/divergence alignment; the two objectives are naturally orthogonal.

### Loss & Training
The total objective is $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{base}}+\lambda\mathcal{L}_{\text{GAD}}$. Each iteration samples a pair $(\mathbf{z}, \mathbf{z}+h\mathbf{v})$, performs forward passes for both teacher and student, and the teacher forward uses `torch.no_grad` + stop-grad as an anchor. Values for $h$ and $\lambda$ are detailed in Appendix D. The training follows the base framework's timestep schedule, CFG, and optimizer, with nearly zero migration cost.

## Key Experimental Results

### Main Results
GAD was integrated into 3 types of backbones (SD2 UNet / PixArt-α DiT / SANA Flow-DiT) × 3 types of distillation paradigms (LADD / TDM / SiD), evaluating 11 distilled baselines.

**Seed Identifiability (Tab. 2, SD2 Architecture)**: Training a classifier to predict which seed an image came from; higher scores represent stronger sensitivity.

| Model | Self-Identifiability ↑ | Teacher Alignment ↑ |
|------|------------------------|---------------------|
| SD2 Teacher (Multi-step) | 93.70% | - |
| SD-Turbo | 77.80% | 63.20% |
| SwiftBrush | 52.90% | 45.80% |
| TCD | 87.30% | 84.50% |
| LADD | 87.60% | 83.70% |
| LADD + GAD (Ours) | **92.40%** | **87.40%** |

**General Generation Quality (Tab. 3)**: GAD not only maintains CLIP/PickScore but generally improves them slightly. Notably, SiD+GAD increased the CLIP score on SANA from 32.75 to 34.40.

**Layout Control (Tab. 5, COCO 800 prompts + bbox)**:

| Model | AP ↑ | AP50 ↑ | CLIP ↑ |
|------|------|--------|--------|
| SD2 Teacher | 6.6 | 21.3 | 0.3333 |
| SD-Turbo | 3.0 | 8.4 | 0.3237 |
| LADD | 5.0 | 17.4 | 0.3187 |
| LADD + GAD | **5.8** | **20.6** | 0.3184 |

GAD restores 87% of the teacher's layout accuracy.

### Ablation Study
**Direct Measurement of Geometric Alignment (Tab. 1, PixArt-α)**:

| Config | JVP Cos ↑ | Jac Norm ↑ | Spec KL ↓ | JVP MSE ↓ |
|------|-----------|-----------|-----------|-----------|
| Teacher | 1.000 | 1.000 | 0.000 | 0.000 |
| TDM | 0.012 | 0.98 | 0.008 | 0.0003 |
| TDM + GAD | 0.014 | 0.99 | 0.006 | 0.0002 |

**Trajectory Cumulative Deviation (Tab. 4, PixArt-α, 200 unseen prompts)**: GAD achieves lower cumulative deviation across four time intervals, with the terminal error at t=0 dropping from 0.491 to 0.427 (−13%), indicating that GAD makes the student more closely follow the teacher's denoising trajectory on unseen prompts.

### Key Findings
- Pointwise MSE is already compressed very low by existing methods (Fig 2a); the room for improvement lies in "geometric alignment." GAD barely changes pointwise metrics but pulls JVP behavior close to the teacher (Fig 2b), proving "low MSE ≠ correct dynamics."
- Restoring noise sensitivity while improving general quality: The authors attribute this to GAD forcing local consistency in the neighborhood of $\mathbf{z}$, acting as a smoothness regularizer that improves generalization to unseen prompts (Tab. 4 terminal error −13%).
- Downstream zero-shot transfer (Tab. 6, NoiseQuery feeding the teacher's optimal $\mathbf{z}^*$ to the student) shows that baseline distilled models can hardly use noise selected by the teacher, whereas GAD allows direct enjoyment of teacher-side test-time enhancement by aligning the noise-to-image geometry.
- Diversity / Fidelity trade-off (Fig. 5): Baselines are stuck in the disadvantageous zone of the Vendi vs CLIP axes, while GAD moves results for all three backbones toward the top-right corner, close to the teacher.

## Highlights & Insights
- The framing of "distilled away noise sensitivity" is highly valuable—making an implicit degradation long obscured by FID/CLIP explicit, and designing direct metrics like seed classification, JVP cos, Spec KL, and trajectory deviation to make the problem "visible and measurable" for the first time.
- Approximating JVP with finite difference and paired forwards reduces the theoretically expensive "Jacobian matching" to almost zero extra engineering cost (just 4 forwards), making it transferable to any distillation framework—true plug-and-play.
- Using stop-gradient to treat teacher response as an anchor while using $\lambda$ to decouple "global fidelity" from "local curvature" answers a often-ignored question: what to teach is more important than how accurately to teach in knowledge distillation.
- This logic is transferable: any scenario where a student is distilled into a trajectory/distribution using MSE/KL (speech synthesis distillation, policy distillation, video generation distillation) can benefit from a JVP-style paired response alignment term to preserve the teacher's local input sensitivity.

## Limitations & Future Work
- Training cost: Each step requires two additional teacher forwards + two additional student forwards, resulting in roughly 1.5x to 2x the training time compared to the base loss; the paper lacks a wall-clock comparison.
- Perturbation $h$ is a critical hyperparameter: if too small, it is submerged in numerical noise; if too large, it breaks the first-order approximation. The paper only provides configurations in the appendix and lacks a systematic sensitivity analysis for $h$ (which might need retuning across models).
- Hutchinson estimation requires random directions $\mathbf{v}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ across the full space. For DiTs with 64×64×4 latents, the variance of an unbiased estimate from a single direction is still significant, yet variance reduction for multiple $\mathbf{v}$ samples is not discussed.
- Experimental limitations: Layout control only used YOLOv4 AP instead of verifying with stronger detectors like ControlNet or Grounded-SAM; NoiseQuery was only performed on DrawBench.
- No assessment of impacts on "sister problems" like negative prompt adherence or trajectory invertibility—these are highly related to noise sensitivity and are natural next steps for evaluation.

## Related Work & Insights
- **vs Standard Distillation (ADD / LADD / DMD / TDM / SiD)**: These use pointwise alignment (MSE / Reverse KL / Fisher divergence). GAD does not replace them but overlaps as an orthogonal regularization term. The experiments use "base + GAD," emphasizing "complementing rather than replacing."
- **vs Relational Knowledge Distillation (Park 2019 RKD / Tung 2019 SP)**: Classic KD also emphasizes matching relationships between samples using pairwise distances/angles. GAD extends this to the continuous input space of generative models, using JVP to characterize relationships in "infinitesimal neighborhoods," which is better suited for generative mappings.
- **vs Diversity Enhancement (Diverse Distillation / Gandikota & Bau 2025)**: Those methods often directly regularize the entropy of the output distribution or explicitly diffuse seeds. GAD does not explicitly optimize diversity but intervenes at the upstream "local response to noise," restoring diversity as a byproduct—more elegant and simultaneously unlocking layout, retrieval, and diversity.
- **vs NoiseQuery / Noise Retrieval (Wang et al. 2025)**: NoiseQuery assumes the teacher's $\mathbf{z}\to\mathbf{x}$ geometry is preserved in the student, which actually fails. GAD is the first to directly solve this assumption mismatch, allowing techniques like NoiseQuery to achieve zero-shot transfer to 1-4 step students.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematizes "noise sensitivity collapse" for the first time and provides a clean geometric alignment framework; JVP + finite difference is not new math, but it is a new combination in T2I distillation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 backbones × 3 distillation paradigms × 11 baselines × 3 downstream tasks (layout / diversity / NoiseQuery), plus direct geometric measurements like JVP cos / Spec KL / trajectory deviation; very comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative line from motivation → diagnosis → formula → instantiation → experiments. Conceptual diagrams for "smooth path vs preserved curvature" in Figs 1/2/3 are intuitive; some formula layouts are slightly dense.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, nearly zero migration cost, almost no loss in base quality, and significant restoration of downstream controllability—this is a rare distillation contribution that is "add and use, ecosystem-friendly" and deserves to be merged into existing T2I distillation pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[CVPR 2026\] Resolving Endpoint Underfitting in Diffusion Bridges via Noise Alignment](../../CVPR2026/image_generation/resolving_endpoint_underfitting_in_diffusion_bridges_via_noise_alignment.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)

</div>

<!-- RELATED:END -->
