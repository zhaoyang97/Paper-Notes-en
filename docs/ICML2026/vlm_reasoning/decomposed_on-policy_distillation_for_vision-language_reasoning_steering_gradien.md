---
title: >-
  [Paper Note] Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
description: >-
  [ICML 2026][vlm_reasoning][Paper Note] The authors decompose the KL loss of multimodal on-policy distillation into "language prior" and "visual grounding" sub-objectives based on a Bayesian chain. They find that the gradients of these two are nearly orthogonal, and standard distillation merely takes a passive bisector. Consequently, they propose Visual Grad
tags:
  - ICML 2026
  - vlm_reasoning
date: 2026-05-08
content_hash: 8a9dc6138e130bb3
---
# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

**Conference**: ICML2026 Spotlight  
**arXiv**: [2606.00564](https://arxiv.org/abs/2606.00564)  
**Code**: https://github.com/hee-suk-yoon/Decomposed_OPD  
**Area**: Multimodal VLM / LLM Reasoning  
**Keywords**: On-Policy Distillation, Visual Grounding, Gradient Orthogonality, Visual Gradient Steering, VLM Reasoning  

## TL;DR
The authors decompose the KL loss of multimodal on-policy distillation into "language prior" and "visual grounding" sub-objectives based on a Bayesian chain. They find that the gradients of these two are nearly orthogonal, and standard distillation merely takes a passive bisector. Consequently, they propose Visual Gradient Steering (VGS) to actively bias the update direction toward the visual subspace, achieving average gains of +2.37%/+1.56% across seven multimodal reasoning benchmarks for Qwen3-VL 8B→2B/4B.

## Background & Motivation

**Background**: The two mainstream paths for small models to acquire reasoning capabilities are RLVR and On-Policy Distillation. On-Policy Distillation involves dense token-level supervision by a teacher model on trajectories sampled by the student, avoiding the cold-start problem of sparse rewards in RLVR, and has been proven effective for text-only LLMs.

**Limitations of Prior Work**: When directly applying on-policy distillation to VLMs, the industry either avoids it entirely (e.g., Qwen3-VL explicitly restricts distillation to text-only data to fine-tune the LLM backbone, abandoning visual alignment) or treats the multimodal conditional KL as a single monolithic objective, leading to insufficient transfer of visual grounding capabilities.

**Key Challenge**: Through Bayesian decomposition $\log p(\tau\mid I,x)=\log p(\tau\mid x)+\log p(I\mid\tau,x)-\log p(I\mid x)$, the authors reveal that the overall KL actually comprises "language prior alignment" and "visual grounding alignment." Geometric analysis of gradients shows that these two are nearly orthogonal on tokens with high visual dependence (the angle $\approx 92^\circ$ in the 9th bin of highest visual dependence). The standard monolithic loss gradient remains at a compromise direction of approximately $42^\circ$ to $50^\circ$ between the two, acting as a static bisector that underserves the visual subspace where the model is actually bottlenecked.

**Goal**: To actively steer the optimization direction toward the visual subspace without breaking the language prior, spending the update budget on resolving perceptual ambiguity rather than general language modeling.

**Key Insight**: The authors propose the "Asymmetric Maturity Hypothesis"—VLM students after pre-training and GRPO already possess strong language priors; the true bottleneck is visual perception. If the two gradients are already orthogonal, a "weighted sum" will be pointlessly diluted by the language term in extreme visual regions. Explicitly rotating the update direction toward the visual component provides a "free" improvement.

**Core Idea**: An additional KL term $\gamma\ell_{\text{Vis}}$ targeting "visual information gain" is added on top of the standard KL. A language preservation regularizer $\lambda\ell_{\text{LP}}$ is further applied to tokens with high visual dependence. Finally, a gradient norm normalization constant $\eta_{\text{VGS}}(\gamma)$ ensures that only the direction is changed, not the step size.

## Method

### Overall Architecture

VGS follows the standard setup for on-policy distillation: the student policy $p_S^\theta$ samples trajectories $\tau$ given multimodal inputs $(I,x)$, and the teacher $q_T$ (a GRPO fine-tuned 8B Qwen3-VL) provides token-level supervision. The pipeline differs from standard practices only in the loss function: the original multimodal Reverse KL $\ell_{\text{Standard}}(\tau)$ is replaced by the sum of three terms $\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)$, multiplied by a gradient norm normalization coefficient $\eta_{\text{VGS}}(\gamma)$. Trajectory sampling, model architecture, training framework, and batch scheduling remain unchanged, ensuring low engineering overhead.

### Key Designs

**1. Visual Information Gain Sub-objective $\ell_{\text{Vis}}$: A separate handle for "Visual Grounding"**

Standard distillation treats the multimodal KL as a monolith, diluting the visual term with the language term. The authors first decompose it: according to the Bayesian chain rule $\log p(I\mid\tau,x)=[\log p(\tau\mid I,x)-\log p(\tau\mid x)]+\log p(I\mid x)$. By performing the same decomposition on the teacher, they construct a "target distribution" $q_T^*$—which retains the student's own language prior but replaces the visual likelihood with the teacher's: $q_T^*(\tau\mid I,x)\propto p_S^\theta(\tau\mid x)\cdot q_T(I\mid\tau,x)$. In logit space, this is equivalent to:

$$\log q_T^*=\log p_S^\theta(\tau\mid x)+\big(\log q_T(\tau\mid I,x)-\log q_T(\tau\mid x)\big)-\log Z^*$$

Each term can be obtained by running forward passes of the student/teacher on "multimodal context" and "text-only context" without extra networks. The visual sub-objective is then the KL divergence of the student from this target distribution: $\ell_{\text{Vis}}(\tau)=\frac{1}{|\tau|}\sum_t D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},I,x)\,\|\,q_T^*(\cdot\mid\tau_{<t},I,x))$, punishing only the gap in "visual information gain." This separation is necessary because empirical findings show that for tokens with higher visual dependence, the angle between $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ increases monotonically from $\sim60^\circ$ to $\sim92^\circ$—the two gradients are geometrically independent, and a monolithic KL cannot satisfy the visual branch.

**2. Visual Gradient Steering + Norm Normalization $\eta_{\text{VGS}}(\gamma)$: Rotating direction without changing step size**

With an independent visual term, the update direction can be explicitly rotated toward the visual subspace: let $\ell_{\text{VGS}}(\tau)=\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)$, where $\gamma\ge 0$ controls the rotation magnitude. However, direct weighting changes the gradient norm as well, which is equivalent to implicitly changing the learning rate. the authors use a normalization constant to maintain the step size:

$$\eta_{\text{VGS}}(\gamma)=\frac{\|\nabla_\theta\mathcal{L}_{\text{Standard}}\|_2}{\|\nabla_\theta\mathcal{L}_{\text{Standard}}+\gamma\nabla_\theta\mathcal{L}_{\text{Vis}}\|_2}$$

After multiplying the overall loss by this constant, the steered gradient norm always equals the standard gradient norm, making $\gamma$ a pure "directional knob." Intuitively, this ratio remains nearly constant throughout training, so the authors fix it as a constant ($\eta=0.41$ for 2B, $\eta=0.36$ for 4B). Decoupling direction from magnitude is a robustness standard in multi-task learning (like GradNorm), which here makes $\gamma$ ablation clean and controllable.

**3. Language Preservation Regularizer $\ell_{\text{LP}}$: Protecting the language prior on "back-biting" tokens**

Visual steering is not always safe. Geometric analysis reveals that in the most extreme bin of visual dependence, $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ form an obtuse angle ($>90^\circ$). In this case, pushing toward the visual direction inversely reduces the language prior matching, leading to divergence of $\ell_{\text{Lang}}$. The solution is a language KL regularizer applied only to these tokens—specifically the top 30% ranked by Visual Dependence Score (VDS, where $\text{VDS}_t=D_{KL}(q_T(\cdot\mid\tau_{<t},I,x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$). For these, $\ell_{\text{LP}}(\tau)=\frac{1}{|\tau|}\sum_t \mathbf{1}[\text{VDS}_t>Q_{0.7}]\cdot D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$ is applied with a conservative weight $\lambda\approx 0.01$. Protecting only these few tokens prevents catastrophic forgetting without diluting the visual steering for general tokens.

### Loss & Training

The final training objective is $\mathcal{L}_{\text{VGS-LP}}=\eta_{\text{VGS}}(\gamma)\cdot\mathbb{E}_{\tau\sim p_S^\theta(\cdot\mid I,x)}[\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)]$. All experiments fix $\gamma=2.0$ and $\lambda=0.01$. The teacher is a Qwen3-VL-8B-Instruct trained on Vision-SR1-47K using GRPO for 2 epochs; the students are Qwen3-VL-2B/4B-Instruct, with a mandatory uniform system prompt to separate the reasoning chain from the final answer. The same loss can be seamlessly integrated into a GRPO training loop for hybrid RL+Distillation.

## Key Experimental Results

### Main Results

Qwen3-VL-8B teacher distilled to 2B/4B students on Vision-SR1-47K. Measured by average Acc@1 (greedy) and Acc@16 (T=1.0) across 7 benchmarks.

| Setting | Teacher 8B | Standard Distill Acc@1 | VGS Acc@1 | Gain |
|------|-----------|-----------------------|-----------|------|
| 2B Student Avg (7 benchmarks) | 61.37 | 43.74 | 46.10 | +2.37 |
| 4B Student Avg (7 benchmarks) | 61.37 | 56.64 | 58.12 | +1.56 |
| 2B / VisualPuzzles | 43.15 | 28.08 | 31.76 | +3.68 |
| 2B / LogicVista | 60.01 | 45.53 | 48.88 | +3.35 |
| 2B / MathVerse-VD | 79.63 | 56.02 | 58.10 | +2.08 |
| 4B / MathVerse-VD | 79.63 | 71.53 | 74.31 | +2.78 |
| 4B / MathVision | 44.14 | 37.96 | 40.59 | +2.63 |

The gain from VGS is more significant when the teacher-student capacity gap is larger (2B avg. +2.37 vs 4B +1.56), consistent with the hypothesis that visual perception is the primary bottleneck.

### Ablation Study

GRPO + Distillation ablation (2B Student, Vision-SR1-47K):

| Configuration | Avg Acc@1 | Avg Acc@16 | Description |
|------|-----------|-----------|------|
| Initial Student (2B) | 31.32 | – | Starting point |
| Pure GRPO | 44.83 | 45.68 | RL-only baseline |
| GRPO + Standard-KD | 45.41 | 45.22 | Adding monolithic distillation |
| GRPO + VGS (full) | 47.20 | 46.57 | +1.79 / +1.35 over Standard-KD |

### Key Findings

- For tokens with higher visual dependence, the value of $\ell_{\text{Vis}}$ decreases faster under VGS, confirming that the directional rotation indeed spends the update budget on the visual subspace.
- Disabling the LP regularizer while setting $\gamma=2.0$ causes $\ell_{\text{Lang}}$ to rise significantly in high VDS bins. Adding LP suppresses this curve and prevents drops in multimodal reasoning accuracy, proving the necessity of selective activation.
- An "Inverse Language Steering" experiment shows that rotating the steering direction toward the language subspace actually lowers average accuracy, falsifying the idea of "symmetric investment" and validating the asymmetric hypothesis that the visual subspace is the bottleneck.
- The $\eta_{\text{VGS}}$ norm normalization is critical: without it, changing $\gamma$ is equivalent to changing the learning rate, making hyperparameters difficult to tune.

## Highlights & Insights

- **Returns model distillation to an optimization geometry perspective**: While previous discussions on KL distillation often remained at the probabilistic level (Forward vs. Reverse KL, mode-covering vs. mode-seeking), this work examines the cosine relationship of gradient directions. It reveals the structural flaw of "Standard KL = Static Bisector" and provides insights transferable to any multi-conditional generation scenario (e.g., audio-language, video-language).
- **Computable Visual Information Gain**: Constructing the target distribution $q_T^*$ requires only the student/teacher logit ratios between multimodal and text-only contexts. It requires no extra networks or sampling, making it engineering-friendly for any base model supporting image token removal.
- **VDS Binning Analysis**: Measuring token-level visual dependence via $\text{VDS}_t=D_{KL}(q_T(\cdot\mid I,x)\,\|\,q_T(\cdot\mid x))$ provides a clean metric that can serve as an independent VLM interpretability tool for data filtering or attention analysis.

## Limitations & Future Work

- Experiments are limited to the Qwen3-VL series (2B/4B/8B) and lack cross-validation on architectures like LLaVA, InternVL, or Gemma to test the stability of $q_T^*$.
- The concept of a "text-only context" in VLMs assumes the model provides a valid distribution without image tokens, which might not hold for architectures using learned image tokens or soft prompts (e.g., Q-Former before word embeddings).
- Values for $\gamma=2.0$, $\lambda=0.01$, and threshold $Q_{0.7}$ are empirical; no automated tuning scheme based on student size is provided.
- Evaluation is restricted to static images and math/logic reasoning; the transferability of VGS to real-world multimodal dialogues or long-video understanding remains unknown.

## Related Work & Insights

- **vs. On-Policy Distillation (Agarwal et al., 2024)**: This work is a multimodal extension of their text-only Reverse KL + on-policy sampling framework, but it attacks the geometric flaws of monolithic KL rather than changing the KL type.
- **vs. GradNorm / PCGrad / GradVac**: Traditional multi-task gradient surgery involves projection or re-weighting after computing per-task gradients. VGS does not require per-task gradients (visual and language terms share the same tokens), incurring near-zero overhead with explicit interpretability.
- **vs. RLVR / GRPO**: RL provide sparse outcome rewards while distillation provides dense token-level supervision; the two are complementary. GRPO + VGS achieves implicit length regularization compared to the length explosion seen in pure GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work in VLM distillation to explicitly decompose "Language/Visual" components to examine gradient geometry.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers seven reasoning benchmarks, two student scales, and GRPO hybrid training, though restricted to one model family.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations; the geometric diagrams (Fig.3) and training dynamics (Fig.4) consistently support the core biological hypothesis.
- Value: ⭐⭐⭐⭐ Can be integrated into existing on-policy distillation frameworks with near-zero engineering cost, providing a plug-and-play boost for the small VLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation](../../CVPR2026/vlm_reasoning/vold_reasoning_transfer_from_llms_to_vision-language_models_via_on-policy_distil.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning](../../CVPR2026/vlm_reasoning/pdcr_perception-decomposed_confidence_reward_for_vision-language_reasoning.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)
- [\[CVPR 2026\] Self-Critical Distillation Network for Video-based Commonsense Captioning](../../CVPR2026/vlm_reasoning/self-critical_distillation_network_for_video-based_commonsense_captioning.md)

</div>

<!-- RELATED:END -->
