---
title: >-
  [Paper Note] Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
description: >-
  [ICML 2026][Multimodal VLM][Paper Note] The authors decompose the KL loss of multimodal online distillation into two sub-objectives, "language prior" and "visual grounding," based on the Bayes chain rule. They find that the gradients of these two are nearly orthogonal and that standard distillation merely takes a passive bisector. Consequently, they propose
tags:
  - ICML 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: dd18fa31409612ad
---
# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

**Conference**: ICML2026 Spotlight  
**arXiv**: [2606.00564](https://arxiv.org/abs/2606.00564)  
**Code**: https://github.com/hee-suk-yoon/Decomposed_OPD  
**Area**: Multimodal VLM / LLM Reasoning  
**Keywords**: Online Distillation, Visual Grounding, Gradient Orthogonality, Visual Gradient Steering, VLM Reasoning  

## TL;DR
The authors decompose the KL loss of multimodal online distillation into two sub-objectives, "language prior" and "visual grounding," based on the Bayes chain rule. They find that the gradients of these two are nearly orthogonal and that standard distillation merely takes a passive bisector. Consequently, they propose Visual Gradient Steering (VGS) to actively bias the update direction toward the visual subspace, achieving average improvements of +2.37%/+1.56% across seven multimodal reasoning benchmarks for Qwen3-VL 8B→2B/4B distillation.

## Background & Motivation

**Background**: Two mainstream paths for small models to acquire reasoning capabilities are RLVR and On-Policy Distillation. In on-policy distillation, a teacher model provides token-level dense supervision for trajectories sampled by the student, circumventing the cold-start problem of sparse rewards in RLVR. This has been validated as effective for pure-text LLMs.

**Limitations of Prior Work**: When migrating on-policy distillation directly to VLMs, the industry either avoids it entirely (e.g., Qwen3-VL explicitly limits distillation to pure-text data to fine-tune the LLM backbone, abandoning visual alignment) or crudely fits the multimodal conditional KL as a single objective, leading to insufficient transfer of visual grounding capabilities.

**Key Challenge**: Through Bayes decomposition $\log p(\tau\mid I,x)=\log p(\tau\mid x)+\log p(I\mid\tau,x)-\log p(I\mid x)$, the authors reveal that the overall KL actually comprises "language prior alignment" and "visual grounding alignment." Geometric gradient analysis shows that these two are nearly orthogonal on high visual-dependency tokens (angle $\approx 92^\circ$ in the 9th bucket of highest visual dependency). The standard monolithic loss gradient remains at a compromise direction of approximately $42^\circ$ to $50^\circ$ between the two, acting as a static bisector that underserves the visual subspace where the model is truly stuck.

**Goal**: To actively tilt the optimization direction toward the visual subspace without breaking the language prior, spending the update budget on resolving perceptual ambiguity rather than general language modeling.

**Key Insight**: The authors propose the "asymmetric maturity hypothesis"—VLM students trained with pre-training + GRPO already possess strong language priors; the real bottleneck is visual perception. Since the two gradients are naturally orthogonal, a "weighted sum" causes the visual term to be pointlessly diluted by the language term in extreme visual regions. Explicitly rotating the update direction toward vision provides a free performance boost.

**Core Idea**: Add an extra KL term $\gamma\ell_{\text{Vis}}$ targeting "visual information gain" atop the standard KL, and apply an additional language preservation regularizer $\lambda\ell_{\text{LP}}$ for high visual-dependency tokens. A gradient norm normalization constant $\eta_{\text{VGS}}(\gamma)$ is then used to ensure that only the direction is changed, not the step size.

## Method

### Overall Architecture

VGS follows the standard setup for on-policy distillation: the student policy $p_S^\theta$ samples trajectories $\tau$ under multimodal input $(I,x)$, and the teacher $q_T$ (an 8B Qwen3-VL fine-tuned via GRPO) provides token-level supervision. The difference from the standard pipeline lies solely in the loss function: the original multimodal Reverse KL $\ell_{\text{Standard}}(\tau)$ is replaced by the sum of three terms $\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)$, multiplied by a gradient norm normalization coefficient $\eta_{\text{VGS}}(\gamma)$. Trajectory sampling, model architecture, training framework, and batch scheduling remain unchanged, resulting in very low engineering overhead.

### Key Designs

**1. Visual Information Gain Sub-objective $\ell_{\text{Vis}}$: A handle for independent visual grounding optimization**

Standard distillation treats multimodal KL as a single entity, where the visual term is diluted by the language term. The authors first decompose it: via the Bayes chain rule $\log p(I\mid\tau,x)=[\log p(\tau\mid I,x)-\log p(\tau\mid x)]+\log p(I\mid x)$. By applying the same decomposition to the teacher, a "target distribution" $q_T^*$ can be constructed—it preserves the student's own language prior while replacing only the visual likelihood with the teacher's: $q_T^*(\tau\mid I,x)\propto p_S^\theta(\tau\mid x)\cdot q_T(I\mid\tau,x)$. In logit space, this is equivalent to:

$$\log q_T^*=\log p_S^\theta(\tau\mid x)+\big(\log q_T(\tau\mid I,x)-\log q_T(\tau\mid x)\big)-\log Z^*,$$

Every term can be obtained by running the student/teacher through forward passes on "multimodal context" and "pure-text context" once, requiring no extra networks. The visual sub-objective is then the KL divergence of the student relative to this target distribution: $\ell_{\text{Vis}}(\tau)=\frac{1}{|\tau|}\sum_t D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},I,x)\,\|\,q_T^*(\cdot\mid\tau_{<t},I,x))$, which penalizes only the gap in "visual information gain." This is isolated because empirical findings show that for tokens with higher visual dependency, the angle between $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ increases monotonically from $\sim60^\circ$ to $\sim92^\circ$—the two gradients are geometrically nearly independent, and passive averaging via monolithic KL fails to satisfy the visual branch.

**2. Visual Gradient Steering + Norm Normalization $\eta_{\text{VGS}}(\gamma)$: Rotating direction without changing step size**

With an independent visual term, the update direction can be explicitly steered toward the visual subspace: let $\ell_{\text{VGS}}(\tau)=\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)$, where $\gamma\ge 0$ controls the rotation magnitude. However, direct weighting has a side effect—$\gamma$ amplifies the gradient magnitude while changing the direction, effectively modifying the learning rate. To prevent direction and step size from contaminating each other during tuning, the authors use a normalization constant to scale the step size back:

$$\eta_{\text{VGS}}(\gamma)=\frac{\|\nabla_\theta\mathcal{L}_{\text{Standard}}\|_2}{\|\nabla_\theta\mathcal{L}_{\text{Standard}}+\gamma\nabla_\theta\mathcal{L}_{\text{Vis}}\|_2},$$

After multiplying this with the overall loss, the norm of the steered gradient always equals the norm of the standard gradient, making $\gamma$ a pure "directional knob." In practice, this ratio remains nearly constant throughout training; the authors fix it as a constant ($\eta=0.41$ for 2B, $\eta=0.36$ for 4B) to avoid dynamic computation. Decoupling direction and step size is a robustness common sense in multi-task learning (e.g., GradNorm), and here it makes the $\gamma$ ablation clean and controllable.

**3. Language Preservation Regularization $\ell_{\text{LP}}$: Guarding the language prior on "back-biting" tokens**

Visual steering is not universally safe. Geometric analysis shows that in the bucket with the most extreme visual dependency, $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ form an obtuse angle ($>90^\circ$). In such cases, pushing forcedly toward the visual direction will inversely decrease the language prior match, leading to divergence in $\ell_{\text{Lang}}$ on the training curve. The authors' strategy is to add a language KL regularizer only to these tokens—selecting the top 30% based on Visual Dependency Score (VDS) ($\text{VDS}_t=D_{KL}(q_T(\cdot\mid\tau_{<t},I,x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$). For these, they apply $\ell_{\text{LP}}(\tau)=\frac{1}{|\tau|}\sum_t \mathbf{1}[\text{VDS}_t>Q_{0.7}]\cdot D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$ with a conservative weight $\lambda\approx 0.01$. Protecting only these few back-biting tokens prevents catastrophic forgetting without adding excessive supervision to general tokens or diluting the visual steering intensity.

### Loss & Training

The final training objective is $\mathcal{L}_{\text{VGS-LP}}=\eta_{\text{VGS}}(\gamma)\cdot\mathbb{E}_{\tau\sim p_S^\theta(\cdot\mid I,x)}[\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)]$. All experiments fix $\gamma=2.0$ and $\lambda=0.01$. The teacher is obtained by training Qwen3-VL-8B-Instruct on Vision-SR1-47K for 2 epochs using GRPO; students are Qwen3-VL-2B/4B-Instruct, forced with a unified system prompt to separate the reasoning chain from the final answer. The same loss can be seamlessly integrated into a GRPO training loop for hybrid RL+distillation.

## Key Experimental Results

### Main Results

Main result: Qwen3-VL-8B teacher distilled to 2B / 4B students on Vision-SR1-47K, showing average Acc@1 (greedy) and Acc@16 (T=1.0) across 7 multimodal reasoning benchmarks.

| Setting | Teacher 8B | Standard Distill Acc@1 | VGS Acc@1 | Gain |
|------|-----------|-----------------------|-----------|------|
| 2B Student Avg (7 benchmarks) | 61.37 | 43.74 | 46.10 | +2.37 |
| 4B Student Avg (7 benchmarks) | 61.37 | 56.64 | 58.12 | +1.56 |
| 2B / VisualPuzzles | 43.15 | 28.08 | 31.76 | +3.68 |
| 2B / LogicVista | 60.01 | 45.53 | 48.88 | +3.35 |
| 2B / MathVerse-VD | 79.63 | 56.02 | 58.10 | +2.08 |
| 4B / MathVerse-VD | 79.63 | 71.53 | 74.31 | +2.78 |
| 4B / MathVision | 44.14 | 37.96 | 40.59 | +2.63 |

VGS gains are more significant when the teacher-student capacity gap is larger (Avg +2.37 for 2B vs. +1.56 for 4B), consistent with the hypothesis that visual perception is the true bottleneck.

### Ablation Study

Ablation of GRPO + distillation (2B student, Vision-SR1-47K):

| Configuration | Avg Acc@1 | Avg Acc@16 | Description |
|------|-----------|-------------|------|
| Initial Student (2B) | 31.32 | – | Starting Point |
| Pure GRPO | 44.83 | 45.68 | Single RL baseline |
| GRPO + Standard-KD | 45.41 | 45.22 | Added monolithic distillation |
| GRPO + VGS (full) | 47.20 | 46.57 | +1.79 / +1.35 over Standard-KD |

### Key Findings

- For tokens with higher visual dependency, VGS results in a faster decline in $\ell_{\text{Vis}}$, confirming that directional rotation indeed spends the update budget on the visual subspace.
- Disabling the LP regularizer while setting $\gamma$ to 2.0 causes $\ell_{\text{Lang}}$ in high VDS buckets to rise significantly; adding LP pushes the curve back down without hurting multimodal reasoning accuracy, proving the necessity of selective LP activation.
- An inverse "Language Steering" experiment shows that rotating the steering direction toward the language subspace actually lowers average accuracy, directly disproving naive "symmetrical investment" ideas and validating the asymmetrical hypothesis that the visual subspace is the bottleneck.
- $\eta_{\text{VGS}}$ norm normalization is critical: removing it makes changing $\gamma$ equivalent to changing the learning rate, making hyperparameters difficult to tune.

## Highlights & Insights

- **Returning model distillation to an optimization geometry perspective**: Previous discussions on KL distillation mostly stayed at the probabilistic level (forward vs. backward KL, mode-covering vs. mode-seeking). This work shifts to the cosine relationship of gradient directions, revealing structural flaws like "Standard KL = Static Bisector" that were previously unclearly explained. This idea is transferable to any multi-conditional generation distillation scenario, such as audio-language or video-language.
- **Computable Visual Information Gain**: Constructing the target distribution $q_T^*$ only requires the logit ratios of the student/teacher across multimodal and pure-text contexts, requiring no additional networks or sampling. It is engineering-friendly as long as the base model supports the removal of image tokens.
- **VDS Bucket Analysis**: Using $\text{VDS}_t=D_{KL}(q_T(\cdot\mid I,x)\,\|\,q_T(\cdot\mid x))$ to measure token-level visual dependency is a very clean metric that can be used independently as a VLM interpretability tool (e.g., for training data filtering or attention analysis).

## Limitations & Future Work

- The experiments only cover the Qwen3-VL series (2B/4B/8B) and lack cross-validation on the stability of $q_T^*$ construction across LLaVA, InternVL, or Gemma architectures. Since vision-text fusion methods vary greatly across base models, the transferability of the conclusions requires further validation.
- The concept of "pure-text context" in VLMs assumes the model can still provide a valid distribution without image tokens, which might not hold for architectures using learned image tokens or soft prompts (e.g., Q-Former inserted before word embeddings).
- $\gamma=2.0$, $\lambda=0.01$, and threshold $Q_{0.7}$ are empirical values. No scheme for automatic tuning based on student size is provided; in particular, the setting of $\alpha$ during the RL phase might be strongly coupled with $\gamma$.
- Evaluation is currently limited to static image + math/logic reasoning tasks. The transferability of VGS to real-world multimodal dialogues or long-video understanding remains unknown.

## Related Work & Insights

- **vs On-Policy Distillation (Agarwal et al., 2024)**: They proposed a pure-text distillation framework using Reverse KL + on-policy sampling; this work is a multimodal extension. The difference is the direct attack on the geometric flaws of monolithic KL rather than changing the KL type.
- **vs GradNorm / PCGrad / GradVac etc. multi-task gradient surgery**: Traditional methods perform projection/re-weighting after per-task gradients are obtained. This work does not require per-task gradients (visual and language terms share the same tokens), incurs nearly zero overhead, and the directional correction is explicitly interpretable (always tilting toward $\nabla\ell_{\text{Vis}}$).
- **vs RLVR / GRPO**: RL provides sparse outcome rewards while distillation provides dense token-level supervision; this work treats them as complementary. GRPO + VGS achieves implicit length regularization aligned with the teacher, which is more stable than the length explosion seen in pure GRPO.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bayes decomposition + gradient orthogonal analysis is the first work in VLM distillation literature to explicitly separate language/vision to examine gradient geometry.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven reasoning benchmarks + two student scales + GRPO hybrid training, though only covering one model family.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations; geometric diagrams (Fig.3) and training dynamics plots (Fig.4) consistently support the core hypothesis.
- Value: ⭐⭐⭐⭐ Can be integrated into existing on-policy distillation frameworks at almost zero cost, providing a plug-and-play boost for the small VLM training community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VOLD: Reasoning Transfer from LLMs to Vision-Language Models via On-Policy Distillation](../../CVPR2026/multimodal_vlm/vold_reasoning_transfer_from_llms_to_vision-language_models_via_on-policy_distil.md)
- [\[CVPR 2026\] PDCR: Perception-Decomposed Confidence Reward for Vision-Language Reasoning](../../CVPR2026/multimodal_vlm/pdcr_perception-decomposed_confidence_reward_for_vision-language_reasoning.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] CodeV: Code with Images for Faithful Visual Reasoning via Tool-Aware Policy Optimization](../../CVPR2026/multimodal_vlm/codev_code_with_images_for_faithful_visual_reasoning_via_tool-aware_policy_optim.md)
- [\[CVPR 2026\] Multimodal Distribution Matching for Vision-Language Dataset Distillation](../../CVPR2026/multimodal_vlm/multimodal_distribution_matching_for_vision-language_dataset_distillation.md)

</div>

<!-- RELATED:END -->
