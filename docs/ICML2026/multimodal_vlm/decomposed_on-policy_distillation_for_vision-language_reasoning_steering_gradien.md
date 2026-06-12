---
title: >-
  [Paper Note] Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
description: >-
  [ICML2026][Multimodal VLM][On-policy Distillation] The authors decompose the KL loss of multimodal on-policy distillation into two sub-objectives, "language prior" and "visual grounding"…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "On-policy Distillation"
  - "Visual Grounding"
  - "Gradient Orthogonality"
  - "Visual Gradient Steering"
  - "VLM Reasoning"
date: 2026-05-08
content_hash: 5aff5af099b0654b
---

# Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding

**Conference**: ICML2026  
**arXiv**: [2606.00564](https://arxiv.org/abs/2606.00564)  
**Code**: https://github.com/hee-suk-yoon/Decomposed_OPD  
**Area**: Multimodal VLM / LLM Reasoning  
**Keywords**: On-policy Distillation, Visual Grounding, Gradient Orthogonality, Visual Gradient Steering, VLM Reasoning  

## TL;DR
The authors decompose the KL loss of multimodal on-policy distillation into two sub-objectives, "language prior" and "visual grounding", based on a Bayesian chain. They discover that their gradients are nearly orthogonal and that standard distillation merely acts as a passive bisector. They propose Visual Gradient Steering (VGS) to actively bias the update direction toward the visual subspace, achieving average gains of +2.37%/+1.56% across seven multimodal reasoning benchmarks for Qwen3-VL 8B→2B/4B distillation.

## Background & Motivation

**Background**: There are two mainstream paths for small models to acquire reasoning capabilities: RLVR and On-Policy Distillation. In on-policy distillation, a teacher model provides token-level dense supervision on trajectories sampled by the student itself, bypassing the cold-start problem of sparse rewards in RLVR. This has been proven effective for text-only LLMs.

**Limitations of Prior Work**: When directly applying on-policy distillation to VLMs, the industry either completely avoids it (e.g., Qwen3-VL explicitly restricts distillation to text-only data to fine-tune the LLM backbone, abandoning visual alignment) or treats the multimodal conditional KL as a single monolithic objective, resulting in insufficient transfer of visual grounding capabilities.

**Key Challenge**: Through Bayesian decomposition $\log p(\tau\mid I,x)=\log p(\tau\mid x)+\log p(I\mid\tau,x)-\log p(I\mid x)$, the authors reveal that the overall KL actually consists of "language prior alignment" and "visual grounding alignment" sub-objectives. Gradient geometric analysis shows that these two are nearly orthogonal on tokens with high visual dependence (an angle of $\approx 92^\circ$ in the 9th bin of highest visual dependency). The standard monolithic loss gradient consistently maintains a compromise direction of approximately $42^\circ$-$50^\circ$ between them, acting as a static bisector that fails to invest sufficiently in the critical visual subspace.

**Goal**: To actively steer the optimization direction toward the visual subspace without breaking the language prior, thereby spending the update budget on resolving perceptual ambiguity rather than general language modeling.

**Key Insight**: The authors propose the "asymmetric maturity hypothesis"—a VLM student that has undergone pre-training + GRPO already possesses a strong language prior; the true bottleneck is visual perception. Since the two gradients are already orthogonal, a "weighted sum" would be pointlessly diluted by the language term in visual-heavy regions. Explicitly rotating the update direction toward vision provides a "free" performance boost.

**Core Idea**: Add an additional KL term $\gamma\ell_{\text{Vis}}$ targeting "visual information gain" on top of the standard KL, and apply an extra language preservation regularizer $\lambda\ell_{\text{LP}}$ for high visual-dependency tokens. A gradient norm normalization constant $\eta_{\text{VGS}}(\gamma)$ is then used to ensure that only the direction is modified, not the step size.

## Method

### Overall Architecture

VGS follows the standard setup for on-policy distillation: a student policy $p_S^\theta$ samples trajectories $\tau$ given multimodal input $(I,x)$, and a teacher $q_T$ (an 8B Qwen3-VL fine-tuned with GRPO) provides token-level supervision. The difference from the standard pipeline lies solely in the loss function: the original multimodal Reverse KL $\ell_{\text{Standard}}(\tau)$ is replaced by the sum of three terms $\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)$, scaled by a gradient norm normalization coefficient $\eta_{\text{VGS}}(\gamma)$. Trajectory sampling, model architecture, training framework, and batching remain unchanged, ensuring extremely low engineering overhead.

### Key Designs

1. **Objective Decomposition and Visual Information Gain Construction $\ell_{\text{Vis}}$**:

    - **Function**: To isolate a sub-objective focused purely on "visual grounding" from the monolithic KL, quantifying the perceptual gap between the student and teacher.
    - **Mechanism**: Based on the Bayesian chain rule $\log p(I\mid\tau,x)=[\log p(\tau\mid I,x)-\log p(\tau\mid x)]+\log p(I\mid x)$, a similar decomposition for the teacher allows the construction of a "target distribution" $q_T^*$. This distribution preserves the student's own language prior while replacing visual likelihood with the teacher's: $q_T^*(\tau\mid I,x)\propto p_S^\theta(\tau\mid x)\cdot q_T(I\mid\tau,x)$. In logit space, this is equivalent to $\log q_T^*=\log p_S^\theta(\tau\mid x)+(\log q_T(\tau\mid I,x)-\log q_T(\tau\mid x))-\log Z^*$. All quantities can be obtained by running the student and teacher through forward passes on both multimodal and text-only contexts. Finally, $\ell_{\text{Vis}}(\tau)=\frac{1}{|\tau|}\sum_t D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},I,x)\,\|\,q_T^*(\cdot\mid\tau_{<t},I,x))$ only penalizes the gap between student and teacher "visual information gain."
    - **Design Motivation**: Empirical results show that for tokens with higher visual dependency, the angle between $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ monotonically increases from $\sim60^\circ$ to $\sim92^\circ$. Since they are geometrically independent, they require independent optimization levers rather than relying on the passive average of a monolithic gradient.

2. **Visual Gradient Steering and Norm Normalization $\eta_{\text{VGS}}(\gamma)$**:

    - **Function**: To explicitly rotate the standard update direction into the visual subspace while maintaining the magnitude of the update step to avoid disrupting the learning rate schedule.
    - **Mechanism**: Let $\ell_{\text{VGS}}(\tau)=\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)$, where $\gamma\ge 0$ is the steering coefficient. To prevent $\gamma$ from amplifying the gradient norm (which is coupled with the learning rate), $\eta_{\text{VGS}}(\gamma)=\|\nabla_\theta\mathcal{L}_{\text{Standard}}\|_2/\|\nabla_\theta\mathcal{L}_{\text{Standard}}+\gamma\nabla_\theta\mathcal{L}_{\text{Vis}}\|_2$ is defined and applied to the total loss. This ensures the steered gradient norm always equals the standard gradient norm. In practice, the authors found this ratio to be stable throughout training, so it is fixed as a constant ($\eta=0.41$ for 2B, $\eta=0.36$ for 4B) to save dynamic computation costs.
    - **Design Motivation**: Decoupling "direction" from "magnitude" is a robustness principle in multi-task learning (cf. GradNorm); otherwise, increasing $\gamma$ changes two things simultaneously, causing ablation and $\gamma$ tuning to interfere with each other.

3. **Language Preservation Regularization $\ell_{\text{LP}}$**:

    - **Function**: To suppress the "negative projection" of the visual gradient on the language subspace for high visual-dependency tokens, preventing catastrophic forgetting.
    - **Mechanism**: Geometric analysis reveals that in bins with extreme visual dependency, $\nabla\ell_{\text{Lang}}$ and $\nabla\ell_{\text{Vis}}$ form an obtuse angle ($>90^\circ$). At this point, pure visual steering would inversely reduce language prior matching, leading to significant divergence in $\ell_{\text{Lang}}$ observed in training curves. The authors apply a language KL regularizer $\ell_{\text{LP}}(\tau)=\frac{1}{|\tau|}\sum_t \mathbf{1}[\text{VDS}_t>Q_{0.7}]\cdot D_{KL}(p_S^\theta(\cdot\mid\tau_{<t},x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$ for the top 30% of tokens based on the Visual Dependency Score ($\text{VDS}_t=D_{KL}(q_T(\cdot\mid\tau_{<t},I,x)\,\|\,q_T(\cdot\mid\tau_{<t},x))$), with a conservative weight $\lambda\approx 0.01$.
    - **Design Motivation**: To protect the language prior only on the few tokens where "backlash" actually occurs, avoiding heavy supervision on general tokens that would dilute visual steering.

### Loss & Training

The final training objective is $\mathcal{L}_{\text{VGS-LP}}=\eta_{\text{VGS}}(\gamma)\cdot\mathbb{E}_{\tau\sim p_S^\theta(\cdot\mid I,x)}[\ell_{\text{Standard}}(\tau)+\gamma\ell_{\text{Vis}}(\tau)+\lambda\ell_{\text{LP}}(\tau)]$. $\gamma=2.0$ and $\lambda=0.01$ are fixed for all experiments. The teacher is a Qwen3-VL-8B-Instruct trained on Vision-SR1-47K with GRPO for 2 epochs; the students are Qwen3-VL-2B/4B-Instruct, using a standardized system prompt to separate the reasoning chain from the final answer. This loss can also seamlessly integrate into the GRPO training loop for hybrid RL+distillation.

## Key Experimental Results

### Main Results

Primary findings: Distilling the Qwen3-VL-8B teacher onto 2B / 4B students using Vision-SR1-47K, showing average Acc@1 (greedy) and Acc@16 (T=1.0) across 7 multimodal reasoning benchmarks.

| Setting | Teacher 8B | Standard Distill Acc@1 | VGS Acc@1 | Gain |
|------|-----------|-----------------------|-----------|------|
| 2B Student Avg (7 benchmarks) | 61.37 | 43.74 | 46.10 | +2.37 |
| 4B Student Avg (7 benchmarks) | 61.37 | 56.64 | 58.12 | +1.56 |
| 2B / VisualPuzzles | 43.15 | 28.08 | 31.76 | +3.68 |
| 2B / LogicVista | 60.01 | 45.53 | 48.88 | +3.35 |
| 2B / MathVerse-VD | 79.63 | 56.02 | 58.10 | +2.08 |
| 4B / MathVerse-VD | 79.63 | 71.53 | 74.31 | +2.78 |
| 4B / MathVision | 44.14 | 37.96 | 40.59 | +2.63 |

The gain from VGS is more significant when the teacher-student capacity gap is larger (2B avg +2.37 vs 4B +1.56), consistent with the hypothesis that visual perception is the true bottleneck.

### Ablation Study

GRPO + distillation ablation (2B Student, Vision-SR1-47K):

| Configuration | Avg Acc@1 | Avg Acc@16 | Notes |
|------|-----------|-------------|------|
| Initial Student (2B) | 31.32 | – | Starting point |
| Pure GRPO | 44.83 | 45.68 | RL-only baseline |
| GRPO + Standard-KD | 45.41 | 45.22 | Plus monolithic distillation |
| GRPO + VGS (full) | 47.20 | 46.57 | +1.79 / +1.35 over Standard-KD |

### Key Findings

- For tokens with higher visual dependency, VGS leads to a faster decrease in $\ell_{\text{Vis}}$ (Fig. 4), confirming that directional rotation effectively allocates the update budget to the visual subspace.
- Disabling the LP regularizer while setting $\gamma$ to 2.0 causes $\ell_{\text{Lang}}$ in high VDS bins to rise significantly; adding LP suppresses this curve without dropping multimodal reasoning accuracy, proving that selective activation of LP is necessary.
- An "Inverse Language Steering" experiment showed that rotating the steering direction toward the language subspace actually reduced average accuracy, falsifying the naive idea of "symmetric investment" and validating the asymmetric hypothesis that the visual subspace is the bottleneck.
- $\eta_{\text{VGS}}$ norm normalization is critical: without it, changing $\gamma$ is equivalent to changing the learning rate, making hyperparameter tuning extremely difficult.

## Highlights & Insights

- **Returning model distillation to an optimization geometry perspective**: Previous discussions on KL distillation mostly focused on the probabilistic level (Forward vs. Reverse KL, mode-covering vs. mode-seeking). This work instead examines the cosine relationship of gradient directions, revealing structural flaws—like "Standard KL = Static Bisector"—that have not been previously clarified. This idea is transferable to any multi-conditional generation distillation scenario, such as audio-language or video-language.
- **Computable Visual Information Gain**: Constructing the target distribution $q_T^*$ requires only the logit ratios of the student and teacher across multimodal and text-only contexts. It requires no additional networks or sampling, making it engineering-friendly; it can be used with any base model that supports removing image tokens.
- **VDS Binning Analysis**: Measuring token-level visual dependency with $\text{VDS}_t=D_{KL}(q_T(\cdot\mid I,x)\,\|\,q_T(\cdot\mid x))$ provides a very clean metric that can be used independently as a VLM interpretability tool (e.g., for data filtering or attention analysis).

## Limitations & Future Work

- Experiments are limited to the Qwen3-VL series (2B/4B/8B), without cross-validation on architectures like LLaVA, InternVL, or Gemma to check the stability of $q_T^*$ construction. The transferability of the conclusions needs further validation given the varied vision-text fusion methods in different base models.
- The meaning of "text-only context" in VLMs assumes the model can still provide a valid distribution without image tokens, which might not hold for architectures using learned image tokens or soft prompts (e.g., Q-Former placed before word embeddings).
- $\gamma=2.0$, $\lambda=0.01$, and threshold $Q_{0.7}$ are empirical values; no scheme was provided for automatic tuning based on student model size. Particularly in the RL phase, the setting of $\alpha$ for distillation might be strongly coupled with $\gamma$.
- Current evaluations are all on static images and math/logic reasoning tasks. The transferability of VGS to real-world multimodal dialogues or long-video understanding remains unknown.

## Related Work & Insights

- **vs On-Policy Distillation (Agarwal et al., 2024)**: They proposed a the Reverse KL + on-policy sampling framework for text-only distillation; this work is a multimodal extension that directly addresses the geometric flaws of monolithic KL rather than changing the KL type.
- **vs GradNorm / PCGrad / GradVac (multi-task gradient surgery)**: Traditional methods project or re-weight gradients after obtaining per-task gradients. This work does not require per-task gradients (visual and language terms share the same tokens), incurring almost zero overhead, and the directional correction is explicitly interpretable (always biased toward $\nabla\ell_{\text{Vis}}$).
- **vs RLVR / GRPO**: RL provides sparse outcome rewards while distillation provides dense token-level supervision; this work treats them as complementary. GRPO + VGS achieves implicit length regularization aligned with the teacher (Fig. 6), making it more stable than the length explosion often seen in pure GRPO.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Bayes decomposition + gradient orthogonality analysis is the first work in VLM distillation literature to explicitly separate language/visual gradients geometrically.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers seven reasoning benchmarks, two student sizes, and GRPO hybrid training, but only within one model family.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear derivations; geometric diagrams (Fig. 3) and training dynamics plots (Fig. 4) consistently support the core hypothesis.
- **Value**: ⭐⭐⭐⭐ Can be integrated into existing on-policy distillation frameworks at almost zero cost, providing a "plug-and-play" boost for the small VLM training community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[ICML 2026\] Contextualized Visual Personalization in Vision-Language Models](contextualized_visual_personalization_in_vision-language_models.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](../../AAAI2026/multimodal_vlm/towards_long-window_anchoring_in_vision-language_model_distillation.md)
- [\[ICML 2026\] CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models](clip_tricks_you_training-free_token_pruning_for_efficient_pixel_grounding_in_lar.md)

</div>

<!-- RELATED:END -->
