---
title: >-
  [Paper Note] Entropy-Aware On-Policy Distillation of Language Models
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Addressing the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation, this paper proposes an adaptive strategy that mixes forward and reverse KL based on token-level teacher entropy, achieving up to a +5.05 improvement in Pass@8 across s
tags:
  - ICML 2026
  - Model Compression
  - Knowledge Distillation
date: 2026-05-08
content_hash: 005e20f894cddfa0
---
# Entropy-Aware On-Policy Distillation of Language Models

**Conference**: ICML 2026  
**arXiv**: [2603.07079](https://arxiv.org/abs/2603.07079)  
**Code**: TBC  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, On-Policy Distillation, KL Divergence, Entropy-Aware, Language Models

## TL;DR
Addressing the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation, this paper proposes an adaptive strategy that mixes forward and reverse KL based on token-level teacher entropy, achieving up to a +5.05 improvement in Pass@8 across six mathematical reasoning benchmarks.

## Background & Motivation

**Background**: On-policy distillation is a mainstream paradigm for knowledge transfer in language models, where the student model learns using dense token-level signals provided by the teacher on its own sampled trajectories. The standard practice employs reverse KL divergence $D_{\mathrm{KL}}(p_\theta \| p_T)$ as the training objective to encourage the student to focus on the high-confidence modes of the teacher distribution.

**Limitations of Prior Work**: Reverse KL is mode-seeking, forcing the student to concentrate probability mass on the peaks of the teacher distribution. When the teacher distribution has high entropy—meaning multiple reasonable continuations exist (e.g., several reasoning paths for a math problem)—reverse KL forces the student to fit only one, leading to a sharp drop in generation diversity. More critically, gradient signals from the teacher in high-entropy regions exhibit high variance, resulting in training instability.

**Key Challenge**: There is a fundamental trade-off between mode-seeking (reverse KL, precise but narrow) and mode-covering (forward KL, comprehensive but spread). Existing methods apply reverse KL uniformly, ignoring the fact that teacher output uncertainty fluctuates dynamically across token positions.

**Goal**: To design an on-policy distillation framework that adaptively perceives teacher uncertainty, switching to forward KL in high-entropy scenarios while maintaining the efficiency advantages of on-policy training.

**Key Insight**: The authors observe that the teacher's token-level entropy $H(p_T(\cdot|x_{<t}))$ serves as a natural indicator for when to perform mode-seeking versus mode-covering. Low entropy (teacher is certain) warrants precise imitation via reverse KL, while high entropy (teacher is uncertain) requires forward KL to preserve diversity.

**Core Idea**: Adaptively augment the standard reverse KL objective with forward KL based on per-token teacher entropy values to balance precise imitation and diversity preservation within a unified on-policy framework.

## Method

### Overall Architecture
The core problem addressed is that uniform use of reverse KL in on-policy distillation forces the student to collapse into a single continuation for high-entropy tokens where the teacher is uncertain, leading to unstable gradients (§3 measurements show distilled students retain only 6.8% of high-entropy tokens compared to the teacher's 18.5%). This method (EOPD) maintains the standard on-policy distillation workflow (student sampling $\rightarrow$ teacher scoring $\rightarrow$ loss calculation $\rightarrow$ gradient update) but modifies the loss calculation step. While reverse KL is always retained for precise imitation and efficient convergence, an additional forward KL term is "plugged in" for high-entropy tokens where the teacher's conditional distribution entropy $H_t^{\mathrm{te}}=H(p_T(\cdot|x_{<t}))$ exceeds a threshold $\tau$. This mechanism uses a hard-threshold gate rather than smooth interpolation to recover multi-modal coverage.

### Key Designs

**1. Entropy-Aware KL Augmentation: Gated Forward KL Plug-in**

Standard on-policy distillation (OPD) uniformly optimizes a (PPO-truncated) reverse KL, which is mode-seeking. This is efficient when the teacher is certain but fails when the teacher faces multiple reasoning paths, causing diversity collapse and gradient instability (in §3 toy experiments, the student's top-1 index changes 84 times per step under a high-entropy teacher vs. 7 times under a low-entropy teacher). EOPD delegates the decision of when to preserve diversity to the teacher's token-level entropy $H_t^{\mathrm{te}}=-\sum_x \pi_{\mathrm{te}}(x|\mathbf{c}_t)\log\pi_{\mathrm{te}}(x|\mathbf{c}_t)$. It augments reverse KL with a gated forward KL term:

$$\mathcal{L}_t^{\mathrm{EOPD}} = \mathcal{L}_t^{\mathrm{OPD}} + \alpha \cdot \mathbb{I}\!\left[H_t^{\mathrm{te}} > \tau\right] \mathcal{L}_t^{\mathrm{FKL}}$$

where $\mathcal{L}_t^{\mathrm{OPD}}$ is the truncated reverse KL and $\mathcal{L}_t^{\mathrm{FKL}}=D_{\mathrm{KL}}(\pi_{\mathrm{te}}\|\pi_\theta)$ is the forward KL. This **hard-threshold gating** ensures that for low-entropy tokens, the objective simplifies to standard reverse KL for efficiency. Only when entropy crosses $\tau$ does forward KL intervene with weight $\alpha$ to force the student to retain probability mass for multiple reasonable continuations. Experiments use $\tau=0.8$ and $\alpha=1.0$.

**2. Efficient Top-k Approximation of Forward KL: Zero-Cost Entropy Sensing**

Forward KL typically requires calculating expectations over the teacher distribution, which traditionally involves sampling from the teacher and can force the student to fit low-probability long tails. EOPD avoids sampling by approximating forward KL using only the **top-k (k=16)** teacher tokens with a renormalized distribution $\tilde{\pi}_{\mathrm{te}}$:

$$\mathcal{L}_t^{\mathrm{FKL}} \approx \sum_{x\in\mathcal{S}_t^k} \tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)\,\log\frac{\tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)}{\pi_\theta(x|\mathbf{c}_t)}$$

Restricting to top-k excludes the long tail and saves memory. The entropy $H_t^{\mathrm{te}}$ is computed directly from teacher logits already obtained during the standard OPD forward pass, making the mechanism nearly free in terms of computation. This allows the entropy-aware mechanism to fit into existing pipelines without extra forward passes or teacher sampling, preserving the $10\times$ efficiency advantage of on-policy distillation over GRPO.

### Loss & Training
The training follows a PPO-style implementation (Algorithm 1): per-round sampling using the old policy $\pi_{\theta_{\mathrm{old}}}$, followed by querying the teacher for $(\log\pi_{\mathrm{te}}(x_t|\mathbf{c}_t),\,H_t^{\mathrm{te}},\,\text{top-}k \text{ set})$ and storing them in a rollout buffer. The student parameters are updated using the EOPD loss. The teacher model is Qwen3-8B (non-thinking mode), and students are Qwen3-0.6B-Base, 1.7B-Base, and 4B-Base. The 0.6B and 1.7B models are trained on MATH, while the 4B model uses the more challenging DAPO-Math-14k.

## Key Experimental Results

### Main Results

| Student Model | Method | 6 Math Benchmarks Pass@8 (avg) | vs. Baseline |
| :--- | :--- | :--- | :--- |
| Qwen3-0.6B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-0.6B-Base | Entropy-Aware (Ours) | baseline + 1.37 | **+1.37** |
| Qwen3-1.7B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-1.7B-Base | Entropy-Aware (Ours) | baseline + 2.39 | **+2.39** |
| Qwen3-4B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-4B-Base | Entropy-Aware (Ours) | baseline + 5.05 | **+5.05** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Pure Reverse KL | Baseline | Standard on-policy distillation; poor diversity in high-entropy regions. |
| Pure Forward KL | < Baseline | Global mode-covering leads to imprecise fitting in low-entropy regions. |
| Fixed Mix Weight | Slight gain | Static mixing fails to adapt to token-level entropy variations. |
| Entropy-Aware (Ours) | Optimal | Dynamic switching achieves both precision and diversity. |

### Key Findings
- Gains increase with student model scale (0.6B: +1.37, 1.7B: +2.39, 4B: +5.05), indicating larger students benefit more from high-entropy diversity.
- Token-level analysis confirms the method maintains student entropy, preventing diversity collapse.
- Forward KL between student and teacher is significantly reduced on high-entropy tokens, showing better alignment.
- Improvements in Pass@8 are more significant than Pass@1, verifying the importance of diversity preservation.

## Highlights & Insights
- Using teacher token-level entropy as a switching signal for mode-seeking/covering is simple yet effective, incurring negligible computational overhead.
- Reveals the "high-entropy blind spot" of reverse KL in LM distillation, offering a new perspective on objective selection.
- High generalizability: the method can be applied as a plug-in to any on-policy distillation framework without modifying sampling strategies or architectures.

## Limitations & Future Work
- Currently validated only on math reasoning; generalization to code generation or open-domain dialogue remains to be tested.
- Threshold $\tau$ and weight $\alpha$ rely on empirical tuning; automated threshold selection warrants exploration.
- Main experiments focus on the Qwen3 family; cross-family distillation (e.g., Llama $\rightarrow$ Qwen) requires more extensive study.
- Cooperation with other distillation enhancements (data augmentation, curriculum learning) has not been explored.

## Related Work & Insights
This work extends the line of LM knowledge distillation, contrasting with on-policy methods like GKD and MiniLLM. The key insight is that distillation objectives should be dynamic rather than globally fixed, adjusting to local teacher uncertainty. This can potentially be transferred to reward shaping in RL or hard negative mining in contrastive learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)

</div>

<!-- RELATED:END -->
