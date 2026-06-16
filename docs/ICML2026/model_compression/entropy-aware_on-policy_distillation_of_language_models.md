---
title: >-
  [Paper Note] Entropy-Aware On-Policy Distillation of Language Models
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] Addressing the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation, this paper proposes an adaptive strategy to mix forward and reverse KL based on teacher token-level entropy. It achieves up to a +5.05 improvement in Pass@8 across six
tags:
  - ICML 2026
  - Model Compression
  - Knowledge Distillation
date: 2026-05-08
content_hash: 13998108dd792032
---
# Entropy-Aware On-Policy Distillation of Language Models

**Conference**: ICML 2026  
**arXiv**: [2603.07079](https://arxiv.org/abs/2603.07079)  
**Code**: TBD  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, On-Policy Distillation, KL Divergence, Entropy-Aware, Language Models

## TL;DR
Addressing the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation, this paper proposes an adaptive strategy to mix forward and reverse KL based on teacher token-level entropy. It achieves up to a +5.05 improvement in Pass@8 across six mathematical reasoning benchmarks.

## Background & Motivation

**Background**: On-policy distillation is a primary paradigm for knowledge transfer in language models, where student models learn from dense token-level signals provided by a teacher on their own sampled trajectories. Standard practices employ reverse KL divergence $D_{\mathrm{KL}}(p_\theta \| p_T)$ as the training objective to encourage students to focus on high-confidence modes of the teacher distribution.

**Limitations of Prior Work**: Reverse KL is mode-seeking, forcing the student to concentrate probability mass on the peaks of the teacher distribution. When the teacher distribution has high entropy—i.e., multiple reasonable continuations exist (such as multiple reasoning paths in math problems)—reverse KL forces the student to fit only one mode, leading to a sharp drop in generation diversity. More severely, teacher gradient signals exhibit high variance in high-entropy regions, causing training instability.

**Key Challenge**: There is a fundamental trade-off between mode-seeking (reverse KL, precise but narrow) and mode-covering (forward KL, comprehensive but dispersed). Existing methods apply reverse KL indiscriminately, ignoring the fact that teacher output uncertainty changes dynamically with token positions.

**Goal**: Design a distillation framework that can adaptively perceive teacher uncertainty during distillation, switching to forward KL when entropy is high while maintaining the efficiency advantages of on-policy training.

**Key Insight**: The authors observe that teacher token-level entropy $H(p_T(\cdot|x_{<t}))$ serves as a natural indicator of when to mode-seek versus mode-cover. Low entropy = teacher certainty $\rightarrow$ precise imitation with reverse KL; High entropy = teacher uncertainty $\rightarrow$ preservation of diversity with forward KL.

**Core Idea**: Adaptively enhance the standard reverse KL objective with forward KL based on teacher token-level entropy values to balance precise imitation and diversity preservation within a unified on-policy framework.

## Method

### Overall Architecture
The core problem addressed is that uniform use of reverse KL in on-policy distillation forces student collapse and causes gradient instability on high-entropy tokens where the teacher is uncertain (§3 empirical evidence: distilled students retain only 6.8% of high-entropy tokens compared to the teacher's 18.5%). EOPD (Entropy-aware On-Policy Distillation) does not alter the standard on-policy distillation workflow (student sampling $\rightarrow$ teacher token-level scoring $\rightarrow$ loss calculation $\rightarrow$ gradient update) but modifies the loss calculation step. While reverse KL is always retained to ensure efficient convergence and precise imitation in low-entropy regions, an additional forward KL term is **gated** for tokens where the teacher conditional entropy $H_t^{\mathrm{te}}=H(p_T(\cdot|x_{<t}))$ exceeds a threshold $\tau$ to recover coverage of the teacher's multi-modal distribution.

### Key Designs

**1. Entropy-Aware KL Augmentation: Gating Forward KL by Teacher Entropy**

Standard On-Policy Distillation (OPD) optimizes a (clipped) reverse KL objective, which is mode-seeking and pushes student probability mass toward teacher peaks. While efficient when the teacher is certain (low entropy), it forces a single continuation when multiple paths exist (high entropy), leading to diversity collapse and instability (§3 toy experiments show that with high-entropy teachers, student top-1 indices change an average of 84 times per step, compared to only 7 times in low-entropy regions). EOPD uses teacher token-level entropy $H_t^{\mathrm{te}}=-\sum_x \pi_{\mathrm{te}}(x|\mathbf{c}_t)\log\pi_{\mathrm{te}}(x|\mathbf{c}_t)$ as the arbiter for preserving diversity, augmenting the reverse KL with a forward KL term gated by an indicator function:

$$\mathcal{L}_t^{\mathrm{EOPD}} = \mathcal{L}_t^{\mathrm{OPD}} + \alpha \cdot \mathbb{I}\!\left[H_t^{\mathrm{te}} > \tau\right] \mathcal{L}_t^{\mathrm{FKL}}$$

where $\mathcal{L}_t^{\mathrm{OPD}}$ is the clipped reverse KL and $\mathcal{L}_t^{\mathrm{FKL}}=D_{\mathrm{KL}}(\pi_{\mathrm{te}}\|\pi_\theta)$ is the forward KL. This uses **hard-threshold gating** rather than smooth interpolation: for low-entropy tokens, the indicator is 0 and the objective reverts to standard reverse KL for efficiency; only when entropy exceeds $\tau$ does forward KL intervene with weight $\alpha$ to force the student to maintain probability mass across multiple valid continuations. Experiments use $\tau=0.8$ and $\alpha=1.0$. This directly addresses the failure of reverse KL to account for dynamic uncertainty across token positions.

**2. Efficient Top-k Approximation of Forward KL: No Extra Sampling Cost**

A challenge of adding forward KL is that its expectation is defined over the teacher distribution, which typically requires sampling from the teacher and fitting a long-tail distribution—increasing overhead and decreasing efficiency. EOPD avoids sampling by approximating forward KL using only the **top-k (k=16)** teacher tokens with a renormalized teacher distribution $\tilde{\pi}_{\mathrm{te}}$:

$$\mathcal{L}_t^{\mathrm{FKL}} \approx \sum_{x\in\mathcal{S}_t^k} \tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)\,\log\frac{\tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)}{\pi_\theta(x|\mathbf{c}_t)}$$

Focusing on top-k tokens filters out low-probability tails (preventing small students from wasting capacity on noise) and saves memory. The entropy $H_t^{\mathrm{te}}$ is computed directly from teacher logits already produced during standard OPD. Since these logits are required for log-probability queries anyway, computing the entropy is nearly free. This allows the entropy-aware mechanism to integrate into the existing pipeline without additional forward passes or teacher sampling, preserving the 10x efficiency advantage of on-policy distillation over methods like GRPO.

### Loss & Training
The training follows a PPO-style implementation of on-policy distillation (Algorithm 1). In each round, the student samples trajectories using its old policy $\pi_{\theta_{\mathrm{old}}}$. For each token, the teacher is queried for $(\log\pi_{\mathrm{te}}(x_t|\mathbf{c}_t),\,H_t^{\mathrm{te}},\,\text{top-}k\text{ set})$, which are stored in a rollout buffer. The EOPD loss is then computed—always including reverse KL and conditionally adding forward KL when $H_t^{\mathrm{te}}>\tau$—and used to update student parameters. The teacher model is Qwen3-8B (non-thinking mode), and students are Qwen3-0.6B-Base, Qwen3-1.7B-Base, and Qwen3-4B-Base. The 0.6B and 1.7B models are trained on MATH, while the 4B model uses the more difficult DAPO-Math-14k.

## Key Experimental Results

### Main Results

| Student Model | Method | 6 Math Benchmarks Pass@8 (avg) | vs. Baseline |
|----------|------|-------------------------|----------|
| Qwen3-0.6B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-0.6B-Base | Entropy-Aware (Ours) | baseline + 1.37 | **+1.37** |
| Qwen3-1.7B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-1.7B-Base | Entropy-Aware (Ours) | baseline + 2.39 | **+2.39** |
| Qwen3-4B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-4B-Base | Entropy-Aware (Ours) | baseline + 5.05 | **+5.05** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Pure Reverse KL | Baseline | Standard on-policy distillation; poor diversity in high-entropy regions |
| Pure Forward KL | Slightly below baseline | Global mode-covering leads to imprecise fitting in low-entropy regions |
| Fixed Mix Weight | Slight gain | Static mixing independent of token entropy is sub-optimal |
| Entropy-Aware Adaptive (Ours) | Optimal | Dynamic switching achieves both precision and diversity |

### Key Findings
- Gains increase with student model size (0.6B: +1.37, 1.7B: +2.39, 4B: +5.05), suggesting larger students benefit more from maintaining diversity in high-entropy regions.
- Token-level analysis confirms the proposed method successfully maintains student token-level entropy, preventing diversity collapse.
- On high-entropy tokens, forward KL between student and teacher is significantly reduced, indicating better student-teacher alignment.
- Pass@8 improvements are more significant than Pass@1, further validating the importance of diversity—there is a higher probability of at least one correct path among multiple reasoning chains.

## Highlights & Insights
- Using teacher token-level entropy as a mode-seeking/mode-covering switch is simple yet effective, incurring virtually no extra computational cost while yielding significant gains.
- Reveals the "high-entropy blind spot" of reverse KL in LM distillation, offering a new perspective on objective function selection.
- High generalizability: the method can be applied as a plug-in to any on-policy distillation framework without modifying sampling strategies or network architectures.

## Limitations & Future Work
- Currently validated only on mathematical reasoning; generalization to other diversity-sensitive tasks like code generation or open-domain dialogue remains to be tested.
- The threshold $\tau$ and weight $\alpha$ rely on empirical tuning ($\tau=0.8, \alpha=1.0$); adaptive threshold selection mechanisms are worth exploring.
- Main experiments focus on the Qwen3 family; though Llama-3.1-8B $\rightarrow$ Llama-3.2-3B cross-family validation is provided in the appendix, larger or more heterogeneous pairings require more study.
- The interaction with other distillation enhancements (e.g., data augmentation, curriculum learning) has not been explored.

## Related Work & Insights
This work continues the line of research on LM knowledge distillation, contrasting with on-policy methods like GKD (Generalized Knowledge Distillation) and MiniLLM. The key insight is that distillation objectives should not be globally fixed but dynamically adjusted based on the teacher's local uncertainty. This insight could be transferred to reward shaping in RL (weighting rewards lower in uncertain regions) and hard negative mining in contrastive learning (selecting negative strategies based on anchor entropy).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/model_compression/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)

</div>

<!-- RELATED:END -->
