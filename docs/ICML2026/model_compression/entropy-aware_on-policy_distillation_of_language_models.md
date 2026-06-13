---
title: >-
  [Paper Note] Entropy-Aware On-Policy Distillation of Language Models
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] To address the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "On-Policy Distillation"
  - "KL Divergence"
  - "Entropy-Aware"
  - "Language Model"
date: 2026-05-08
content_hash: 3a0c19200ec07d92
---

# Entropy-Aware On-Policy Distillation of Language Models

**Conference**: ICML 2026  
**arXiv**: [2603.07079](https://arxiv.org/abs/2603.07079)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, On-Policy Distillation, KL Divergence, Entropy-Aware, Language Model

## TL;DR
To address the issues of diversity collapse and gradient instability caused by reverse KL in high-entropy teacher regions during on-policy distillation, this paper proposes an adaptive strategy that mixes forward KL and reverse KL based on teacher token-level entropy, achieving up to +5.05 improvement in Pass@8 across six mathematical reasoning benchmarks.

## Background & Motivation

**Background**: On-policy distillation is a mainstream paradigm for language model knowledge transfer—student models learn from dense token-level signals provided by the teacher on their own sampled trajectories. The standard practice uses reverse KL divergence $D_{\mathrm{KL}}(p_\theta \| p_T)$ as the training objective, encouraging students to focus on the high-confidence modes of the teacher distribution.

**Limitations of Prior Work**: Reverse KL is mode-seeking, forcing the student to concentrate probability mass on the peaks of the teacher distribution. When the teacher distribution has high entropy—meaning multiple plausible continuations exist (e.g., various reasoning paths for a math problem)—reverse KL forces the student to fit only one, leading to a sharp drop in generation diversity. More severely, teacher gradient signals have high variance in high-entropy regions, causing training instability.

**Key Challenge**: There is a fundamental trade-off between mode-seeking (reverse KL, precise but narrow) and mode-covering (forward KL, comprehensive but broad). Existing methods uniformly select reverse KL, ignoring the reality that teacher output uncertainty changes dynamically with token positions.

**Goal**: Design a distillation framework that adaptively perceives teacher uncertainty, switches to forward KL during high entropy, and maintains the efficiency advantages of on-policy training.

**Key Insight**: The authors observe that teacher token-level entropy $H(p_T(\cdot|x_{<t}))$ serves as a natural indicator for "when to mode-seek and when to mode-cover." Low entropy = Teacher certain → utilize reverse KL for precise imitation; High entropy = Teacher uncertain → utilize forward KL to preserve diversity.

**Core Idea**: Adaptively augment the standard reverse KL objective with forward KL based on teacher per-token entropy to balance precise imitation and diversity preservation within a unified on-policy framework.

## Method

### Overall Architecture
The input consists of a pre-trained teacher model $p_T$ and a student model $p_\theta$ to be trained. The student calculates the entropy of the teacher distribution at each token position on its sampled trajectories and dynamically mixes reverse KL and forward KL as the distillation loss based on the entropy value. The overall pipeline follows standard on-policy distillation: student sampling → teacher scoring → mixed loss calculation → gradient update.

### Key Designs

1. **Teacher Entropy-Aware KL Objective Switching**:

    - **Function**: Adaptively selects the distillation objective based on the teacher's output entropy at each token position.
    - **Mechanism**: For each token position $t$, the entropy of the teacher's conditional distribution $H_t = H(p_T(\cdot|x_{<t}))$ is calculated. When $H_t$ is below a threshold, reverse KL $D_{\mathrm{KL}}(p_\theta \| p_T)$ is used (mode-seeking, precise fitting of teacher peaks); when $H_t$ is above the threshold, it switches to forward KL $D_{\mathrm{KL}}(p_T \| p_\theta)$ (mode-covering, covering the multi-modal distribution of the teacher). The final loss is a weighted mixture: $\mathcal{L} = (1-\lambda_t) \cdot D_{\mathrm{KL}}^{\mathrm{rev}} + \lambda_t \cdot D_{\mathrm{KL}}^{\mathrm{fwd}}$, where the weight $\lambda_t$ is monotonically determined by the teacher's entropy.
    - **Design Motivation**: To solve the issues of high gradient variance and diversity collapse of reverse KL in high-entropy regions, while maintaining the advantage of precise imitation in low-entropy regions.

2. **Efficient Integration of On-Policy Training**:

    - **Function**: Achieves entropy-aware distillation without increasing additional sampling costs.
    - **Mechanism**: Forward KL usually requires sampling from the teacher distribution, but this work utilizes importance weighting on student trajectories to approximate the forward KL gradient, avoiding extra teacher sampling overhead. Teacher logits are already calculated in standard on-policy distillation, so entropy calculation only requires one additional softmax normalization, making the overhead negligible.
    - **Design Motivation**: To maintain the training efficiency advantage of on-policy distillation without introducing extra forward passes or sampling steps.

3. **Gradient Stabilization for Diversity Preservation**:

    - **Function**: Stabilizes gradient signals in high-entropy regions to prevent training oscillations.
    - **Mechanism**: On high-entropy tokens, the gradient of reverse KL $\nabla_\theta D_{\mathrm{KL}}(p_\theta \| p_T)$ has high variance and unstable direction because the log-ratio $\log(p_\theta / p_T)$ is sensitive to parameter changes when the teacher distribution is flat. After switching to forward KL, the gradient becomes $\nabla_\theta [-\sum p_T \log p_\theta]$, equivalent to cross-entropy with the teacher distribution as the target, where the gradient direction is anchored by the teacher distribution and is significantly more stable.
    - **Design Motivation**: Indirectly increases the student model's token-level entropy (generation diversity) through gradient stabilization to improve student-teacher alignment.

### Loss & Training
The training process follows standard on-policy distillation: the student samples generated sequences, the teacher calculates the conditional distribution and entropy for each token, the mixed KL loss is calculated based on entropy, and student parameters are updated using a standard optimizer (e.g., AdamW). The teacher model is Qwen3-32B, and the student models are Qwen3-0.6B-Base, Qwen3-1.7B-Base, and Qwen3-4B-Base.

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
| Pure Reverse KL | Baseline level | Standard on-policy distillation, poor diversity in high-entropy regions |
| Pure Forward KL | Slightly lower than baseline | Global mode-covering leads to imprecise fitting in low-entropy regions |
| Fixed Mixture Weight | Small Gain | Static mixture without variation based on token entropy cannot adapt optimally |
| Entropy-Aware Adaptive Mixture (Ours) | Optimal | Dynamic switching achieves both precision and diversity |

### Key Findings
- Gains increase as the student model scale increases (0.6B: +1.37, 1.7B: +2.39, 4B: +5.05), indicating that larger student models benefit more from maintaining diversity in high-entropy regions.
- Token-level analysis shows that this method significantly maintains the token-level entropy of the student model, avoiding generation diversity collapse.
- On high-entropy tokens, the forward KL between student and teacher is significantly reduced, indicating better student-teacher alignment.
- The improvement in the Pass@8 metric is more significant than Pass@1, further verifying the importance of diversity preservation—there is a higher probability that at least one reasoning path is correct among the eight paths.

## Highlights & Insights
- Using teacher token-level entropy as the switching signal for mode-seeking / mode-covering is simple yet effective—this design adds almost no computational overhead (entropy is calculated directly from existing logits) but yields significant gains.
- It reveals the overlooked "high-entropy blind spot" issue of reverse KL in language model distillation, providing a new perspective on the choice of distillation objectives.
- Method generality is high: it can be applied as a plugin to any on-policy distillation framework without modifying sampling strategies or network architectures.

## Limitations & Future Work
- Currently validated only on mathematical reasoning tasks; generalization to other tasks requiring high diversity, such as code generation or open-domain dialogue, has not yet been verified.
- The choice of entropy threshold relies on empirical tuning; adaptive threshold selection mechanisms remain to be explored.
- The teacher model is Qwen3-32B; the effectiveness of other teacher-student architecture combinations (e.g., cross-family distillation) is unknown.
- The combined effect with other distillation enhancement techniques (e.g., data augmentation, curriculum learning) has not been explored.

## Related Work & Insights
This work continues the research line of knowledge distillation for language models, contrasting with on-policy distillation methods like GKD (Generalized Knowledge Distillation) and MiniLLM. The key insight is that the distillation objective should not be globally fixed but dynamically adjusted according to the teacher's local uncertainty. This insight can be transferred to reward shaping in reinforcement learning (reducing reward weight in uncertain regions) and hard negative mining in contrastive learning (selecting negative sample strategies based on anchor entropy).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/model_compression/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)

</div>

<!-- RELATED:END -->
