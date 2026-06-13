---
title: >-
  [Paper Note] Distillation of Large Language Models via Concrete Score Matching
description: >-
  [ICLR 2026][Model Compression][knowledge distillation] This paper proposes Concrete Score Distillation (CSD), a knowledge distillation loss for LLMs grounded in discrete score matching. By matching the relative logit dif…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "knowledge distillation"
  - "LLM compression"
  - "score matching"
  - "logit distillation"
  - "discrete score matching"
date: 2026-05-08
content_hash: 4dc1ba2625191522
---

# Distillation of Large Language Models via Concrete Score Matching

**Conference**: ICLR 2026
**arXiv**: [2509.25837](https://arxiv.org/abs/2509.25837)  
**Code**: [GitHub](https://github.com/aailab-kaist/CSD)  
**Area**: Model Compression / Knowledge Distillation
**Keywords**: knowledge distillation, LLM compression, score matching, logit distillation, discrete score matching

## TL;DR

This paper proposes Concrete Score Distillation (CSD), a knowledge distillation loss for LLMs grounded in discrete score matching. By matching the relative logit differences between all vocabulary token pairs across the student and teacher, CSD simultaneously overcomes the softmax-smoothing problem and the solution-space restriction inherent in direct logit distillation.

## Background & Motivation

Deploying LLMs incurs prohibitive inference costs, making knowledge distillation (KD) a central technique for transferring capabilities from large models to smaller ones. Existing KD methods predominantly align student and teacher distributions via softmax probability matching (e.g., KL divergence); however, the softmax transformation severely smooths logit information. With large vocabularies (e.g., 128K tokens), the vast majority of token probabilities approach zero (only 0.0023% exceed 0.01), rendering the rich knowledge encoded in teacher logits nearly indistinguishable after softmax.

Direct Logit Distillation (DLD) avoids softmax smoothing by matching raw logits via MSE, yet it introduces another critical flaw: **logit-shift invariance**. Because softmax depends only on relative logit differences, the condition $f_\theta[y_t] = f_T[y_t] + C$ yields identical probabilities for all tokens, yet the MSE loss of DLD is non-zero in this case, artificially restricting the feasible solution space. This restriction is especially harmful when there is a large capacity gap between teacher and student.

The core idea of this paper draws on the insight from energy-based models where score matching circumvents the normalization constraint: discrete concrete score matching is introduced into LLM distillation to design a logit-level objective that is neither affected by softmax smoothing nor constrained in its solution space.

## Method

### Overall Architecture

The core of CSD is to reformulate concrete score (discrete probability ratio) matching as logit residual matching. For each token position, CSD does not directly align logit values; instead, it aligns the relative logit differences between all vocabulary pairs $(y_t, x)$.

### Key Designs

1. **Log-transformed Concrete Score Matching**: Directly using the concrete score (probability ratio $q_\theta(x)/q_\theta(y_t)$) causes training instability when the denominator approaches zero. Applying a logarithmic transformation converts probability-ratio matching into logit-difference matching:

$$\mathcal{L}_{\text{CSD}} = \frac{1}{2} \sum_{y_t \in \mathcal{V}} \sum_{x \in \mathcal{V}} w(y_t, x) \left( f_\theta[x] - f_\theta[y_t] - f_T[x] + f_T[y_t] \right)^2$$

The log transformation yields two benefits: (1) probability ratios need not be computed explicitly, ensuring training stability; and (2) a logit-level loss formulation arises naturally.

2. **Superset Property of the Solution Space (Theorem 2)**: The optimal solution set of CSD strictly contains that of DLD, i.e., $\Theta_{\text{CSD}}^* \supsetneq \Theta_{\text{DLD}}^*$. When $f_\theta[y_t] = f_T[y_t] + C$, the CSD loss is zero whereas the DLD loss is not. This means CSD allows the student to search over a strictly larger space, which is particularly advantageous under limited model capacity.

3. **Linear-time Gradient Computation (Theorem 3)**: Although CSD nominally requires a double summation over the vocabulary ($O(|\mathcal{V}|^2)$), decomposing the weight function as $w(y_t, x) = w_1(y_t) \cdot w_2(x)$ enables analytic gradient computation in $O(|\mathcal{V}|)$ time. The key step is performing weighted centering normalization on the logits and then computing a weighted residual sum.

4. **Flexible Weight Design Space**: By choosing different combinations of $w_1, w_2$ from teacher probability (T), student probability (S), and uniform distribution (U), CSD flexibly interpolates between mode-seeking and mode-covering behavior. The $(S,S)$ configuration favors high fidelity, while $(U,S)$ and $(T,S)$ promote diversity.

### Loss & Training

- The gradient structure resembles that of KL divergence, but replaces softmax normalization with **centered normalization**, avoiding softmax smoothing.
- CSD is orthogonal to and compatible with on-policy techniques (ImitKD, GKD, DistiLLM).
- Monte Carlo gradient estimation is also supported (for joint weight functions), though analytic gradients converge faster.

## Key Experimental Results

### Main Results

**Table 1: Pure loss comparison for GPT-2-1.5B → GPT-2-0.1B (ROUGE-L)**

| Loss | Dolly | Self-Instruct | Vicuna | Super-NI | UnNI | Avg. |
|------|-------|--------------|--------|----------|------|------|
| KL | 23.52 | 10.02 | 14.57 | 16.76 | 18.55 | 16.68 |
| RKL | 24.26 | 11.19 | 15.80 | 20.17 | 22.99 | 18.88 |
| SRKL | 24.53 | 12.19 | 15.63 | 23.37 | 24.28 | 20.00 |
| **CSD (Ours)** | **24.94** | **12.06** | **15.78** | **24.60** | **25.88** | **20.65** |

**Table 2: Task-specific distillation Gemma-7B-IT → Gemma-2B-IT**

| Loss | Summarization ROUGE-L | Translation COMET | GSM8K Acc |
|------|-----------------------|-------------------|-----------|
| KL | 35.02 | 73.96 | 24.03 |
| **CSD (T,S)** | **35.67** | **74.14** | **25.78** |
| RKL | 0.00 | 45.02 | 0.00 |

### Ablation Study

- **CSD vs. DLD under identical weights**: CSD (T,T)/(U,U)/(S,S) consistently outperforms DLD T/U/S, validating the benefit of the enlarged solution space.
- **Weight selection**: $(S,S)$ achieves the highest fidelity; $(U,S)$ provides the best diversity–fidelity trade-off; $(T,S)$ yields the best probability calibration.
- **General-purpose dialogue distillation**: On Qwen2.5-7B→1.5B, MT-Bench improves from 5.75 (DistiLLM-2) to 5.95 (CSD).
- **Analytic vs. Monte Carlo gradients**: Analytic computation converges slightly faster and achieves marginally better performance.

### Key Findings

- Softmax discards a substantial portion of teacher knowledge in large-vocabulary LLMs (99.99%+ of token probabilities are below 0.01).
- Logit-shift invariance is the fundamental flaw of DLD; CSD resolves it naturally through its differential structure.
- Mode-seeking losses (RKL, SKL) are prone to collapse under low-data distillation regimes; CSD avoids this via weight selection.
- CSD is orthogonally complementary to on-policy methods, with the largest gains observed in purely on-policy settings.

## Highlights & Insights

- Reframing LLM distillation from the perspective of energy-based model score matching is both conceptually novel and theoretically rigorous.
- The analysis of logit-shift invariance is concise and incisive, directly identifying the fundamental deficiency of DLD.
- The weight function design space provides a unified framework for understanding the mode-seeking versus mode-covering spectrum.
- Analytic gradient computation reduces the nominally $O(|\mathcal{V}|^2)$ CSM objective to a practically feasible $O(|\mathcal{V}|)$ procedure.

## Limitations & Future Work

- Exploration of the weight function space remains limited; joint weight function designs have not been fully investigated.
- The factorization assumption $w(y_t,x)=w_1(y_t)w_2(x)$ constrains the expressiveness of the weighting scheme.
- Validation on larger-scale models (teacher >10B parameters) is insufficient.
- Integration with other compression methods such as quantization and pruning has not been explored.

## Related Work & Insights

- **DistiLLM** (Ko et al., 2024): Proposes SKL/SRKL smoothed KL distillation; CSD builds upon and further improves this line of work.
- **GKD** (Agarwal et al., 2024): An f-divergence framework; CSD can be viewed as a new framework that transcends probability matching.
- **Concrete Score Matching** (Meng et al., 2022): Score matching for discrete diffusion models; this paper innovatively adapts it to autoregressive LLM distillation.
- Insight: The "bypassing normalization" principle of score matching may find broader applicability in other NLP tasks that require distribution matching.

## Rating

- Novelty: ⭐⭐⭐⭐ — Transferring EBM score matching to LLM distillation is creative, though the core mechanism ultimately reduces to logit differential matching.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers diverse backbones, task types, and on-policy combinations with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, figures are clear, and motivation is well articulated.
- Value: ⭐⭐⭐⭐ — Provides a unified framework for logit distillation design; empirical gains are consistent though modest in magnitude.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)

</div>

<!-- RELATED:END -->
