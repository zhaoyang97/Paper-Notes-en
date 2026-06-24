---
title: >-
  [Paper Note] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation
description: >-
  [ACL 2025][Model Compression][LLM Alignment] AlignDistil theoretically proves the equivalence between the RLHF objective and a token-level distillation process. Based on this, it designs a simple distillation method: constructing a teacher distribution through a linear combination of logit distributions from a DPO model and a reverse DPO model, and combining this with a token-adaptive extrapolation mechanism to achieve token-level reward optimization. It outperforms existing…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "LLM Alignment"
  - "DPO"
  - "Distillation"
  - "Token-Level Reward"
  - "Preference Optimization"
date: 2026-05-08
content_hash: 7be03e801a85eb80
---

# AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation

**Conference**: ACL 2025  
**arXiv**: [2503.02832](https://arxiv.org/abs/2503.02832)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: LLM Alignment, DPO, Distillation, Token-Level Reward, Preference Optimization  

## TL;DR
AlignDistil theoretically proves the equivalence between the RLHF objective and a token-level distillation process. Based on this, it designs a simple distillation method: constructing a teacher distribution through a linear combination of logit distributions from a DPO model and a reverse DPO model, and combining this with a token-adaptive extrapolation mechanism to achieve token-level reward optimization. It outperforms existing methods on AlpacaEval 2.0, MT-Bench, and Arena-Hard while achieving faster convergence.

## Background & Motivation

**Background**: LLM alignment is primarily achieved through RLHF and DPO, but these methods utilize sparse response-level reward/preference annotations to optimize all tokens.

**Limitations of Prior Work**: Response-level feedback is coarse-grained and fails to reflect the individual contribution of each token, which may incorrectly penalize high-quality tokens or encourage low-quality tokens, leading to suboptimal performance and slow convergence.

**Key Challenge**: Fine-grained token-level reward signals are needed, but human annotations can only provide response-level preferences.

**Goal**: To theoretically decompose the response-level RLHF objective into token-level optimization and achieve efficient token-level alignment.

**Key Insight**: Leveraging the token-level decomposability of DPO reward to prove that the RLHF objective is equivalent to a token-level distillation process.

**Core Idea**: RLHF = token-level distillation, and the teacher distribution = a linear combination of DPO logits and reference model logits.

## Method

### Overall Architecture

Starting from the RLHF objective, the method incorporates the token-level decomposition of DPO reward $\rightarrow$ derives the equivalent token-level distillation objective $\rightarrow$ the student policy $\pi_\theta$ learns from the teacher distribution $\pi^*$, which is composed of the adaptive logit extrapolation of a DPO model and a reverse DPO model.

### Key Designs

1. **RLHF-Distillation Equivalence**:

    - **Function**: To prove that the sequence-level RLHF objective can be decomposed into token-level KL divergence distillation.
    - **Mechanism**: The implicit DPO reward can be decomposed to each token position $r_t = \beta \log \frac{\pi_{DPO}(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$. Substituting this into the RLHF objective, the optimal policy at each token position is equivalent to the teacher distribution $\pi^*(t) \propto \exp(\text{logit}_{ref}(t) + \alpha \cdot \text{logit}_{DPO}(t))$.
    - **Design Motivation**: To convert an intractable RL optimization problem into a simple distillation task.

2. **Contrastive DPO Reward**:

    - **Function**: To improve the accuracy of the implicit DPO reward.
    - **Mechanism**: Training a standard DPO model and a reverse DPO model (swapping chosen/rejected), and constructing a more robust reward utilizing their contrast: the forward DPO strengthens good tokens, while the reverse DPO weakens bad tokens.
    - **Design Motivation**: A standalone DPO reward has worse accuracy than a pure reward model, and the contrastive strategy bridges this gap.

3. **Token-Adaptive Logit Extrapolation**:

    - **Function**: To construct a teacher distribution with an appropriate strength for each token position.
    - **Mechanism**: Dynamically adjusting the extrapolation weight based on the divergence degree $\alpha_t$ between the DPO model and the reference model at each token position—tokens with larger divergence receive smaller weights to avoid over-optimization, while tokens with smaller divergence receive larger weights to enhance alignment.
    - **Design Motivation**: Using a uniform weight across all positions leads to over-optimization of some tokens and under-optimization of others.

### Loss & Training
Distillation loss: $\mathcal{L} = \text{KL}(\pi^*(t) \| \pi_\theta(t))$, summed over all token positions. Supports the flexible switching between on-policy (self-sampling) and off-policy (using existing data) training modes.

## Key Experimental Results

### Main Results

Base model: Llama-3-8B-Instruct. Preference data: UltraFeedback.

| Method | AlpacaEval 2.0 LC WR ↑ | MT-Bench ↑ | Arena-Hard ↑ |
|------|----------------------|------------|-------------|
| DPO | 25.4 | 7.82 | 28.1 |
| SimPO | 30.2 | 7.88 | 33.5 |
| TDPO | 28.9 | 7.85 | 31.2 |
| **AlignDistil** | **34.7** | **8.05** | **37.8** |

### Ablation Study

| Configuration | AlpacaEval 2.0 LC WR | Description |
|------|---------------------|------|
| AlignDistil (Full) | **34.7** | Contrastive DPO + adaptive extrapolation |
| Without contrast (forward DPO only) | 31.5 | Affected by the inaccuracy of DPO rewards |
| Without adaptation (fixed $\alpha$) | 32.3 | Partial tokens over/under-optimized |
| Response-level reward distillation | 29.1 | Validates the advantage of token-level alignment |

### Key Findings
- **Token-level distribution reward > Token-level scalar reward > Response-level reward**: Distribution-based rewards provide the richest gradient signals.
- **Significantly faster convergence**: AlignDistil reaches the final performance of DPO in approximately half the training steps.
- **The contrastive DPO reward effectively compensates for the shortcomings of DPO acting as a reward model.**

## Highlights & Insights
- **Theoretical equivalence of RLHF and distillation**: Elegantly converts complex RL optimization into a standard knowledge distillation problem, greatly simplifying the implementation.
- **Clever use of the reverse DPO model**: By swapping chosen/rejected to train a model that does the "opposite," the contrast between the two models enhances the accuracy of the reward signal.

## Limitations & Future Work
- Requires training two DPO models (forward and reverse), which increases the upfront computing cost.
- The theoretical equivalence relies on the assumption of token-level decomposition of DPO rewards, which may be imprecise in practice.
- Has not been compared with recent methods such as GRPO/RLVR.

## Related Work & Insights
- **vs DPO**: DPO directly optimizes via response-level preferences, while AlignDistil achieves token-level optimization through distillation, offering finer granularity.
- **vs TDPO**: TDPO also performs token-level DPO, but lacks the distillation perspective and the adaptive mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical finding of the RLHF-distillation equivalence is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Benchmark comparisons and ablation experiments are comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The theoretical derivations are rigorous, and equations are clear.
- Value: ⭐⭐⭐⭐ Provides a fresh theoretical perspective and a practical methodology for LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](../../ICML2025/model_compression/confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](quantification_of_large_language_model_distillation.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)
- [\[ACL 2026\] SRA: Span Representation Alignment for Large Language Model Distillation](../../ACL2026/model_compression/sra_span_representation_alignment_for_large_language_model_distillation.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)

</div>

<!-- RELATED:END -->
