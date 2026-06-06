---
title: >-
  [Paper Note] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning
description: >-
  [ACL 2026][LLM Safety][LLM unlearning] This paper proposes PALU (Prefix-Aware Localized Unlearning), which achieves localized entropy maximization for unlearning along two dimensions: temporally…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM unlearning"
  - "localized entropy maximization"
  - "prefix-awareness"
  - "vocabulary sparsity optimization"
  - "privacy protection"
date: 2026-05-08
content_hash: 3c8059ef4b406361
---

# Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning

**Conference**: ACL 2026
**arXiv**: [2601.03190](https://arxiv.org/abs/2601.03190)  
**Code**: [GitHub](https://github.com/nxZhai/PALU)  
**Area**: LLM Safety / Machine Unlearning
**Keywords**: LLM unlearning, localized entropy maximization, prefix-awareness, vocabulary sparsity optimization, privacy protection

## TL;DR

This paper proposes PALU (Prefix-Aware Localized Unlearning), which achieves localized entropy maximization for unlearning along two dimensions: temporally, unlearning objectives are applied only to sensitive prefix tokens; in the vocabulary dimension, only top-K logits are flattened. This approach enables effective unlearning with minimal parameter perturbation while preserving the model's general capabilities.

## Background & Motivation

**Background**: LLMs inevitably memorize sensitive, private, and copyrighted information present in training data. Machine Unlearning aims to selectively remove specific knowledge from a model without retraining from scratch. Existing methods are primarily based on negated cross-entropy (negated CE) and its variants.

**Limitations of Prior Work**: (1) The negated CE objective suppresses only the top-1 token probability, but the suppressed probability mass may shift to closely related synonyms, leaving the distribution still sharp (low entropy) — the model has not truly "forgotten"; (2) Existing methods apply unlearning gradients indiscriminately to all response tokens, including content-irrelevant function words such as "is" and "for," causing unnecessary degradation of linguistic capabilities; (3) Full-vocabulary entropy maximization methods (e.g., PDU), while theoretically superior, require gradient computation over the entire $|V|$-dimensional vocabulary, incurring prohibitive computational cost.

**Key Challenge**: Effective unlearning demands precise intervention, yet existing methods apply globally indiscriminate optimization across both the temporal dimension (token sequences) and the vocabulary dimension — redundant optimization wastes computation and damages general model capabilities.

**Goal**: To achieve effective unlearning with the minimum necessary perturbation by simultaneously enforcing sparsity in both the temporal and vocabulary dimensions.

**Key Insight**: Two key observations: (i) sensitive semantics are triggered by a small number of prefix tokens, and intervening only on these "onset tokens" is sufficient to deflect the generation trajectory; (ii) autoregressive decoding is dominated by a small set of high-probability candidates, so flattening only the top-K logits is sufficient to effectively introduce uncertainty.

**Core Idea**: Bidirectional localization — intervening only on sensitive prefix tokens in the temporal dimension, and flattening only top-K logits toward a uniform target value $c$ in the vocabulary dimension, reducing unlearning complexity from $O(T|V|)$ to $O(TK)$.

## Method

### Overall Architecture

PALU operates through two levels of optimization: (1) Token level — semantically aware filtering identifies sensitive spans, from which only the first $N$ "onset tokens" per span are selected as unlearning targets; remaining tokens are either kept via KL divergence or skipped entirely; (2) Vocabulary level — for selected onset tokens, a localized entropy maximization objective (top-K logit flattening) replaces the negated CE.

### Key Designs

1. **Sparse Onset Token Selection (Temporal Sparsity)**:

    - Function: Precisely locates the minimal token subset requiring unlearning within the response sequence.
    - Mechanism: DistilBERT or GPT-4 is used to identify sensitive spans, yielding a binary mask $m_t$. Only the first $N$ tokens from each sensitive span are selected as "onset targets" $\mathcal{I}_{\text{init}}$. Tokens are categorized into three classes: onset targets (subject to unlearning loss), ordinary tokens (preserved via KL divergence), and redundant sensitive tokens (skipped during computation).
    - Design Motivation: Even within a sensitive span, typically only the first few tokens determine the semantic direction, with subsequent tokens merely unfolding along the already-established trajectory — intervening at the onset tokens suffices to redirect the entire generation path.

2. **Localized Entropy Maximization (Vocabulary Sparsity)**:

    - Function: Maximizes predictive uncertainty within the critical subspace of the vocabulary.
    - Mechanism: For onset token positions $t \in \mathcal{I}_{\text{init}}$, the top-K logit indices $V_{\text{top}}$ are extracted from a frozen reference model. The variance of top-K logits relative to a target value $c$ is then minimized: $\mathcal{L}_{\text{local}}(z_t) = \frac{1}{K}\sum_{i \in V_{\text{top}}}(z_{t,i} - c)^2$. This flattens the top-K logits (increasing local entropy) while globally suppressing the probability mass of top-K candidates by selecting a smaller value of $c$.
    - Design Motivation: Negated CE suppresses only the top-1 token while probability mass may shift to synonyms; full-vocabulary entropy maximization carries $O(T|V|)$ complexity; localized entropy maximization requires only $O(TK)$, achieving structured uncertainty within the decoding-critical subspace.

3. **Unified Unlearning Loss**:

    - Function: Integrates token-level and vocabulary-level sparsification.
    - Mechanism: $\mathcal{L}_f = \mathbb{E}_{t \in \mathcal{I}_{\text{init}}}[\mathcal{L}_{\text{local}}(z_t)] + \lambda \mathbb{E}_{t \notin \mathcal{I}_{\text{sens}}}[\text{KL}(P_{\theta_{\text{ref}}} \| P_\theta)]$. Gradients are nonzero only at onset tokens and ordinary tokens; gradients at redundant sensitive tokens are zero.
    - Design Motivation: Strictly adheres to the principle of minimal intervention — unlearning and retention objectives act on disjoint token subsets.

### Loss & Training

The total loss is $\mathcal{L}_{\text{all}} = \mathcal{L}_f + \lambda \mathcal{L}_r$, where $\mathcal{L}_r$ is the standard CE loss on the retain set. Base models are Llama-2-7B and Llama-3.1-8B. Top-K indices are extracted from the frozen reference model and held fixed throughout the unlearning process.

## Key Experimental Results

### Main Results

**TOFU Forget 5% Benchmark (Llama-2-7B)**

| Method | FQ ↑ | MU ↑ | Fluency ↑ | EM ↓ |
|--------|------|------|-----------|------|
| GA | 5.95E-11 | 0.5580 | 0.7423 | 0.9215 |
| NPO | 0.6284 | 0.5920 | 0.8115 | 0.6574 |
| TPO | 0.6284 | 0.5862 | 0.7929 | 0.6621 |
| PDU | 0.0021 | 0.5111 | 0.4834 | 0.6498 |
| **PALU** | **0.7126** | **0.6238** | 0.8122 | **0.5935** |
| Retain (Oracle) | 1.0000 | 0.6266 | 0.8889 | 0.6670 |

**TOFU Forget 5% Benchmark (Llama-3.1-8B)**

| Method | FQ ↑ | MU ↑ |
|--------|------|------|
| NPO | 0.6284 | 0.6006 |
| TPO | 0.7216 | 0.5921 |
| **PALU** | **0.9238** | **0.6162** |
| Retain (Oracle) | 1.0000 | 0.6323 |

### Ablation Study

**Dual Sparsity Ablation**

| Configuration | FQ ↑ | MU ↑ |
|---------------|------|------|
| Global negated CE (baseline) | ~0.63 | ~0.59 |
| + Token sparsity (prefix only) | Improved | Maintained |
| + Vocabulary sparsity (top-K only) | Improved | Maintained |
| + Dual sparsity (PALU) | **Best** | **Best** |

**Key Hyperparameter Sensitivity**

- Top-K cutoff size: $K=50$ achieves the optimal FQ/MU balance; excessively large $K$ ($K \to |V|$) degrades to global entropy maximization.
- Prefix length $N$: $N=3$–$5$ effectively disrupts sensitive generation; larger $N$ degrades MU.
- Target value $c$: The Local Mean strategy outperforms both Uniform and Global Mean strategies.

### Key Findings

- PALU achieves FQ of 0.9238 on Llama-3.1-8B, a 28% improvement over the strongest baseline TPO (0.7216).
- MU reaches 0.6162, nearly matching the theoretical upper bound of the Retain model at 0.6323 — breaking the conventional trade-off between unlearning efficacy and general capability retention.
- Performance remains stable across Forget 1% and 10% settings, whereas competing methods (NPO, DPO) degrade sharply under the 10% setting.
- Computational complexity is reduced from $O(T|V|)$ to $O(TK)$, yielding approximately a thousandfold speedup at $K=50$.

## Highlights & Insights

- The observation that "intervening only at prefix tokens suffices to deflect the entire generation trajectory" is highly insightful, revealing the causal chain structure inherent in autoregressive generation.
- Localized entropy maximization represents an elegant compromise between negated CE and global entropy maximization — avoiding the probability shift problem while maintaining computational efficiency.
- PALU's advantage is more pronounced on stronger models (Llama-3.1), suggesting that the method scales favorably with model capacity.

## Limitations & Future Work

- Relies on external models (DistilBERT/GPT-4) for sensitive span identification, introducing additional computational overhead and potential annotation errors.
- Top-K indices are extracted from the frozen reference model and held fixed; as the unlearning process progresses, logit distributions may shift, causing the fixed indices to become misaligned.
- Evaluation is conducted primarily on the synthetic TOFU dataset; real-world unlearning scenarios are considerably more complex.
- Robustness under adversarial attacks is not addressed — it remains unclear whether attackers could bypass prefix-level unlearning to recover sensitive information.

## Related Work & Insights

- **vs. GA/GD**: Negated CE leads to unbounded probability decay and catastrophic collapse; PALU achieves bounded, stable unlearning via entropy maximization.
- **vs. PDU**: Full-vocabulary entropy maximization is theoretically optimal but computationally intractable at $O(T|V|)$; PALU localizes it to the top-K subspace.
- **vs. TPO**: TPO achieves token-level sparsity but still employs negated CE over the full vocabulary; PALU simultaneously enforces sparsity in both the token and vocabulary dimensions.
- **vs. SU (Selective Unlearning)**: SU selects informative tokens but neglects vocabulary-level redundancy; PALU enforces sparsity along both dimensions simultaneously.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The bidirectional localization insight is precise and well-motivated; the paper reframes the unlearning problem from an "intervention efficiency" perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-setting TOFU + MUSE evaluations, two base models, and detailed ablations are provided, though adversarial robustness evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The derivation from the two sparsity observations to the method design is natural and fluent.
- Value: ⭐⭐⭐⭐⭐ Breaks the unlearning–general capability trade-off, offering a practical solution for deploying LLM unlearning in real-world settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/llm_safety/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_safety/llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)

</div>

<!-- RELATED:END -->
