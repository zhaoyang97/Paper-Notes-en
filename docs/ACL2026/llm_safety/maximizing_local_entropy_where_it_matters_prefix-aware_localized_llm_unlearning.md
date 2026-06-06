---
title: >-
  [Paper Note] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning
description: >-
  [ACL 2026][LLM Safety][LLM Unlearning] This paper proposes PALU (Prefix-Aware Localized Unlearning), which achieves localized entropy maximization across temporal and vocabulary dimensions: it applies unlearning objectiv…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM Unlearning"
  - "Local Entropy Maximization"
  - "Prefix-Aware"
  - "Vocabulary Sparsity Optimization"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 9e795420c5de72b9
---

# Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning

**Conference**: ACL 2026  
**arXiv**: [2601.03190](https://arxiv.org/abs/2601.03190)  
**Code**: [GitHub](https://github.com/nxZhai/PALU)  
**Area**: LLM Security / Machine Unlearning  
**Keywords**: LLM Unlearning, Local Entropy Maximization, Prefix-Aware, Vocabulary Sparsity Optimization, Privacy Protection

## TL;DR

This paper proposes PALU (Prefix-Aware Localized Unlearning), which achieves localized entropy maximization across temporal and vocabulary dimensions: it applies unlearning objectives only to sensitive prefix tokens in the temporal dimension and flattens only the top-K logits in the vocabulary dimension, achieving efficient unlearning with minimal parameter perturbation while maintaining general model capabilities.

## Background & Motivation

**Background**: LLMs inevitably memorize sensitive, private, and copyrighted information from training data. Machine Unlearning aims to selectively remove specific knowledge from a model without retraining from scratch. Existing methods are primarily based on negated cross-entropy (negated CE) and its variants.

**Limitations of Prior Work**: (1) Negated CE objectives only suppress the top-1 token probability, but the suppressed probability mass may shift to highly correlated synonyms, leaving the distribution sharp (low entropy) and the model without true "unlearning"; (2) Existing methods apply unlearning gradients indiscriminately to all response tokens, including content-irrelevant function words like "is" or "for," leading to unnecessary degradation of linguistic capabilities; (3) Full-vocabulary entropy maximization methods (e.g., PDU) are theoretically superior but demand gradients calculated over a $|V|$-dimensional vocabulary, incurring prohibitive computational costs.

**Key Challenge**: Efficient unlearning requires precise intervention, yet existing methods perform global, indiscriminate optimization in both the temporal (token sequence) and vocabulary dimensions—redundant optimization wastes computation and harms general model capabilities.

**Goal**: To achieve effective unlearning with the minimum necessary perturbation—implementing sparsity in both temporal and vocabulary dimensions.

**Key Insight**: Two key observations—(i) Sensitive semantics are triggered by a small number of prefix tokens; applying unlearning to these "start tokens" alone is sufficient to deflect the generation path; (ii) Autoregressive decoding is dominated by a few high-probability candidates; flattening only the top-K logits effectively introduces uncertainty.

**Core Idea**: Bi-directional localization—intervening only on sensitive prefix tokens in the temporal dimension and flattening only top-K logits in the vocabulary dimension to approach a uniform value $c$, realizing an unlearning complexity of $O(TK)$ instead of $O(T|V|)$.

## Method

### Overall Architecture

PALU optimizes at two levels: (1) Token-level—identifying sensitive spans via semantic-aware filtering and selecting only the first N "start tokens" of each span as unlearning targets, while other tokens are kept invariant via KL divergence or skipped; (2) Vocabulary-level—using a local entropy maximization objective (top-K logit flattening) instead of negated CE for selected start tokens.

### Key Designs

1.  **Sparse Start Token Selection (Temporal Sparsity)**:
    - **Function**: Precisely locating the minimum token subset for unlearning within the response sequence.
    - **Mechanism**: Using DistilBERT or GPT-4 to identify sensitive spans, obtaining a binary mask $m_t$. Only the first N tokens of each sensitive span are selected as "initial targets" $\mathcal{I}_{\text{init}}$. Tokens are categorized into: initial targets (unlearning loss), normal tokens (KL divergence maintenance), and redundant sensitive tokens (skipped).
    - **Design Motivation**: Even within a sensitive span, the first few tokens usually determine the semantic direction, with subsequent tokens merely unfolding along the established path—intervening at start tokens is enough to deflect the entire generation trajectory.

2.  **Local Entropy Maximization (Vocabulary Sparsity)**:
    - **Function**: Maximizing predictive uncertainty within key subspaces of the vocabulary.
    - **Mechanism**: For start token positions $t \in \mathcal{I}_{\text{init}}$, top-K logit indices $V_{\text{top}}$ are extracted from a frozen reference model. The variance between top-K logits and a target value $c$ is minimized: $\mathcal{L}_{\text{local}}(z_t) = \frac{1}{K}\sum_{i \in V_{\text{top}}}(z_{t,i} - c)^2$. This flattens top-K logits (increasing local entropy) and depresses the top-K probability mass as a whole by choosing a small $c$.
    - **Design Motivation**: Negated CE only suppresses top-1 while probability may shift to synonyms; full-vocabulary entropy maximization is computationally heavy ($O(T|V|)$); localized entropy maximization requires only $O(TK)$, achieving structured uncertainty within the critical decoding subspace.

3.  **Unified Unlearning Loss**:
    - **Function**: Integrating token-level and vocabulary-level sparsity.
    - **Mechanism**: $\mathcal{L}_f = \mathbb{E}_{t \in \mathcal{I}_{\text{init}}}[\mathcal{L}_{\text{local}}(z_t)] + \lambda \mathbb{E}_{t \notin \mathcal{I}_{\text{sens}}}[\text{KL}(P_{\theta_{\text{ref}}} \| P_\theta)]$. Gradients are non-zero only for start tokens and normal tokens, while gradients for redundant sensitive tokens are zero.
    - **Design Motivation**: Strictly adhering to the principle of minimal intervention—unlearning and maintenance act on distinct token subsets.

### Loss & Training

Total loss $\mathcal{L}_{\text{all}} = \mathcal{L}_f + \lambda \mathcal{L}_r$, where $\mathcal{L}_r$ is standard CE loss on the retain set. Base models are Llama-2-7B and Llama-3.1-8B. Top-K indices are extracted from a frozen reference model and fixed during the unlearning process.

## Key Experimental Results

### Main Results

**TOFU Forget 5% Benchmark (Llama-2-7B)**

| Method | FQ ↑ | MU ↑ | Fluency ↑ | EM ↓ |
| :--- | :--- | :--- | :--- | :--- |
| GA | 5.95E-11 | 0.5580 | 0.7423 | 0.9215 |
| NPO | 0.6284 | 0.5920 | 0.8115 | 0.6574 |
| TPO | 0.6284 | 0.5862 | 0.7929 | 0.6621 |
| PDU | 0.0021 | 0.5111 | 0.4834 | 0.6498 |
| **PALU** | **0.7126** | **0.6238** | 0.8122 | **0.5935** |
| Retain (Ideal) | 1.0000 | 0.6266 | 0.8889 | 0.6670 |

**TOFU Forget 5% Benchmark (Llama-3.1-8B)**

| Method | FQ ↑ | MU ↑ |
| :--- | :--- | :--- |
| NPO | 0.6284 | 0.6006 |
| TPO | 0.7216 | 0.5921 |
| **PALU** | **0.9238** | **0.6162** |
| Retain (Ideal) | 1.0000 | 0.6323 |

### Ablation Study

**Dual Sparsity Ablation**

| Configuration | FQ ↑ | MU ↑ |
| :--- | :--- | :--- |
| Global Negated CE (baseline) | ~0.63 | ~0.59 |
| + Token Sparsity (Prefix-only) | Gain | Maintain |
| + Vocabulary Sparsity (Top-K only)| Gain | Maintain |
| + Dual Sparsity (PALU) | **Highest** | **Highest** |

**Key Hyperparameter Impact**

- **Top-K truncation size**: $K=50$ provides the best FQ/MU balance; excessively large values ($K \to |V|$) degrade to global entropy maximization.
- **Prefix length N**: $N=3-5$ is sufficient to effectively disrupt sensitive generation; larger values harm MU.
- **Target value c**: The Local Mean strategy outperforms Uniform and Global Mean strategies.

### Key Findings

- PALU achieves an FQ of 0.9238 on Llama-3.1-8B, a 28% improvement over the strongest baseline TPO (0.7216).
- MU reaches 0.6162, nearly approaching the theoretical upper bound of 0.6323 from the Retain model—breaking the trade-off where more unlearning typically degrades general performance.
- Results remain stable across Forget 1% and 10% settings, whereas other methods (NPO, DPO) significantly degrade in the 10% setting.
- Computational complexity is reduced from $O(T|V|)$ to $O(TK)$, achieving approximately 1000x speedup when $K=50$.

## Highlights & Insights

- The observation that "intervening only on prefixes can deflect the entire generation trajectory" is highly insightful—revealing the causal chain characteristic of autoregressive generation.
- Local entropy maximization is a sophisticated compromise between negated CE and global entropy maximization—avoiding probability mass shifts while maintaining computational efficiency.
- PALU demonstrates a greater advantage on stronger models (Llama-3.1), suggesting the method is scalable as model capabilities increase.

## Limitations & Future Work

- Relies on external models (DistilBERT/GPT-4) to identify sensitive spans, introducing additional computation and potential errors.
- Top-K indices are extracted from a frozen model and fixed; logit distributions may shift during unlearning, potentially making fixed indices inaccurate.
- Primarily evaluated on the synthetic TOFU dataset; real-world unlearning scenarios are more complex.
- Adversarial robustness is not discussed—it is unclear if attackers could bypass prefix unlearning to recover sensitive information.

## Related Work & Insights

- **vs GA/GD**: Negated CE causes unbounded probability decreases and catastrophic collapse; PALU achieves bounded, stable unlearning via entropy maximization.
- **vs PDU**: Full-vocabulary entropy maximization is theoretically optimal but computationally unacceptable ($O(T|V|)$); PALU localizes this to top-K.
- **vs TPO**: TPO implements token-level sparsity but still uses negated CE and calculates over the full vocabulary; PALU implements dual sparsity (token + vocabulary).
- **vs SU (Selective Unlearning)**: SU selects important tokens but ignores vocabulary redundancy; PALU sparsifies across both dimensions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The insight into bi-directional localization is precise, redefining the unlearning problem through the lens of "intervention efficiency."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes TOFU multi-settings, MUSE, two base models, and detailed ablations, but lacks adversarial evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The derivation from dual-sparsity observations to method design is natural and fluent.
- **Value**: ⭐⭐⭐⭐⭐ Breaks the unlearning-utility trade-off, providing a feasible solution for practical LLM unlearning deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/llm_safety/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[ACL 2026\] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem](modeling_llm_unlearning_as_an_asymmetric_two-task_learning_problem.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)

</div>

<!-- RELATED:END -->
