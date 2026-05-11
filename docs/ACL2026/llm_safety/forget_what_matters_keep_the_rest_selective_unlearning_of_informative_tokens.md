---
title: >-
  [Paper Note] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens
description: >-
  [ACL 2026][LLM Safety][machine unlearning] This paper proposes Entropy-guided Token Weighting (ETW), which uses the entropy of the predictive distribution as a proxy for token informativeness. ETW selectively imposes str…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "machine unlearning"
  - "informative tokens"
  - "entropy guidance"
  - "token weighting"
  - "selective unlearning"
date: 2026-05-08
content_hash: 6f09395c86383170
---

# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens

**Conference**: ACL 2026
**arXiv**: [2604.17785](https://arxiv.org/abs/2604.17785)
**Code**: None
**Area**: LLM Efficiency / Machine Unlearning
**Keywords**: machine unlearning, informative tokens, entropy guidance, token weighting, selective unlearning

## TL;DR

This paper proposes Entropy-guided Token Weighting (ETW), which uses the entropy of the predictive distribution as a proxy for token informativeness. ETW selectively imposes stronger unlearning penalties on informative tokens, enabling effective removal of target knowledge while better preserving general model utility.

## Background & Motivation

**Background**: LLM machine unlearning aims to selectively remove specific knowledge (e.g., private data) from a model while retaining other capabilities. Gradient ascent (GA)-based methods are the dominant paradigm, but applying uniform unlearning loss degrades model utility unnecessarily.

**Limitations of Prior Work**: Existing token-level regularization methods either rely on ground-truth confidence (WGA, SatImp)—failing to distinguish tokens with identical confidence but different semantic importance—or depend on external linguistic parsers (spaCy's SCN/SCE), which cannot capture contextual information or the model's overall predictive state.

**Key Challenge**: How can one reliably identify "informative tokens" (words carrying core semantic content) versus "structural tokens" (words primarily serving grammatical functions) within a sequence? Neither confidence scores nor part-of-speech categories provide sufficient discriminative power.

**Goal**: Design a more effective token-level informativeness measure to guide selective unlearning, allowing the model to precisely forget critical information while minimizing loss of general capability.

**Key Insight**: The entropy of the model's predictive distribution is leveraged as the key signal—structural tokens (e.g., "the") yield highly confident, low-entropy predictions, whereas informative tokens (e.g., a person's name like "Carmen") admit multiple plausible alternatives, resulting in higher entropy.

**Core Idea**: Entropy exploits the full probability distribution over the vocabulary rather than only the ground-truth confidence, providing a richer representation of token informativeness and more accurately distinguishing informative tokens from structural ones.

## Method

### Overall Architecture

ETW introduces token-level weights into the standard unlearning training pipeline: a weighted gradient ascent loss is applied to the forget set, while a standard cross-entropy loss is applied to the retain set. Weights are determined by the entropy of the model's predictive distribution—tokens with higher entropy receive larger unlearning penalties.

### Key Designs

1. **Entropy-guided Token Weight Computation**:

    - Function: Compute an informativeness weight for each token to guide unlearning intensity.
    - Mechanism: The predictive distribution entropy $H(y_i | y_{<i}, \mathbf{x}; \hat{\boldsymbol{\theta}})$ is computed at each position $i$, then normalized so that the weights sum to the sequence length $n$: $\omega_i^{\text{ETW}} = n \cdot H_i / \sum_j H_j$
    - Design Motivation: Compared to focusing solely on ground-truth confidence, entropy reflects how probability mass is distributed across all candidate tokens. Even when two tokens share the same GT confidence, their entropy values can differ by as much as $H_{\max} - H_{\min}$.

2. **Stop-gradient Weight Computation**:

    - Function: Ensure that weight computation does not interfere with gradient updates.
    - Mechanism: Entropy is computed using a stop-gradient copy $\hat{\boldsymbol{\theta}}$, with probability distributions obtained via a softmax with temperature $T$.
    - Design Motivation: Decoupling weight computation from the unlearning training process avoids spurious gradient interference introduced by the weight computation.

3. **Normalization to Preserve Loss Scale**:

    - Function: Redistribute unlearning intensity across tokens without altering the total magnitude.
    - Mechanism: After normalization, $\sum_i \omega_i = n$, preserving the same overall unlearning loss scale as unweighted GA.
    - Design Motivation: This avoids introducing additional hyperparameters to control overall unlearning intensity, making ETW a drop-in replacement for uniform weighting in existing methods.

### Loss & Training

The overall unlearning objective is $\mathcal{L} = \mathcal{L}_r + \lambda \mathcal{L}_f$, where the retain loss $\mathcal{L}_r$ is standard cross-entropy, the forget loss $\mathcal{L}_f = \sum_i \omega_i^{\text{ETW}} \log p(y_i | y_{<i}, \mathbf{x}; \boldsymbol{\theta})$ is the ETW-weighted gradient ascent loss, and $\lambda$ controls the unlearning strength.

## Key Experimental Results

### Main Results (TOFU Forget 10% – Llama 3.2-1B)

| Method | -log(FQ) ↓ | ΔMU ↓ | Agg. ↓ | \|Priv.\| ↓ |
|--------|-----------|-------|--------|------------|
| GA | 2.639 | 4.271 | 11.273 | 48.59 |
| WGA | 2.309 | 5.365 | 12.388 | 3.14 |
| SatImp | 2.871 | 5.258 | 15.093 | 19.65 |
| SCN | 2.754 | 3.548 | 9.772 | 39.98 |
| **ETW** | **0.492** | **3.471** | **1.707** | 9.56 |

### Ablation Study (ROC-AUC – Distinguishing Informative vs. Structural Tokens)

| Method | AUC |
|--------|-----|
| ETW | Highest (≥ 0.06 above runner-up) |
| SCN | Runner-up |
| Imp | 0.66 |
| WGA/TNPO | ≤ Random |

### Key Findings
- ETW consistently outperforms all baselines on ROC-AUC, demonstrating superior ability to distinguish informative from structural tokens.
- In qualitative analysis, ETW is the only method that successfully identifies core answer spans (e.g., "Love Inspired").
- Tokens with identical confidence can exhibit large entropy differences due to divergent distributions over non-GT tokens, validating entropy as a superior proxy for informativeness.
- After unlearning, the ETW model's predictive probabilities on informative tokens most closely approximate those of a model trained from scratch.

## Highlights & Insights
- The motivation is rigorously argued: through mathematical analysis of $H_{\min}$ and $H_{\max}$, the paper formally demonstrates that entropy provides a strictly richer representational space than confidence alone.
- The method is remarkably simple—requiring only entropy computation and normalization of the predictive distribution, with no external tools, reference models, or additional hyperparameters.
- Qualitative visualizations are intuitive and compelling, clearly illustrating token-level differences across methods.
- The normalization design makes ETW a plug-and-play regularizer that can directly replace uniform weighting in existing pipelines.

## Limitations & Future Work
- Experiments are primarily conducted on the TOFU dataset; generalization to larger-scale and more diverse unlearning scenarios (e.g., safety alignment) remains to be validated.
- Entropy computation requires an additional forward pass (softmax over the full vocabulary), which may incur non-trivial computational overhead for models with large vocabularies.
- The potential of combining ETW with DPO-based unlearning methods is not explored.
- When information is uniformly distributed across a sequence, ETW degrades to uniform weighting and may offer no advantage.

## Related Work & Insights
- **vs. WGA/TNPO**: These methods weight tokens by GT confidence, failing to distinguish tokens with different semantic importance but identical confidence.
- **vs. SCN/SCE (spaCy)**: POS-based binary masking ignores context and may incorrectly tag entities appearing in repeated question content.
- **vs. SatImp**: Combines confidence with complementary signals, but is fundamentally bounded by the discriminative ceiling of confidence-based measures.
- **Insights**: The application of entropy for token-level control generalizes naturally to other settings such as fine-tuning and reinforcement learning.

## Rating
- Novelty: ⭐⭐⭐⭐ — The insight of using entropy to measure token informativeness is novel and intuitively clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative (ROC-AUC) and qualitative (visualization) analyses are thorough, with comprehensive multi-baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is rigorously argued, progressing systematically from intuition to mathematical analysis to empirical validation.
- Value: ⭐⭐⭐⭐ — The method is simple yet effective, offering practical guidance for LLM unlearning and token-level control research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[CVPR 2026\] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions](../../CVPR2026/llm_safety/damp_class_unlearning_via_depth_aware_removal_of_forget_specific_directions.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[CVPR 2026\] ⊘ Source Models Leak What They Shouldn't ↛: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization](../../CVPR2026/llm_safety/oslash_source_models_leak_what_they_shouldnt_nrightarrow_unlearning_zero-shot_tr.md)

</div>

<!-- RELATED:END -->
