---
title: >-
  [Paper Note] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens
description: >-
  [ACL 2026][LLM Safety][Machine Unlearning] Proposes Entropy-guided Token Weighting (ETW), which utilizes the entropy of the predictive distribution as a proxy indicator for token informativeness. It selectively imposes s…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Informative Tokens"
  - "Entropy-guided"
  - "Token Weighting"
  - "Selective Forgetting"
date: 2026-05-08
content_hash: 04ec1b64bcb31d41
---

# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens

**Conference**: ACL 2026  
**arXiv**: [2604.17785](https://arxiv.org/abs/2604.17785)  
**Code**: None  
**Area**: LLM Efficiency / Machine Unlearning  
**Keywords**: Machine Unlearning, Informative Tokens, Entropy-guided, Token Weighting, Selective Forgetting

## TL;DR

Proposes Entropy-guided Token Weighting (ETW), which utilizes the entropy of the predictive distribution as a proxy indicator for token informativeness. It selectively imposes stronger forgetting penalties on informative tokens to effectively unlearn target knowledge while better preserving the model's general capabilities.

## Background & Motivation

**Background**: LLM machine unlearning aims to selectively remove specific knowledge (e.g., private data) from a model while retaining other capabilities. Methods based on Gradient Ascent (GA) are the mainstream paradigm, but applying forgetting loss uniformly can unnecessarily degrade model utility.

**Limitations of Prior Work**: Existing token-level regularization methods either rely on ground-truth confidence (WGA, SatImp), which fails to distinguish tokens with the same confidence but different semantic importance, or depend on external language parsers (SCN/SCE in spaCy), which fail to capture context and the model's overall predictive state.

**Key Challenge**: How to accurately distinguish between "informative tokens" (words carrying core semantic content) and "structural tokens" (words primarily serving grammatical functions) in a sequence? Neither confidence nor part-of-speech tags are sufficient for reliable differentiation.

**Goal**: Design a more effective token-level informativeness measure to guide selective forgetting, enabling the model to precisely forget key information while minimizing the loss of general capabilities.

**Key Insight**: Leveraging the entropy of the model's predictive distribution—structural tokens (e.g., "the") have high predictive certainty and low entropy, whereas informative tokens (e.g., the name "Carmen") have multiple plausible alternatives and high entropy.

**Core Idea**: Entropy utilizes the probability distribution over the entire vocabulary rather than just the ground-truth confidence, providing a richer representation of informativeness that more accurately distinguishes informative tokens from structural tokens.

## Method

### Overall Architecture

ETW introduces token-level weights into the standard unlearning training pipeline: a weighted gradient ascent loss is used for the forget set, while a standard cross-entropy loss is used for the retain set. Weights are determined by the entropy of the model's predictive distribution—high-entropy tokens receive a larger forgetting penalty.

### Key Designs

1.  **Entropy-guided token weight calculation**:
    - **Function**: Calculates an informativeness weight for each token to guide forgetting intensity.
    - **Mechanism**: Computes the entropy of the predictive distribution at each position $i$, $H(y_i | y_{<i}, \mathbf{x}; \hat{\boldsymbol{\theta}})$, and then normalizes it so the sum of weights equals the sequence length $n$: $\omega_i^{\text{ETW}} = n \cdot H_i / \sum_j H_j$.
    - **Design Motivation**: Compared to focusing only on ground-truth confidence, entropy reflects the distribution of probability mass across all candidate tokens. Even if two tokens have identical GT confidence, their entropy can still differ within the range of $H_{\max} - H_{\min}$.

2.  **Stop-gradient weight calculation**:
    - **Function**: Ensures that weight calculation does not affect gradient updates.
    - **Mechanism**: Uses a stop-gradient copy $\hat{\boldsymbol{\theta}}$ to calculate the entropy, obtaining the probability distribution via softmax with temperature $T$.
    - **Design Motivation**: Decouples the weight calculation from the unlearning training process to avoid introducing additional gradient interference.

3.  **Normalization to keep loss scale invariant**:
    - **Function**: Redistributes forgetting intensity among tokens without changing the total amount.
    - **Mechanism**: After normalization, $\sum_i \omega_i = n$, maintaining the same overall forgetting loss scale as unweighted GA.
    - **Design Motivation**: Avoids introducing additional hyperparameters to control overall forgetting intensity, allowing ETW to directly replace uniform weights in existing methods.

### Loss & Training

The total forgetting objective is $\mathcal{L} = \mathcal{L}_r + \lambda \mathcal{L}_f$, where the retain loss $\mathcal{L}_r$ is standard cross-entropy, the forget loss $\mathcal{L}_f = \sum_i \omega_i^{\text{ETW}} \log p(y_i | y_{<i}, \mathbf{x}; \boldsymbol{\theta})$ is the ETW-weighted gradient ascent loss, and $\lambda$ controls the forgetting intensity.

## Key Experimental Results

### Main Results (TOFU Forget 10% - Llama 3.2-1B)

| Method | -log(FQ) ↓ | ΔMU ↓ | Agg. ↓ | |Priv.| ↓ |
| :--- | :--- | :--- | :--- | :--- |
| GA | 2.639 | 4.271 | 11.273 | 48.59 |
| WGA | 2.309 | 5.365 | 12.388 | 3.14 |
| SatImp | 2.871 | 5.258 | 15.093 | 19.65 |
| SCN | 2.754 | 3.548 | 9.772 | 39.98 |
| **ETW** | **0.492** | **3.471** | **1.707** | 9.56 |

### Ablation Study (ROC-AUC - Distinguishing Informative/Structural Tokens)

| Method | AUC |
| :--- | :--- |
| ETW | Highest (Exceeds second best by $\ge 0.06$) |
| SCN | Second Best |
| Imp | 0.66 |
| WGA/TNPO | $\le$ Random |

### Key Findings
- ETW comprehensively outperforms all baselines in ROC-AUC, demonstrating superior performance in distinguishing informative and structural tokens.
- In visualization analysis, ETW is the only method that successfully identifies core answer segments (e.g., "Love Inspired").
- Tokens with the same confidence can have vastly different entropy values due to the distribution of non-GT tokens, validating the advantage of entropy as a proxy for informativeness.
- Models after ETW unlearning show predictive probabilities for informative tokens that are closest to those of models trained from scratch.

## Highlights & Insights
- Compelling motivation: Rigorously proves through mathematical analysis of $H_{\min}$ and $H_{\max}$ that entropy provides a richer representation space than confidence.
- Exceptional simplicity: Requires only calculating the entropy of the predictive distribution and normalizing it, without needing external tools, reference models, or additional hyperparameters.
- Intuitive and powerful visualization analysis clearly demonstrates token-level differences across methods.
- The normalization design makes ETW a plug-and-play regularizer.

## Limitations & Future Work
- Experiments were primarily validated on the TOFU dataset; generalization to larger-scale and more diverse unlearning scenarios (e.g., safety alignment) remains to be verified.
- Entropy calculation requires an additional forward pass (performing softmax over the entire vocabulary), which may incur computational overhead for models with large vocabularies.
- Potential integration with DPO-like unlearning methods was not discussed.
- When information is uniformly distributed across a sequence, ETW degrades to uniform weighting and may lose its advantage.

## Related Work & Insights
- **vs WGA/TNPO**: These methods weight based on GT confidence and cannot distinguish tokens with different semantic importance but identical confidence.
- **vs SCN/SCE (spaCy)**: POS-based binary masks ignore context and may incorrectly label entities in repeated question content.
- **vs SatImp**: Combines confidence and complementary values but remains limited by the discriminative capacity of confidence.
- **Insight**: Application of entropy for token-level control could be extended to other scenarios such as fine-tuning and reinforcement learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of using entropy to measure token informativeness is novel and intuitively clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive quantitative (ROC-AUC) and qualitative (visualization) analyses with comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous argumentation, progressing logically from intuition to mathematical analysis to experiments.
- Value: ⭐⭐⭐⭐ Simple and effective method with practical guidance for LLM unlearning and token-level control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)

</div>

<!-- RELATED:END -->
