---
title: >-
  [Paper Note] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens
description: >-
  [ACL 2026][LLM Safety][Paper Note] Ours proposes Entropy-guided Token Weighting (ETW), which utilizes the entropy of the prediction distribution as a proxy for token informativeness. It selectively applies stronger unlearning penalties to informative tokens, effectively unlearning target knowledge while better maintaining the general capabilities of the
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: f25a0a759198a2a2
---
# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens

**Conference**: ACL 2026  
**arXiv**: [2604.17785](https://arxiv.org/abs/2604.17785)  
**Code**: None  
**Area**: LLM Efficiency / Machine Unlearning  
**Keywords**: Machine Unlearning, Informative Tokens, Entropy-guided, Token Weighting, Selective Unlearning

## TL;DR

Ours proposes Entropy-guided Token Weighting (ETW), which utilizes the entropy of the prediction distribution as a proxy for token informativeness. It selectively applies stronger unlearning penalties to informative tokens, effectively unlearning target knowledge while better maintaining the general capabilities of the model.

## Background & Motivation

**Background**: LLM machine unlearning aims to selectively remove specific knowledge (e.g., private data) from a model while preserving other capabilities. Methods based on Gradient Ascent (GA) are the mainstream paradigm, but applying unlearning loss uniformly reduces model utility unnecessarily.

**Limitations of Prior Work**: Existing token-level regularization methods are either based on ground-truth confidence (WGA, SatImp), which fails to distinguish between tokens with identical confidence but different semantic importance, or rely on external language parsers (SCN/SCE from spaCy), which fail to capture contextual information and the overall prediction state of the model.

**Key Challenge**: How to accurately identify "informative tokens" (words carrying core semantic content) versus "structural tokens" (words primarily serving grammatical functions) within a sequence? Neither confidence nor part-of-speech categories are sufficient for reliable differentiation.

**Goal**: Design a more effective token-level measure of informativeness to guide selective unlearning, enabling the model to precisely unlearn key information while minimizing losses in general capability.

**Key Insight**: Utilize the entropy of the model's prediction distribution—structural tokens (e.g., "the") have high prediction certainty and low entropy, while informative tokens (e.g., the name "Carmen") have multiple plausible alternatives and high entropy.

**Core Idea**: Entropy utilizes the probability distribution over the entire vocabulary rather than just the ground-truth confidence, providing a richer representation of informativeness that more accurately distinguishes between informative and structural tokens.

## Method

### Overall Architecture

The problem ETW addresses is that standard Gradient Ascent (GA) unlearning spreads the unlearning loss uniformly across every token in a sequence, even applying heavy unlearning to structural words like "the" and "of." Consequently, core information is not thoroughly forgotten, while the general capability of the model is unnecessarily degraded. ETW's mechanism adds a token-level weighting layer to the standard unlearning workflow: the forget set undergoes weighted Gradient Ascent, while the retain set follows standard cross-entropy. Weights are directly determined by the entropy of the model's prediction distribution: tokens with high entropy (informative) receive a heavier unlearning penalty, while tokens with low entropy (structural) remain largely unchanged. This modification introduces no external tools or reference models; it only introduces a temperature $T$ into the softmax for calculating entropy, effectively replacing the "uniform brush" of previous methods with a "brush that allocates intensity based on informativeness."

### Key Designs

**1. Entropy-guided token weight calculation: Using the distribution over the entire vocabulary, not just GT confidence, to judge token "importance"**

Existing token-level regularization (WGA, SatImp) relies on ground-truth confidence weighting. However, confidence only reflects the probability of the correct answer, failing when two tokens have the same confidence but vastly different semantic importance. ETW uses the entropy of the prediction distribution instead: for each position $i$, it calculates $H(y_i\mid y_{<i},\mathbf{x};\hat{\boldsymbol{\theta}})$, then normalizes it so that the sum of weights equals the sequence length $n$:

$$\omega_i^{\text{ETW}}=n\cdot H_i\Big/\sum_j H_j$$

Entropy reflects the dispersion of probability mass across all candidate tokens—structural tokens like "the," where the next word is nearly certain, have low entropy, whereas informative tokens like the name "Carmen," which have several plausible alternatives, have high entropy. Even if two tokens have identical GT confidence, their entropy can still differ within the range of $H_{\max}-H_{\min}$, providing a discriminative power that confidence lacks.

**2. Stop-gradient weight calculation: Ensuring weight calculation and unlearning do not interfere**

If the weights themselves are involved in backpropagation, the weight calculation could inject additional gradient noise into the unlearning training. ETW uses a stop-gradient parameter copy $\hat{\boldsymbol{\theta}}$ to calculate entropy, obtaining the entropy from the probability distribution after a softmax with temperature $T$. In this way, the weight is merely a passively read scalar, completely decoupling the decision of "how hard to forget each token" from the "actual parameter update."

**3. Normalization to maintain loss scale: Redistributing unlearning intensity among tokens without changing the total amount**

If the total scale of the unlearning loss changes after weighting, an additional hyperparameter would be required to adjust the intensity, increasing the tuning burden. ETW's normalization ensures $\sum_i\omega_i=n$, making the total weighted unlearning loss equivalent to unweighted GA. Only the distribution of intensity across tokens changes, while the total workload remains constant. Because of this, ETW can replace uniform weights in existing methods without modifying any other hyperparameters, making it a truly plug-and-play regularizer.

### Loss & Training

The total unlearning objective is $\mathcal{L} = \mathcal{L}_r + \lambda \mathcal{L}_f$, where the retain loss $\mathcal{L}_r$ is standard cross-entropy and the forget loss $\mathcal{L}_f = \sum_i \omega_i^{\text{ETW}} \log p(y_i | y_{<i}, \mathbf{x}; \boldsymbol{\theta})$ is the ETW-weighted Gradient Ascent loss, with $\lambda$ controlling the unlearning intensity.

## Key Experimental Results

### Main Results (TOFU Forget 10% - Llama 3.2-1B)

| Method | -log(FQ) ↓ | ΔMU ↓ | Agg. ↓ | |Priv.| ↓ |
|------|-----------|-------|--------|-----------|
| GA | 2.639 | 4.271 | 11.273 | 48.59 |
| WGA | 2.309 | 5.365 | 12.388 | 3.14 |
| SatImp | 2.871 | 5.258 | 15.093 | 19.65 |
| SCN | 2.754 | 3.548 | 9.772 | 39.98 |
| **ETW** | **0.492** | **3.471** | **1.707** | 9.56 |

### Ablation Study (ROC-AUC - Distinguishing Informative/Structural Tokens)

| Method | AUC |
|------|-----|
| ETW | Highest (Exceeds second best by ≥ 0.06) |
| SCN | Second best |
| Imp | 0.66 |
| WGA/TNPO | ≤ Random |

### Key Findings
- ETW significantly outperforms all baselines in ROC-AUC, performing best in distinguishing between informative and structural tokens.
- In visualization analysis, ETW is the only method that successfully identifies core answer segments (e.g., "Love Inspired").
- Tokens with the same confidence can have vastly different entropy values due to different distributions of non-GT tokens, validating the advantage of entropy as a proxy for informativeness.
- Models after ETW unlearning have informative token prediction probabilities closest to those of models trained from scratch.

## Highlights & Insights
- Precise motivation: Rigorously proves that entropy provides a richer representation space than confidence through mathematical analysis of $H_{\min}$ and $H_{\max}$.
- Extremely simple method: Requires only calculating and normalizing the entropy of the prediction distribution, without external tools, reference models, or extra hyperparameters.
- Intuitive and powerful visualization analysis: Clearly demonstrates token-level differences between methods.
- Normalization design: Makes ETW a plug-and-play regularizer.

## Limitations & Future Work
- Experiments were mainly validated on the TOFU dataset; generalization to larger scales and more diverse unlearning scenarios (such as safety alignment) needs verification.
- Entropy calculation requires an additional forward pass (softmax over the entire vocabulary), which may incur computational overhead for models with large vocabularies.
- The potential for integration with DPO-style unlearning methods has not been discussed.
- When information is uniformly distributed across a sequence, ETW degrades to uniform weighting and may lose its advantage.

## Related Work & Insights
- **vs WGA/TNPO**: These methods weight based on GT confidence and cannot distinguish between tokens with different semantic importance but the same confidence.
- **vs SCN/SCE (spaCy)**: Binary masks based on part-of-speech ignore context and may incorrectly label entities in repeated question content.
- **vs SatImp**: Combines confidence and complementary values but is limited by the upper bound of confidence's discriminative ability.
- **Inspiration**: The application of entropy in token-level control can be extended to other scenarios such as fine-tuning and reinforcement learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of using entropy to measure token informativeness is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient quantitative (ROC-AUC) and qualitative (visualization) analysis with comprehensive baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous motivation logic, progressing from intuition to mathematical analysis to experiments.
- Value: ⭐⭐⭐⭐ Simple and effective method with practical guiding significance for LLM unlearning and token-level control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)

</div>

<!-- RELATED:END -->
