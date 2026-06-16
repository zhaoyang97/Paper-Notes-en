---
title: >-
  [Paper Note] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens
description: >-
  [ACL 2026][LLM Safety][Paper Note] The authors propose Entropy-guided Token Weighting (ETW), which utilizes the entropy of the predictive distribution as a proxy for token informativeness. By selectively applying stronger unlearning penalties to informative tokens, the method effectively removes target knowledge while better preserving the model's gener
tags:
  - ACL 2026
  - LLM Safety
date: 2026-05-08
content_hash: da8eb4b3ca63423b
---
# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens

**Conference**: ACL 2026  
**arXiv**: [2604.17785](https://arxiv.org/abs/2604.17785)  
**Code**: None  
**Area**: LLM Efficiency / Machine Unlearning  
**Keywords**: Machine Unlearning, Informative Tokens, Entropy Guidance, Token Weighting, Selective Unlearning

## TL;DR

The authors propose Entropy-guided Token Weighting (ETW), which utilizes the entropy of the predictive distribution as a proxy for token informativeness. By selectively applying stronger unlearning penalties to informative tokens, the method effectively removes target knowledge while better preserving the model's general capabilities.

## Background & Motivation

**Background**: LLM machine unlearning aims to selectively remove specific knowledge (e.g., private data) from a model while retaining other capabilities. Gradient Ascent (GA) based methods are the dominant paradigm, but applying unlearning losses uniformly often leads to unnecessary degradation of model utility.

**Limitations of Prior Work**: Existing token-level regularization methods either rely on ground-truth confidence (WGA, SatImp), which fails to distinguish between tokens with identical confidence but different semantic importance, or depend on external language parsers (SCN/SCE using spaCy), which cannot capture contextual information and the model's overall predictive state.

**Key Challenge**: How to accurately identify "informative tokens" (words carrying core semantic content) versus "structural tokens" (words primarily serving grammatical functions) within a sequence? Neither confidence levels nor part-of-speech categories are sufficient for reliable differentiation.

**Goal**: To design a more effective token-level measure of informativeness to guide selective unlearning, enabling precise removal of key information while minimizing the loss of general capabilities.

**Key Insight**: Leveraging the entropy of the model's predictive distribution—structural tokens (e.g., "the") have high prediction certainty and low entropy, whereas informative tokens (e.g., the name "Carmen") have multiple plausible alternatives and thus exhibit high entropy.

**Core Idea**: Entropy utilizes the probability distribution across the entire vocabulary rather than just the ground-truth confidence. This provides a richer representation of informativeness, allowing for more accurate differentiation between informative and structural tokens.

## Method

### Overall Architecture

ETW addresses the limitation where standard Gradient Ascent (GA) distributes unlearning loss uniformly across every token in a sequence, including structural words like "the" or "of." This results in incomplete removal of core information and collateral damage to general utility. ETW introduces a token-level weight layer into the standard unlearning pipeline: weighted GA is applied to the forget set, while standard cross-entropy is used for the retain set. Weights are directly determined by the entropy of the predictive distribution—high-entropy (informative) tokens receive heavier unlearning penalties, while low-entropy (structural) tokens remain largely unaffected. This modification requires no external tools or reference models; it simply introduces a temperature parameter $T$ into the softmax during entropy calculation, effectively replacing the "uniform brush" of traditional methods with an "informative-aware" one.

### Key Designs

**1. Entropy-Guided Token Weight Calculation: Utilizing the full vocabulary distribution instead of just GT confidence**

Existing token-level regularization (WGA, SatImp) relies on ground-truth (GT) confidence weighting. However, confidence only considers the probability of the correct answer, failing when two tokens have the same confidence but vastly different semantic importance. ETW calculates the entropy $H(y_i\mid y_{<i},\mathbf{x};\hat{\boldsymbol{\theta}})$ for each position $i$, then normalizes it so the sum of weights equals the sequence length $n$:

$$\omega_i^{\text{ETW}}=n\cdot H_i\Big/\sum_j H_j$$

Entropy reflects how probability mass is spread across all candidate tokens. Structural tokens like "the," where the next word is nearly certain, have low entropy. Conversely, informative tokens like the name "Carmen," which have several plausible alternatives, exhibit high entropy. Even if two tokens share identical GT confidence, their entropy can differ significantly within the range of $H_{\max}-H_{\min}$, capturing discriminative information that confidence ignores.

**2. Stop-Gradient Weight Calculation: Decoupling weight estimation from unlearning updates**

If the weights themselves are involved in backpropagation, the weight calculation could inject additional gradient noise into the unlearning training. ETW uses a stop-gradient parameter copy $\hat{\boldsymbol{\theta}}$ to compute the entropy. The probability distribution is obtained via softmax with temperature $T$ before calculating entropy. This treats the weight as a passively observed scalar, fulfilling the decoupling of "deciding how much to forget" from "actually updating parameters."

**3. Normalization to Maintain Loss Scale: Redistributing unlearning intensity without changing total magnitude**

If weighting changes the overall scale of the unlearning loss, an additional hyperparameter would be needed to recalibrate the intensity. ETW’s normalization ensures $\sum_i\omega_i=n$, keeping the total weighted unlearning loss equivalent to that of unweighted GA. Only the distribution of intensity across tokens changes. Consequently, ETW acts as a plug-and-play regularizer that can replace uniform weights without modifying other hyperparameters.

### Loss & Training

The total unlearning objective is $\mathcal{L} = \mathcal{L}_r + \lambda \mathcal{L}_f$, where the retain loss $\mathcal{L}_r$ is standard cross-entropy, and the forget loss $\mathcal{L}_f = \sum_i \omega_i^{\text{ETW}} \log p(y_i | y_{<i}, \mathbf{x}; \boldsymbol{\theta})$ is the ETW-weighted gradient ascent loss. $\lambda$ controls the unlearning intensity.

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
| SCN | Second Best |
| Imp | 0.66 |
| WGA/TNPO | ≤ Random |

### Key Findings
- ETW outperforms all baselines in ROC-AUC, demonstrating superior performance in distinguishing informative tokens from structural ones.
- In visualization analysis, ETW is the only method that successfully identifies core answer segments (e.g., "Love Inspired").
- Tokens with identical confidence can have vastly different entropy values due to the distribution of non-GT tokens, validating entropy as a superior proxy for informativeness.
- Models unlearned via ETW exhibit predictive probabilities for informative tokens that are closest to those of a model trained from scratch.

## Highlights & Insights
- **Rigorous Motivation**: Mathematical analysis of $H_{\min}$ and $H_{\max}$ strictly proves that entropy provides a richer representation space than confidence.
- **Simplicity**: The method is extremely concise, requiring only entropy calculation and normalization without external tools, reference models, or extra hyperparameters.
- **Strong Visualization**: Qualitative analysis clearly illustrates the token-level differences between methods.
- **Plug-and-Play**: The normalization design makes ETW a seamless regularizer for existing unlearning frameworks.

## Limitations & Future Work
- Experiments were primarily conducted on the TOFU dataset; generalization to larger scales and diverse scenarios (e.g., safety alignment) remains to be verified.
- Entropy calculation requires an additional forward pass (softmax over the entire vocabulary), which may incur overhead for models with large vocabularies.
- The potential for combining ETW with DPO-style unlearning methods has not been explored.
- In sequences where information is uniformly distributed, ETW reverts to uniform weighting and may provide no advantage.

## Related Work & Insights
- **vs WGA/TNPO**: These methods use GT-confidence weighting, failing to distinguish between tokens with different semantic importance but identical confidence.
- **vs SCN/SCE (spaCy)**: POS-based binary masks ignore context and may incorrectly label entities within repeated question content.
- **vs SatImp**: While combining confidence and complementary values, it remains limited by the upper bound of confidence-based discriminative power.
- **Insight**: The application of entropy for token-level control can be extended to other domains such as fine-tuning and reinforcement learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of using entropy to measure token informativeness is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative (ROC-AUC) and qualitative (visualization) analyses across multiple baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous motivation, progressing logically from intuition to mathematical analysis to experiments.
- Value: ⭐⭐⭐⭐ A simple yet effective method with practical implications for LLM unlearning and token-level control.

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
