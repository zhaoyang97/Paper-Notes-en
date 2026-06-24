---
title: >-
  [Paper Note] Improving LLM Safety Alignment with Dual-Objective Optimization
description: >-
  [ICML 2025][LLM Safety][Safety Alignment] Through gradient analysis, this work reveals two major limitations of DPO in safety alignment (learning rate saturation and poor OOD generalization). It proposes the DOOR/W-DOOR dual-objective optimization framework (incorporating robust refusal training, targeted unlearning of harmful knowledge, and token-level weighting). On Llama-3-8B and Gemma-2-2B, this approach significantly reduces the attack success rate (ASR) of multiple jail…
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Safety Alignment"
  - "Jailbreak Attack Defense"
  - "DPO Limitations"
  - "Dual-Objective Optimization"
  - "Token-level Weighting"
date: 2026-05-08
content_hash: 2e7b5be71a97412d
---

# Improving LLM Safety Alignment with Dual-Objective Optimization

**Conference**: ICML 2025  
**arXiv**: [2503.03710](https://arxiv.org/abs/2503.03710)  
**Code**: [https://github.com/wicai24/DOOR-Alignment](https://github.com/wicai24/DOOR-Alignment)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Safety Alignment, Jailbreak Attack Defense, DPO Limitations, Dual-Objective Optimization, Token-level Weighting

## TL;DR
Through gradient analysis, this work reveals two major limitations of DPO in safety alignment (learning rate saturation and poor OOD generalization). It proposes the DOOR/W-DOOR dual-objective optimization framework (incorporating robust refusal training, targeted unlearning of harmful knowledge, and token-level weighting). On Llama-3-8B and Gemma-2-2B, this approach significantly reduces the attack success rate (ASR) of multiple jailbreak styles (such as prefilling, suffix, and multi-turn attacks) while preserving general capabilities.

## Background & Motivation
DPO has become one of the primary methods for LLM safety alignment. However, DPO-aligned models remain vulnerable to jailbreak attacks. While prior works have noted the weaknesses of DPO in safety contexts, a systematic theoretical analysis has been lacking.

This paper pinpointed two systematic flaws through mathematical analysis of DPO gradient dynamics:

**Learning Rate Imbalance**: The effective learning rate of DPO, $r_\theta^\beta(y^h|x)/[r_\theta^\beta(y^s|x)+r_\theta^\beta(y^h|x)]$, decays exponentially ($\lesssim e^{-\beta C}$) as the gap between safe and harmful responses widens. This leads to premature saturation of the logit growth for safety tokens. The model stops optimizing once it is "good enough," failing to push the refusal probability sufficiently high.

**OOD Generalization Failures**: DPO does not explicitly penalize out-of-distribution (OOD) responses. Its gradient terms can be positively correlated with OOD data, which inadvertently increases the logits of OOD responses and correspondingly lowers the probability of safe responses. This issue is critical given the nearly limitless attack surface of text-based interfaces.

A crucial observation is that successful jailbreak attacks often do not completely bypass the refusal mechanism. Instead, they induce the model to generate a portion of harmful content before continuing along the harmful direction (prefilling attacks). Existing methods, which atomically classify responses as either safe or unsafe, fail to handle such "partial deviation" cases.

Core Idea: **Decouple safety alignment into two complementary objectives—robust refusal training (learning to refuse even after generating partial harmful content) + targeted unlearning of harmful knowledge (reducing the generation probability of harmful content at the source), and reinforce key refusal tokens through token-level weighting.**

## Method

### Overall Architecture
The DOOR framework consists of three core components consolidated into a unified loss function:
1. Robust Refusal Training: Employs data augmentation and SFT to train the model to "halt at the brink" even after already generating partial harmful content.
2. Targeted Unlearning with NPO: Uses negative preference optimization to reduce the generation probability of harmful content.
3. Token-level Weighted Optimization (W-DOOR Extension): Allocates token-level weights based on rewards to reinforce key refusal tokens.

### Key Designs
1. **Robust Refusal Training + Data Augmentation**:

    - Function: Trains the model to switch to a refusal response even after generating partial harmful content.
    - Mechanism: For each harmful prompt $x$, the first $k$ tokens of the known harmful response $y^h$ are appended to the prompt as an augmented input $x' = x \oplus y^h_{<k}$ (where $k$ is uniformly sampled from $\{1,...,C\}$), while the target remains the safe refusal response $y^s$. Minimize:
    $\mathbb{E}_{(x,y^h,y^s)\sim\mathcal{D}, k\sim\text{Uniform}[1,C]} \left[-\log\pi_\theta(y^s \mid x \oplus y^h_{<k})\right]$
    - Design Motivation: Directly simulates prefilling attack scenarios, forcing the model to learn to "brake" at any position. This is much deeper than traditional SFT, which only trains refusal immediately following the prompt.

2. **Targeted Unlearning with NPO**:

    - Function: Actively removes harmful knowledge pathways from the model, reducing the probability of generating harmful content.
    - Mechanism: Employs Negative Preference Optimization (NPO) to penalize harmful outputs relative to a reference model:
    $\mathcal{L}_{\text{NPO}} = -\frac{2}{\beta}\mathbb{E}_{(x,y^h)\sim\mathcal{D}}\left[\log\sigma\left(-\beta\log\frac{\pi_\theta(y^h \mid x)}{\pi_{\text{ref}}(y^h \mid x)}\right)\right]$
    - Design Motivation: Naive gradient ascent severely damages the general capabilities of the model (experimentally confirmed). NPO avoids training instability by applying constraints relative to the reference model.
    - Harmful response data is scalably generated by first fine-tuning a "jailbroken model."

3. **Token-level Weighting (W-DOOR)**:

    - Function: Assigns different weights to different tokens during SFT refusal training, prioritizing the reinforcement of key refusal tokens.
    - Mechanism: Based on the principles of KL-regularized optimization, token-level reward is defined as $r(s_t, a_t) = \log\frac{\pi^*(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}$, with the weight defined as:
    $\beta_t = \exp\left(\frac{1}{\tau}r(s_t, a_t)\right) = \left(\frac{\pi^*(y_t|x,y_{<t})}{\pi_{\text{ref}}(y_t|x,y_{<t})}\right)^{1/\tau}$
      where $\pi^*$ is approximated by a DPO-aligned model, and $\tau$ is a temperature parameter.
    - Design Motivation: Key refusal tokens (such as "Sorry", "cannot", etc.) exhibit the largest probability ratio between $\pi^*$ and $\pi_{\text{ref}}$, and should thus receive the strongest gradient updates. Unimportant tokens (which might overlap with harmful tokens) should not be excessively reinforced to avoid interfering with the unlearning process.

### Loss & Training
**Total DOOR Loss**:
$$\mathcal{L}_{\text{DOOR}} = \mathbb{E}_{(x,y^s,y^h),k}\left[-\log\pi_\theta(y^s \mid x\oplus y^h_{<k}) - \frac{2}{\beta}\log\sigma\left(-\beta\cdot\log\frac{\pi_\theta(y^h|x)}{\pi_{\text{ref}}(y^h|x)}\right)\right]$$

**Total W-DOOR Loss** (with token-level weighting):
$$\mathcal{L}_{\text{W-DOOR}} = \mathbb{E}\left[\sum_{t=1}^T\left(-\beta_t\log\pi_\theta(y_t^s|x,y_{<t}^s) - \frac{2}{\beta}\log\sigma\left(-\beta\log\frac{\pi_\theta(y_t^h|x,y_{<t}^h)}{\pi_{\text{ref}}(y_t^h|x,y_{<t}^h)}\right)\right)\right]$$

**Capability Retention Loss**:
$$\mathcal{L}_{\text{Total}} = \alpha\mathcal{L}_{\text{Align}} + (1-\alpha)\mathcal{L}_{\text{Retain}}$$

Training configurations: 10 epochs, H100 GPUs, batch size of 2, learning rate of $1\times10^{-5}$, $\beta=0.5$, $\alpha=0.2$, and $\tau=5$ in W-DOOR.

**Divergence Analysis of DOOR Gradient Advantages**:
- The effective learning rate for safe tokens remains constant at 1 (whereas it exponentially decays in DPO), ensuring continuous reinforcement of refusal behavior.
- The gradient includes the term $\mathbb{E}_{y\sim\pi_\theta}[\nabla s_{\theta,y}(x)]$, which acts as regularization on the logits of all responses and improves OOD generalization.

## Key Experimental Results

### Main Results

| Method | Multi-turn ASR↓ | Prefilling ASR↓ | GCG ASR↓ | AutoDAN ASR↓ | HellaSwag↑ | XSTest Refusal↓ |
|------|----------------|----------------|----------|-------------|-----------|----------------|
| **Llama-3-8B** | | | | | | |
| Original | 0.521 | 0.547 | 0.307 | 0.198 | 0.577 | 0.409 |
| DPO | 0.521 | 0.210 | 0.133 | 0.138 | 0.564 | 0.456 |
| RR | 0.213 | 0.338 | 0.045 | 0.000 | 0.574 | 0.404 |
| SFT (Qi et al.) | 0.511 | 0.071 | 0.143 | 0.136 | 0.564 | 0.396 |
| DOOR | 0.489 | 0.055 | 0.093 | 0.095 | 0.565 | 0.407 |
| W-DOOR | 0.447 | 0.034 | 0.093 | 0.088 | 0.573 | 0.440 |
| **Gemma-2-2B** | | | | | | |
| Original | 0.554 | 0.346 | 0.190 | 0.098 | 0.536 | 0.422 |
| DPO | 0.446 | 0.060 | 0.148 | 0.048 | 0.478 | 0.438 |
| SFT (Qi et al.) | 0.505 | 0.010 | 0.156 | 0.020 | 0.513 | 0.400 |
| DOOR | 0.525 | 0.009 | 0.106 | 0.015 | 0.504 | 0.407 |
| W-DOOR | 0.347 | 0.005 | 0.103 | 0.020 | 0.507 | 0.440 |

### Ablation Study

| Configuration | Key Metrics | Notes |
|------|---------|------|
| Data Augmentation vs. No Augmentation | Significant decrease in ASR | Augmentation is highly effective for SFT-based methods, but DPO benefits less from it |
| Naive Gradient Ascent Unlearning | Severe drop in HellaSwag/MMLU | Naive GA leads to model degradation; NPO is more stable |
| W-DOOR $\tau=1/3/5/10$ | Robustness insensitive to $\tau$ | Exponential/sigmoid variants perform similarly |
| Replacing Reference Model with Jailbroken Model | Slight improvement | Provides a stronger contrastive signal but requires more consideration |

### Key Findings
- **Data augmentation is key**: Exposing the model to harmful prefixes while training it to refuse is the core factor in improving robustness against prefilling attacks.
- **DPO indeed lags behind SFT-based methods in learning refusal**: The ASR pattern of DPO is closer to NPO than SFT, validating the theoretical analysis.
- **W-DOOR demonstrates a unique advantage in multi-turn attacks**: While the ASR of other methods increases with the number of turns, W-DOOR remains stable or even decreases.
- **W-DOOR preserves general capabilities best**: It maintains Hellaswag accuracy with almost no decline, whereas DPO suffers the largest loss in capability.
- **KL divergence is strongly correlated with robustness**: W-DOOR shows the highest KL divergence relative to the base model, and its impact on deep-layer token positions is more uniform.
- **Pareto analysis**: Most robustness gains are obtained in the 1st epoch, while subsequent training mainly restores general capabilities.
- **Over-refusal does not increase with training**: Prolonging training actually reduces both the ASR and the over-refusal rate.

## Highlights & Insights
- **Gradient decomposition analysis** is compelling: It mathematically and clearly details the two fundamental flaws of DPO in safety scenarios, providing theoretical backing rather than just empirical observation.
- **The effective learning rate of DOOR's gradient remains constant at 1**: This is the most crucial advantage over DPO, ensuring the probability of safe responses can continuously grow.
- **Token-level weighting offers general value**: Automatically identifying key refusal tokens through the ratio $\pi^*/\pi_{\text{ref}}$ eliminates the need for manual design.
- **W-DOOR avoids gradient conflicts between refusal training and unlearning training**: By discounting the weights of unimportant tokens, it avoids excessively reinforcing unimportant safe tokens that might overlap with harmful ones.
- **t-SNE visualization** intuitively demonstrates that W-DOOR achieves "deeper" alignment—with safer/harmful representations being more clearly separated.
- Prefilling ASR serves as a reliable proxy metric for overall robustness, simplifying evaluation.

## Limitations & Future Work
- Defense effectiveness against multi-turn attacks remains limited (all methods exhibit relatively high ASR), possibly requiring multi-turn or long-context training data.
- Uniformly sampling prefix length $k$ in data augmentation may lead to over-refusal (as some benign prefixes might be falsely learned as harmful signals).
- Although performance is insensitive to the token-level weight parameter $\tau$, a principled method for choosing it is still lacking.
- Validated only on Gemma-2-2B and Llama-3-8B; has not been tested on larger-scale models.
- The training dataset is small (~400 safe + 400 harmful + 400 general); performance under large-scale data remains to be validated.
- W-DOOR's over-refusal rate is slightly higher than DOOR's, suggesting the weighted design introduces a subtle trade-off between safety and helpfulness.

## Related Work & Insights
- **vs DPO**: DOOR theoretically corrects the two fundamental flaws of DPO in safety scenarios, which is comprehensively validated by experiments.
- **vs Decoupled Refusal Training (Yuan et al.)**: Uses a similar data augmentation concept, but DOOR additionally incorporates NPO unlearning and token-level weighting.
- **vs Safe Unlearning (Zhang et al.)**: Also combines unlearning with SFT, but does not utilize data augmentation.
- **vs Circuit Breakers (Zou et al.)**: Formulates a representation engineering approach, and is orthogonally complementary to training-level methods.
- **vs TAR (Tamirisa et al.)**: A tamper-resistance method, but experiments show it is more vulnerable to suffix attacks.

## Rating
- Novelty: ⭐⭐⭐⭐ Gradient analysis provides solid theoretical insights, the DOOR framework is well-designed, and token-level weighting is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 attack types, 2 models, and multiple baselines, including Pareto analysis and representation visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical progression from theoretical analysis to method design and experimental verification is highly coherent, with rich and informative figures and tables.
- Value: ⭐⭐⭐⭐ Provides practical guidance for the LLM safety alignment community; DOOR/W-DOOR can be directly applied to improve existing safety training pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Safety Alignment Can Be Not Superficial With Explicit Safety Signals](safety_alignment_can_be_not_superficial_with_explicit_safety_signals.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](../../NeurIPS2025/llm_safety/learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)

</div>

<!-- RELATED:END -->
