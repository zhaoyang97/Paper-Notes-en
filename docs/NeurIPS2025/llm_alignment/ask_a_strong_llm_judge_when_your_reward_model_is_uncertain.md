---
title: >-
  [Paper Note] Ask a Strong LLM Judge when Your Reward Model is Uncertain
description: >-
  [NeurIPS 2025][LLM Alignment][Reward Model] This paper proposes an uncertainty-based routing framework that applies SNGP to a pairwise reward model for uncertainty quantification, routing high-epistemic-uncertainty samples to a strong LLM judge (DeepSeek-R1). At a judge invocation cost of only 9.2%–42.5%, the approach significantly outperforms random routing in accuracy and demonstrably improves downstream online RLHF alignment.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - Reward Model
  - LLM-as-Judge
  - Uncertainty Quantification
  - SNGP
  - Routing
  - RLHF
date: 2026-05-08
content_hash: 6d122cfac9313d90
---

# Ask a Strong LLM Judge when Your Reward Model is Uncertain

**Conference**: NeurIPS 2025
**arXiv**: [2510.20369](https://arxiv.org/abs/2510.20369)
**Code**: [GitHub](https://github.com/zhenghaoxu-gatech/uncertainty-router)
**Area**: Alignment / RLHF
**Keywords**: Reward Model, LLM-as-Judge, Uncertainty Quantification, SNGP, Routing, RLHF

## TL;DR
This paper proposes an uncertainty-based routing framework that applies SNGP to a pairwise reward model for uncertainty quantification, routing high-epistemic-uncertainty samples to a strong LLM judge (DeepSeek-R1). At a judge invocation cost of only 9.2%–42.5%, the approach significantly outperforms random routing in accuracy and demonstrably improves downstream online RLHF alignment.

## Background & Motivation
**Background**: Reward models (RMs) are a central component of RLHF, yet standard RMs (pointwise/pairwise) generalize poorly on out-of-distribution data and are susceptible to reward hacking. Strong LLM judges (e.g., DeepSeek-R1, GPT-4) provide more reliable preference judgments via chain-of-thought reasoning.

**Limitations of Prior Work**: RMs are cheap but unreliable—on the hard subset of RM-Bench, the state-of-the-art 8B RM achieves only 46.6% accuracy, worse than random guessing (50%). LLM judges are accurate but expensive—long CoT inference introduces latency tens of times greater than scalar RMs, making them infeasible for online RLHF.

**Key Challenge**: How can preference judgment accuracy be maximized under a limited LLM judge invocation budget? Random routing wastes budget on samples the RM already judges correctly.

**Goal**: Design an adaptive routing strategy that precisely identifies samples where the RM is uncertain (and thus most likely to err), routes them to a strong judge, and handles the remainder efficiently with the RM.

**Key Insight**: The approach begins from uncertainty quantification—the preference classification problem of pairwise RMs is naturally amenable to UQ methods (in contrast to pointwise RMs, where uncertainty under the Bradley–Terry model is ill-defined). SNGP is adopted to efficiently quantify epistemic uncertainty in a single forward pass without ensembling.

**Core Idea**: Equip a pairwise RM with uncertainty awareness via SNGP. Pairs with high epistemic uncertainty are automatically routed to an LLM judge; those with low uncertainty are handled directly by the RM.

## Method

### Overall Architecture
Given a prompt $x$ and two responses $y_1, y_2$, the SNGP-PM (pairwise preference model with spectral-normalized GP) computes a preference score $p$ and an uncertainty estimate $u$. If $u > \bar{u}$ (a threshold), the pair is routed to the DeepSeek-R1 judge for a more reliable judgment; otherwise, the PM result is used directly. The resulting preference difference is used to construct advantage estimates for RLOO/GRPO, driving downstream policy gradient updates.

### Key Designs

1. **SNGP-PM (Uncertainty-Aware Pairwise RM)**:

    - **Function**: Simultaneously outputs a preference score and an epistemic uncertainty estimate.
    - **Mechanism**: Spectral normalization is applied to the LLM backbone (to preserve distance awareness), followed by a GP layer (approximated via random features). The logit $g(h)$ is divided by the standard deviation $u = \sqrt{1 + \lambda \cdot \phi(h)^\top \Sigma \phi(h)}$ to obtain a calibrated preference score. $\Sigma$ denotes the GP posterior covariance and is computed during an additional frozen epoch.
    - **Design Motivation**: SNGP requires only a single model and a single forward pass—unlike MC Dropout or ensembles, which require multiple passes—so its latency is nearly identical to a standard PM. It additionally separates aleatoric uncertainty (intrinsic noise in the BT model, irreducible) from epistemic uncertainty (insufficient data coverage, addressable by the judge).

2. **Uncertainty-Based Routing Strategy**:

    - **Function**: Determines whether to use the PM or the judge based on an uncertainty threshold $\bar{u}$.
    - **Mechanism**: When $u > \bar{u}$, the sample is routed to DeepSeek-R1. The judge returns one of three labels ($y_1$ preferred, $y_2$ preferred, or tie), which are mapped to high-confidence logits or a zero logit, respectively.
    - **Design Motivation**: High epistemic uncertainty indicates OOD data, where the PM is most likely to err, enabling precise allocation of the judge budget.

3. **Pairwise Advantage Estimation (Compatible with RLOO/GRPO)**:

    - **Function**: Converts pairwise reward differences into advantages usable by policy gradient methods.
    - **Mechanism**: The RLOO advantage $A_i = \frac{1}{K-1}\sum_{j \neq i}(r(x,y_i) - r(x,y_j))$ depends only on reward differences, making it naturally compatible with pairwise PMs without requiring absolute pointwise reward values.
    - **Design Motivation**: This avoids the ill-defined uncertainty problem of pointwise RMs under the BT model (where adding an arbitrary offset $s(x)$ does not change preferences).

### Loss & Training
- **Base model**: Llama-3.1-8B-Instruct
- **Training data**: HelpSteer2-Preference (7,118 pairs with preference intensity annotations)
- **Data augmentation**: Response order is swapped with labels flipped to eliminate position bias
- **Training**: 2 epochs + 1 frozen epoch to compute the GP covariance matrix
- **Judge**: DeepSeek-R1 (78.9% accuracy on RM-Bench hard)

## Key Experimental Results

### Main Results: Routing Strategies on RewardBench

| Routing Strategy | Judge Calls | Chat Hard | Reasoning | Overall Avg (vs. Random) |
|---|---|---|---|---|
| No routing | 0 | 73.8 | 90.0 | 87.3 |
| Uncertainty routing | 274 (9.2%) | **76.8** | **93.7** | **89.2** (+1.7) |
| Random routing | 274 (9.2%) | 73.7 | 90.4 | 87.5 |
| Uncertainty routing | 1270 (42.5%) | **81.2** | **97.0** | **91.6** (+2.5) |
| Random routing | 1270 (42.5%) | 77.5 | 91.9 | 89.1 |
| DeepSeek-R1 100% | All | 85.8 | 96.9 | 92.3 |

### Ablation Study

| Configuration | Description |
|---|---|
| SNGP-PM vs. standard PM | Accuracy difference < 1%; the uncertainty component does not degrade performance |
| Threshold 1.30 vs. 1.45 | Lower threshold → more judge calls → higher accuracy, but with diminishing returns |
| Uncertainty routing vs. random routing | Uncertainty routing consistently outperforms random routing across all judge invocation ratios (+0.8–2.5 pp) |

### Key Findings
- Epistemic uncertainty is strongly negatively correlated with RM accuracy (Spearman $p < 10^{-29}$), validating the hypothesis that high uncertainty implies high error probability.
- OOD data (RewardBench, RM-Bench) systematically exhibits higher uncertainty than in-distribution data (HelpSteer2 validation set).
- With only 9.2% judge invocations, RewardBench accuracy improves from 87.3% to 89.2%, demonstrating strong cost-effectiveness.
- In downstream RLHF alignment, uncertainty routing likewise outperforms random routing, confirming end-to-end effectiveness.

## Highlights & Insights
- The theoretical analysis of **pointwise vs. pairwise RM uncertainty quantification** is particularly insightful—pointwise RMs have ill-defined uncertainty under the BT model (invariant to additive bias), which provides a principled, rather than merely empirical, justification for adopting a pairwise PM.
- The choice of **SNGP** is highly practical—single-model, single-pass inference incurs almost no additional latency compared to a standard PM, in contrast to ensembles that require $N$-fold overhead, making it suitable for online RLHF.
- The **three-label judge design** (preferred / dispreferred / tie) elegantly handles aleatoric uncertainty—ties return $\sigma^{-1}(1/2) = 0$, contributing no noisy signal.
- The routing framework is general and can be combined with any judge and any UQ method.

## Limitations & Future Work
- The judge (DeepSeek-R1) itself carries biases (e.g., length preference), so routing to the judge does not guarantee ground-truth labels.
- The GP layer in SNGP relies on random feature approximation, whose quality depends on the feature dimensionality.
- Experiments are conducted only at the 8B scale; whether larger PMs would still benefit from routing remains an open question.
- The threshold $\bar{u}$ requires manual tuning; an adaptive threshold would be more practical.
- More sophisticated routing strategies (e.g., batch selection in an active learning fashion) are not explored.

## Related Work & Insights
- **vs. LoRA Ensemble RM**: Ensemble methods require multiple models and multiple inference passes, incurring high cost; SNGP's single-pass approach is more suitable for online settings.
- **vs. Pure LLM-as-Judge RLHF**: Full judge invocation is prohibitively expensive (Table 3 shows 10× latency increase); the proposed routing scheme offers a practical compromise.
- **vs. OAIF (Online AI Feedback)**: OAIF replaces RMs with LLM judges but suffers from high latency; the proposed framework can directly improve OAIF's efficiency.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of pairwise RM + SNGP UQ + routing to an LLM judge is novel and theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage spans RM benchmarks, downstream RLHF, and ablations; large-scale model experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical motivation is clear, figures are information-dense, and Remarks are well-explained.
- **Value**: ⭐⭐⭐⭐ High practical utility—directly pluggable into existing RLHF pipelines; budget-aware judge invocation has strong engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

<!-- RELATED:END -->
