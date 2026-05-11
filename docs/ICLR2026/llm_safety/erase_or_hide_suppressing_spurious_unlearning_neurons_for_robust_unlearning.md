---
title: >-
  [Paper Note] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning
description: >-
  [ICLR 2026][LLM Safety][machine unlearning] This paper exposes the "shallow alignment" problem in mainstream LLM unlearning methods — rather than truly erasing target knowledge…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "machine unlearning"
  - "spurious neurons"
  - "shallow alignment"
  - "attribution"
  - "privacy"
date: 2026-05-08
content_hash: 6787deef981e0138
---

# Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning

**Conference**: ICLR 2026
**arXiv**: [2509.22263](https://arxiv.org/abs/2509.22263)
**Code**: None
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, spurious neurons, shallow alignment, attribution, privacy

## TL;DR
This paper exposes the "shallow alignment" problem in mainstream LLM unlearning methods — rather than truly erasing target knowledge, these methods generate "spurious unlearning neurons" that suppress its expression, allowing the knowledge to be readily recovered via subsequent fine-tuning. The proposed method, Ssiuu, employs attribution-guided regularization to prevent the growth of negative influence, achieving robust unlearning.

## Background & Motivation
**Background**: LLM training data may contain private information, and machine unlearning methods aim to remove specific knowledge from model parameters. Mainstream approaches include Gradient Ascent (GA), Gradient Difference (GD), DPO, NPO, and RMU.

**Limitations of Prior Work**: Prior studies have observed that unlearned models are vulnerable to knowledge recovery via prompt-based attacks or continued training, yet the underlying reasons remain unclear.

**Key Challenge**: Unlearning methods prevent the model from outputting target knowledge, but the question is whether the knowledge is truly forgotten or merely concealed. If the original neurons encoding the knowledge remain intact and only new suppressive neurons are introduced, the knowledge has not been erased.

**Goal**: (1) Diagnose whether unlearning constitutes "erasure" or "hiding"; (2) Design a method that genuinely erases knowledge.

**Key Insight**: Attribution methods are used to quantify the change in each neuron's positive/negative contribution to target knowledge — positive influence should decrease (knowledge erased) while negative influence should not increase (no spurious suppression).

**Core Idea**: Existing unlearning methods increase negative influence rather than reducing positive influence ("hiding" rather than "erasing"). Ssiuu achieves genuine unlearning by regularizing against the growth of negative influence.

## Method

### Overall Architecture
Attribution analysis is first applied to diagnose the problem (explaining why existing methods exhibit shallow alignment), followed by the design of the Ssiuu regularization term to address it.

### Key Designs

1. **Attribution-Driven Unlearning Diagnosis**:

    - Function: Quantifies the change in each neuron's positive/negative contribution to target knowledge before and after unlearning.
    - Mechanism: The attribution score is defined as $A_{\theta_i,k}^{(x,y)} = h_{\theta_i,k} \times \frac{\partial P_\theta(y|x)}{\partial h_{\theta_i,k}}$. The reduction in positive influence $D_i^+$ and the increase in negative influence $D_i^-$ measure the degree of "erasure" and "hiding," respectively.
    - Key Findings: Across GA, GD, DPO, NPO, and RMU, $D_i^-$ (negative increase) substantially exceeds $D_i^+$ (positive reduction), indicating that these methods primarily generate spurious unlearning neurons to conceal knowledge.

2. **Two Retraining Attack Scenarios**:

    - **Harmful attack**: A small subset ($p = 0.1$ or $0.3$) of the forget set is used to fine-tune the model, and recovery of disjoint forgotten knowledge is assessed.
    - **Benign attack**: Unrelated instruction-following data (e.g., Alpaca) is used for fine-tuning, and inadvertent knowledge recovery is examined.
    - Results: All methods exhibit substantial knowledge recovery under both attacks (up to >75%), confirming shallow alignment.

3. **Ssiuu Regularization Method**:

    - Function: Prevents the growth of negative influence during the unlearning process.
    - Loss function: $\arg\min_{\theta^t} \mathcal{L}_{\theta^t} + \lambda \sum_{i \in \mathcal{I}^-} \sum_{(x,y) \in C_f} \|A_{\theta_i^{t-1}}^{(x,y)} - A_{\theta_i^t}^{(x,y)}\|_2$
    - $\mathcal{I}^-$ denotes the set of neurons with negative attribution at the current step. The regularization term minimizes the change in negative attribution across consecutive steps, preventing the emergence of new suppressive neurons.
    - Efficiency optimization: Parameter-times-gradient approximation replaces per-token attribution computation.

## Key Experimental Results

### Main Results: FaithUn Dataset (Llama-3.2-3B)

| Method | FS↓ | RS↑ | Harmful p=0.1↓ | Harmful p=0.3↓ | Benign↓ |
|--------|-----|-----|----------------|----------------|---------|
| GA | 0.0 | 58.4 | 68.4 | 73.3 | 16.7 |
| GD | 0.0 | 81.0 | 48.1 | 54.8 | 33.3 |
| DPO | 0.0 | 81.5 | 31.6 | 46.7 | 15.3 |
| NPO | 0.0 | 77.6 | 18.3 | 18.8 | 18.6 |
| RMU | 0.0 | 77.8 | 52.6 | 75.5 | 14.3 |
| **Ssiuu** | **0.0** | **84.7** | **14.8** | **14.3** | **13.3** |

### Key Findings
- All mainstream unlearning methods exhibit recovery rates of 18–75% under harmful attacks; Ssiuu reduces this to 14–15%.
- Ssiuu simultaneously achieves the highest retain score (84.7%), demonstrating no sacrifice of general capability.
- Attribution analysis confirms that Ssiuu produces a substantially larger reduction in positive influence (genuine erasure) relative to the increase in negative influence (spurious suppression), whereas the opposite holds for all other methods.
- Findings are consistent on Qwen-2.5-3B and the TOFU dataset.
- A 99.63% monotonic decline phenomenon (resonating with SquaredPO from a prior batch) suggests that once a knowledge-hiding pattern is established, subsequent training continuously reinforces it.

## Highlights & Insights
- **Precise diagnosis of "erase vs. hide"**: By translating the vague question of unlearning effectiveness into quantifiable metrics ($D_i^+$ vs. $D_i^-$) via attribution analysis, this work introduces an original analytical framework generalizable to other unlearning evaluation settings.
- **Universality of shallow alignment**: Five methodologically distinct approaches — GA, GD, DPO, NPO, and RMU — all exhibit the same pattern, indicating a structural deficiency in the current unlearning paradigm rather than a flaw of any single method.
- **Real-world threat of benign attacks**: The fact that even benign instruction fine-tuning can recover forgotten knowledge implies that openly released unlearned models may leak private information under ordinary use — a compelling security warning.

## Limitations & Future Work
- Validation is limited to 3B-scale models; unlearning dynamics in larger models (7B+) may differ.
- Ssiuu incurs additional computational cost from attribution score computation, despite the parameter-times-gradient approximation.
- Evaluation relies primarily on accuracy-based metrics without assessing finer-grained knowledge residuals (e.g., representation-level probing).
- The impact of the hyperparameter $\lambda$ on performance is not sufficiently discussed.

## Related Work & Insights
- **vs. RMU (Li et al., 2024)**: RMU removes knowledge via representation engineering, yet this paper demonstrates that it also generates spurious unlearning neurons, with recovery rates reaching 75.5% under harmful attacks.
- **vs. DPO-based unlearning**: DPO performs unlearning through preference optimization, but exhibits significant increases in negative influence, making it the method most prone to "hiding" rather than erasing.
- **Resonance with AlphaSteer**: AlphaSteer preserves benign activations in the null space; Ssiuu preserves negative influence from growing in attribution space. Both approaches share the principle that "what should not change must not be changed."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of "spurious unlearning neurons" is original and compelling; the attribution-based diagnostic framework is a genuine contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 2 models × 2 datasets × 6 baselines × 3 attack scenarios — comprehensive, though model scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; the "Erase or Hide" framing provides a strong and incisive entry point.
- Value: ⭐⭐⭐⭐⭐ Exposes a fundamental flaw in current unlearning research with significant implications for privacy and security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[ICLR 2026\] Gaussian Certified Unlearning in High Dimensions: A Hypothesis Testing Approach](gaussian_certified_unlearning.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)
- [\[ICLR 2026\] Attention Smoothing Is All You Need For Unlearning](attention_smoothing_is_all_you_need_for_unlearning.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](unlearning_evaluation_through_subset_statistical_independence.md)

</div>

<!-- RELATED:END -->
