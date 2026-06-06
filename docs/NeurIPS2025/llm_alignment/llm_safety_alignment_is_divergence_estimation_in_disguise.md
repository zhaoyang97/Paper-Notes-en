---
title: >-
  [Paper Note] LLM Safety Alignment is Divergence Estimation in Disguise
description: >-
  [NeurIPS 2025][LLM Alignment][safety alignment] This paper establishes a unified theoretical framework demonstrating that alignment methods such as RLHF, DPO, KTO…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "safety alignment"
  - "divergence estimation"
  - "DPO"
  - "KTO"
  - "KLDO"
  - "representation separation"
date: 2026-05-08
content_hash: e58f2295873bb378
---

# LLM Safety Alignment is Divergence Estimation in Disguise

**Conference**: NeurIPS 2025
**arXiv**: [2502.00657](https://arxiv.org/abs/2502.00657)  
**Code**: None  
**Area**: Alignment / RLHF
**Keywords**: safety alignment, divergence estimation, DPO, KTO, KLDO, representation separation

## TL;DR
This paper establishes a unified theoretical framework demonstrating that alignment methods such as RLHF, DPO, KTO, and BCO are essentially estimating the divergence between a safe distribution $\mathcal{D}^+$ and an unsafe distribution $\mathcal{D}^-$. This perspective explains the latent-space separation phenomenon observed after alignment. Building on this insight, the paper proposes KLDO, a KL divergence-based alignment method that achieves state-of-the-art robustness across 5 models.

## Background & Motivation

**Background**: Mainstream LLM safety alignment methods include RLHF, DPO, KTO, and BCO, yet a unified theoretical explanation connecting these approaches is lacking. Prior work has observed that aligned models exhibit clearly separated clusters for safe and harmful prompts in their latent spaces.

**Limitations of Prior Work**: This "separation effect" has been leveraged in attack and defense strategies, but its root cause remains unclear—whether it is an incidental phenomenon or an intrinsic consequence of alignment. Systematic theoretical analysis of the connections among different alignment methods is also absent.

**Key Challenge**: All existing alignment methods aim to make models prefer safe responses, yet a unified perspective for understanding their shared mechanism, explaining the separation phenomenon, and guiding the design of new methods is missing.

**Goal**: ① Why does alignment lead to latent-space separation? ② What is the unified mathematical nature of different alignment methods? ③ Can this understanding guide the design of better alignment methods?

**Key Insight**: Reinterpreting alignment losses as variational problems in divergence estimation, where different divergences (TV, JS, KL) correspond to different alignment methods.

**Core Idea**: Alignment ≈ divergence estimation; separation is a natural consequence of divergence estimation; KL divergence is most sensitive to large distributional shifts and is therefore best suited for safety alignment.

## Method

### Overall Architecture
The paper builds on the mathematical framework of variational divergence estimation, connecting the loss functions of alignment methods to the divergence between $\mathcal{D}^+$ (safe/preferred distribution) and $\mathcal{D}^-$ (unsafe/dispreferred distribution). The implicit reward is defined as $r_\theta(x,y) = \beta\log(\pi_\theta(y|x)/\pi_{ref}(y|x))$.

### Key Designs

1. **Unified Divergence Estimation Perspective (Theorem 4.1)**:

    - Function: Proves that each alignment method corresponds to a specific divergence estimate at its optimal solution.
    - Mechanism: $\mathcal{L}_{KTO}(\theta^*) = -\mathbb{D}_{TV}(\mathcal{D}^+\|\mathcal{D}^-) + 1$, $\mathcal{L}_{BCO}(\theta^*) = \ln 4 - 2\cdot\mathbb{D}_{JS}$, $\mathcal{L}_{DPO}(\theta^*) = \Omega(-\mathbb{D}_{TV})$.
    - Design Motivation: The mathematical properties of divergences (convexity, sensitivity) are used to compare the relative merits of each method.

2. **Analysis of DPO Saturation**:

    - Function: Reveals that the implicit divergence of DPO saturates in regions of large distributional shift.
    - Mechanism: $\mathbb{D}_{DPO}$ exhibits an S-shaped curve that saturates at both ends, causing sensitivity to drop sharply when the gap between safe and unsafe distributions is large.
    - Design Motivation: Provides a fundamental theoretical explanation for DPO's relatively poor performance in safety alignment.

3. **KLDO (KL-Divergence Optimizer)**:

    - Function: Proposes a new alignment loss based on the Donsker–Varadhan variational representation.
    - Mechanism: $\mathcal{L}_{KLDO}(\theta) = -\mathbb{E}_{\mathcal{D}^+} r_\theta + \ln\mathbb{E}_{\mathcal{D}^-} e^{r_\theta}$, with a MINE-style moving average to handle gradient bias.
    - Design Motivation: KL divergence is most sensitive to large distributional shifts and is therefore the optimal choice for safety alignment.

4. **Alignment Consistency and Separation Theorems (Theorems 4.3 & 4.5)**:

    - Function: Proves that alignment-consistent methods can perfectly recover safety labels, and that CR data outperforms Pref data.
    - Mechanism: The optimal policy satisfies $\pi_{\theta^*}(y|x) = Z(x)^{-1}\cdot\pi_{ref}(y|x)\cdot h(R(x,y))$, where $h$ is non-decreasing and non-constant. Furthermore, $p^{CR}(z=z_x|x,\theta^*) \geq p^{Pref}(z=z_x|x,\theta^*) > 0.5$.

### Loss & Training
The general FDO framework is defined as $\mathcal{L}_{FDO(f,g)}(\theta) = -\mathbb{E}_{\mathcal{D}^+}g(r_\theta) + \mathbb{E}_{\mathcal{D}^-}f^*\circ g(r_\theta)$, which recovers KTO, BCO, and KLDO as special cases.

## Key Experimental Results

### Main Results (Bhattacharyya Distance and Robustness)

| Model | Method | $D_B$↑ | AdvBench ASR↓ | SALAD ASR↓ | ToxiGen↑ | Overall↑ |
|-------|--------|--------|-------------|-----------|---------|---------|
| Qwen2.5-1.5B | DPO | 4.10 | 4.62% | 59.13% | 45.91% | 5.59 |
| Qwen2.5-1.5B | KTO | 4.25 | 0.96% | 56.90% | 53.48% | 41.83 |
| Qwen2.5-1.5B | BCO | **11.77** | 0.58% | **45.42%** | 53.83% | 76.01 |
| Qwen2.5-1.5B | **KLDO** | 9.19 | **0.19%** | 49.78% | **56.97%** | **92.04** |

KLDO achieves the best average rank across all 5 models (1.4), followed by BCO (1.6), with DPO performing worst (3.8).

### Ablation Study (CR vs. Pref Data)

| Data Type | Qwen $D_B$ | Qwen Overall Robustness | LLaMA3.2 $D_B$ | LLaMA3.2 Overall Robustness |
|-----------|-----------|------------------------|---------------|----------------------------|
| CR | 9.19 | 92.04 | 5.75 | 95.02 |
| Pref | 3.34 | 60.76 | 4.53 | 31.10 |

### Key Findings
- **Separation correlates strongly with robustness**: The Pearson correlation between $D_B$ and SALAD ASR is $r=-0.82$ ($p<0.001$), and $r=0.70$ with overall robustness.
- **Divergence sensitivity ordering**: DPO < TV (KTO) < JS (BCO) ≈ KL (KLDO).
- **KLDO preserves utility**: No degradation in helpfulness is observed on AlpacaEval and MT-Bench.

## Highlights & Insights
- **Unified perspective of "alignment = divergence estimation"**: This framework unifies DPO, KTO, and BCO under a common lens, enabling future alignment methods to be designed directly by choosing an appropriate divergence.
- **Theoretical explanation for DPO saturation**: The S-shaped divergence curve saturates in large-shift regions, providing a fundamental explanation for DPO's poor safety alignment performance.
- **Bhattacharyya distance as a safety proxy metric**: This computable latent-space measure correlates strongly with actual attack success rates.

## Limitations & Future Work
- Alignment consistency for DPO is not rigorously proven, and closed-form solutions for the divergences are unavailable.
- Experiments are conducted on relatively small models (up to 7B); validation on larger models is lacking.
- The moving-average estimator for KLDO gradients introduces additional hyperparameters.
- Constructing CR data incurs relatively high cost.
- Systematic exploration of the optimal divergence choice within the FDO framework remains an open problem.

## Related Work & Insights
- **vs. DPO**: DPO's implicit divergence saturates under large distributional shifts, making it theoretically unsuitable for safety alignment; KLDO maintains high sensitivity via KL divergence.
- **vs. KTO**: KTO corresponds to TV divergence with a discrete $h$ function, which cannot capture fine-grained differences in the degree of safety.
- **vs. representation engineering**: Prior work exploits the separation effect for attack and defense; this paper provides a theoretical account of the origin of that separation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — "Alignment = divergence estimation" is a profound theoretical insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and alignment methods.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and clearly presented.
- Value: ⭐⭐⭐⭐⭐ — Makes important theoretical contributions to understanding and designing alignment methods; KLDO demonstrates strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[ICML 2026\] Curriculum Learning for Safety Alignment](../../ICML2026/llm_alignment/curriculum_learning_for_safety_alignment.md)
- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](../../ICLR2026/llm_alignment/align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)

</div>

<!-- RELATED:END -->
