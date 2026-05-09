---
title: >-
  [Paper Note] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment
description: >-
  [ICLR 2026][LLM Alignment][multilingual safety] This paper proposes Multi-Lingual Consistency (MLC), an auxiliary loss that manipulates the singular values of a multilingual representation matrix via SVD to drive it toward rank-1 (i.e., collinear multilingual representations). Using only multilingual prompt translations—without requiring target-language responses—MLC consistently transfers safety alignment from one language to all others.
tags:
  - ICLR 2026
  - LLM Alignment
  - multilingual safety
  - consistency alignment
  - singular value decomposition
  - cross-lingual transfer
  - DPO
date: 2026-05-08
content_hash: b0833ed5d47fb745
---

# Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment

**Conference**: ICLR 2026
**arXiv**: [2602.16660](https://arxiv.org/abs/2602.16660)
**Code**: None
**Area**: LLM Alignment
**Keywords**: multilingual safety, consistency alignment, singular value decomposition, cross-lingual transfer, DPO

## TL;DR
This paper proposes Multi-Lingual Consistency (MLC), an auxiliary loss that manipulates the singular values of a multilingual representation matrix via SVD to drive it toward rank-1 (i.e., collinear multilingual representations). Using only multilingual prompt translations—without requiring target-language responses—MLC consistently transfers safety alignment from one language to all others.

## Background & Motivation
**Background**: LLM safety alignment (SFT/DPO) is predominantly conducted on high-resource languages such as English. As a result, models behave safely in English but may exhibit drastically reduced safety rates in low-resource languages (e.g., Swahili, Kurdish), dropping from 93% to as low as 6–12%.

**Limitations of Prior Work**: The two mainstream approaches to multilingual alignment both have significant drawbacks: (a) collecting high-quality safety data for each target language incurs prohibitive resource costs; (b) pairwise transfer using a high-resource anchor language (e.g., SDRRL/MPO) scales poorly and yields inconsistent results, leaving certain languages well behind.

**Key Challenge**: If all languages are aligned to the same anchor language, they should theoretically achieve comparable safety levels. The observed performance disparity indicates that existing methods fail to fully exploit the safety signal already present in the anchor language.

**Goal**: How can multiple languages be aligned simultaneously in a single training run, without requiring response data in the target languages?

**Key Insight**: Multilingual behavioral consistency is determined by representational consistency. If the internal representations of the same query across different languages are directionally aligned (collinear), the model will produce consistent safety behaviors across those languages.

**Core Idea**: Constrain the multilingual representation matrix to be rank-1 via singular value analysis, realizing "align once, benefit multilingually."

## Method

### Overall Architecture
Given a training prompt $q$ and its translations $\{q^{(\ell)}\}_{\ell=1}^m$ across $m$ languages, the hidden state of the last prompt token is extracted for each language, projected via a trainable linear layer, normalized, and stacked into a matrix $\mathbf{Z} \in \mathbb{R}^{d \times m}$. The total loss is $\mathcal{L}_{total} = \mathcal{L}_{align} + \lambda_{aux} \mathcal{L}_{cons}$, where $\mathcal{L}_{align}$ is the original alignment loss (DPO/SFT) and $\mathcal{L}_{cons}$ is the MLC auxiliary loss.

### Key Designs

1. **MLC Consistency Loss (Singular Value Manipulation)**:

    - Function: Drive the multilingual representation matrix $\mathbf{Z}$ toward rank-1.
    - Mechanism: SVD is applied to $\mathbf{Z}$ to obtain singular values $\{\sigma_j\}$. When $\sigma_1$ dominates the remaining singular values, all language representations are approximately collinear. Treating the singular values as logits, a temperature-scaled softmax cross-entropy encourages the distribution to concentrate on $\sigma_1$: $\mathcal{L}_{cons} = -\frac{1}{N}\sum_{n=1}^N \log \frac{\exp(\sigma_1^{(n)}/\tau)}{\sum_j \exp(\sigma_j^{(n)}/\tau)}$.
    - Design Motivation: The rank-1 constraint is equivalent to minimizing $\|\mathbf{Z} - \tilde{\mathbf{Z}}\|_F^2$ (by the Eckart–Young theorem), i.e., making multilingual representations as close as possible to their optimal rank-1 approximation. The softmax formulation ensures smooth and differentiable gradient computation.
    - Theoretical Basis: Proposition 1 proves that minimizing the reconstruction error is equivalent to maximizing the relative dominance of $\sigma_1$.

2. **Linear Representation Extractor**:

    - Function: Project the hidden state of the last prompt token of the LLM into a low-dimensional representation space.
    - Mechanism: $\mathbf{r}^{(\ell)} = \mathbf{W}\mathbf{h}^{(\ell)} + b$, where $\mathbf{W} \in \mathbb{R}^{d \times d_h}$ is trained jointly with the LLM.
    - Design Motivation: A simple linear projection suffices to capture the cross-lingual semantic consistency direction; experiments show it outperforms more complex extractors.

3. **Plug-and-Play Integration**:

    - Function: Seamlessly integrates with arbitrary alignment algorithms (SFT/DPO/SimPO/ORPO).
    - Mechanism: MLC requires only multilingual prompt translations and does not alter the original training data format; $\lambda_{aux}$ controls the loss weight.

### Loss & Training
Only English response data and multilingual prompt translations are required. Translations can be obtained via machine translation at minimal cost. During training, forward passes are conducted over all language prompts simultaneously; the MLC loss is computed and added to the original alignment loss before backpropagation.

## Key Experimental Results

### Main Results: PKU-SafeRLHF Multilingual Safety Rate

| Method | EN | ZH | RU | JA | AR | BN | SW | UR | PS | KU | Avg↑ | Var↓ | PAG↑ |
|--------|----|----|----|----|----|----|----|----|----|----|------|------|------|
| Qwen Raw | 93.3 | 96.1 | 93.3 | 92.2 | 93.9 | 53.3 | 6.1 | 33.9 | 21.1 | 12.2 | 59.6 | 13.14 | 0.50 |
| DPO | 99.4 | 98.3 | 97.2 | 97.8 | 96.1 | 70.6 | 7.2 | 50.6 | 30.0 | 17.2 | 66.4 | 12.44 | 0.54 |
| MPO | 81.1 | 81.7 | 82.2 | 77.2 | 78.3 | 77.2 | 4.4 | 70.6 | 59.4 | 42.8 | 65.5 | 5.53 | 0.70 |
| **DPO+MLC** | **99.4** | **96.7** | **97.8** | **98.3** | **98.3** | **95.0** | **92.8** | **92.8** | **91.1** | **97.2** | **95.9** | **0.07** | **0.97** |

### Ablation Study (MultiJail OOD, ASR↓)

| Method | EN | ZH | AR | BN | SW | Avg ID↓ | KO | IT | JV | TH | VI | Avg OOD↓ |
|--------|----|----|----|----|----|---------|----|----|----|----|----|----------|
| DPO | 2.9 | 1.9 | 5.1 | 17.8 | 25.1 | 10.5 | 3.2 | 3.5 | 4.1 | 5.4 | 1.9 | 3.6 |
| **DPO+MLC** | **0.6** | **0.3** | **1.0** | **1.0** | **0.6** | **0.7** | **0.6** | **0.6** | **0.3** | **1.0** | **0.0** | **0.5** |

### Key Findings
- DPO+MLC raises Qwen's average safety rate across 10 languages from 59.6% to 95.9%, reducing variance from 13.14 to 0.07.
- Critically, DPO alone improves English safety but leaves low-resource languages nearly unchanged (Swahili: 7.2%), whereas MLC elevates Swahili to 92.8%.
- On the OOD benchmark MultiJail, DPO+MLC reduces ASR to 0.5–0.7%, generalizing to languages unseen during training (Korean, Italian, etc.).
- MLC is compatible with all four alignment paradigms (SFT/DPO/SimPO/ORPO), consistently yielding positive gains.
- Impact on general capability (MMLU) is less than 1%.

## Highlights & Insights
- **Cross-lingual consistency via singular value perspective**: Framing multilingual representational consistency as a matrix rank minimization problem yields a theoretically grounded and practically simple solution (one SVD + softmax loss). The core insight is that collinear multilingual representations induce consistent model behavior across all languages.
- **High resource efficiency**: No target-language response data is required; only prompt translations are needed. This represents a substantial breakthrough for low-resource language safety, as translating prompts is far cheaper than generating high-quality aligned data.
- **Significant improvement in safety lower bound**: More meaningful than the rise in average safety rate is the fact that low-resource language safety rates jump from 6–12% to 91–97%, effectively eliminating the "weakest link" in multilingual safety.

## Limitations & Future Work
- Performance depends on the quality of prompt translations; machine translation for low-resource languages may introduce noise (the paper uses GPT-4 translations, and practical deployment costs and quality may vary).
- An overly strict collinearity constraint may be problematic in scenarios requiring culturally differentiated safety policies (e.g., content acceptable in some cultures but not in others).
- Validation is limited to 7–9B models.
- Whether the linear extractor is optimal across all architectures remains an open question; no evaluation on substantially different architectures (e.g., MoE) is provided.

## Related Work & Insights
- **vs. MPO (Zhao et al., 2025)**: MPO supervises transfer using reward gaps from high-resource languages, a pairwise approach that yields unstable multilingual performance (e.g., Swahili at only 4.4%). MLC achieves simultaneous alignment of all languages via representational consistency constraints.
- **vs. SDRRL (Zhang et al., 2024b)**: SDRRL uses self-distillation to generate target-language responses, requiring additional data and computation. MLC requires no target-language responses whatsoever.
- **vs. AlphaSteer**: AlphaSteer uses null-space projection to preserve utility; MLC uses collinearity constraints to align multilingual representations. Both exploit linear structure in the representation space but target different objectives.

## Rating
- Novelty: ⭐⭐⭐⭐ The SVD-based multilingual consistency constraint is an elegant theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 languages × 2 models × 4 alignment paradigms × ID+OOD evaluation × MMLU utility
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are complete, though the dense notation requires careful reading
- Value: ⭐⭐⭐⭐⭐ Addresses the core bottleneck of multilingual safety alignment; practical and high-impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)
- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)

</div>

<!-- RELATED:END -->
