---
title: >-
  [Paper Note] Impact of Layer Norm on Memorization and Generalization in Transformers
description: >-
  [NeurIPS 2025][Layer Normalization] This work systematically reveals the **fundamentally distinct** roles of LayerNorm in Pre-LN and Post-LN Transformers: in Pre-LN…
tags:
  - "NeurIPS 2025"
  - "Layer Normalization"
  - "memorization"
  - "generalization"
  - "Pre-LN"
  - "Post-LN"
date: 2026-05-08
content_hash: e523455fb9da3d40
---

# Impact of Layer Norm on Memorization and Generalization in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2511.10566](https://arxiv.org/abs/2511.10566)  
**Code**: [GitHub](https://github.com/JEKimLab/NeurIPS2025_LayernormMemorization)  
**Area**: Transformer Architecture Analysis / Deep Learning Theory
**Keywords**: Layer Normalization, memorization, generalization, Pre-LN, Post-LN
**Authors**: Rishi Singhal, Jung-Eun Kim
**Institution**: North Carolina State University

## TL;DR

This work systematically reveals the **fundamentally distinct** roles of LayerNorm in Pre-LN and Post-LN Transformers: in Pre-LN, LN is essential for learning and its removal disrupts generalization; in Post-LN, LN drives memorization and its removal suppresses memorization while recovering true labels.

## Background & Motivation

Layer Normalization (LN) is a fundamental component of Transformers, responsible for stabilizing training and improving optimization. Two primary LN placement strategies exist: Post-LN (Vaswani 2017, LN after residual connections) and Pre-LN (Xiong 2020, LN before sublayers). The latter has become the default choice in modern architectures such as GPT, LLaMA, and ViT due to its more stable gradient flow.

Although prior work has examined the roles of attention heads and FFNs in memorization, the **impact of LN on memorization and learning** remains largely unexplored. Xu et al. (2019) only vaguely suggested that LN may cause Pre-LN models to overfit. This paper demonstrates that LN plays **qualitatively different** roles in the two architectures: in Pre-LN it is critical for learning, while in Post-LN it is critical for memorization. This finding is supported theoretically through gradient analysis.

## Method

### Overall Architecture

The analysis proceeds at three levels: (1) removing LN learnable parameters (retaining the normalization operation $N(x)$) to compare learning and memorization behavior between full and LN-free models; (2) layer-wise removal (early/middle/late) to localize the most critical LN layers; (3) gradient norm analysis to explain the observed phenomena.

### Key Designs

1. **LN Parameter Removal Experimental Design**: The standard normalization operation $N(x) = (x-\mu)/\sigma$ is retained while the learnable scale $w$ and bias $b$ parameters are removed. Models are trained with 1% noisy labels to 100% training accuracy to ensure memorization occurs. Four metrics are defined: learning accuracy (test set), memorization score (proportion of noisy labels memorized), recovery score (proportion of true labels recovered after LN removal), and random prediction score. **Design Motivation**: causal attribution of LN's role is established through removal rather than addition.

2. **Layer-wise Analysis (Section 5)**: An $N$-layer Transformer is partitioned into early ($1..N/3$), middle ($N/3+1..2N/3$), and late ($2N/3+1..N$) groups, with LN parameters removed one group at a time. Early-layer LN is found to be the most critical — in Pre-LN, removing early LN causes the greatest learning degradation ($\Delta_{\text{overfit}}^{\text{Pre, early}} > \Delta_{\text{overfit}}^{\text{Pre, middle}} > \Delta_{\text{overfit}}^{\text{Pre, later}}$), while in Post-LN, removing early LN most effectively suppresses memorization ($\Delta_{\text{overfit}}^{\text{Post, early}} < \Delta_{\text{overfit}}^{\text{Post, middle}} < \Delta_{\text{overfit}}^{\text{Post, later}}$).

3. **Gradient Norm Analysis (Section 6)**: The gradient of the loss with respect to LN inputs, $g_x = \partial\mathcal{L}/\partial x$, is computed and averaged separately over test samples and noisy-label samples to obtain $\|g_x^{\text{learn}}\|_2$ and $\|g_x^{\text{mem}}\|_2$. **Theorem 1** proves that $\|g_x^{\text{learn}}\|_2 \geq \|g_x^{\text{mem}}\|_2$ across all layers. A key observation is that the ratio of learning to memorization gradients is much larger in Pre-LN than in Post-LN ($\frac{\|g_x^{\text{learn}}\|}{\|g_x^{\text{mem}}\|}|_{\text{Pre-LN}} \gg \frac{\|g_x^{\text{learn}}\|}{\|g_x^{\text{mem}}\|}|_{\text{Post-LN}}$), which explains why removing LN in Pre-LN primarily disrupts learning while removing it in Post-LN primarily suppresses memorization.

4. **Early-Layer Gradient Upper Bound Analysis (Theorems 2 & 3)**: Upper bounds on gradient norms are derived. For Pre-LN: $\|g_{x_i}\|_2 \leq s_{\max}(P_2) \cdot \prod_{j=i}^N (1 + s_{\max}(J_{\text{FFN}}^{\text{LN}_2(x_j')} J_{\text{LN}_2}^{x_j'})) \cdot \prod_{j=i}^N (1 + s_{\max}(J_{\text{MHSA}}^{\text{LN}_1(x_j)} J_{\text{LN}_1}^{x_j}))$. Since the product terms decrease as fewer layers remain, it is proven that $\text{UB}(\|g_{x_1}\|_2) \geq \text{UB}(\|g_{x_2}\|_2) \geq \cdots$.

### Loss & Training

Cross-entropy loss $\mathcal{L} = -\sum_k y_k \log(\hat{y}_k)$ is used, with learning rate 2e-5, batch size 16, 40 training epochs for Post-LN, and 70 epochs for Pre-LN (to allow sufficient observation of recovery after LN removal impairs learning). All experiments are run with 3 random seeds.

## Key Experimental Results

### Main Results

| Model Type | Representative Model | Learning Acc. after LN Removal | Memorization Score Change | Recovery Score | Notes |
|-----------|---------------------|-------------------------------|--------------------------|---------------|-------|
| Post-LN | ELECTRA (News) | Nearly unchanged (~stable) | Large decrease | High (green bar) | LN removal suppresses memorization |
| Post-LN | BERT (Emotions) | Remains stable | Large decrease | High | Consistent across models |
| Pre-LN | Qwen2 (News) | Large decrease (~30%↓) | Remains high | Extremely low | LN removal disrupts learning |
| Pre-LN | ViT-B (CIFAR10) | Large decrease | Remains high | Extremely low | Consistent in vision models |
| Post-LN | DistilBERT | Unaffected | No significant decrease | Low | Only exception |

### Ablation Study (Layer-wise Removal)

| Configuration | Pre-LN Learning Impact | Post-LN Memorization Impact | $\Delta_{\text{overfit}}$ |
|--------------|----------------------|----------------------------|--------------------------|
| Remove Early LN | Most severe learning degradation | Most effective memorization suppression | Largest for Pre / Smallest for Post |
| Remove Middle LN | Moderate impact | Moderate effect | Moderate |
| Remove Late LN | Least impact | Least effective | Smallest for Pre / Largest for Post |

### Key Findings

- **Fundamental distinction between Pre-LN and Post-LN**: In Pre-LN, gradient norms are extremely high at the first layer and nearly zero in subsequent layers (concentrated), so learning cannot recover after early LN removal. In Post-LN, gradient norms decay gradually across layers (distributed), allowing later layers to compensate for early-layer deficits.
- **Consistency across 13 models × 6 datasets**: BERT/RoBERTa/DeBERTa/ELECTRA/Longformer (Post-LN) and GPT2/GPTNeo/Qwen2/ViT-B/ViT-S/DeiT/RoBERTa-PreLN (Pre-LN) all validate the core findings.
- **DistilBERT exception**: Likely attributable to knowledge distillation training, which may cause other components to play a larger role in memorization.

## Highlights & Insights

- The paper identifies a "dual" role of LN across Pre/Post architectures — critical for learning in one and critical for memorization in the other — representing a genuinely novel insight.
- The conclusion that early-layer LN is most critical aligns with recent layer-pruning literature suggesting that deeper layers are less important, while providing more precise attribution.
- The gradient norm ratio offers a concise theoretical explanatory framework.
- Large-scale validation across 13 models and 6 datasets provides strong empirical credibility.

## Limitations & Future Work

- Only 1% noisy labels are studied as the memorization-inducing mechanism; naturally occurring memorization (e.g., long-tail distributions) may exhibit different behavior.
- The theoretical proof relies on the assumption in Theorem 1 that samples of the same class have similar features, which may not hold exactly in practice.
- Post-LN is predominantly used in language models (vision Transformers are almost exclusively Pre-LN), which limits the direct applicability of Post-LN conclusions.
- The differential impact of RMSNorm (used in Qwen2) versus standard LayerNorm is not examined.

## Related Work & Insights

- This work extends the preliminary observations of Xu et al. (2019) on LN-induced overfitting, providing a more precise Pre/Post distinction.
- It complements recent layer pruning work (Men et al. 2024, Lad et al. 2024), which finds that deeper layers are less useful; this paper pinpoints LN as the critical factor.
- Practical implication: memorization mitigation and privacy protection in Post-LN models may be achieved by removing LN parameters.

## Quick Reference Summary

- Pre-LN: LN removal → learning collapse + persistent memorization + increased overfitting
- Post-LN: LN removal → learning preserved + memorization suppressed + true label recovery
- Early-layer LN is most critical, as gradient norm upper bounds decrease with layer depth
- Gradient ratio $\|g_x^{\text{learn}}\|/\|g_x^{\text{mem}}\|$ is substantially larger in Pre-LN than in Post-LN
- Experimental coverage: BERT/RoBERTa/DeBERTa/ELECTRA/Longformer/DistilBERT (Post-LN) + GPT2/GPTNeo/Qwen2/ViT-B/ViT-S/DeiT/RoBERTa-PreLN (Pre-LN)

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "duality" of LN roles in Pre/Post architectures is an important and unexpected finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 13 models × 6 datasets spanning NLP and CV, with highly consistent conclusions.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, though the main text is somewhat lengthy and could be condensed.
- **Value**: ⭐⭐⭐⭐ Substantially advances the understanding of Transformer architecture and has direct practical implications for memorization mitigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](generalized_linear_mode_connectivity_for_transformers.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[NeurIPS 2025\] A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation](a_theoretical_framework_for_grokking_interpolation_followed_by_riemannian_norm_m.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)

</div>

<!-- RELATED:END -->
