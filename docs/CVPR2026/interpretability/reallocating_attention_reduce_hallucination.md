---
title: >-
  [Paper Note] Reallocating Attention Across Layers to Reduce Multimodal Hallucination
description: >-
  [CVPR 2026][Multimodal hallucination] This paper decomposes multimodal reasoning model hallucinations into two failure modes — shallow-layer perceptual bias and deep-layer reasoning drift — and selectively amplifies the contributions of identified perception/reasoning functional heads in a plug-and-play, training-free manner, achieving an average accuracy improvement of 4.2% with only ~1% additional computational overhead.
tags:
  - CVPR 2026
  - Multimodal hallucination
  - attention head reallocation
  - perception-reasoning layer decomposition
  - training-free
  - MLRM
date: 2026-05-08
content_hash: f4ac53569e26f327
---

# Reallocating Attention Across Layers to Reduce Multimodal Hallucination

**Conference**: CVPR 2026
**arXiv**: [2510.10285](https://arxiv.org/abs/2510.10285)
**Code**: None
**Area**: Multimodal VLM / Hallucination Mitigation / Interpretability
**Keywords**: Multimodal hallucination, attention head reallocation, perception-reasoning layer decomposition, training-free, MLRM

## TL;DR
This paper decomposes multimodal reasoning model hallucinations into two failure modes — shallow-layer perceptual bias and deep-layer reasoning drift — and selectively amplifies the contributions of identified perception/reasoning functional heads in a plug-and-play, training-free manner, achieving an average accuracy improvement of 4.2% with only ~1% additional computational overhead.

## Background & Motivation
Hallucinations in multimodal large reasoning models (MLRMs) such as Kimi-VL and Ocean-R1 arise not only from insufficient visual evidence, but also from **functional misalignment** between perception and reasoning within the model. Interpretability research has revealed that Transformers exhibit staged attention: shallow layers attend to visual tokens for perception, while deeper layers shift to text tokens for reasoning. However, this functional division is often insufficiently robust — overly diffuse visual attention in shallow layers dilutes critical evidence (perceptual bias), and deep-layer attention fails to maintain consistency across intermediate reasoning steps, causing conclusions to deviate from premises (reasoning drift). Manual inspection of 200 hallucination cases reveals that perceptual bias accounts for 16%, reasoning drift for 20%, and their co-occurrence for 10%.

## Core Problem
Can both hallucination modes be simultaneously mitigated — without retraining or architectural modification — by identifying and enhancing existing perception/reasoning functional heads within the model?

## Method

### Overall Architecture
A two-step plug-and-play module: (1) **Functional head identification**: compute the visual attention ratio for each attention head, and combine it with layer-depth boundaries to classify heads into perception heads (shallow layers + high visual attention ratio) and reasoning heads (deep layers + low visual attention ratio); (2) **Category-conditional scaling**: apply multiplicative gains $g_{perc}/g_{reas}$ to amplify the output of identified functional heads, leaving all other heads unchanged.

### Key Designs
1. **Layer-modality attention analysis**: For each head $h$ at layer $\ell$, compute the visual attention ratio $S_v^{(\ell)}(h) = \sum_{j \in T_v} a_{i^*j}^{(h,\ell)}$. Depth boundaries $\ell_{perc}$ and $\ell_{reas}$ together with ratio thresholds $\tau_{perc}$ and $\tau_{reas}$ are used to classify heads: $\mathcal{H}_{perc}^{(\ell)} = \{h: \ell \leq \ell_{perc} \wedge S_v \geq \tau_{perc}\}$, $\mathcal{H}_{reas}^{(\ell)} = \{h: \ell \geq \ell_{reas} \wedge S_v \leq \tau_{reas}\}$. Typical settings: $\ell_{perc}=7, \ell_{reas}=3$ (overlapping allowed), with approximately 6.4% of heads selected.

2. **Selective enhancement strategy**: Theoretical analysis based on the minimum-edit principle demonstrates that enhancement-only (Strategy A) outperforms enhancement-plus-suppression (Strategies C/D), since most non-target heads are not harmful and suppressing them causes collateral damage. Gain factors $g_{perc}=1.16$ and $g_{reas}=1.30$ (moderate amplification) are applied prior to the head output projection. The core formula is:

$$g^{(h,\ell)} = g_{perc} \cdot \mathbb{1}[h \in \mathcal{H}_{perc}] + g_{reas} \cdot \mathbb{1}[h \in \mathcal{H}_{reas}] + 1 \cdot \mathbb{1}[\text{otherwise}]$$

3. **Task-dependent boundary bands**: The optimal $\ell_{perc}$ and $\ell_{reas}$ form a range rather than a single point — visual tasks favor shallower boundaries, while mathematical reasoning favors deeper ones. A performance trough exists in the transition zone $\ell \in [10,17]$, indicating that perception and reasoning functions are interleaved rather than cleanly separated in this region.

### Loss & Training
Entirely training-free. Only the scaling of attention head outputs is modified at inference time; no weights are altered. The additional computation consists solely of visual attention ratio calculations with $O(H \cdot N^2)$ complexity, leaving the asymptotic complexity unchanged at $O(N^2)$, with empirically measured additional latency below 5%.

## Key Experimental Results

| Model | Method | MathVista | MathVision | HallusBench | MMStar | SEED | Avg Gain |
|--------|------|------|----------|------|------|------|------|
| Kimi-VL | Vanilla | 63.48 | 56.24 | 64.76 | 59.76 | 66.26 | — |
| Kimi-VL | +VCD | 63.51 | 56.14 | 65.68 | 59.48 | 66.52 | +0.2% |
| Kimi-VL | +Ours | **69.20** | **58.54** | **67.41** | **66.47** | **69.55** | **+4.7%** |
| Ocean-R1 | Vanilla | 71.90 | 22.42 | 60.03 | 59.40 | 65.46 | — |
| Ocean-R1 | +Ours | **73.73** | **27.71** | **61.16** | **62.27** | **66.28** | **+3.5%** |

Efficiency comparison (HallusBench, Kimi-VL): VCD +62% latency, CGD +392%, AGLA +21%, **Ours +5.1%**.

### Ablation Study
- **Joint perception+reasoning enhancement is necessary**: Enhancing perception heads alone yields large gains on visual tasks but modest gains on mathematical reasoning, and vice versa. On R1-OneVision, enhancing either head type alone causes a 3.91% drop, whereas joint enhancement yields +5.58%, confirming that hallucination results from **cross-stage interactions**.
- **Optimal gain factor $g_{reas}=1.30$**: Improvements of ~10% begin as early as 1.10, peaking at 1.30; $g_{perc}$ exhibits greater sensitivity, with 1.16 being optimal.
- **Boundary sweep**: Over 150 boundary configurations were evaluated; the gap between the best and worst configurations reaches 27.4%, validating the critical importance of boundary selection.
- **Head proportion**: Selecting approximately 6.4% of heads via $\tau_{reas}$ is optimal; including too many heads dilutes the functional head signal.
- **Model-specific configurations required**: On Kimi-VL MMStar, enhancing only perception heads yields +6.71%; on Ocean-R1, the same setting yields −1.51%.

## Highlights & Insights
- The dual-factor hallucination framework of "perceptual bias + reasoning drift" is more comprehensive and precise than explanations based solely on insufficient visual information.
- The theoretical analysis grounded in the **minimum-edit principle** is elegant — it proves that amplification is preferable to suppression (since most heads are not harmful), thereby avoiding collateral damage.
- Empirically measured latency overhead of only 5% far outperforms VCD (62%), CGD (392%), and AGLA (21%), making this a genuinely practical plug-and-play solution.
- Validation is conducted on MLRMs (long-chain reasoning models), where hallucination problems are particularly pronounced.
- The Contribution Map, derived via gate parameters and gradient backpropagation, can serve as a general-purpose head attribution tool.

## Limitations & Future Work
- The boundaries $\ell_{perc}$ and $\ell_{reas}$ require manual tuning, although performance remains stable over a relatively wide range.
- Validation is limited to Qwen-based architectures (both Ocean-R1 and Kimi-VL are Qwen derivatives); generalization to other families (e.g., LLaMA) remains to be verified.
- Uniform gain values are applied across all inputs; **input-adaptive** gain adjustment may yield further improvements.
- The optimal gain factor $g$ varies across tasks, and cross-task robustness leaves room for improvement.
- The binary shallow/deep layer decomposition is overly simplistic — intermediate layers exhibit both perceptual and reasoning functions, and finer-grained partitioning warrants further exploration.

## Related Work & Insights
- **vs. VCD (contrastive decoding)**: VCD constructs counterfactual visual contrasts but does not distinguish between failure types, yielding limited gains on mathematical reasoning. The proposed method mitigates both perceptual and reasoning failures simultaneously.
- **vs. AGLA (global-local attention fusion)**: AGLA generates locally enhanced views via Grad-CAM to improve perception, but leaves the reasoning side unchanged. The proposed method directly enhances reasoning heads in deep layers.
- **vs. CGD (CLIP-guided decoding)**: CGD employs an external CLIP model for sentence-level verification, incurring substantial latency overhead (392%). The proposed method requires no external models and adds only 5% latency.
- **vs. OPERA/DAMRO (attention probing)**: These works diagnose attention sink and hallucination patterns, but the proposed method is the first to classify attention heads by function and perform selective enhancement.

The finding that "layer depth → functional specialization" aligns with the observation in V2Drop that "token variation → importance"; combining variation-based functional head discrimination with fixed thresholds may yield greater robustness. The method could also be extended to **video understanding**, where temporal inter-frame hallucinations in video VLMs may exhibit analogous perception/reasoning layer-wise failure modes.

## Rating
- Novelty: ⭐⭐⭐⭐ The perception/reasoning dual-factor framework and selective enhancement strategy are novel, though head importance analysis is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three models, five benchmarks, over 150 boundary configurations, four strategy comparisons, efficiency analysis, and detailed case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, and the logical chain from motivation → analysis → method → experiments is complete.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play hallucination mitigation approach with only 5% latency overhead is highly valuable for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events](cut_to_the_chase_training-free_multimodal_summarization_via_chain-of-events.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)

</div>

<!-- RELATED:END -->
