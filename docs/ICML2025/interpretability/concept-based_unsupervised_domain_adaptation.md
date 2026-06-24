---
title: >-
  [Paper Note] Concept-Based Unsupervised Domain Adaptation
description: >-
  [ICML 2025][Interpretability][Concept Bottleneck Models] Proposes the CUDA framework, which combines Concept Bottleneck Models (CBMs) with Unsupervised Domain Adaptation (UDA). By aligning concept representations via relaxed consistency (allowing minor domain discrepancies) and inferring unlabeled concepts in the target domain, CUDA simultaneously provides interpretability and cross-domain generalization under domain shift for the first time, backed by theoretical guarantees.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Concept Bottleneck Models"
  - "Domain Adaptation"
  - "Relaxed Alignment"
  - "Adversarial Training"
date: 2026-05-08
content_hash: b7d8c22802cb6825
---

# Concept-Based Unsupervised Domain Adaptation

**Conference**: ICML 2025  
**arXiv**: [2505.05195](https://arxiv.org/abs/2505.05195)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Concept Bottleneck Models, Domain Adaptation, Interpretability, Relaxed Alignment, Adversarial Training

## TL;DR
Proposes the CUDA framework, which combines Concept Bottleneck Models (CBMs) with Unsupervised Domain Adaptation (UDA). By aligning concept representations via relaxed consistency (allowing minor domain discrepancies) and inferring unlabeled concepts in the target domain, CUDA simultaneously provides interpretability and cross-domain generalization under domain shift for the first time, backed by theoretical guarantees.

## Background & Motivation

**Background**: Concept Bottleneck Models (CBMs) enhance interpretability by using human-understandable concepts as intermediate representations (e.g., first predicting "black eyes" + "solid belly" before classifying as "Laysan Albatross"). However, CBMs assume that training and testing data are identically distributed.

**Limitations of Prior Work**:
   - CBM accuracy drops drastically from ~80% to ~66% under domain shift (e.g., background shift on the CUB dataset).
   - Naive integration of CBM and DA performs poorly because: (a) class alignment and concept alignment are conducted separately without unification; (b) strict consistency alignment overlooks reasonable cross-domain concept discrepancies.
   - The target domain lacks concept annotations, preventing CBMs from directly training concept predictors on the target domain.

**Key Challenge**: CBMs require concept consistency to ensure interpretability, yet concept distributions inherently differ under domain shift. Over-alignment compromises concept accuracy.

**Goal**: Achieve interpretable classification under domain shift.

**Key Insight**: Relaxed concept alignment, which allows minor discrepancies in concept distributions across domains (e.g., admitting that "predominantly brown" occupying 19% of the source domain and 17% of the target domain is reasonable) instead of enforcing strict identity.

**Core Idea**: Relaxed consistency alignment + target-domain concept inference (leveraging the aligned embedding space to infer target concepts) + a theoretical generalization error bound for CBM×DA.

## Method

### Overall Architecture
CUDA pipeline:
1. Train a concept and label predictor on the source domain (standard CBM).
2. Align source and target concept embeddings using adversarial training, while allowing relaxation.
3. Infer target domain concepts within the aligned embedding space.
4. Perform target domain classification using the inferred concepts and the label predictor.

### Key Designs

1. **Relaxed Consistency Alignment Loss**:

    - **Function**: Aligns source and target concept distributions while allowing minor discrepancies.
    - **Mechanism**: $\mathcal{L}_{\text{relax}} = \max(0, d(P_s^c, P_t^c) - \epsilon)$, where $\epsilon > 0$ represents the relaxation threshold.
    - Distinction from strict alignment: Strict alignment ($d(P_s^c, P_t^c) \to 0$) imposes over-constraints that lead to concept distortion, whereas relaxed alignment permits a "reasonable range of misalignment."
    - **Design Motivation**: Empirical results demonstrate that concept distributions predicted after relaxed alignment are closer to the ground truth, leading to higher classification accuracy.
    - Theoretical Support: Provides a generalization error bound for CBMs under DA, where the error is bounded by the distance of concept embeddings.

2. **Unlabeled Concept Inference in Target Domain**:

    - **Function**: Infers concepts in the target domain without target concept annotations.
    - **Mechanism**: The source concept predictor directly transfers to the target domain within the aligned embedding space.
    - Consistency Regularization: $\mathcal{L}_{\text{consist}} = \|c_s(g(x_s)) - c_t(g(x_t))\|$ predicts concepts using aligned features.
    - **Design Motivation**: The target domain lacks concept labels, requiring the model to "borrow" concept knowledge from the source domain via the aligned embedding space.

3. **Unified Concept-Class Alignment**:

    - **Function**: Unifies concept alignment and class alignment into a single feature space.
    - **Mechanism**: The domain discriminator in adversarial training simultaneously considers both concept and class dimensions.
    - **Design Motivation**: Aligning concepts and classes separately leads to a fragmented feature space. Thus, domain adaptation should be conducted in a unified concept embedding space.

### Loss & Training
- Concept prediction loss (labeled on the source domain)
- Label prediction loss (labeled on the source domain)
- Adversarial domain alignment loss (relaxed version)
- Target domain concept consistency regularization
- End-to-end training

## Key Experimental Results

### Main Results
CUB-200 (Bird classification, background shift):

| Method | Source Accuracy | Target Accuracy↑ | Interpretable? |
|------|----------|-----------|---------|
| CBM (No DA) | 80.2% | 66.3% | ✓ |
| DANN (No Concept) | - | 75.8% | ✗ |
| CBM + DANN (Naive Combo) | - | 70.5% | ✓ (poor) |
| CBM + Strict Alignment | - | 72.1% | ✓ (poor) |
| **CUDA (Relaxed Alignment)** | - | **78.5%** | **✓ (good)** |

### Concept Prediction Accuracy (Target Domain)

| Method | Concept Prediction F1↑ | Note |
|------|-----------|------|
| Strict Alignment | 0.72 | Over-alignment causes concept distortion |
| **Relaxed Alignment** | **0.84** | Retains reasonable domain discrepancies |

### Ablation Study

| Configuration | Target Accuracy | Note |
|------|-----------|------|
| No Relaxation (Strict Alignment) | 72.1% | Over-constrained |
| **Relaxation $\epsilon=0.05$** | **78.5%** | Optimal relaxation |
| Relaxation $\epsilon=0.2$ | 76.3% | Over-relaxation |
| No Concept Inference | 73.8% | Lack of concept information in target domain |
| **Full CUDA** | **78.5%** | Relaxation + Concept Inference + Unified Alignment |

### Key Findings
- Relaxed alignment improves accuracy by +6.4% compared to strict alignment — "imperfect but accurate concepts" outperform "perfectly aligned but distorted concepts."
- CUDA enhances target-domain concept prediction accuracy by +17% (from 0.72 to 0.84) — relaxation preserves the semantic plausibility of concepts.
- Theoretical bounds are empirically validated — the concept embedding distance is positively correlated with classification error.
- Consistent improvements are achieved on genetic data and medical images, demonstrating that the method is not limited to natural images.

## Highlights & Insights
- **"Relaxation is better than perfection"** — allowing minor discrepancies in concept alignment paradoxically yields more accurate concept predictions. This counterintuitive finding is supported by both theory and experiments.
- The theoretical generalization bound for CBM×DA is the first of its kind, establishing a foundation for future work.
- Unlabeled concept inference on the target domain makes the method highly practical, as annotating target-domain concepts is often prohibitively expensive in real-world scenarios.
- Directly valuable for the reliable deployment of explainable AI under distribution shifts.

## Limitations & Future Work
- The relaxation threshold $\epsilon$ is a hyperparameter requiring tuning.
- The concept set is pre-defined by humans; exploring extensions to automated concept discovery remains for future work.
- Addresses only covariate shift; scenarios where both label shift and concept shift co-occur are not yet covered.
- Only validated on classification tasks.

## Related Work & Insights
- **vs Standard CBM**: Standard CBM does not handle domain shift; CUDA adds DA capabilities.
- **vs Standard DA (DANN)**: Standard DA lacks interpretability; CUDA incorporates interpretability through a concept bottleneck.
- **vs Naive CBM+DANN Combination**: Aligning concepts and classes separately performs poorly; CUDA provides unified alignment and relaxation.
- **Insight**: Interpretability and robustness do not have to be a trade-off — both can be simultaneously improved through concept-level domain adaptation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The first systematic integration of CBM×DA holds significant value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets, with concept-level analysis and theoretical validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear intuitive illustrations of relaxed alignment.
- **Value**: ⭐⭐⭐⭐⭐ Advances the deployment of explainable AI in real-world scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Features Adaptation in Networking: Toward Flexible Training and Explainable Inference](../../NeurIPS2025/interpretability/dynamic_features_adaptation_in_networking_toward_flexible_training_and_explainab.md)
- [\[ACL 2026\] From Weights to Activations: Is Steering the Next Frontier of Adaptation?](../../ACL2026/interpretability/from_weights_to_activations_is_steering_the_next_frontier_of_adaptation.md)
- [\[AAAI 2026\] Unsupervised Feature Selection Through Group Discovery](../../AAAI2026/interpretability/unsupervised_feature_selection_through_group_discovery.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[CVPR 2025\] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction](../../CVPR2025/interpretability/tide_domain_generalization.md)

</div>

<!-- RELATED:END -->
