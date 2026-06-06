---
title: >-
  [Paper Note] Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Unsupervised Domain Generalization] MS-UDG operates without class or domain labels, decomposing representations into semantic and variation components via an Information Disentang…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Unsupervised Domain Generalization"
  - "Information Disentanglement"
  - "Semantic Sufficiency"
  - "Minimality"
  - "Fourier Augmentation"
date: 2026-05-08
content_hash: 850e1fcc4999ec62
---

# Minimal Semantic Sufficiency Meets Unsupervised Domain Generalization

**Conference**: NeurIPS 2025
**arXiv**: [2509.15791](https://arxiv.org/abs/2509.15791)  
**Code**: To be confirmed  
**Area**: Self-Supervised Learning / Domain Generalization
**Keywords**: Unsupervised Domain Generalization, Information Disentanglement, Semantic Sufficiency, Minimality, Fourier Augmentation

## TL;DR
MS-UDG operates without class or domain labels, decomposing representations into semantic and variation components via an Information Disentanglement Module (IDM). Coupled with a Semantic Representation Optimization Module (SROM) that simultaneously maximizes semantic information and minimizes variation interference, the method achieves 72.89% accuracy on PACS (+1.5% vs. CycleMAE). Theoretical analysis proves that minimally sufficient semantic representations minimize the downstream Bayes error rate.

## Background & Motivation

**Background**: Self-supervised learning (SSL) representations entangle semantic content with variation factors such as style and texture. Unsupervised domain generalization (UDG) requires learning robust cross-domain representations without any labels.

**Limitations of Prior Work**: Existing methods either require domain labels (impractical) or fail to explicitly separate semantics from variation (e.g., MAE performs reconstruction without explicit disentanglement), leading to performance degradation under domain shift.

**Key Challenge**: Without class labels, "what constitutes semantics" cannot be directly supervised; without domain labels, cross-domain alignment cannot be performed explicitly. Defining and optimizing a *minimally sufficient semantic representation* must therefore be grounded in an information-theoretic framework.

**Goal**: Learn representations that retain only task-relevant semantics while discarding domain-related variation, in a fully self-supervised setting with neither class nor domain labels.

**Key Insight**: An information-theoretic framework — *sufficiency* requires the semantic representation $s$ to preserve all prediction-relevant information, i.e., $I(s;T) = I(x;T)$; *minimality* requires $s$ to exclude variation information irrelevant to prediction, i.e., $I(s;v) \to 0$.

**Core Idea**: IDM decomposes representations into semantic component $s$ and variation component $v$; SROM achieves minimally sufficient semantic representations via mutual information minimization/maximization, with theoretical guarantees on minimizing the Bayes error rate.

## Method

### Overall Architecture
Input image $x$ → Fourier augmentation generates domain-shifted views $(x_1, x_2)$ → ViT encoder → sufficient representation $z$ → **IDM** (two MLPs decompose $z$ into $s \oplus v$) → **SROM** ($\mathcal{L}_{min}$ minimizes $I(s;v)$ + $\mathcal{L}_{max}$ maximizes $I(v;x|S)$ + $\mathcal{L}_{suff}$ ensures semantic sufficiency) → downstream fine-tuning uses $s$

### Key Designs

1. **Information Disentanglement Module (IDM)**:

    - **Function**: Decomposes the encoder output $z$ into semantic component $s$ and variation component $v$.
    - **Mechanism**: Two parallel MLPs project $z$ into $s$ and $v$ respectively, satisfying $z = s \oplus v$ (concatenation recovers $z$).
    - **Design Motivation**: Performs disentanglement directly in representation space without requiring additional encoders or generative models.

2. **Semantic Representation Optimization Module (SROM)**:

    - **Function**: Jointly optimizes three mutual information objectives to achieve minimal sufficiency.
    - **Mechanism**: $\mathcal{L}_{min}$ (modified InfoNCE) minimizes $I(s;v)$ while maximizing $I(s_1;s_2)$, encouraging semantic consistency across views while removing variation dependence. $\mathcal{L}_{max}$ (reconstruction) maximizes $I(v;x|S)$ by reconstructing the input from $v$ via decoder $D$, ensuring the variation component captures sufficient non-semantic information. $\mathcal{L}_{suff}$ (InfoNCE) guarantees that $s$ retains all semantic information.
    - **Design Motivation**: All three losses are indispensable — $\mathcal{L}_{min}$ alone causes $s$ to degenerate to a constant; $\mathcal{L}_{suff}$ alone cannot exclude variation information; $\mathcal{L}_{max}$ prevents variation information from leaking into $s$.

3. **Fourier Domain Augmentation**:

    - **Function**: Generates image views with different domain styles.
    - **Mechanism**: Swaps low-frequency components (which govern global style) in the frequency domain to simulate domain shift.
    - **Design Motivation**: In the absence of domain labels, Fourier augmentation provides a principled way to simulate domain variation.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_{suff} + \mathcal{L}_{min} + \mathcal{L}_{max}$
- ViT-S/16 backbone, lr=1e-4, batch size=32, warm-up + 50 epochs
- Theoretical guarantee: A formal theorem proves that minimally sufficient semantic representations minimize an upper bound on the Bayes error rate.

## Key Experimental Results

### Main Results

| Dataset | Label Ratio | MS-UDG | CycleMAE | SimCLR |
|---------|------------|--------|----------|--------|
| PACS | 100% | **72.89%** | 71.41% | 65.30% |
| DomainNet | 1% | **34.92%** | — | — |
| PACS | 1% | **56.2%** | 54.8% | 51.3% |

### Ablation Study

| Configuration | PACS Accuracy |
|---------------|--------------|
| $\mathcal{L}_{suff}$ only | 68.5% |
| + $\mathcal{L}_{min}$ | 70.8% |
| + $\mathcal{L}_{max}$ | **72.89%** |

### Key Findings
- Consistently outperforms baselines across all label ratios (1%/5%/10%/100%).
- The reconstruction loss $\mathcal{L}_{max}$ is critical for preventing information leakage — removing it causes a 2% drop.
- Generalizes consistently across all 6 domains of DomainNet.

## Highlights & Insights
- **Completeness of the information-theoretic framework**: Sufficiency + minimality yields optimal semantic representations, with theory and experiments mutually consistent.
- **No domain labels required**: Fourier augmentation combined with information disentanglement circumvents the need for domain annotations.
- **General-purpose framework**: IDM + SROM can be plugged into any SSL method.

## Limitations & Future Work
- Assumes semantic information is fully shared between two augmented views — extreme augmentations may violate this assumption.
- Fourier augmentation has limited capacity to simulate diverse domain shifts.
- Validation is restricted to the image domain.

## Related Work & Insights
- **vs. CycleMAE**: CycleMAE performs cyclic reconstruction but does not explicitly separate semantics from variation.
- **vs. SimCLR**: SimCLR relies solely on contrastive learning without disentanglement, allowing domain information to remain entangled in representations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Information-theoretic unsupervised domain generalization with theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple datasets, multiple label ratios, and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous theoretical derivations.
- **Value**: ⭐⭐⭐⭐ Provides a theoretically grounded new approach for UDG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2026/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[ICML 2026\] Statistical Consistency and Generalization of Contrastive Representation Learning](../../ICML2026/self_supervised/statistical_consistency_and_generalization_of_contrastive_representation_learnin.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](../../CVPR2026/self_supervised/geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)

</div>

<!-- RELATED:END -->
