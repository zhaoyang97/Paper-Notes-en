---
title: >-
  [Paper Note] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper proposes FedDAP, a domain-aware prototype federated learning framework that addresses global model performance degradation caused by client-side domain shift in federated learning. FedDAP constructs domain-specific global prototypes and employs a dual prototype alignment strategy comprising intra-domain alignment and cross-domain contrastive learning.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Learning
  - Domain Shift
  - Prototype Learning
  - Contrastive Learning
  - Domain-Aware
date: 2026-05-08
content_hash: 03e5b542694bab20
---

# FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift

**Conference**: CVPR 2026
**arXiv**: [2604.06795](https://arxiv.org/abs/2604.06795)
**Code**: [GitHub](https://github.com/quanghuy6997/FedDAP)
**Area**: AI Security
**Keywords**: Federated Learning, Domain Shift, Prototype Learning, Contrastive Learning, Domain-Aware

## TL;DR

This paper proposes FedDAP, a domain-aware prototype federated learning framework that addresses global model performance degradation caused by client-side domain shift in federated learning. FedDAP constructs domain-specific global prototypes and employs a dual prototype alignment strategy comprising intra-domain alignment and cross-domain contrastive learning.

## Background & Motivation

Federated learning (FL) enables multiple clients to collaboratively train models without exposing private data. In real-world scenarios, however, data across clients often originates from different domains (e.g., varying sensors, environments, or image styles), leading to severe domain shift.

Existing prototype-driven FL methods exhibit two critical limitations:

**Semantic dilution from a single global prototype**: Constructing a single global prototype per class by naively averaging local prototypes across all clients discards domain information. When "dog" appears with natural textures in the photo domain but as simplified line strokes in the sketch domain, the averaged prototype fails to faithfully represent either domain.

**Domain-agnostic alignment strategy**: Forcing all clients to align their features toward the same global prototype, regardless of domain origin, introduces a form of domain-blind supervision that ignores the semantic discrepancy between local distributions and global prototypes.

## Method

### Overall Architecture

The FedDAP framework consists of three stages:
1. Clients compute local prototypes and upload them to the server.
2. The server constructs domain-specific global prototypes for each (class, domain) pair via a cosine-similarity-weighted aggregation mechanism.
3. Clients download the global prototypes and perform local training using a dual prototype alignment strategy.

### Key Designs

1. **Domain-Specific Global Prototype Aggregation**: Rather than simple averaging, an independent prototype is constructed for each (class, domain) pair. The server collects all client prototypes belonging to the same class within the same domain, computes a cosine similarity consistency score $S_j$ among them, derives attention weights $\alpha_j$ via softmax normalization, and obtains the domain-specific global prototype $\mathbf{P}^{(c,d)}$ as a weighted sum. A temperature parameter $\tau_{agg}$ controls the sharpness of the weight distribution. This mechanism emphasizes semantically consistent prototypes within a domain while suppressing the influence of outlier prototypes.

2. **Domain-consistent Prototype Alignment (DPA)**: Client local features are aligned with same-domain prototypes (intra-domain prototypes) using a cosine similarity loss $\mathcal{L}_{DPA} = \sum_{c}(1 - \cos(z_i^c, \mathbf{P}^{(c,d_m)}))$, ensuring semantic consistency between features and class-specific, domain-relevant prototypes. This formulation is more robust to scale variations.

3. **Cross-domain Prototype Contrastive Learning (CPCL)**: Prototypes from other domains are leveraged for contrastive learning. Positive samples are same-class prototypes from other domains; negative samples are different-class prototypes from other domains. An InfoNCE loss pulls features toward cross-domain same-class prototypes and pushes them away from cross-domain different-class prototypes, encouraging the model to learn domain-invariant semantic representations and enhancing cross-domain generalization.

### Loss & Training

The total loss is a weighted combination of three terms:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{DPA} + \lambda_2 \mathcal{L}_{CPCL}$$

- $\mathcal{L}_{CE}$: Standard cross-entropy classification loss
- $\lambda_1 = 10$: Intra-domain alignment weight (a larger value stabilizes domain-specific prototypes)
- $\lambda_2 = 1$: Cross-domain contrastive weight (an excessively large value over-regularizes and disrupts domain-specific structure)
- Training runs for 100 communication rounds with 10 local epochs per round.

## Key Experimental Results

### Main Results

| Dataset | Metric | FedDAP | FedPLVM | FedRDN | FedAvg | Gain (vs FedAvg) |
|--------|------|--------|---------|--------|--------|----------------|
| DomainNet | Avg Acc | **65.20** | 62.22 | 61.01 | 59.59 | +5.61 |
| Office-10 | Avg Acc | **72.53** | 68.77 | 65.54 | 57.47 | +15.06 |
| PACS | Avg Acc | **84.63** | 82.06 | 83.17 | 77.07 | +7.56 |

### Ablation Study

| Configuration (DPA / CPCL) | DomainNet | Office-10 | PACS |
|-------------------|-----------|-----------|------|
| ✗ / ✗ (FedAvg) | 59.59 | 57.47 | 77.07 |
| ✗ / ✓ | 62.86 | 62.18 | 81.87 |
| ✓ / ✗ | 62.61 | 68.53 | 78.74 |
| ✓ / ✓ (Full) | **65.20** | **72.53** | **84.63** |

### Key Findings

1. **Both components are complementary and necessary**: DPA improves performance through intra-domain consistency, while CPCL enhances generalization through cross-domain contrastive learning; their combination yields the best results. Notably, DPA delivers a substantially larger gain on Office-10 (+11.06) than CPCL (+4.71), indicating that intra-domain alignment is especially critical when domain discrepancy is pronounced.

2. **Domain-specific aggregation vs. simple averaging**: Cosine-similarity-weighted aggregation outperforms simple averaging by 1.05%/1.04%/1.41% on the three datasets, respectively. Even domain-aware prototypes aggregated via simple averaging significantly outperform domain-agnostic prototypes, validating the core value of domain-specific prototypes.

3. **Faster convergence**: FedDAP reaches higher accuracy in fewer communication rounds across all three datasets.

4. **Cross-domain generalization**: Under leave-one-domain-out evaluation, FedDAP outperforms the second-best method by +1.88% and +1.17% on DomainNet and Office-10, respectively.

5. **t-SNE visualization**: Compared to the diffuse and overlapping feature clusters produced by FedProto, FedDAP yields more compact and better-separated class clusters.

## Highlights & Insights

- The core design philosophy is clear: expanding the prototype space from a one-dimensional "class" space to a two-dimensional "class × domain" space is a natural and effective approach to handling domain shift in FL.
- The dual alignment strategy offers a principled design philosophy worth emulating: intra-domain alignment maintains stability, while cross-domain contrastive learning enhances generalization, and the two are mutually reinforcing.
- Cosine-similarity-weighted aggregation outperforms simple averaging, though the margin is moderate (~1%), suggesting that the construction of domain-specific prototypes is more critical than the choice of aggregation method.

## Limitations & Future Work

- The framework assumes that domain labels are known for each client (requiring a domain identifier $d$); additional domain discovery mechanisms would be needed for implicit domain shift scenarios.
- The number of domains $D$ must be specified in advance, which may be unsuitable for real-world settings with ambiguous domain boundaries.
- Experiments are conducted solely on image classification tasks and have not been extended to more complex vision tasks such as detection or segmentation.
- Scalability with respect to the number of clients is insufficiently analyzed (experiments involve at most 20 clients).
- The additional bandwidth overhead introduced by prototype communication is not quantitatively analyzed in the paper.

## Related Work & Insights

- FedProto first introduced prototypes into FL but relies on domain-agnostic single global prototypes.
- FedPLVM and FPL improve prototype quality but still disregard domain information.
- FedRDN (CVPR'25) is a recent FL method addressing domain shift, but underperforms FedDAP in the reported experiments.
- Compared to domain generalization methods (COPA, FedGA), FedDAP directly models domain-specific semantic structures rather than learning domain-invariant representations.

## Rating

- Novelty: ⭐⭐⭐⭐ (Domain-specific prototypes combined with a dual alignment strategy constitute a meaningful extension of prototype-based FL)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, extensive ablation studies, and parameter sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-motivated problem formulation)
- Value: ⭐⭐⭐⭐ (Practically applicable, open-sourced code, meaningful contribution to the FL domain shift problem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2026\] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance](federated_active_learning_extreme_noniid.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)

</div>

<!-- RELATED:END -->
