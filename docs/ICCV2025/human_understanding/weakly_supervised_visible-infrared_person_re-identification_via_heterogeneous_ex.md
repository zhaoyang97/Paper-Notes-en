---
title: >-
  [Paper Note] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning
description: >-
  [ICCV 2025][Human Understanding][Visible-Infrared Person Re-Identification] This paper proposes the first weakly supervised paradigm for visible-infrared person re-identification (VIReID), which relies solely on intra-modality identity annotations (without cross-modal correspondence labels). A heterogeneous expert collaborative consistency learning framework is introduced to establish cross-modal identity correspondences, achieving performance close to fully supervised methods.
tags:
  - ICCV 2025
  - Human Understanding
  - Visible-Infrared Person Re-Identification
  - Weakly Supervised Learning
  - Cross-Modal Matching
  - Heterogeneous Experts
  - Collaborative Consistency Learning
date: 2026-05-08
content_hash: 8be8b39e8ade12cd
---

# Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning

**Conference**: ICCV 2025
**arXiv**: [2507.12942](https://arxiv.org/abs/2507.12942)
**Code**: [GitHub](https://github.com/KongLingqi2333/WSL-VIReID)
**Area**: Human Understanding
**Keywords**: Visible-Infrared Person Re-Identification, Weakly Supervised Learning, Cross-Modal Matching, Heterogeneous Experts, Collaborative Consistency Learning

## TL;DR

This paper proposes the first weakly supervised paradigm for visible-infrared person re-identification (VIReID), which relies solely on intra-modality identity annotations (without cross-modal correspondence labels). A heterogeneous expert collaborative consistency learning framework is introduced to establish cross-modal identity correspondences, achieving performance close to fully supervised methods.

## Background & Motivation

Visible-infrared person re-identification (VIReID) is a core task in intelligent surveillance, requiring the matching of the same pedestrian across visible and infrared images. Existing methods face annotation challenges at three levels:

**Fully supervised methods**: Require accurate cross-modal identity correspondence annotations. However, visible and infrared cameras typically operate at different periods (day/night), making it costly and difficult to establish direct cross-modal pedestrian correspondences.

**Semi-supervised methods**: Utilize single-modality annotations combined with unlabeled data from the other modality. However, pseudo-label noise in the unlabeled modality severely degrades performance.

**Unsupervised methods**: Generate pseudo-labels via clustering without any annotations. Cross-modal pseudo-label quality is poor, and noise accumulation limits the performance ceiling.

**The proposed weakly supervised setting**: Each modality has intra-modality identity annotations (i.e., identity labels are available within the visible modality and within the infrared modality separately), but the cross-modal identity correspondences remain unknown. This setting offers two key advantages:
1. Intra-modality annotation is relatively easy (identity recognition is straightforward under the same camera and spectral characteristics).
2. It avoids the cumulative pseudo-label noise inherent in unsupervised methods.

## Method

### Overall Architecture

The method consists of two stages:
1. **Heterogeneous Expert Learning (HEL)**: Identity classification experts are trained independently within each modality.
2. **Collaborative Consistency Learning (CCL)**: The experts perform cross-modal identity prediction to establish correspondences and enable collaborative training.

### Key Designs

1. **Heterogeneous Expert Learning (HEL)**:

    - **Function**: Independently trains an identity classification expert for each modality.
    - **Mechanism**: ResNet-50 is adopted as the backbone. The early layers are modality-specific (non-shared), while subsequent convolutional layers share parameters. Separate classifiers $\boldsymbol{W}^v$ and $\boldsymbol{W}^r$ are built for the visible and infrared modalities, respectively. Training employs a joint loss of cross-entropy and weighted regularized triplet loss:
    $\mathcal{L}_{phase1} = \mathcal{L}_{id}^{exp} + \lambda_1 \mathcal{L}_{wrt}^{intr}$
      where the cross-entropy loss is: $\mathcal{L}_{id}^{exp} = -\sum_{t \in \{v,r\}} \frac{1}{n^t} \sum_i \boldsymbol{y}_i^t \log \boldsymbol{p}_i^t$
    - **Design Motivation**: The two experts are trained on different modalities and thus focus on different identity-relevant cues, hence the term "heterogeneous experts." Each expert attains strong intra-modality discriminability, providing the foundation for subsequent cross-modal prediction.

2. **Cross-Modal Relation Establishment (CRE)**:

    - **Function**: Fuses cross-modal predictions from both experts to establish reliable cross-modal identity correspondences.
    - **Mechanism**: Each expert predicts identities of samples from the other modality. A Count Priority Selection strategy yields decision matrices $\boldsymbol{M}^{t \to \bar{t}}$. Cross-modal correspondences are categorized into three types:
        - **Consistent matches $\boldsymbol{M}_c$**: Both experts agree on the identity correspondence — most reliable.
       $$\boldsymbol{M}_c = \boldsymbol{M}^{t \to \bar{t}} \odot (\boldsymbol{M}^{\bar{t} \to t})^T$$
        - **Single matches $\boldsymbol{M}_s$**: Only one expert provides a correspondence prediction while the other is indecisive.
        - **Contradictory matches $\boldsymbol{M}_w$**: The two experts give conflicting correspondence predictions.
       $$\boldsymbol{M}_w = \boldsymbol{M}^{v \to r} + (\boldsymbol{M}^{r \to v})^T - 2\boldsymbol{M}_c - \boldsymbol{M}_s$$
    - **Design Motivation**: A single expert may produce inaccurate cross-modal predictions, but mutual agreement between two experts yields high confidence. Categorizing correspondences into three types enables differentiated training strategies according to reliability level.

3. **Collaborative Consistency Learning (CCL)**:

    - **Function**: Uses cross-modal correspondences to constrain the encoder to learn modality-invariant features, while progressively enhancing experts' cross-modal discriminability.
    - **Mechanism** comprises two components:
        - **Cross-Modal Consistency Learning (CMCL)**:
       - For consistent/single-match samples, a strong constraint (cross-modal cross-entropy) is applied: $\mathcal{L}_{id}^{stro}$
       - For contradictory-match samples, a weak constraint (excluding impossible identities) is applied: $\mathcal{L}_{id}^{weak} = -\frac{1}{n_w^v} \sum_i \boldsymbol{m}_i \log(1 - \boldsymbol{W}^c(\boldsymbol{f}_i^v) + \epsilon)$
        - **Collaborative Learning of Asymmetric Experts (CLAE)**:
       - Identity prototype features are maintained for each modality: $\mathcal{P}_i^t \leftarrow \lambda \mathcal{P}_i^t + (1-\lambda) \bar{\boldsymbol{f}}_i^t$
       - A collaborative consistency loss encourages the experts to produce consistent predictions for cross-modal positive pairs: $\mathcal{L}_{homo}^v = \frac{1}{n^c \times C^v} \sum_i \| \boldsymbol{p}_i^{v \to v} - \boldsymbol{p}_i^{r \to v} \|_2^2$
       - Information entropy is used to adaptively modulate constraint strength: the more uncertain the prediction, the stronger the collaborative constraint.
    - **Design Motivation**: CMCL drives the encoder to learn cross-modal consistent features, while CLAE progressively improves experts' cross-modal prediction capability. The two components reinforce each other in a positive feedback loop. The weak constraint design prevents contradictory labels from harming training.

### Loss & Training

The total loss for the collaborative consistency learning stage:
$$\mathcal{L}_{phase2} = \mathcal{L}_{id}^{exp} + \mathcal{L}_{id}^{stro} + \mathcal{L}_{homo} + \lambda_1 \mathcal{L}_{wrt}^{cros} + \lambda_2 \mathcal{L}_{id}^{weak}$$

Training hyperparameters:
- Momentum update coefficient $\lambda = 0.8$, $\lambda_1 = \lambda_2 = 0.25$
- Initial learning rate: $3 \times 10^{-4}$ for the encoder, $6 \times 10^{-4}$ for experts and shared classifiers
- CCL stage: 120 epochs with 10 warmup epochs

## Key Experimental Results

### Main Results

**SYSU-MM01 (All Search mode)**:

| Method | Type | Rank-1 | mAP | Gain |
|--------|------|--------|-----|------|
| DPIS (ICCV'23) | Semi-supervised | 58.4 | 55.6 | - |
| GUR (ICCV'23) | Unsupervised | 63.5 | 61.6 | - |
| DEEN (CVPR'23) | Fully supervised | 74.7 | 71.8 | - |
| **Ours** | **Weakly supervised** | **70.4** | **66.6** | +12.0/+11.0 vs DPIS |

**LLCM (VIS to IR mode)**:

| Method | Type | Rank-1 | mAP | Gain |
|--------|------|--------|-----|------|
| OTLA (ECCV'22) | Semi-supervised | 44.2 | 48.2 | - |
| PGM (CVPR'23) | Unsupervised | 44.9 | 49.0 | - |
| DEEN (CVPR'23) | Fully supervised | 62.5 | 65.8 | - |
| **Ours** | **Weakly supervised** | **55.3** | **58.7** | +11.1/+10.5 vs OTLA |

### Ablation Study

**Contribution of each module (SYSU-MM01 All Search)**:

| Configuration | Rank-1 | mAP | Note |
|---------------|--------|-----|------|
| Baseline (HEL only) | 47.8 | 47.2 | Intra-modality training only |
| B + CMCL\CRE | 66.7 | 62.8 | CMCL with consistent predictions only |
| B + CRE + CMCL | 68.3 | 64.5 | Adding relation fusion |
| B + CRE + CMCL + CLAE | **70.4** | **66.6** | Full model |

### Key Findings

1. Intra-modality training alone (Baseline) already achieves non-trivial cross-modal retrieval (47.8% Rank-1), indicating that shared-parameter layers capture preliminary modality-invariant features.
2. CMCL contributes the largest gain (+18.9% Rank-1), confirming that establishing cross-modal correspondences is the core contribution.
3. Relation fusion via CRE further improves Rank-1 by 1.6%, demonstrating that fusing predictions from two experts outperforms relying on a single expert.
4. CLAE provides an additional 2.1% Rank-1 gain, showing that experts continuously improve cross-modal discriminability during training.
5. The weakly supervised performance (70.4%) approaches that of the fully supervised method DEEN (74.7%), with only a 4.3% gap, while substantially reducing annotation cost.

## Highlights & Insights

- **The weakly supervised paradigm is elegantly designed**: it avoids the expensive annotations of fully supervised methods while providing more reliable intra-modality supervision signals than semi-supervised or unsupervised approaches.
- **Hierarchical treatment of three correspondence types** is the core innovation: consistent, single, and contradictory matches are handled with strong, moderate, and weak (exclusion-based) constraints, respectively, mitigating the harmful effects of noisy labels.
- **Expert collaborative learning** forms a positive feedback loop: better correspondences → better features → more accurate expert predictions → better correspondences.
- Entropy-based adaptive weighting realizes the intuition that "greater uncertainty warrants stronger constraint."

## Limitations & Future Work

- The heterogeneous experts employ simple classifiers; more sophisticated expert architectures (e.g., Mixture of Experts) could be explored.
- Correspondence establishment is a discrete, one-shot process; end-to-end soft alignment approaches (e.g., continuous relaxation via optimal transport) are worth investigating.
- Validation is conducted only on SYSU-MM01 and LLCM; generalizability requires further verification on additional datasets.
- The weakly supervised setting assumes complete intra-modality identity annotations, which may still incur substantial annotation cost in extreme scenarios.

## Related Work & Insights

- The optimal transport strategy in OTLA (ECCV 2022) is complementary to the proposed CRE and could be integrated in future work.
- Contrastive learning frameworks (e.g., InfoNCE) may replace cross-entropy loss for more robust cross-modal alignment.
- The weakly supervised paradigm has potential for generalization to other cross-modal matching tasks (e.g., RGB-depth, RGB-text person re-identification).
- The expert collaboration mechanism can draw inspiration from the momentum teacher paradigm in self-supervised learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Pioneers the weakly supervised VIReID paradigm; the three-category correspondence design is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablation study; thorough comparison across fully supervised, semi-supervised, and unsupervised paradigms.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation and method description are clear; mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Significantly reduces annotation cost in practical deployment while maintaining competitive performance.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Modality-Aware Bias Mitigation and Invariance Learning for Unsupervised Visible-Infrared Person Re-Identification](../../AAAI2026/human_understanding/modality-aware_bias_mitigation_and_invariance_learning_for_unsupervised_visible-.md)
- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ICCV 2025\] IDFace: Face Template Protection for Efficient and Secure Identification](idface_face_template_protection_for_efficient_and_secure_identification.md)

<!-- RELATED:END -->
