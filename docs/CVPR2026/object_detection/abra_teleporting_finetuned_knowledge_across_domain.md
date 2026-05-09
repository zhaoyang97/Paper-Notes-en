---
title: >-
  [Paper Note] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection
description: >-
  [CVPR 2026][Object Detection][Open-vocabulary detection] This paper formulates cross-domain category transfer as SVD rotation alignment in weight space: domain-agnostic experts are trained via Objectification, lightweight class residuals are extracted with SVFT, and a closed-form orthogonal Procrustes solution is used to "teleport" source-domain class knowledge to a target domain with no data for that class.
tags:
  - CVPR 2026
  - Object Detection
  - Open-vocabulary detection
  - domain adaptation
  - weight-space transfer
  - SVD rotation alignment
  - Objectification
  - SVFT
date: 2026-05-08
content_hash: e21f9122ee887769
---

# ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.12409](https://arxiv.org/abs/2603.12409)
**Code**: To be released
**Area**: Object Detection / Domain Adaptation / Open-Vocabulary
**Keywords**: Open-vocabulary detection, domain adaptation, weight-space transfer, SVD rotation alignment, Objectification, SVFT

## TL;DR

This paper formulates cross-domain category transfer as SVD rotation alignment in weight space: domain-agnostic experts are trained via Objectification, lightweight class residuals are extracted with SVFT, and a closed-form orthogonal Procrustes solution is used to "teleport" source-domain class knowledge to a target domain with no data for that class.

## Background & Motivation

**Background**: Open-vocabulary detectors (e.g., Grounding DINO) can detect arbitrary categories via text prompts, but suffer severe performance degradation under domain shift (night/fog/rain). Conventional DAOD methods rely on Mean Teacher with pseudo-labels, which become unreliable under large domain shifts.

**Limitations of Prior Work**:

1. Standard DAOD assumes target-domain images exist for all categories (even if unannotated), whereas in practice rare categories may have no target-domain data whatsoever—neither annotations nor images.
2. Weight-space methods such as Task Arithmetic ignore the rotational difference between source and target SVD subspaces, making naive residual addition/subtraction ineffective.
3. There is no existing method to transfer a category across domains when that category is completely absent from the target domain.

**Key Challenge**: The target domain must detect a certain category, yet that category is entirely invisible in the target domain (zero-shot class transfer across domains).

**Goal**: Transfer the category-detection capability learned in the source domain to a target domain that contains no data for that category.

**Key Insight**: Domain knowledge and class knowledge are disentanglable—domain experts capture visual statistics (illumination/texture/weather), while class experts capture category semantics. By aligning the SVD bases of the two domains, class residuals can be "teleported" across domains.

**Core Idea**: $\theta_T^{(c)} \approx U_T(\Sigma_T + U_T^\top U_S \cdot \Delta\Sigma_S^{(c)} \cdot V_S^\top V_T) V_T^\top$, yielding a closed-form solution that requires no training.

## Method

### Overall Architecture

Grounding DINO pretrained weights $\theta_0$ → Objectification training of source/target domain experts $\theta_S, \theta_T$ → SVFT training of lightweight class residuals $\Delta\Sigma_S^{(c)}$ on the source domain → orthogonal Procrustes closed-form teleportation to the target domain → target-domain class expert $\hat{\theta}_T^{(c)}$.

### Key Designs

1. **Objectification**

    - Annotations for the top-3 most frequent categories are replaced with a unified "object" label, training a class-agnostic domain expert.
    - The model is forced to learn domain visual statistics (illumination patterns, texture features, weather conditions) rather than category semantics.
    - One domain expert is trained per domain: $\theta_S = \text{Fine-Tune}(\theta_0, \tilde{\mathcal{D}}_S)$.
    - Annotations for non-top-3 categories are discarded to prevent low-frequency classes from introducing class bias.

2. **SVFT Class Expert Training**

    - SVD decomposition is applied to the domain expert: $\theta_{S,\ell} = U_{S,\ell} \Sigma_{S,\ell} V_{S,\ell}^\top$.
    - $U$, $\Sigma$, and $V$ are frozen; only the extremely lightweight singular-value residual $\Delta\Sigma_{S,\ell}^{(c)}$ (diagonal or banded matrix) is trained.
    - Forward pass: $f_\ell(x) = U_{S,\ell}(\Sigma_{S,\ell} + \Delta\Sigma_{S,\ell}^{(c)}) V_{S,\ell}^\top x$.
    - Each class expert requires only a small amount of data and few parameters to train.
    - During training, only images containing the target category are retained, and annotations for other categories are masked.

3. **Orthogonal Procrustes Teleportation**

    - Core operation: rotate the source class residual into the target-domain SVD basis: $\pi_{S \rightarrow T}(\Delta\Sigma_S^{(c)}) = L \Delta\Sigma_S^{(c)} R^\top$.
    - The orthogonal Procrustes problem is solved as: $L^* = U_T^\top U_S$, $R^* = V_T^\top V_S$.
    - The resulting target-domain class expert is: $\theta_{T,\ell}^{(c)} = U_T(\Sigma_T + U_T^\top U_S \Delta\Sigma_S^{(c)} V_S^\top V_T) V_T^\top$.
    - The solution is entirely closed-form, requiring no training iterations or target-domain class data.

### Loss & Training

- Domain experts: encoder attention layers, 10 epochs, lr=1e-4, batch=2.
- Class experts: SVFT, 12 epochs, lr=1e-2, batch=4.
- Teleportation stage: no training; pure matrix operations.

## Key Experimental Results

### Main Results

**Cityscapes → Foggy Cityscapes (average over 5 unseen classes)**

| Method | mAP | AP50 |
|--------|-----|------|
| Zero shot | 27.66 | 44.12 |
| Source (direct evaluation of source-domain class expert) | 38.25 | 57.34 |
| Task Analogy | 18.12 | 26.79 |
| ParamΔ | 28.29 | 44.42 |
| **ABRA** | **40.54** | **61.06** |
| Fine-tuning (upper bound) | 41.36 | 62.48 |

ABRA achieves on average 98% of the fine-tuning upper bound across the 5 transferred classes.

**SDGOD — four domain shifts (average)**

| Method | mAP | AP50 |
|--------|-----|------|
| Zero shot | 20.65 | 34.82 |
| Source | 27.76 | 48.99 |
| ParamΔ | 8.87 | 13.85 |
| **ABRA** | **28.10** | **50.57** |
| Fine-tuning | 29.20 | 51.93 |

ParamΔ collapses on SDGOD (the identity mapping fails under aggressive domain shift), while ABRA remains robust.

### Ablation Study

| Ablation | mAP |
|----------|-----|
| Zero Shot w/ Obj. (merged detection boxes) | 36.20 |
| Supervised (retaining semantic labels) | 38.88 |
| **Objectification** | **40.54** |

- Objectification outperforms retaining original semantic labels, confirming that class-agnostic training is critical for cross-domain transfer.
- Fine-tuning and FDA initialized with ABRA both outperform standard $\theta_0$ initialization (FFT: 42.80 vs. 41.36).

### Key Findings

- The failure of Task Analogy and ParamΔ demonstrates that naive weight addition/subtraction is insufficient; basis alignment is necessary.
- ABRA consistently outperforms standard pretraining as an initialization in few-shot settings.
- Training independent class experts per category outperforms a single expert trained on all categories jointly.
- ABRA remains competitive on Night Rainy, the most challenging domain.

## Highlights & Insights

- SVD rotation alignment in weight space offers an elegant domain adaptation paradigm that completely bypasses feature alignment and adversarial training.
- Objectification is a clever class–domain disentanglement strategy: assigning a unified "object" label strips away class semantics.
- SVFT residuals are extremely lightweight (diagonal matrices only), enabling parallel training and efficient storage across multiple classes.
- The closed-form solution yields zero-latency teleportation, making it deployment-friendly for rapid adaptation to new domains.

## Limitations & Future Work

- The orthogonal Procrustes assumption requires well-corresponding SVD subspaces between source and target domains; validity under extreme domain shifts (e.g., CT → ultrasound) remains to be verified.
- Objectification uses only the top-3 categories; the optimal number of categories may vary across datasets.
- Experiments are conducted solely on Grounding DINO; generalization to other OVD architectures (e.g., YOLO-World) requires further validation.
- Comparisons with recent multi-source domain adaptation methods are absent.

## Related Work & Insights

- **vs. Task Arithmetic**: Ignoring SVD rotational differences causes transfer failure (mAP drops from 40.54 to 18.12).
- **vs. ParamΔ**: The identity-mapping assumption collapses on SDGOD (mAP drops to 8.87), demonstrating that rotation alignment cannot be omitted.
- **vs. Mean Teacher DAOD**: Mean Teacher requires target-domain images, whereas ABRA requires no target-domain class data whatsoever.
- **vs. Model Rebasin**: Shares a similar conceptual basis, but ABRA specializes it for detection domain adaptation and introduces Objectification.
- Insight: Weight-space operations and SVFT are broadly applicable to cross-domain deployment of segmentation and classification models.

## Rating

- Novelty: ⭐⭐⭐⭐ Modeling domain adaptation as weight rotation is a novel perspective; the Objectification design is ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple domain-shift scenarios, comprehensive ablations, and few-shot analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation and methodological derivation with consistent mathematical notation.
- Value: ⭐⭐⭐⭐ High practical value for weight-space transfer; the closed-form solution is deployment-friendly.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation_for_open_world_object_dete.md)
- [\[CVPR 2026\] A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps](a_closer_look_at_cross-domain_few-shot_object_detection_fine-tuning_matters_and_.md)

<!-- RELATED:END -->
