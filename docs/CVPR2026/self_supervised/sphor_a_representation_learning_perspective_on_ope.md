---
title: >-
  [Paper Note] SpHOR: A Representation Learning Perspective on Open-set Recognition
description: >-
  [CVPR 2026 (Findings)][Self-Supervised Learning][open-set recognition] SpHOR proposes a two-stage decoupled training framework: Stage 1 performs OSR-tailored representation learning via orthogonal label embeddings, spherical constraints (vMF distribution), and Mixup/Label Smoothing; Stage 2 freezes the encoder and trains a linear classifier. The method achieves up to 5.1%/5.2% gains in OSCR/AUROC on the Semantic Shift Benchmark, and introduces two new metrics: Angular Separability and Norm Separability.
tags:
  - CVPR 2026 (Findings)
  - Self-Supervised Learning
  - open-set recognition
  - von Mises-Fisher
  - orthogonal embeddings
  - spherical representation
  - familiarity trap
date: 2026-05-08
content_hash: 5ccc06eef242c4e8
---

# SpHOR: A Representation Learning Perspective on Open-set Recognition

**Conference**: CVPR 2026 (Findings)
**arXiv**: [2503.08049](https://arxiv.org/abs/2503.08049)
**Code**: [https://github.com/nadarasarbahavan/SpHOR](https://github.com/nadarasarbahavan/SpHOR)
**Area**: Representation Learning / Open-set Recognition
**Keywords**: open-set recognition, von Mises-Fisher, orthogonal embeddings, spherical representation, familiarity trap

## TL;DR
SpHOR proposes a two-stage decoupled training framework: Stage 1 performs OSR-tailored representation learning via orthogonal label embeddings, spherical constraints (vMF distribution), and Mixup/Label Smoothing; Stage 2 freezes the encoder and trains a linear classifier. The method achieves up to 5.1%/5.2% gains in OSCR/AUROC on the Semantic Shift Benchmark, and introduces two new metrics: Angular Separability and Norm Separability.

## Background & Motivation
Open-set recognition (OSR) requires models to accurately classify known classes while detecting unseen unknown classes at test time. The core issue with existing OSR methods is that feature representations are not explicitly designed for unknowns: (1) most methods jointly train the backbone and classifier end-to-end, allowing only implicit adaptation of the feature space to unknown data; (2) unbounded feature magnitudes in Euclidean space lead to uncontrollable open-space risk; (3) general-purpose representation learning objectives such as SupCon are not designed for OSR. Vaze et al. showed that a simple closed-set classification baseline can match many OSR methods, with the key factor being representation quality. Whether explicitly designing the representation can further improve OSR performance is the starting point of SpHOR.

## Core Problem
How to tailor representation learning objectives specifically for OSR, so that the feature space explicitly reserves open space for unknown classes while preventing the *Familiarity Trap* — the high-confidence misclassification of unknowns that are semantically similar to known classes?

## Method

### Overall Architecture
Two-stage decoupled training: Stage 1 learns a spherical representation (Encoder + Projection network); Stage 2 discards the Projection network, freezes the Encoder, and trains a linear classifier on non-normalized features. At inference, scoring rules (MaxLogit/KNN/PostMax/NNGuide) are used for known/unknown binary classification.

### Key Designs
1. **Spherical Constraint + vMF Alignment Loss**: Features are L2-normalized onto a hypersphere, and each class is modeled as a vMF distribution. The vMFAL loss (Eq. 7) aligns the projected sample $z_i$ with the corresponding class label embedding $\mu_c$, while being compatible with soft labels from Mixup and Label Smoothing. Theorem 2 theoretically shows that this loss decomposes into an Alignment term (pulling samples toward the correct class embedding) and a Uniformity term (spreading samples uniformly around embeddings). For ambiguous samples ($\max(S_{ik}) \to 1/|C|$), the Uniformity term dominates and pushes ambiguous samples away from class centers, thereby alleviating the Familiarity Trap.

2. **Orthogonal Regularization $\mathcal{R}_{Ortho}$**: Prevents label embedding collapse — even when vMFAL optimizes feature-embedding alignment, all $\mu_k$ may tend toward collinearity. The regularizer enforces pairwise orthogonality of label embeddings: $\mathcal{R}_{Ortho} = \log \frac{1}{|C|^2 - |C|} \sum_{j \neq i} \exp(\frac{1}{\tau}(\mu_j \cdot \mu_i)^2)$. Compared to ETF-based methods, orthogonal constraints avoid negative correlations and feature redundancy.

3. **Mixup + Label Smoothing in Representation Learning**: The key innovation is shifting these two techniques from the classifier level to the representation learning stage. Mixup generates semantically ambiguous samples (simulating unknown classes), while Label Smoothing softens class labels. Ablations reveal complementary effects: Mixup improves Angular Separability (AS) and Label Smoothing improves Norm Separability (NS), with both metrics improving when used jointly.

### Loss & Training
- Stage 1: Encoder + 1024-dim linear Projection network, jointly trained with vMFAL + $\mathcal{R}_{Ortho}$
- Stage 2: Frozen Encoder extracts non-normalized features $f_i$; a linear classifier is trained with standard cross-entropy at negligible computational cost
- Training complexity $O(B \cdot C)$, significantly more efficient than SupCon's $O(B^2)$

## Key Experimental Results

| SSB (ImageNet pretrained) | Method | Avg Acc↑ | Avg AUROC (Easy/Hard)↑ | Avg OSCR (Easy/Hard)↑ |
|---|---|---|---|---|
| | MLS+MaxLogit | 84.9 | 84.12/74.78 | 75.24/70.83 |
| | MLS+Mixup+MaxLogit | 87.0 | 86.93/78.56 | 78.53/74.84 |
| | SupCon+MaxLogit | 82.9 | 87.48/78.21 | 78.67/71.44 |
| | **SpHOR+MaxLogit** | **92.6** | **93.00/83.20** | **88.40/80.00** |

- Up to 5.1% OSCR and 5.2% AUROC improvement on SSB (vs. SupCon)
- Legacy Benchmark A: average AUROC 94.6 (+0.8 over ConOSR at 93.9)
- Legacy Benchmark B: average AUROC 94.0 (+1.0 over RCSSR at 93.0)
- Without pretraining, SpHOR remains robust: MLS+Mixup AUROC drops 20–30%, while SpHOR drops only marginally
- Small-batch robustness: at B=16, SpHOR OSCR is 81.8 vs. SupCon's 62.9

### Ablation Study
- Mixup + LS combined: Avg Acc 89.56→92.60, AUROC (Easy) 86.94→93.00, OSCR (Easy) 81.72→88.40
- $\mathcal{R}_{Ortho}$ increases label embedding Dispersion (inter-class angular distance), improving AUROC on 3/4 datasets
- AS and NS metrics reveal complementary roles: Mixup improves AS (angular separation), LS improves NS (norm separation)
- MaxLogit is the most stable scoring rule; SpHOR is the least sensitive to scoring rule choice (std 0.99/0.51 vs. SupCon 5.70/3.40)

## Highlights & Insights
- Decoupled training with OSR-tailored representation: distinct from general-purpose methods like SupCon, with theoretical analysis of how vMFAL promotes Alignment and Uniformity
- The insight of using Mixup at the representation learning stage to generate "surrogate unknown" samples is elegant — semantic ambiguity in Mixup samples naturally models open space
- AS and NS metrics provide an analytical toolkit for explaining the complementary mechanisms of Mixup and Label Smoothing, benefiting future work
- High training efficiency: $O(B \cdot C)$ vs. SupCon's $O(B^2)$, with stability under small batch sizes

## Limitations & Future Work
- Orthogonal constraints require embedding dimension $p \geq |C|$, which may be limiting for large-scale fine-grained classification (e.g., 1000+ classes)
- Validation is primarily on ResNet50; Transformer backbones (e.g., ViT) are insufficiently evaluated
- $\mathcal{R}_{Ortho}$ yields limited improvement on some datasets (slight AUROC drop on Aircraft), and dataset-dependent behavior warrants further investigation
- Validation is confined to image classification; generalization to other modalities (text, multimodal) remains unexplored

## Related Work & Insights
- **vs. MLS (Vaze et al.)**: MLS is a closed-set training baseline; SpHOR explicitly designs spherical representations, improving Avg OSCR from ~75 to 88
- **vs. SupCon (ConOSR)**: SupCon uses general-purpose contrastive learning without OSR-specific design; SpHOR with vMF + orthogonal embeddings is more robust to scoring rule choice and more stable under small batches
- **vs. ARPL**: ARPL uses reciprocal points in Euclidean space with unbounded open space; SpHOR's spherical constraint naturally bounds open space
- **vs. HAFrame/Hier-COS**: These methods focus on hierarchical classification; SpHOR targets known/unknown binary detection with a different objective for orthogonal embeddings

The vMF + orthogonal embedding framework may generalize to other scenarios requiring explicit open-space reservation (e.g., anomaly detection, novel category discovery). The AS/NS metrics can be applied to analyze open-set separability in arbitrary feature spaces.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of three innovations (spherical constraints + orthogonal embeddings + Mixup/LS at the representation stage) is effective, with thorough theoretical analysis
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ SSB + two Legacy benchmarks, multiple scoring rules, detailed ablations, and new metric analysis
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and ablation analysis is in-depth, though notation is dense
- **Value**: ⭐⭐⭐⭐ Provides a systematic methodology for representation design in OSR; AS/NS metrics have independent value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)
- [\[AAAI 2026\] Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](../../AAAI2026/self_supervised/let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)

</div>

<!-- RELATED:END -->
