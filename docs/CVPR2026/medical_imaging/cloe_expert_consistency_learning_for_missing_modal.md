---
title: >-
  [Paper Note] CLoE: Expert Consistency Learning for Missing Modality Segmentation
description: >-
  [CVPR 2026][Medical Imaging][missing modality] This work reformulates the robustness problem under missing modalities as decision-level expert consistency control. It proposes a dual-branch consistency learning scheme (global MEC + regional REC) coupled with a lightweight gating network that converts consistency scores into modality reliability weights, achieving an average WT Dice of 88.09% across 15 missing-modality combinations on BraTS 2020, surpassing all prior state-of-the-art methods.
tags:
  - CVPR 2026
  - Medical Imaging
  - missing modality
  - consistency learning
  - expert fusion
  - reliability gating
  - brain tumor
date: 2026-05-08
content_hash: 73faacbfa785a171
---

# CLoE: Expert Consistency Learning for Missing Modality Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.09316](https://arxiv.org/abs/2603.09316)
**Code**: N/A
**Area**: Medical Image Segmentation / Multimodal Learning
**Keywords**: missing modality, consistency learning, expert fusion, reliability gating, brain tumor

## TL;DR

This work reformulates the robustness problem under missing modalities as decision-level expert consistency control. It proposes a dual-branch consistency learning scheme (global MEC + regional REC) coupled with a lightweight gating network that converts consistency scores into modality reliability weights, achieving an average WT Dice of 88.09% across 15 missing-modality combinations on BraTS 2020, surpassing all prior state-of-the-art methods.

## Background & Motivation

**Background**: Multimodal MRI segmentation (e.g., four-modality T1/T1c/T2/FLAIR for brain tumor) typically assumes full modality availability at training time, employing encoder–decoder architectures such as U-Net and V-Net. In clinical practice, however, modality absence due to interrupted scans, protocol differences, or quality issues is extremely common.

**Limitations of Prior Work**: (1) GAN-based synthesis of missing modalities is computationally expensive and prone to hallucinations; (2) arithmetic fusion methods such as HeMIS cause attention mechanisms to fail when zero-padding is applied; (3) the spatial priors in RFNet are passive—they specify where to attend but provide no signal as to which expert is trustworthy; (4) consistency learning approaches such as Mean Teacher are dominated by background voxels in volumetric MRI, so global consistency does not imply alignment on small foreground structures.

**Key Challenge**: Modality absence is not merely a problem of reduced information; it also amplifies prediction disagreement among modality experts. Naïve fusion can amplify such disagreement rather than resolve it, particularly for small foreground structures of clinical importance.

**Goal**: How can inter-expert consistency be quantified as a reliability signal, and how can that signal guide dynamic fusion?

**Key Insight**: Elevate robustness from the representation level to the decision level—rather than learning better features, the goal is to control which expert's output is trusted.

**Core Idea**: Cosine similarity between expert predictions is used to quantify consistency at two levels—global (MEC) and foreground-region (REC)—and a gating network maps these consistency scores to fusion weights.

## Method

### Overall Architecture

CLoE comprises four core components: (1) parallel modality encoders $\Phi_m$ that extract multi-scale features; (2) a shared expert decoder $D^{sep}$ that generates single-modality predictions $p^{(m)}$; (3) an ECL module that computes MEC/REC consistency scores and maps them to fusion weights $w_m$ via a gating network; and (4) a fusion decoder $D^{fuse}$ that takes the weighted fused features as input to produce the final segmentation.

### Key Designs

1. **Modality Expert Consistency (MEC) + Regional Expert Consistency (REC)**
   - MEC: Probability prediction vectors from available experts are flattened and their pairwise cosine similarities are computed; the average over all available pairs serves as the loss: $\mathcal{L}_{MEC} = \frac{1}{|\mathcal{P}|}\sum_{(a,b)}(1-\mathcal{S}(p^{(a)}, p^{(b)}))$
   - REC: A lightweight projection head aggregates shallow features from available experts to generate a probabilistic region map $r=\sigma(\pi(\frac{1}{|\mathcal{A}|}\sum f_1^{(m)}))$; cosine similarity is then computed on $r$-weighted predictions.
   - **Design Motivation**: MEC enforces global distributional alignment to prevent expert drift, while REC focuses on clinically critical foreground regions to avoid domination by background voxels.

2. **Consistency-Driven Dynamic Gating**
   - For each expert $m$, the global consistency score $u_m$ and regional consistency score $v_m$ are fed into a lightweight gating network $\mathcal{G}$.
   - The network outputs reliability logits $g_m = \mathcal{G}(u_m, v_m)$; softmax normalization over available experts yields fusion weights $w_m$.
   - Multi-scale features are fused as $f_\ell = \sum w_m \odot f_\ell^{(m)}$.
   - **Design Motivation**: Consistency is not only used as a training constraint but is also recycled as a reliability signal to guide fusion—consistent experts are trusted, while divergent ones are suppressed.

### Loss & Training

The overall objective is $\mathcal{L}_{total} = \mathcal{L}_{seg} + \alpha \mathcal{L}_{ECL} + \beta \mathcal{L}_{contrast}$:
- $\mathcal{L}_{seg}$: Weighted cross-entropy + Dice loss on fused predictions.
- $\mathcal{L}_{ECL}$: Per-expert supervision (WCE + DL) + $\eta(\mathcal{L}_{MEC} + \lambda_{rec}\mathcal{L}_{REC})$.
- $\mathcal{L}_{contrast}$: Contrastive representation loss (SSIM for anatomical content alignment + cosine for modality style clustering + KL regularization).
- Adam optimizer, lr = 0.0002, weight decay = 0.0001, batch size = 1, 500 epochs, input size = $112^3$ 3D patch.

## Key Experimental Results

### Main Results

| Dataset | Metric | CLoE | DC-Seg | M³AE | RFNet | HeMIS |
|---|---|---|---|---|---|---|
| BraTS2020 (15-combo Avg) | WT Dice% | **88.09** | 87.54 | 86.90 | 86.98 | 75.10 |
| BraTS2020 (15-combo Avg) | TC Dice% | **80.23** | 79.63 | 79.10 | 78.23 | 65.45 |
| BraTS2020 (15-combo Avg) | ET Dice% | **65.06** | 65.00 | 61.70 | 61.47 | - |
| BraTS2020 Full | WT Dice% | **91.30** | 90.95 | 90.40 | 91.11 | 85.19 |
| MSD Prostate PZ Avg | Dice% | **80.12** | 79.59 | - | 77.35 | - |

### Ablation Study

| Configuration | Avg Dice Change | ET Dice Change | Note |
|---|---|---|---|
| w/o REC | −1.98% | −3.41% | Regional consistency is critical for small foreground structures |
| w/o Weight Fusion | −2.47% | −3.96% | Dynamic fusion weights contribute the most |
| w/o MEC | −0.70% | — | Global consistency plays a fine-tuning role |
| w/o Gating Network | −0.47% | — | Parameterized contribution is limited but directionally correct |

### Key Findings

- REC and Weight Fusion are the two most critical components of CLoE, yielding the largest gains on ET (the clinically most important enhancing tumor subregion).
- CLoE does not sacrifice performance in the full-modality setting (WT 91.30%), demonstrating that consistency constraints do not introduce degradation.
- Cross-dataset generalization (BraTS → MSD Prostate) validates the framework's applicability across domains.
- Even with bounding-box prompts, MedSAM fails to produce clear tumor boundaries, confirming the continued value of dedicated multimodal segmentation frameworks.

## Highlights & Insights

- Reformulating robustness as decision-level consistency control is a clear and intuitively sound perspective.
- REC automatically identifies foreground regions of interest via a probabilistic region map without requiring manual ROI annotations, providing an elegant solution to the background-dominated consistency problem.
- The design of converting consistency scores into reliability weights effectively recycles the consistency signal as a fusion signal rather than treating it solely as a training constraint.
- Notable improvements on ET segmentation (+3.59% vs. RFNet), where foreground is extremely small, confirm the effectiveness of REC for clinically critical small targets.

## Limitations & Future Work

- Ablation results indicate that MEC and the gating network each contribute modestly in isolation (−0.70% / −0.47%), suggesting potential for architectural simplification.
- Validation is limited to two datasets; evaluation on additional organs and modality combination patterns would be valuable.
- Under extreme missing-modality conditions (only one modality available), pairwise consistency comparison is infeasible and the meaning of consistency scores degrades.
- The probabilistic region map $r$ is derived from shallow features, which may be unstable in early training stages.
- The gating network is highly compact (2D input → 1D output), and whether its expressive capacity is sufficient remains an open question.

## Related Work & Insights

- **vs. DC-Seg**: DC-Seg performs latent-space disentanglement via VAE-based contrastive learning; CLoE adds decision-level consistency control and dynamic fusion on top of this, making the two approaches complementary.
- **vs. M³AE**: M³AE relies on large-scale masked autoencoder pre-training, yet CLoE surpasses it with a substantially lighter framework.
- **vs. RFNet**: RFNet employs region-aware spatial priors in a passive manner, whereas CLoE's REC actively learns foreground regions and explicitly quantifies expert reliability.
- The consistency-to-reliability framework is broadly applicable to other multi-source fusion scenarios, such as multi-sensor fusion in autonomous driving.
- The foreground-focused regional consistency idea offers transferable insights for small-object detection and segmentation tasks.

## Rating

- **Novelty**: ⭐⭐⭐ — Each individual component is not entirely novel, but the problem formulation is clear and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Full coverage of 15 missing-modality combinations, cross-dataset validation, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐ — Methodology is described clearly, though space constraints leave some implementation details to the algorithm box.
- **Value**: ⭐⭐⭐ — Missing modality segmentation is an important clinical problem; performance gains are solid if not dramatic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

</div>

<!-- RELATED:END -->
