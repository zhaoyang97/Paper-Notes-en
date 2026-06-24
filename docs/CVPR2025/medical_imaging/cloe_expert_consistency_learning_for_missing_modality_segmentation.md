---
title: >-
  [Paper Note] CLoE: Expert Consistency Learning for Missing Modality Segmentation
description: >-
  [CVPR 2025][Medical Imaging][missing modality] This paper proposes the CLoE framework, which reformulates the robustness challenge of missing modality segmentation as a decision-level expert consistency control problem. It reduces expert drift through dual-branch constraints: Modality Expert Consistency (MEC) globally and Region Expert Consistency (REC) regionally. A lightweight gating network is employed to convert consistency scores into reliability weights to guide feature…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "missing modality"
  - "multimodal segmentation"
  - "expert consistency"
  - "brain tumor"
  - "MRI"
date: 2026-05-08
content_hash: 7e4055dc99cc31b2
---

# CLoE: Expert Consistency Learning for Missing Modality Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2603.09316](https://arxiv.org/abs/2603.09316)  
**Code**: None  
**Area**: Medical Images  
**Keywords**: missing modality, multimodal segmentation, expert consistency, brain tumor, MRI

## TL;DR
This paper proposes the CLoE framework, which reformulates the robustness challenge of missing modality segmentation as a decision-level expert consistency control problem. It reduces expert drift through dual-branch constraints: Modality Expert Consistency (MEC) globally and Region Expert Consistency (REC) regionally. A lightweight gating network is employed to convert consistency scores into reliability weights to guide feature fusion, outperforming SOTA methods on BraTS 2020 and MSD Prostate.

## Background & Motivation
**Background**: Multimodal MRI segmentation (e.g., T1, T1c, T2, FLAIR) is widely used in brain tumor diagnosis. Mainstream methods assume all modalities are available and perform voxel-level segmentation based on U-Net/V-Net encoder-decoder architectures.

**Limitations of Prior Work**: Missing modalities frequently occur in clinical practice (due to equipment failure, protocol variations, or quality issues). Existing strategies to address this include GAN-based synthesis of missing modalities (CollaGAN), arithmetic fusion (HeMIS), latent representation learning (DC-Seg), and attention-based adaptive fusion (SE, CBAM, RFNet). However, these methods either cannot determine which expert is more reliable or fail in attention mechanisms when missing modalities are zero-filled.

**Key Challenge**: Missing modalities lead to disagreements in individual modality expert predictions (expert disagreement). Fixed weights or unconstrained attention-based fusion amplify such disagreements, particularly in small yet critical foreground areas (e.g., enhancing tumors). Although consistency learning is effective in semi-supervised learning, global consistency is easily dominated by background pixels, neglecting small tumor sub-regions.

**Goal**: (a) How to quantify and reduce prediction inconsistency among modality experts? (b) How to avoid global consistency being dominated by the background? (c) How to transform consistency signals into reliability weights to guide fusion?

**Key Insight**: Reformulate the robustness problem from the feature level to the decision level—instead of recovering missing features, the focus shifts to controlling the consistency of expert predictions, using the degree of consistency as a proxy for reliability.

**Core Idea**: Use cosine similarity to measure expert prediction consistency, imposing constraints at both the global and foreground region levels, and mapping the consistency scores to fusion weights via a gating network.

## Method

### Overall Architecture
CLoE consists of three components: parallel modality encoders, a consistency-driven gating module, and a shared fusion decoder. Each modality $m$ has an independent encoder $\Phi_m$ to extract multi-scale features. A weight-shared expert decoder $D^{\text{sep}}$ generates independent predictions $p^{(m)}$ for each modality, which are then aggregated through a dynamic gating mechanism for multi-scale features, and the fusion decoder $D^{\text{fuse}}$ outputs the final segmentation.

### Key Designs

1. **Modality Expert Consistency (MEC)**:

    - **Function**: Forces all available modality expert predictions to maintain consistency in terms of global distribution.
    - **Mechanism**: Vectorizes predictions from each pair of available experts and computes the cosine similarity $\mathcal{S}(\mathbf{p}^{(a)}, \mathbf{p}^{(b)})$. The MEC loss is the average inconsistency across all expert pairs: $\mathcal{L}_{\text{MEC}} = \frac{1}{|\mathcal{P}|}\sum(1 - \mathcal{S})$.
    - **Design Motivation**: Under missing modality scenarios, experts are prone to case-wise drift. Global consistency constraints force experts to reach a consensus given the available modalities.

2. **Region Expert Consistency (REC)**:

    - **Function**: Enforces expert consistency in critical foreground regions (e.g., tumors), preventing global consistency from being diluted by massive background pixels.
    - **Mechanism**: Aggregates low-level features to generate a probabilistic region map $r = \sigma(\pi(\frac{1}{|\mathcal{A}|}\sum f_1^{(m)}))$ via a lightweight projection head $\pi(\cdot)$, then computes cosine consistency after weighting the expert predictions with $r$.
    - **Design Motivation**: Background pixels drastically outnumber foreground pixels in volumetric MRI, meaning global consistency might fail to align small tumor sub-regions (e.g., ET). REC focuses specifically on clinically critical structures.

3. **Consistency-Driven Dynamic Gating**:

    - **Function**: Translates consistency scores into modality reliability weights to guide feature fusion.
    - **Mechanism**: Computes global/regional consistency scores $(u_m, v_m)$ for each available expert, maps them to a logit $g_m$ via a gating network $\mathcal{G}$, and normalizes them using softmax to obtain fusion weights $w_m$. Multi-scale features are fusion-weighted as $f_\ell = \sum w_m \odot f_\ell^{(m)}$.
    - **Design Motivation**: Experts with high consistency are more trustworthy and should receive higher weights, whereas those with low consistency may be impacted by missing modalities and should be suppressed.

### Loss & Training
The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \alpha \mathcal{L}_{\text{ECL}} + \beta \mathcal{L}_{\text{contrast}}$, where:
- $\mathcal{L}_{\text{seg}}$: WCE + Dice loss of the fused prediction.
- $\mathcal{L}_{\text{ECL}}$: Independent supervision (WCE + Dice) for each expert + $\eta(\mathcal{L}_{\text{MEC}} + \lambda_{\text{rec}} \mathcal{L}_{\text{REC}})$.
- $\mathcal{L}_{\text{contrast}}$: Contrastive representation loss (content alignment via SSIM + style clustering via Cosine + VAE reconstruction/KL), borrowing the disentanglement perspective from DC-Seg.

## Key Experimental Results

### Main Results

| Dataset | Metrics | CLoE | DC-Seg | M³AE | RFNet | Gain (vs DC-Seg) |
|--------|------|------|--------|------|-------|-----------------|
| BraTS 2020 (15 missing cases) | Avg. WT Dice | **88.09** | 87.54 | 86.90 | 86.98 | +0.55% |
| BraTS 2020 | Avg. TC Dice | **80.23** | 79.63 | 79.10 | 78.23 | +0.60% |
| BraTS 2020 | Avg. ET Dice | **65.06** | 65.00 | 61.70 | 61.47 | +0.06% |
| MSD Prostate (PZ) | T2 Dice | **80.33** | 79.21 | - | 75.18 | +1.12% |
| MSD Prostate (PZ) | ADC Dice | **77.12** | 75.89 | - | 72.07 | +1.23% |
| MSD Prostate (PZ) | T2&ADC Dice | **82.91** | 81.67 | - | 78.00 | +1.24% |
| MSD Prostate (PZ) | Avg Dice | **80.12** | 79.59 | - | 77.35 | +0.53% |

### Ablation Study

| Configuration | WT Dice | TC Dice | ET Dice | Description |
|------|---------|---------|---------|------|
| Full CLoE | **88.09** | **80.23** | **65.06** | Full model |
| w/o REC | 86.40 | 79.39 | 61.65 | w/o region consistency, ET drops by 3.41% |
| w/o Weight Fusion | 86.52 | 78.33 | 61.10 | w/o weighted fusion, ET drops by 3.96% |
| w/o MEC | 87.75 | 80.01 | 63.50 | w/o global consistency, minimal impact |
| w/o Gating | 87.99 | 80.08 | 63.90 | w/o gating network, least impact |

### Key Findings
- **REC and Weight Fusion are core components**: Removing either results in an ET Dice drop of >3%, indicating that foreground region consistency and consistency-driven fusion are critical for small targets like enhancing tumors.
- MEC and Gating have relatively small impacts when removed individually, indicating they act as auxiliary fine-tuning components.
- The method remains effective even in the extreme scenario of MSD Prostate, which has only 2 modalities and 48 training samples, validating its robustness under data-scarce and modality-limited settings.

## Highlights & Insights
- **Shifting robustness from the feature level to the decision level**: Instead of recovering missing features, the method controls the consistency of expert predictions. This perspective is more lightweight and direct than generative completion (e.g., GAN-based synthesis of missing modalities).
- **Region Expert Consistency (REC) cleverly addresses the background dominance problem**: By weighting with a learnable foreground probability map, it automatically focuses on expert consistency in small target regions without requiring manual RoI definition.
- **The mapping from consistency to reliability is transferable**: Using inter-expert consistency as a proxy for reliability can be generalized to multi-view fusion, multi-model ensembles, and other scenarios.

## Limitations & Future Work
- The experiments were validated only on two datasets (BraTS 2020 + MSD Prostate); generalization to other multimodal scenarios (e.g., cardiac or multi-organ abdomen segmentation) remains to be confirmed.
- The contrastive representation loss ($\mathcal{L}_{\text{contrast}}$) directly borrows the disentanglement design from DC-Seg, lacking original contribution.
- MEC contributes minimally in ablation studies (removing it drops only 0.34% WT Dice), making the necessity of the global consistency constraint questionable.
- During inference, the gating network requires all available experts to generate predictions before fusing them, which introduces the overhead of an extra forward pass.
- Training requires sampling all $2^M - 1$ modality combinations, which leads to a significant increase in training cost as the number of modalities grows.
- Whether cosine similarity is the optimal consistency metric has not been compared against alternative options (e.g., KL divergence, JS divergence).

## Related Work & Insights
- **vs DC-Seg**: DC-Seg focuses on latent space disentanglement (content/style separation), while CLoE introduces decision-level consistency control; the two are complementary, with CLoE achieving more pronounced improvements on WT/TC.
- **vs M³AE**: M³AE utilizes a large-scale pre-trained multimodal autoencoder. Although the model is heavier, its ET performance is inferior to CLoE, suggesting that decision-level constraints are more targeted than large-scale pre-training.
- **vs RFNet**: RFNet uses region-aware priors to specify "where to look," whereas CLoE uses consistency to specify "whom to trust," forming two complementary dimensions.
- **vs Semi-Supervised Consistency**: Methods like Mean Teacher enforce consistency between a teacher and student, whereas CLoE enforces consistency among peer-level experts, making it tailored for multimodal fusion scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating missing modality robustness as expert consistency control is a valuable perspective, though the contrastive loss section lacks novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 missing combinations + two datasets + comprehensive ablation, but lacks validation on more organs/modalities.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and complete mathematical derivations, though the paper is brief.
- Value: ⭐⭐⭐⭐ The method is lightweight and effective, and the mapping from consistency to reliability has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](../../ICCV2025/medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[ICML 2025\] Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees](../../ICML2025/medical_imaging/mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)

</div>

<!-- RELATED:END -->
