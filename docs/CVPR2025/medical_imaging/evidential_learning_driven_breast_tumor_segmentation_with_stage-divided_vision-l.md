---
title: >-
  [Paper Note] Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction
description: >-
  [CVPR2025][Medical Imaging][breast tumor segmentation] This paper proposes the TextBCS model, which utilizes text prompts to assist breast tumor segmentation through a Stage-divided Vision-Language Interaction (SVLI) module and an Evidential Learning (EL) strategy. It achieves a Dice score of 85.33% on the Duke-Breast-Cancer-MRI dataset, outperforming all baseline methods.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "breast tumor segmentation"
  - "vision-language"
  - "evidential learning"
  - "cross-attention"
  - "DCE-MRI"
date: 2026-05-08
content_hash: f2acc309d324f583
---

# Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction

**Conference**: CVPR2025  
**arXiv**: [2603.11206](https://arxiv.org/abs/2603.11206)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: breast tumor segmentation, vision-language, evidential learning, cross-attention, DCE-MRI

## TL;DR

This paper proposes the TextBCS model, which utilizes text prompts to assist breast tumor segmentation through a Stage-divided Vision-Language Interaction (SVLI) module and an Evidential Learning (EL) strategy. It achieves a Dice score of 85.33% on the Duke-Breast-Cancer-MRI dataset, outperforming all baseline methods.

## Background & Motivation

Breast cancer is one of the most common causes of cancer death among women worldwide. Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) is widely used for breast tumor detection due to its high sensitivity, but it faces two major challenges: (1) low contrast between tumor regions and normal tissues, making it difficult to precisely locate tumor boundaries; and (2) high segmentation uncertainty caused by blurry boundaries. Existing methods rely solely on learning from the image modality and lack semantic-level guidance. Text prompts can provide prior knowledge regarding the location, shape, and size of the lesion regions, which helps resolve the localization issue in low-contrast scenarios. However, current text-guided methods only perform shallow image-text interactions, which limits effective cross-modal alignment and fusion.

## Method

### Overall Architecture

TextBCS is based on the UNet architecture and includes two core innovative modules:

1. **SVLI (Stage-divided Vision-Language Interaction)**: Executes bidirectional vision-language interaction at each downsampling stage of the encoder.
2. **EL (Evidential Learning)**: Performs pixel-level uncertainty estimation at the decoder side.

The text input is encoded into text embeddings using BioClinicalBERT.

### SVLI Module Design

**(1) Stage-divided Bidirectional Cross-Attention Mechanism**:

Two rounds of interaction are performed at each downsampling stage $s$:
- **Vision Query Module**: Vision features serve as Query, while text features serve as Key/Value. It generates text-aware vision features $F_V^s$ via multi-head cross-attention.
- **Language Query Module**: Text features serve as Query, while text-aware vision features serve as Key/Value. It generates vision-aware text features $F_L^s$.

Each module internally contains two layers of cross-attention + FFN + residual connections.

**(2) Stage-divided Cross-modal Alignment Loss**:

The text-to-image contrastive loss is calculated at each feature level:
$$\mathcal{L}_{con}^{sj} = \begin{cases} -\log(\sigma(\text{Sim}(F_V^{s,j}, F_L^{s,j})/\tau_s)) & j \in Z^+ \\ -\log(1-\sigma(\text{Sim}(F_V^{s,j}, F_L^{s,j})/\tau_s)) & j \in Z^- \end{cases}$$

Unlike previous methods that only perform alignment at the final feature layer, SVLI ensures that both low-level and high-level features undergo cross-modal alignment.

### Evidential Learning Module

After the decoder output, a Softplus activation function is inserted to obtain non-negative evidence $e = [e_1, ..., e_C]$, constructing a Dirichlet distribution $Dir(p|\alpha)$ (where $\alpha = e + 1$) to model the distribution of segmentation probabilities.

- **Belief Mass**: $b_{i,j}^c = e_{i,j}^c / W$
- **Uncertainty**: $u_{i,j} = C / W$ (where $W = \sum_c \alpha_{i,j}^c$)

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{Dice} + \lambda_1 \mathcal{L}_{ice} + \lambda_2 \mathcal{L}_{KL} + \lambda_3 \mathcal{L}_{con}$$

- $\mathcal{L}_{ice}$: Dirichlet-based integrated cross-entropy loss
- $\mathcal{L}_{KL}$: KL divergence regularization term, ensuring lower evidence for incorrect classes
- $\lambda_1 = 10^{-3}$, $\lambda_2 = 5e\text{-}7 \cdot \min\{1, n_{epoch}/100\}$ (gradually increasing), $\lambda_3 = 10^{-3}$

## Key Experimental Results

**Main Results (Duke-Breast-Cancer-MRI dataset, 922 patients, 3876 slices)**:

| Method | Text | Dice (%) | mIoU (%) | Param (M) |
|------|------|----------|----------|-----------|
| UNet | ✗ | 81.54 | 73.22 | 14.8 |
| TransUNet | ✗ | 83.14 | 75.49 | 105 |
| MGCA | ✓ | 84.28 | 75.44 | 135.6 |
| LViT | ✓ | 82.79 | 73.21 | 29.7 |
| **TextBCS**| ✓ | **85.33** | **76.08** | 32.5 |

**Ablation Study**:

| Baseline | SVLI | EL | Dice (%) |
|----------|------|----|----------|
| ✓ | | | 81.54 |
| ✓ | ✓ | | 84.41 (+2.87) |
| ✓ | | ✓ | 83.19 (+1.65) |
| ✓ | ✓ | ✓ | **85.33** (+3.79) |

- The t-test p-values of all comparison methods are $< 0.05$, which is statistically significant.
- The model is robust to variations in text prompt styles.

## Highlights & Insights

- This is the first method to apply text guidance to breast tumor segmentation in DCE-MRI.
- SVLI performs vision-language interaction at each downsampling stage of the encoder, which is more thorough than interacting only at the final layer or skip connections.
- Evidential learning provides pixel-level uncertainty quantification, yielding high uncertainty for blurry boundaries rather than overconfident predictions.
- The parameter count (32.5M) and FLOPs (52.3G) are the lowest among text-guided methods, offering high efficiency.
- Interpretability studies show that SVLI effectively guides the model to focus on cancerous regions.

## Limitations & Future Work

- Text prompts need to be manually provided by radiologists, which limits practical deployment (although automated generation strategies via LLMs are discussed, they are not validated).
- The method is only validated on a single public dataset, leaving generalizability to be verified.
- The format of the text prompts is relatively simple (location/shape/size/number) and does not fully utilize richer clinical descriptions.
- Erroneous or insufficient text prompts can lead to incorrect segmentation (as shown in Fig. 5).
- The method only processes 2D slices and does not utilize 3D volumetric information.

## Related Work & Insights

- **LViT** (Li et al.): Integrates text at the encoder downsampling stage; the proposed method builds on this by adding bidirectional cross-attention.
- **MGCA**: Utilizes text guidance but lacks sufficient image-text interaction and does not evaluate segmentation reliability.
- **CLIP/GLoRIA/ConVIRT**: Use contrastive learning for image-text alignment, but the depth of interaction is insufficient.
- **EDL** (Sensoy et al.): The theoretical foundation of the EL module in this paper, which models predictive uncertainty through Dirichlet distributions.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combined design of SVLI + EL is reasonable, and text-guided breast segmentation is pioneering)
- Experimental Thoroughness: ⭐⭐⭐ (Ablation studies are thorough, but evaluation is only on a single dataset, lacking cross-domain validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, mathematical derivations are complete)
- Value: ⭐⭐⭐⭐ (The combination of text guidance and uncertainty estimation is highly practical in clinical scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[ICCV 2025\] Alleviating Textual Reliance in Medical Language-guided Segmentation via Prototype-driven Semantic Approximation](../../ICCV2025/medical_imaging/alleviating_textual_reliance_in_medical_language-guided_segmentation_via_prototy.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)

</div>

<!-- RELATED:END -->
