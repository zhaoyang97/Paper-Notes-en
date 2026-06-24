---
title: >-
  [Paper Note] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models
description: >-
  [CVPR 2025][Object Detection][Zero-Shot Anomaly Detection] The first MLLM dedicated to zero-shot anomaly detection and reasoning (Anomaly-OV). It generates anomaly saliency maps through a Look-Twice Feature Matching mechanism coupled with a visual token selector to focus on suspicious regions, achieving SOTA zero-shot anomaly detection with an average AUROC of 88.6% across 9 benchmarks.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Zero-Shot Anomaly Detection"
  - "MLLM Expert System"
  - "Visual Feature Matching"
  - "Anomaly Reasoning"
  - "Industrial Inspection"
date: 2026-05-08
content_hash: 83233c8af648360d
---

# Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2502.07601](https://arxiv.org/abs/2502.07601)  
**Code**: [https://xujiacong.github.io/Anomaly-OV/](https://xujiacong.github.io/Anomaly-OV/)  
**Area**: Multimodal VLM  
**Keywords**: Zero-Shot Anomaly Detection, MLLM Expert System, Visual Feature Matching, Anomaly Reasoning, Industrial Inspection

## TL;DR
The first MLLM dedicated to zero-shot anomaly detection and reasoning (Anomaly-OV). It generates anomaly saliency maps through a Look-Twice Feature Matching mechanism coupled with a visual token selector to focus on suspicious regions, achieving SOTA zero-shot anomaly detection with an average AUROC of 88.6% across 9 benchmarks.

## Background & Motivation

**Background**: Zero-shot anomaly detection (ZSAD) aims to detect anomalies without using normal samples from the target categories. Existing methods, such as WinCLIP and AnomalyCLIP, use the CLIP text encoder to construct and match normal/anomaly descriptions, but they suffer from the limited semantic capability of the text encoder.

**Limitations of Prior Work**: (1) General MLLMs (e.g., GPT-4o) can detect anomalies but fail to describe and locate them accurately, achieving about 70% detection accuracy but imprecise reasoning descriptions. (2) Existing ZSAD methods only provide binary classification results and cannot explain "why it is anomalous." (3) Anomaly patterns vary significantly across different domains like industrial, medical, and 3D, making it difficult for a single model to cover all of them.

**Key Challenge**: MLLMs possess reasoning capabilities but lack specialized visual perception for anomaly detection, while dedicated anomaly detection models offer high visual precision but lack explanation and reasoning capabilities.

**Goal**: Inject specialized visual anomaly detection capabilities into MLLMs, enabling them to achieve both high-precision detection and natural language reasoning.

**Key Insight**: Design an "anomaly expert" module that uses multi-layer ViT features and learnable normal/anomaly embeddings to perform Look-Twice Feature Matching, generating anomaly saliency maps. The saliency maps then guide visual token selection, allowing the MLLM to focus on suspicious regions.

**Core Idea**: Use multi-layer visual feature matching to generate anomaly saliency maps as "magnifying glasses" for the MLLM, enabling both precise anomaly localization and natural language explanation.

## Method

### Overall Architecture
Two-stage training: Stage 1 trains the anomaly expert (multi-layer ViT features + learnable $e^+$/$e^-$ embeddings $\rightarrow$ LTFM to generate saliency maps) $\rightarrow$ Stage 2 freezes the expert and visual encoder while training the projection layer + LLM using the Anomaly-Instruct-125K instruction dataset.

### Key Designs

1. **Look-Twice Feature Matching (LTFM)**:

    - **Function**: Generate pixel-wise anomaly saliency maps.
    - **Mechanism**: The first "look" performs covariance matching between multi-layer ViT features and the learnable normal embedding $e^+$ and anomaly embedding $e^-$. The second "look" (look-back path) modulates the original features with the first-stage matching results before matching again, analogous to "looking back carefully." The matching results from both steps are fused to form the final saliency map.
    - **Design Motivation**: A single match is not sensitive enough to subtle anomalies, and the look-back mechanism provides self-correction capability. Ablation studies show that removing the look-back pathway drops the AUROC by 1.2%.

2. **Visual Token Selector**:

    - **Function**: Focus the MLLM on suspicious regions with high saliency.
    - **Mechanism**: Multiply visual tokens with the saliency map $\rightarrow$ perform spatial pooling $\rightarrow$ aggregate into selected tokens via Q-Former. Meanwhile, indicative prompts ($\langle \text{adv} \rangle$ suspicious feature, where $\text{adv} \in \text{\{highly, moderately, slightly\}}$) are used to bridge the original and selected tokens.
    - **Design Motivation**: MLLMs do not need to process all visual tokens—focusing on suspicious regions yields more precise reasoning.

3. **Anomaly-Instruct-125K Dataset**:

    - **Function**: Provide multi-domain instruction tuning data for anomaly detection.
    - **Mechanism**: The 125K samples cover industrial (MVTec, VisA), medical (BrainMRI, HeadCT), 3D (MVTec-3D), and in-the-wild (WebAD with 72K web images) domains. It includes four task types: detection, localization, description, and reasoning.
    - **Design Motivation**: WebAD contributes a +5.5% AUROC improvement on MVTec, demonstrating that in-the-wild anomaly data is crucial for learning general anomaly semantics.

### Loss & Training
Stage 1: The anomaly expert is trained using binary classification + saliency map loss. Stage 2: Freezes the expert + ViT, and trains the LLM + projection layer using standard next-token prediction.

## Key Experimental Results

### Main Results

| Method | MVTec | VisA | AITEX | BrainMRI | HeadCT | 9-Benchmark Avg |
|------|-------|------|-------|----------|--------|----------|
| WinCLIP | 91.8 | 78.8 | 73.0 | 92.6 | 90.0 | 79.2 |
| AnomalyCLIP | 91.5 | 82.1 | 62.2 | 90.3 | 93.4 | 84.5 |
| **Anomaly-OV** | **94.0** | **91.1** | **72.0** | **93.9** | **97.6** | **88.6** |

### Ablation Study

| Configuration | MVTec | VisA | HeadCT |
|------|-------|------|--------|
| Full Model | 94.0 | 91.1 | 97.6 |
| W/o look-back | 92.8 | 90.5 | 96.6 |
| W/o $e^+$/$e^-$ | 92.1 | 90.1 | 94.7 |
| W/o WebAD | 88.5 | 88.9 | 91.2 |

### Key Findings
- **Text encoder is not mandatory**: Anomaly-OV does not use a text encoder for matching (purely visual) but still outperforms all CLIP-based methods.
- **WebAD is key**: 72K in-the-wild anomaly images contribute +5.5% AUROC on MVTec, demonstrating the vital importance of pretraining on general anomaly semantics.
- **Unified detection and reasoning**: GPT-4o achieves 70% detection Acc but only 68% F1 on VisA-D&R, whereas Anomaly-OV reaches 79% Acc and 83% F1.

## Highlights & Insights
- **Ingenious "Anomaly Expert + MLLM" architecture**: The expert provides professional visual perception while the MLLM provides reasoning and textual output, dividing tasks effectively.
- **"Saliency map as a magnifying glass" idea** can be extended to other MLLM applications requiring focus on specific regions (e.g., medical image analysis).
- **Zero-shot cross-domain capability**: SOTA performance achieved in both industrial and medical domains, demonstrating shared visual pattern commonalities in anomalies across domains.

## Limitations & Future Work
- The LTFM of the anomaly expert requires an additional training stage and extra computational overhead.
- The accuracy of pixel-level anomaly localization (segmentation) is not reported in detail.
- Industrial data accounts for a large portion of the 125K training samples, which might be insufficient for natural scene anomalies.

## Related Work & Insights
- **vs WinCLIP / AnomalyCLIP**: These methods use CLIP text-visual matching. Anomaly-OV uses pure visual feature matching plus MLLM, achieving higher accuracy and reasoning capabilities.
- **vs GPT-4o**: GPT-4o can detect anomalies but lacks precision. Anomaly-OV significantly outperforms it in both detection and reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of anomaly expert + MLLM, novel LTFM mechanism
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 anomaly detection benchmarks + VisA-D&R reasoning benchmark + comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Clear method logic, valuable dataset contribution
- Value: ⭐⭐⭐⭐⭐ Direct application value for industrial/medical anomaly detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](../../CVPR2026/object_detection/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](../../CVPR2026/object_detection/back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](../../ICCV2025/object_detection/lmm-det_make_large_multimodal_models_excel_in_object_detection.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)

</div>

<!-- RELATED:END -->
