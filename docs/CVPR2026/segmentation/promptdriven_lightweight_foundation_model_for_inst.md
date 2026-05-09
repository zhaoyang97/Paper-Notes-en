---
title: >-
  [Paper Note] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains
description: >-
  [CVPR 2026][Segmentation][SAM] This paper proposes SAM FTI-FDet, which introduces a Transformer decoder-based Prompt Generator that enables lightweight TinyViT-SAM to automatically generate task-relevant query prompts, achieving instance-level fault detection of freight train components without manual interaction. The method attains 74.6 AP_box / 74.2 AP_mask on a self-constructed dataset.
tags:
  - CVPR 2026
  - Segmentation
  - SAM
  - self-prompt generation
  - lightweight
  - freight train fault detection
  - foundation model transfer
date: 2026-05-08
content_hash: 557f4cb47926f8c2
---

# Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains

**Conference**: CVPR 2026
**arXiv**: [2603.12624](https://arxiv.org/abs/2603.12624)
**Code**: [https://github.com/MVME-HBUT/SAM_FTI-FDet.git](https://github.com/MVME-HBUT/SAM_FTI-FDet.git)
**Area**: Instance Segmentation / Industrial Inspection / Foundation Model Adaptation
**Keywords**: SAM, self-prompt generation, lightweight, freight train fault detection, foundation model transfer

## TL;DR
This paper proposes SAM FTI-FDet, which introduces a Transformer decoder-based Prompt Generator that enables lightweight TinyViT-SAM to automatically generate task-relevant query prompts, achieving instance-level fault detection of freight train components without manual interaction. The method attains 74.6 AP_box / 74.2 AP_mask on a self-constructed dataset.

## Background & Motivation
Critical components of freight trains (e.g., brake shoes, bearing saddles) are prone to wear after prolonged operation, making traditional manual inspection inefficient and experience-dependent. Although CNN/Transformer-based detection methods have been widely deployed, three key limitations remain: (1) **poor generalization**—models trained at one inspection station suffer significant performance degradation when transferred to new stations; (2) **imprecise boundaries**—conventional object detection provides only bounding boxes, which cannot quantitatively assess wear degree (e.g., remaining brake shoe thickness); (3) **deployment constraints**—high-accuracy models are computationally expensive and difficult to run in real time on edge devices along railway lines. While SAM, as a foundation model, possesses strong segmentation generalization capability, it relies on external prompts (clicks, boxes) and is sensitive to prompt location, making it unsuitable for fully automated industrial scenarios.

## Core Problem
How to transfer SAM's general segmentation knowledge to the specific domain of freight train fault detection while addressing three challenges: (1) eliminating SAM's dependence on manual prompts to enable full automation; (2) maintaining a lightweight architecture to meet edge deployment requirements; (3) ensuring instance segmentation accuracy in structurally complex and frequently occluded industrial scenes.

## Method

### Overall Architecture
SAM FTI-FDet is built upon SAM's encoder-decoder architecture. The pipeline is: input image (1024×1024) → TinyViT-SAM encoder for feature extraction → Adaptive Feature Dispatcher for multi-scale feature fusion → Prompt Generator for automatic query prompt generation → Mask Decoder combining prompts and image features to produce instance segmentation masks and bounding boxes. During inference, up to 10 instances are predicted per image; only the final decoder layer output is used, and morphological post-processing is applied to obtain the final masks and boxes.

### Key Designs
1. **Prompt Generator**: The core innovation. A set of learnable query vectors $Q_0$ (of length $N_q$) is initialized and iteratively refined through $L$ Transformer Decoder layers. Each layer first applies self-attention to model inter-query semantic dependencies, then cross-attention to interact with image features, with an attention mask to suppress irrelevant positions. The resulting query vectors are injected into the mask decoder as both sparse and dense prompts, guiding the model to focus on target regions. Unlike the box-based prompts in RSPrompter, these query prompts directly encode target semantic priors, leading to faster convergence and higher accuracy.

2. **Adaptive Feature Dispatcher**: Composed of a Feature Aggregator and a Feature Splitter. The Aggregator applies $1\times1$ convolutions to reduce each TinyViT layer's features to 32 channels, then progressively fuses them via recursive residual aggregation ($m_i = m_{i-1} + \text{Conv2D}(m_{i-1}) + \tilde{F}_i$), and finally restores the channel dimension through multi-layer convolutions to obtain a unified feature $F_\text{agg}$. The Splitter decomposes $F_\text{agg}$ into multi-resolution branches for tasks at different scales. This design compensates for the limited feature expressiveness of the lightweight backbone.

3. **TinyViT-SAM Lightweight Backbone**: The original SAM ViT-B/H is replaced with TinyViT obtained via knowledge distillation in MobileSAM, substantially reducing parameter count and computational cost. A key finding is that freezing the decoder while fine-tuning only the encoder (uf/f configuration) yields the best results—encoder fine-tuning learns domain-specific features while the frozen decoder retains general decoding capability, preventing overfitting.

4. **End-to-End Set Prediction Mechanism**: $N_q = 10$ groups of prompts are generated, each containing $K_p = 4$ point embeddings, directly extracting task-relevant information from global image features. This fixed-count query design, analogous to DETR, eliminates the need for post-processing such as NMS.

### Loss & Training
- AdamW optimizer, initial lr = 1e-4, cosine annealing with linear warmup, trained for 150 epochs
- Batch size = 4, dual RTX 4090 GPUs
- DeepSpeed ZeRO Stage 2 + FP16 mixed-precision training for efficiency
- Data augmentation: horizontal flipping + large-scale jittering
- The Prompt Generator uses only the last 3 smallest-resolution feature maps from the Feature Splitter output

## Key Experimental Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|---------|--------|------|------------|------|
| Freight Train | AP_box | 74.6 | 74.3 (Mask2Former+Swin-T) | +0.3 |
| Freight Train | AP_mask | 74.2 | 73.8 (Mask2Former+Swin-T) | +0.4 |
| Freight Train | Model Size (MB) | 148.2 | 739.5 (Mask2Former+Swin-T) | −80% |
| Freight Train | Params (M) | 36.3 | 49.0 (Mask2Former+Swin-T) | −26% |
| MS-COCO | AP_box | 38.7 | 37.9 (FastSAM) | +0.8 |
| MS-COCO | AP_mask | 33.7 | 32.6 (FastSAM) | +1.1 |
| Noise Test | AP_box | 60.8 | 57.5 (Mask R-CNN) | +3.3 |
| Brake Shoe Wear | Severe Wear Detection | 97.5% | 93.9% (Mask R-CNN) | +3.6% |

### Ablation Study
- **Prompt type is the most critical factor**: Query prompts outperform box prompts by 16.5 AP_mask over SAM-det's bbox prompts (74.2 vs. 57.7), demonstrating that semantic-level prompts are substantially superior to spatial-level prompts.
- **Freezing strategy**: Fine-tuning the encoder while freezing the decoder (uf/f) is optimal; fully freezing drops AP_box by 7.7, and fully fine-tuning drops it by 1.4.
- **Feature layer selection**: Using the last two layers [2, 3] yields the best result (74.6 AP_box); using all four layers [0, 1, 2, 3] causes a 0.8-point drop.
- **Channel dimension**: 256 > 128 > 64; increasing from 64 to 256 yields a 7.3 AP_box improvement.
- **Prompt shape**: $N_q = 10$, $K_p = 4$ is optimal; $N_q$ has a larger impact on performance (coverage), while $K_p$ has a smaller impact (robustness).
- **Pretraining data**: SA-1B pretraining outperforms ImageNet pretraining; TinyViT-5m approaches SAM-B performance with significantly fewer parameters.

## Highlights & Insights
- **Transferable self-prompting paradigm**: Converting SAM from requiring manual clicks to automatically generating prompts is practically valuable and broadly applicable to any industrial scenario that prohibits human interaction (e.g., assembly line inspection, UAV patrol).
- **Freezing the decoder as regularization**: The strategy of fine-tuning only the encoder while freezing the decoder on small datasets is worth adopting, as it leverages the pretrained decoder's general decoding capability to prevent overfitting.
- **Instance segmentation for quantitative assessment**: Beyond fault detection, the paper estimates brake shoe wear severity (slight/moderate/severe) from mask area, offering greater practical industrial value than conventional bounding box regression.
- **Recursive residual feature aggregation is simple yet effective**: $m_i = m_{i-1} + \text{Conv}(m_{i-1}) + \tilde{F}_i$ effectively compensates for the limited feature expressiveness of lightweight backbones.

## Limitations & Future Work
- The dataset is small (4,410 images) and sourced exclusively from the Chinese railway system; cross-national and cross-type generalization remains unvalidated.
- The fixed query count $N_q = 10$ cannot handle dense scenes with more than 10 object instances per image.
- Missed detections persist for very small targets and low-salience defects, as acknowledged by the authors in the Discussion section.
- The method operates on static images only and has not been extended to temporal fault detection in video streams.
- Training still requires 150 epochs on dual RTX 4090 GPUs, falling short of true plug-and-play deployment.

## Related Work & Insights
- **vs. RSPrompter**: RSPrompter uses box prompts to guide SAM, whereas this work uses query prompts. Experiments show that query prompts converge faster (training loss comparison) and achieve higher accuracy (AP_mask 74.2 vs. 71.9), as queries encode semantics rather than spatial constraints.
- **vs. Mask2Former**: Accuracy is comparable (74.6 vs. 74.3 AP_box), but the model is 5× smaller (148 MB vs. 740 MB), making it more suitable for edge deployment.
- **vs. FastSAM**: FastSAM has fewer parameters (9.1M), but AP_mask is 2.2 points lower (72.0 vs. 74.2), and FastSAM lacks domain adaptation as a general-purpose model.

**Further Insights**: The self-prompt generator paradigm can be generalized beyond point/box prompts to generate frequency-domain or text-domain prompts for diverse challenging scenarios. The freeze-decoder/fine-tune-encoder strategy also offers a useful reference for adapting SAM to medical imaging.

## Rating
- **Novelty**: ⭐⭐⭐ — The self-prompting SAM idea is not entirely new (RSPrompter precedes it), but the query-based prompt design and industrial adaptation constitute incremental contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations are highly detailed, covering prompt type, backbone, freezing strategy, channel dimension, prompt shape, noise robustness, and cross-dataset generalization across 10 dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with complete mathematical derivations, though some descriptions are slightly verbose.
- **Value**: ⭐⭐⭐ — High practical value for industrial applications; moderate academic novelty.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)

<!-- RELATED:END -->
