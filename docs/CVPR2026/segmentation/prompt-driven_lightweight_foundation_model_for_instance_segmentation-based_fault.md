---
title: >-
  [Paper Note] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains
description: >-
  [CVPR2026][Segmentation][SAM] This paper proposes SAM FTI-FDet, which transfers SAM's general segmentation capability to freight train fault detection via an automatic prompt generation module and an adaptive feature dispatcher. Using a TinyViT lightweight backbone, the method achieves 74.6 AP^box / 74.2 AP^mask, surpassing existing methods in both accuracy and efficiency.
tags:
  - CVPR2026
  - Segmentation
  - SAM
  - instance segmentation
  - fault detection
  - lightweight foundation model
  - automatic prompt generation
  - freight train inspection
date: 2026-05-08
content_hash: 6317dfe8a1313fa6
---

# Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains

**Conference**: CVPR2026
**arXiv**: [2603.12624](https://arxiv.org/abs/2603.12624)
**Code**: [MVME-HBUT/SAM_FTI-FDet](https://github.com/MVME-HBUT/SAM_FTI-FDet)
**Area**: Semantic Segmentation
**Keywords**: SAM, instance segmentation, fault detection, lightweight foundation model, automatic prompt generation, freight train inspection

## TL;DR

This paper proposes SAM FTI-FDet, which transfers SAM's general segmentation capability to freight train fault detection via an automatic prompt generation module and an adaptive feature dispatcher. Using a TinyViT lightweight backbone, the method achieves 74.6 AP^box / 74.2 AP^mask, surpassing existing methods in both accuracy and efficiency.

## Background & Motivation

**Urgent railway safety demands**: Wear detection of critical components such as brake shoes and bearing saddles in freight trains is directly related to operational safety, while traditional manual inspection is inefficient and operationally complex.

**Limited generalization of traditional detection methods**: Existing CNN-based methods (e.g., anchor-based detection, lightweight anchor-free detection) exhibit significant performance degradation across different station scenarios and are highly sensitive to domain shift.

**Inability to quantify defects via object detection**: Bounding boxes cannot provide pixel-level wear analysis; instance segmentation is required for quantitative evaluation of component area and shape.

**Industrial application bottleneck of SAM**: SAM relies on manual click/box prompts, which cannot satisfy fully automated industrial inspection; direct application to structurally complex train images also yields limited results.

**Computational resource constraints**: Railway surveillance systems must be deployed on edge devices, making the computational overhead of vanilla SAM prohibitive for real-time scenarios.

**Limitations of existing transfer approaches**: Methods such as RSPrompter perform poorly on structurally complex, defect-focused train scenarios, struggling to balance accuracy and efficiency.

## Method

### Overall Architecture

SAM FTI-FDet is built upon SAM's encoder–decoder structure and comprises three core modules: (1) a TinyViT-SAM lightweight backbone for image feature extraction; (2) an Adaptive Feature Dispatcher for multi-scale feature fusion; and (3) a Prompt Generator that automatically produces task-relevant prompts to guide the Mask Decoder in generating instance segmentation results. Through an end-to-end set prediction mechanism, fixed and learnable queries extract task-relevant information directly from global image features.

### Key Designs

**Prompt Generator**: A Transformer decoder based on multi-head attention, initialized with a set of learnable query vectors $Q_0$ of length $N_q$, iteratively refined through $L$ stacked layers of self-attention and cross-attention. Self-attention models semantic dependencies among queries, while cross-attention fuses external image features (with attention masks applied to suppress irrelevant positions). The final output serves as the query input to the Mask Decoder, enabling object-aware mask prediction.

**Adaptive Feature Dispatcher**: Comprises a feature aggregator and a feature splitter. The aggregator reduces each backbone layer's features to 32 channels via $1\times1$ + $3\times3$ convolutions, then performs cross-scale semantic alignment through recursive residual aggregation: $m_i = m_{i-1} + \text{Conv2D}(m_{i-1}) + \tilde{F}_i$, producing a unified feature $F_{\text{agg}}$ via a fusion convolution module. The splitter decomposes the fused features into multi-resolution branches. Experiments show that using the last two backbone layers $[2, 3]$ yields the best performance.

**Mask Decoder**: Structurally similar to the Prompt Generator (stacked Transformer blocks) but functionally distinct — it maps prompt semantics to pixel-level segmentation masks. The prompt embedding $E_{\text{dense}}$ serves as the initial input; through $L$ Transformer layers performing cross-attention with multi-scale image features, the representation is progressively refined to capture both semantic and spatial localization information. At inference, only the last-layer prediction is retained, and morphological post-processing is applied to obtain the final masks and bounding boxes.

### Loss & Training

An end-to-end set prediction training paradigm (analogous to DETR) is adopted. Ten prompt groups ($N_q = 10$) are generated per image, each containing 4 prompt embeddings ($K_p = 4$). The AdamW optimizer (lr = 1e-4) is used with a cosine annealing + linear warmup schedule for 150 epochs. DeepSpeed ZeRO Stage 2 with FP16 mixed precision training is employed. Freezing the decoder while fine-tuning the encoder (uf/f configuration) achieves the best trade-off.

## Experiments

### Main Results

Comparisons on a proprietary freight train fault detection dataset (4,410 images, 15 categories across 6 scenarios, resolution $700\times512$):

| Method | Backbone | AP^box | AP^mask | Params | GFLOPs | FPS |
|---|---|---|---|---|---|---|
| Mask R-CNN | ResNet50 | 70.1 | 70.7 | 44.0M | 234 | 44.6 |
| Mask2Former | ResNet50 | 74.2 | 72.6 | 46.3M | 245 | 13.0 |
| Mask2Former | Swin-T | 74.3 | 73.8 | 49M | 252 | 12.8 |
| RSPrompter-query | SAM-B | 72.7 | 71.9 | 131M | 425 | 7.1 |
| SAM FTI-FDet-PF | TinyViT | 73.2 | 72.9 | 30.1M | 196 | 24.4 |
| **SAM FTI-FDet** | **TinyViT** | **74.6** | **74.2** | **36.3M** | **244** | **16.0** |

SAM FTI-FDet achieves state-of-the-art performance on both AP^box and AP^mask, while requiring only ~1/3 the parameters of the SAM-B-based RSPrompter.

### Ablation Study

**Prompt type comparison (Table III)**: Query-based prompts (74.6/74.2) substantially outperform ground-truth bbox prompts (SAM: 66.3), anchor prompts (RSPrompter: 68.4), and bbox-generated prompts (SAM-det: 57.7), validating the semantic advantage of query-style prompting.

**Backbone and pretraining (Table V)**: SA-1B pretrained TinyViT-5m (5M parameters) achieves 74.6/74.2, significantly outperforming ImageNet-pretrained ResNet101 (45M, 70.7/70.3) with only 1/9 the parameters.

**Feature layer selection (Table VI)**: Using the last two layers $[2, 3]$ (74.6/74.2) outperforms all four layers $[0,1,2,3]$ (73.8/73.2) and a single layer $[3]$ (72.6/72.6), indicating that shallow-layer noise disrupts aggregation.

**Freezing strategy (Table VII)**: Fine-tuning the encoder while freezing the decoder (uf/f) achieves the best result of 74.6/74.2; full fine-tuning degrades performance to 72.2/72.2, suggesting that freezing the decoder acts as a regularizer.

**Channel width (Table VIII)**: 256 channels (74.6/74.2) > 128 (70.9/71.5) > 64 (67.3/67.5), with wider channels providing richer discriminative features.

### Key Findings

- Training converges significantly faster than RSPrompter, with the self-prompting mechanism providing more efficient optimization guidance.
- $N_q = 10$ prompt groups is optimal, approximating the typical number of instances per image; too few ($N_q = 1$, AP: 63.0) leads to insufficient coverage, while too many ($N_q = 30$, AP: 73.2) introduces redundancy.
- The combination of lightweight TinyViT-5m and SA-1B pretraining achieves the optimal balance between accuracy and efficiency.

## Highlights & Insights

- Automatic prompt generation entirely eliminates SAM's dependence on manual interaction, enabling a fully automated industrial inspection pipeline.
- With only 36.3M parameters, the method surpasses RSPrompter at 131M, making it suitable for edge device deployment.
- The recursive residual feature aggregation strategy is concise yet effective, achieving strong cross-scale semantic alignment.
- Ablation studies are highly comprehensive (7 groups covering prompt type / backbone / layer selection / freezing strategy / channel width / prompt shape), demonstrating strong systematic rigor.

## Limitations & Future Work

- The dataset contains only 4,410 images from a single national railway system; cross-national and cross-type generalization remains unvalidated.
- The method targets only 15 component categories across 6 freight train scenarios; applicability to general industrial defect detection is unknown.
- FPS = 16 may be insufficient for real-time detection in high-speed train scenarios (below Mask R-CNN's 44.6 FPS).
- Comparisons with the more recent SAM 2 are absent.
- Robustness under extreme conditions such as adverse weather (rain, snow, strong illumination) is not discussed.

## Related Work & Insights

- **SAM family**: SAM (Kirillov et al., 2023) as the original foundation model; MobileSAM/TinyViT-SAM as lightweight variants; FastSAM as a YOLO-based fast version.
- **SAM domain adaptation**: RSPrompter (Chen et al.) for remote sensing prompt learning; SAM-seg/SAM-det for adapting different prompt modalities.
- **Instance segmentation**: Mask R-CNN (two-stage classic), Mask2Former (unified Transformer segmentation), YOLACT/SOLO (single-stage), CondInst (dynamic masks), SparseInst (sparse convolution).
- **Train inspection**: Zhang et al. (lightweight anchor-free detection), Feng et al. (false detection analysis under OOD conditions), Zhou et al. (NanoDet-based real-time system).

## Rating

- Novelty: ⭐⭐⭐ — The combination of self-prompt generation and lightweight SAM transfer offers engineering novelty, though individual module designs (Transformer decoder queries, feature aggregation) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablation studies are comprehensive (7 groups) and the set of compared methods is extensive; however, the dataset is small-scale and sourced from a single domain.
- Writing Quality: ⭐⭐⭐ — Structure is clear, but the density of mathematical notation makes reading demanding, and some paragraphs are verbose.
- Value: ⭐⭐⭐ — Industrial applicability is well-defined, but the generalizability of the method across broader scenarios warrants further validation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segme.md)
- [\[CVPR 2026\] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment](lod-loc_v3_generalized_aerial_localization_in_dense_cities_using_instance_silhou.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)

<!-- RELATED:END -->
