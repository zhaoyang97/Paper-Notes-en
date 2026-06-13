---
title: >-
  [Paper Note] MobileViCLIP: An Efficient Video-Text Model for Mobile Devices
description: >-
  [ICCV 2025][Video Understanding][video-text model] MobileViCLIP introduces spatiotemporal structural re-parameterization into the efficient image-text model MobileCLIP and trains it on large-scale video-text datasets…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "video-text model"
  - "mobile deployment"
  - "structural re-parameterization"
  - "efficient inference"
  - "video retrieval"
date: 2026-05-08
content_hash: 90720abf862bb323
---

# MobileViCLIP: An Efficient Video-Text Model for Mobile Devices

**Conference**: ICCV 2025  
**arXiv**: [2508.07312](https://arxiv.org/abs/2508.07312)  
**Code**: [https://github.com/MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)  
**Area**: Video Understanding  
**Keywords**: video-text model, mobile deployment, structural re-parameterization, efficient inference, video retrieval

## TL;DR

MobileViCLIP introduces spatiotemporal structural re-parameterization into the efficient image-text model MobileCLIP and trains it on large-scale video-text datasets, yielding a mobile-deployable video-text model that achieves performance comparable to much larger models on zero-shot retrieval and action recognition.

## Background & Motivation

- **Background**: Existing video pre-training models (e.g., InternVideo2) are predominantly built on ViT architectures with hundreds of millions to billions of parameters, resulting in high inference latency that precludes deployment on mobile devices. Parameter-efficient transfer learning (PETL) methods reduce trainable parameters but still rely on heavyweight ViT-B/L backbones. Efficient image-text models for mobile (e.g., MobileCLIP) lack temporal modeling capability and cannot understand video content.
- **Goal**: Design a foundation model that is both efficient enough for mobile-device constraints and capable of strong video-text understanding. The authors extend the existing mobile image-text model MobileCLIP by incorporating lightweight temporal modeling modules and train high-quality video-text representations under a PETL paradigm.

## Method

### Overall Architecture

MobileViCLIP is built upon MobileCLIP, comprising a video encoder and a text encoder. The video encoder encodes $T$ frames independently and aggregates them into a video-level representation via temporal average pooling. Training uses video-text contrastive (VTC) learning with the text encoder frozen; only the video encoder is fine-tuned. Two variants are provided: **Tiny** (based on MobileCLIP-S0) and **Small** (based on MobileCLIP-S2).

### Key Designs

1. **Spatiotemporal RepMixer**: A 1D depthwise separable convolution layer is inserted before the original 2D depthwise separable convolution RepMixer to perform temporal modeling. During training: $X' = \text{DWConv1D}(\text{BN}(X)) + X$; during inference, this can be re-parameterized into a single convolution $X' = \text{DWConv1D}(X)$, introducing virtually no additional inference latency. The design motivation is to add temporal awareness to an efficient image model at minimal cost.

2. **Spatiotemporal Attention**: Learnable temporal positional encodings (TPE) are added to the attention modules of the original MCi blocks alongside the existing conditional positional encodings (CPE), enabling the attention layers to model global spatiotemporal representations. At inference time, TPE can be merged into CPE via re-parameterization.

3. **Temporal Pooling**: A parameter-free frame-level average pooling operation that aggregates $T$ frame features into a video-level representation, keeping the design simple and introducing no additional parameters.

### Loss & Training

- **VTC Loss**: Standard InfoNCE contrastive loss that maximizes cosine similarity for positive video-text pairs and minimizes it for negatives:
    $$L_{VTC} = \frac{1}{2}(L_{V2T} + L_{T2V})$$
    with a learnable temperature parameter $\tau$.
- **Training Setup**: 8-frame input, resolution 256×256, AdamW optimizer, learning rate 1e-5, 3 epochs.
- **Training Resources**: Only 8× RTX 3090 GPUs, training completes in 2 days.
- **Pre-training Data**: InternVid-10M-FLT (10 million high-quality YouTube videos).
- **Data Augmentation**: Random cropping and horizontal flipping.

## Key Experimental Results

### Main Results (Zero-Shot Video-Text Retrieval)

| Model | Params (M) | Mobile Latency (ms) | MSR-VTT T2V | MSR-VTT V2T | DiDeMo T2V | DiDeMo V2T |
|-------|-----------|---------------------|-------------|-------------|------------|------------|
| InternVideo2-S14 | 133 | 282 | 35.6 | 35.9 | 33.7 | 35.5 |
| InternVideo2-L14 | 644 | 2319 | 42.1 | 44.1 | 42.8 | 43.2 |
| MobileViCLIP-Tiny | 54 | 15 | 38.7 | 38.1 | 37.1 | 37.0 |
| **MobileViCLIP-Small** | **99** | **42** | **42.5** | **43.5** | **40.7** | **41.1** |

MobileViCLIP-Small achieves performance comparable to InternVideo2-L14 on MSR-VTT while being 55.4× faster on mobile and using 6.5× fewer parameters.

### Ablation Study (Module Effectiveness)

| Model Configuration | MSR-VTT R@1 |
|--------------------|-------------|
| Baseline (MobileCLIP fine-tuned) | 38.4 |
| + Spatiotemporal RepMixer | 39.5 |
| + Spatiotemporal Attention w/o TPE | 39.1 |
| + Spatiotemporal Attention w/ TPE | 39.6 |
| + RepMixer + Attention w/ TPE | **40.1** |

Both spatiotemporal modules contribute positively; TPE is important for temporal change understanding. Freezing the text branch (vs. training both branches) does not hurt performance but saves 3 GB of GPU memory.

### Key Findings

- MobileViCLIP-Small is 6.75× faster than InternVideo2-S14 on mobile with only half the FLOPs.
- On zero-shot action recognition, MobileViCLIP-Small achieves 63.1% on K400 (vs. InternVideo2-S14: 62.1%) and 53.7% on HMDB-51, surpassing even InternVideo2-L14 (53.2%).
- As a feature extractor for temporal grounding, MobileViCLIP-Small outperforms the combined CLIP+SlowFast features.
- Mobile attention-layer latency grows exponentially with the number of stacked layers, while convolution layers have negligible impact—explaining why hybrid architectures are more mobile-friendly.
- On video captioning, MobileViCLIP-Small outperforms ViT-B/32 (BLEU-4: 48.9 vs. 46.1).

## Highlights & Insights

- **Extreme Efficiency**: A mobile-deployable video-text foundation model can be trained with 8× RTX 3090 GPUs in just 2 days, making this approach highly accessible for academic research groups.
- **Elegant Re-parameterization**: The spatiotemporal modules are fully merged into convolutions at inference time, incurring zero additional latency.
- **In-depth Mobile Latency Analysis**: This work systematically analyzes the latency characteristics of fundamental modules on mobile hardware, revealing that attention operations lack the GPU-level optimizations on mobile and exhibit exponentially growing latency with depth.
- **Strong Generalization**: The model demonstrates robust transfer across diverse downstream tasks including temporal grounding, zero-shot action recognition, and video captioning.

## Limitations & Future Work

- Performance on long-video datasets (e.g., ActivityNet) remains limited, as 8-frame sampling is insufficient to cover videos of 5–10 minutes in duration.
- GPU throughput is not competitive with some ViT-based models, which benefit from deeply optimized Transformer kernels on GPUs.
- Training relies solely on contrastive learning; more complex video-text pre-training objectives (e.g., MLM, VTM) are not explored.
- The fully frozen text encoder may cap the model's ability to learn video-domain-specific textual understanding.

## Related Work & Insights

- Efficient image model designs from MobileCLIP/FastViT provide a strong foundation for mobile video understanding.
- High-quality LLM-generated captions in the InternVid dataset are critical for training small-scale models effectively.
- Structural re-parameterization is a classical strategy for achieving complexity during training while maintaining simplicity at inference, particularly well-suited for mobile deployment.

## Rating

- **Novelty**: ⭐⭐⭐ — The method effectively combines existing techniques, though the spatiotemporal re-parameterization extension is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-task evaluation with thorough latency analysis and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with particularly insightful latency analysis.
- **Value**: ⭐⭐⭐⭐ — The first mobile-deployable video-text foundation model; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
