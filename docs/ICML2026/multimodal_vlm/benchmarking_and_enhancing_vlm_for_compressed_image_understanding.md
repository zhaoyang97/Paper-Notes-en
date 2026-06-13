---
title: >-
  [Paper Note] Benchmarking and Enhancing VLM for Compressed Image Understanding
description: >-
  [ICML 2026][Multimodal VLM][Vision-Language Models] This paper constructs the first large-scale benchmark to evaluate VLM understanding of compressed images (11 codecs, 9 VLMs…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Image Compression"
  - "Compression Artifacts"
  - "Generalization Gap"
  - "Visual Encoder Adapter"
date: 2026-05-08
content_hash: 2e9951958c266c10
---

# Benchmarking and Enhancing VLM for Compressed Image Understanding

**Conference**: ICML 2026  
**arXiv**: [2512.20901](https://arxiv.org/abs/2512.20901)  
**Code**: https://github.com/bblgbr/CompressVLMBench  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Image Compression, Compression Artifacts, Generalization Gap, Visual Encoder Adapter  

## TL;DR

This paper constructs the first large-scale benchmark to evaluate VLM understanding of compressed images (11 codecs, 9 VLMs, 1M+ compressed images). It decomposes performance degradation into an irrecoverable "Information Gap" and a remediable "Generalization Gap." Furthermore, it proposes a lightweight conditional visual encoder adapter that utilizes codec type and compression level embeddings with distillation training, achieving a 10%–30% performance gain across different encoders and bitrates.

## Background & Motivation

**Background**: With the explosive growth of multimedia services and VLM applications, images inevitably undergo compression during transmission and storage. Existing VLM evaluation benchmarks (SEEDBench, MMBench, OCRBench, etc.) primarily use high-quality clear images, while existing image coding standards (JPEG, VVC, learned codecs, generative codecs) are optimized for human perception.

**Limitations of Prior Work**: VLMs in real-world deployments often receive compressed images, yet there is a lack of systematic benchmarks to evaluate their understanding capabilities. Existing Image Coding for Machines (ICM) methods typically target specific codecs and specific vision tasks (e.g., object detection), showing limited generalization.

**Key Challenge**: In VLM performance degradation, how much is due to irreversible information loss from compression (unfixable), and how much is due to the VLM's insufficient generalization to compression artifacts (fixable through adaptation)? Distinguishing these two sources is crucial for deciding whether to "adapt the codec" or "adapt the model."

**Goal**: (1) Build a comprehensive benchmark for compressed image VLM evaluation; (2) Decompose the performance gap into Information Gap and Generalization Gap; (3) Propose a universal adapter to bridge the Generalization Gap.

**Key Insight**: The authors observe that VLM performance drops sharply at low bitrates, but a significant portion of performance can be recovered by fine-tuning on compressed images. This indicates that a substantial percentage of degradation stems from generalization failure rather than information loss.

**Core Idea**: By injecting codec type and compression level as conditions into the VLM visual encoder's positional embeddings and training a unified conditional visual encoder via distillation loss, the VLM can adapt to multiple compression artifacts without modifying the LLM.

## Method

### Overall Architecture

The input consists of a compressed image $\hat{X}$ alongside its codec type and compression level metadata. The output is enhanced visual features aligned with the original uncompressed image features. The mechanism involves fine-tuning only the VLM's visual encoder (ViT), integrating codec conditional information into positional embeddings, and using distillation loss to force features of compressed images to approach those of uncompressed ones. The entire framework requires no modifications to the LLM, maintaining low computational overhead and high versatility.

### Key Designs

1. **Codec Conditional Embedding**:

    - **Function**: Encodes codec type and compression level information into learnable conditional vectors.
    - **Mechanism**: Assuming $m$ codecs and $n$ compression levels, they are first one-hot encoded and then mapped to a $d$-dimensional latent space via an embedding layer to obtain $C_{\mathrm{emb}} = T(m, n, d)$. This is added to the RoPE positional encoding to form a conditional positional encoding $P_{\mathrm{emb}} = \mathrm{RoPE}(h, w, d) + C_{\mathrm{emb}}$, thus injecting compression metadata into all spatial visual tokens.
    - **Design Motivation**: Drawing inspiration from the fusion strategy of time and condition embeddings in conditional diffusion models, this allows the encoder to perceive different distortion types and compression scales, preventing the learning process from being dominated by low-bitrate samples.

2. **Feature Distillation Training**:

    - **Function**: Ensures the features extracted by the Conditional Visual Encoder (CVE) from compressed images are as close as possible to the features extracted by the original Visual Encoder (VE) from uncompressed images.
    - **Mechanism**: The original VE parameters $\theta$ are frozen while CVE parameters $\theta^*$ are trained by minimizing the MSE distillation loss $$\mathcal{L}_d = \| \mathrm{CVE}(\hat{X}, P_{\mathrm{emb}}, \theta^*) - \mathrm{VE}(X, \theta) \|_2^2$$. Training data includes 110k+ COCO images compressed by 3 codecs (JPEG, ELIC, ILLM) at 4 bitrates, forming a 12-dimensional conditional space.
    - **Design Motivation**: Aligning in the feature space rather than the task output layer decouples the adapter from downstream tasks, allowing one adapter to be applicable across multiple VLM tasks (VQA, OCR, Caption, etc.).

3. **Gap Decomposition Framework**:

    - **Function**: Quantifies the VLM performance drop caused by compression into two interpretable components.
    - **Mechanism**: Total Performance Gap $\mathcal{L}(X, \theta) - \mathcal{L}(\hat{X}, \theta)$ = Information Gap $\mathcal{L}(X, \theta) - \mathcal{L}(\hat{X}, \theta^*)$ + Generalization Gap $\mathcal{L}(\hat{X}, \theta^*) - \mathcal{L}(\hat{X}, \theta)$, where $\theta^* = \arg\max_\theta \mathcal{L}(\hat{X}, \theta)$. The Information Gap represents irreversible loss that can only be addressed by improving codecs; the Generalization Gap represents the model's under-adaptation, which can be mitigated via adapters.
    - **Design Motivation**: Provides a diagnostic tool for compressed image VLM research—if the Information Gap dominates, improve codecs; if the Generalization Gap dominates, improve model adaptation.

## Key Experimental Results

### Key Findings from Benchmark

| Finding | Core Conclusion |
|------|---------|
| Finding 1 | VLM semantic understanding drops significantly when bitrate < 0.1 bpp. |
| Finding 2 | Stronger VLMs generally perform better on compressed images, but Janus-pro shows the highest robustness. |
| Finding 3 | Generative codecs (especially diffusion models) provide better semantic reconstruction at low bitrates but perform poorly on fine-grained tasks like OCR. |
| Finding 4 | Model scaling laws do not hold for compressed images: increasing model size does not necessarily reduce compression degradation. |
| Finding 5 | VLM tasks correlate with human perception metrics, but PSNR mainly correlates with OCR, while DISTS/FID correlate more with coarse-grained tasks. |

### Adapter Performance Gain (BD-Metric, QwenVL2.5-3B)

| Codec | POPE | SEEDBench | GQA | MMBench | OCRBench | MME |
|---------|------|-----------|-----|---------|----------|-----|
| JPEG | +12.62 | +12.88 | +11.63 | +14.91 | +52.51 | +285.4 |
| ELIC | +3.42 | +0.69 | +3.88 | +2.45 | +10.51 | +75.97 |
| ILLM | +3.52 | +1.23 | +2.38 | +0.86 | +14.34 | +19.72 |
| StableCodec | +2.87 | +0.63 | +1.34 | +0.09 | +1.30 | +3.18 |

### Generalization to Unseen Codecs and Different VLMs

| VLM | Unseen Codec | POPE | SEEDBench | MME | OCRBench | GQA | MMBench |
|-----|-------------|------|-----------|-----|----------|-----|---------|
| QwenVL2.5-3B | HM | +2.98 | +3.12 | +130.6 | +2.10 | +5.48 | +1.25 |
| QwenVL2.5-3B | MLICpp | +3.32 | +1.22 | +50.0 | +5.73 | +2.01 | +2.52 |
| InternVL3-1B | JPEG | +8.36 | +5.62 | +133.1 | +3.93 | +8.58 | +1.40 |
| InternVL3-1B | ELIC | +2.19 | +1.19 | +25.6 | +6.75 | +4.17 | +0.86 |

### Ablation Study (Condition Comparison)

| Condition Setting | JPEG-POPE | JPEG-SEEDB | ELIC-POPE | ILLM-POPE |
|---------|-----------|------------|-----------|-----------|
| No Conditions | 11.86 | 11.01 | 2.91 | 3.16 |
| Compression Level Only | 12.22 | 11.41 | 3.07 | 3.19 |
| Codec Type Only | 12.43 | 12.54 | 3.28 | 3.41 |
| Full Conditions (Ours) | **12.62** | **12.88** | **3.42** | **3.52** |

## Highlights & Insights

1. **Practical Value of Gap Decomposition**: Quantifying performance loss into Information and Generalization Gaps provides a metric for deciding whether to optimize the codec or the model. On POPE, the Generalization Gap for JPEG is as high as 29.48 (81% of the 36.29 total gap), indicating most degradation can be fixed via adaptation.
2. **Elegant Condition Injection**: Merging codec metadata with RoPE positional encoding via addition flexibly borrows conditional mechanisms from diffusion models, achieving conditionality without altering the ViT architecture.
3. **Strong Generalization**: Adapters trained on only 3 codecs generalize effectively to unseen codecs (HM, MLICpp, DiffEIC) and different VLMs (InternVL3), suggesting the learning of universal distortion-recovery representations.

## Limitations & Future Work

1. Current experiments do not cover the latest closed-source VLMs (e.g., GPT-4V), limiting generalization verification.
2. The adapter requires codec type and compression level metadata as input, which may not always be available in real-world deployments (though ablation shows unconditional versions still work, albeit with potential degradation at high bitrates).
3. Generalization to diffusion-based generative codecs (DiffEIC) is less effective than to traditional or learned codecs due to different distortion patterns between GANs/diffusion and traditional methods.
4. Training is based only on the COCO dataset; generalization to specific domains like medical or remote sensing remains unverified.

## Related Work & Insights

- **Image Coding for Machines (ICM/VCM/FCM)**: Comparison with the ICM method TransTIC shows that simultaneously repairing the Information Gap and Generalization Gap yields additive benefits (BD-POPE increased from 0.18 to 3.02).
- **VLM Robustness Research**: This paper reveals the counter-intuitive phenomenon that VLM scaling laws do not hold under compression artifacts, offering important insights for model deployment strategies.
- **Conditional Feature Alignment**: The paradigm of distillation loss + conditional encoding can be extended to other domain adaptation scenarios (noise, blur, adversarial attacks, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐ (First systematic benchmark + Gap Decomposition framework are significant contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extremely comprehensive: 11 codecs × 9 VLMs × 7 tasks × 4 bitrates, 1M+ images)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-defined findings)
- Value: ⭐⭐⭐⭐ (Directly guides practical VLM deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](../../ICLR2026/multimodal_vlm/enhancing_multi-image_understanding_through_delimiter_token_scaling.md)
- [\[ICML 2026\] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings](timespot_benchmarking_geo-temporal_understanding_in_vision-language_models_in_re.md)
- [\[AAAI 2026\] VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](../../AAAI2026/multimodal_vlm/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)
- [\[AAAI 2026\] SatireDecoder: Visual Cascaded Decoupling for Enhancing Satirical Image Comprehension](../../AAAI2026/multimodal_vlm/satiredecoder_visual_cascaded_decoupling_for_enhancing_satirical_image_comprehen.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](../../ICCV2025/multimodal_vlm/geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)

</div>

<!-- RELATED:END -->
