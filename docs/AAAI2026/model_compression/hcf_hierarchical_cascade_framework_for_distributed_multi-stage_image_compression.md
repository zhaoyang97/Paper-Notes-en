---
title: >-
  [Paper Note] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression
description: >-
  [AAAI 2026][Model Compression][Image Compression] This paper proposes the HCF framework, which performs cross-node transformation directly in the latent space (avoiding pixel-domain recompression) and introduces policy-d…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Image Compression"
  - "Distributed Multi-Stage Compression"
  - "Latent Space Transform"
  - "Quantization Strategy"
  - "Rate-Distortion Optimization"
date: 2026-05-08
content_hash: 38828c1fb09b03f6
---

# HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression

**Conference**: AAAI 2026
**arXiv**: [2508.02051](https://arxiv.org/abs/2508.02051)  
**Code**: N/A  
**Area**: Model Compression
**Keywords**: Image Compression, Distributed Multi-Stage Compression, Latent Space Transform, Quantization Strategy, Rate-Distortion Optimization

## TL;DR

This paper proposes the HCF framework, which performs cross-node transformation directly in the latent space (avoiding pixel-domain recompression) and introduces policy-driven quantization control to achieve up to 12.64% BD-Rate PSNR improvement in distributed multi-stage image compression, while reducing FLOPs by up to 97.8% and GPU memory by up to 96.5%.

## Background & Motivation

### State of the Field

With the rapid growth of digital media consumption, image compression faces dual constraints of bandwidth and storage. In modern transmission scenarios, visual content passes through multiple processing nodes for multi-stage compression, with different quality requirements at each node — a paradigm known as **distributed multi-stage image compression**. For example, in edge computing networks, images are transmitted from source → base station → destination, where each node may require a different compression quality level.

### Limitations of Prior Work

Existing methods exhibit three fundamental limitations:

**Progressive Compression Framework (PCF)**: Achieves quality adaptation via bitstream truncation ("encode once, decode multiple times"), but intermediate nodes can only passively truncate and **cannot leverage their computational resources** for active optimization.

**Distributed Recompression Framework (DRF)**: Performs a full decompress-recompress cycle at each node, utilizing computational resources but **incurring cumulative quality degradation and computational redundancy due to repeated pixel-domain encoding/decoding**.

**Fixed-Parameter Models (SSF)**: Achieve optimal performance in centralized settings but provide only a single operating point, **lacking post-encoding flexibility**.

### Root Cause

PCF wastes the computational capacity of intermediate nodes, while DRF, though utilizing computational resources, squanders them on redundant pixel-domain operations. Both approaches treat quality adaptation as an operation external to the compression pipeline, **never considering adaptation within the compression process itself**.

### Starting Point

This paper reframes quality adaptation from "post-compression adaptation" to "in-compression adaptation" — performing inter-node quality transformation directly in the latent space, thereby leveraging intermediate node computation while avoiding redundant pixel-domain operations. A policy-driven quantization control mechanism is additionally introduced to maximize rate-distortion performance by selecting optimal quantization placement.

## Method

### Overall Architecture

HCF establishes a direct latent-space transformation path for distributed multi-stage compression:
- **Source node**: Analysis transform $g_a^s$ → quantization/encoding (optional) → transmission
- **Intermediate nodes**: Receive latent representation → transformation module $\phi_{k \to k-1}$ → quantization/encoding (optional) → transmission
- **Destination node**: Entropy decoding → synthesis transform $g_s^d$ → reconstructed image

The key innovation is that intermediate nodes **do not need to return to the pixel domain**; instead, quality-level conversion is performed directly in the latent space.

### Key Designs

#### 1. Dual-Mode Processing: Inter-Node and Intra-Node

Each quality level $k$ is processed in one of two modes:

**Inter-node processing**: Involves data transmission, including quantization, entropy encoding, and entropy decoding, followed by the transformation module:

$$\mathcal{T}_k^{\text{inter}}(\tilde{y}^k) = (\phi_{k \to k-1}^{\text{inter}} \circ D^k \circ E^k \circ Q^k)(\tilde{y}^k)$$

**Intra-node processing**: Does not involve transmission; the transformation is applied directly:

$$\mathcal{T}_k^{\text{intra}}(\tilde{y}^k) = \phi_{k \to k-1}^{\text{intra}}(\tilde{y}^k)$$

Both modes share the same network architecture (residual blocks + attention + GDN activations) but are trained separately to handle quantized and unquantized inputs respectively.

- **Design Motivation**: Quantized and unquantized inputs follow different distributions, necessitating separate transformation learning; flexible switching between inter-node/intra-node modes enables optimization of quantization placement.

#### 2. Policy-Driven Quantization Control

A policy vector $\boldsymbol{\pi} = [\pi_s, \pi_{s-1}, \ldots, \pi_d]$ is defined, where $\pi_k \in \{0,1\}$ indicates whether level $k$ undergoes inter-node (1) or intra-node (0) processing. The constraint $\pi_d = 1$ ensures the final level is always transmitted.

The complete end-to-end cascade is:

$$\mathcal{C}(s,d,\boldsymbol{\pi})(x) = (g_s^d \circ D^d \circ E^d \circ Q^d \circ \mathcal{F}_{s \to d}^{\boldsymbol{\pi}} \circ g_a^s)(x)$$

The policy space is defined as:

$$\Pi(s,d;n_q) = \{\boldsymbol{\pi} \in \{0,1\}^{s-d+1} \mid \sum_{k=d}^{s} \pi_k = n_q, \pi_d = 1\}$$

- **Mechanism**: Different policy vectors dynamically determine at which levels quantization and transmission occur.
- **Design Motivation**: Different quantization placement strategies lead to significantly different rate-distortion performance.

#### 3. Edge Quantization Optimality Principle

Through systematic experiments and differential entropy analysis, the **Edge Policy** is found to be consistently optimal:

$$\boldsymbol{\pi}^{\text{edge}} = [1^{(n_q-1)}, 0^{(s-d+1-n_q)}, 1]$$

This concentrates quantization operations at the front of the cascade (the edge), applying quantization only at the initial and final stages.

A **Rate-Quality Sensitivity Index (RQSI)** is proposed to quantify policy efficiency:

$$\eta^{\mathcal{M}}(\boldsymbol{\pi}) = \frac{1}{2}(RQS(\boldsymbol{\pi}, \boldsymbol{\pi}_*) + RQS(\boldsymbol{\pi}, \boldsymbol{\pi}^*))$$

where $RQS$ measures the ratio of quality change to bitrate change; a lower RQSI indicates a more efficient policy.

**Differential entropy analysis reveals the underlying reason**: after early quantization is introduced, subsequent transformation modules can more effectively decorrelate and suppress redundant information. The entropy increases prior to final quantization for the three strategies are 94.1 (edge), 96.1, and 99.3 bits respectively — the edge policy yields the smallest entropy increase.

### Loss & Training

**Two-stage training**:

1. **Transformation module training**: Modules are trained sequentially from $k=s$ to $k=d+1$, minimizing the L2 distance between the transformed latent representation and the analysis transform output of the target quality level:
    - Intra-node: $\mathcal{L}_{\text{intra}}^{k \to k-1} = \|\phi_{k \to k-1}^{\text{intra}}(\tilde{y}^k) - g_a^{k-1}(x)\|_2^2$
    - Inter-node: $\mathcal{L}_{\text{inter}}^{k \to k-1} = \|\phi_{k \to k-1}^{\text{inter}}(\hat{y}^k) - g_a^{k-1}(x)\|_2^2$

2. **End-to-end fine-tuning**: Uses the rate-distortion objective $\mathcal{L}_{\text{RD}}^k = \lambda_k \cdot \mathcal{D}(x, \hat{x}^k) + \mathcal{R}(\hat{y}^k)$

The framework is initialized from pretrained single-stage compression models, with higher-quality-level networks frozen during training.

## Key Experimental Results

### Main Results

**BD-Rate comparison (MLIC++ backbone, relative to HCF $\pi^{\text{edge}}$)**:

| Method | Kodak BD-Rate_P↓ | Kodak BD-PSNR↑ | CLIC BD-Rate_P↓ | CLIC BD-PSNR↑ |
|--------|-----------------|-----------------|-----------------|----------------|
| Presta (PCF SOTA) | +12.64% | -0.46dB | +9.48% | -0.28dB |
| Jeon (PCF) | +13.40% | -0.51dB | +8.62% | -0.27dB |
| Lee (PCF) | +43.84% | -1.76dB | +26.39% | -0.80dB |
| DRF (recompression) | +4.87% | -0.22dB | +5.56% | -0.23dB |

**Computational efficiency (HCF vs. DRF)**:

| Model | FLOPs Reduction↑ | GPU Memory Reduction↑ | Execution Time Reduction↑ |
|-------|-----------------|----------------------|--------------------------|
| cheng2020_attn | 97.8% | 96.5% | 90.0% |
| cheng2020_anchor | 97.5% | 96.5% | 87.1% |

### Ablation Study

**Policy comparison (MLIC++, Kodak, $n_q=2$, target quality $d=2$)**:

| Policy | PSNR↑ | MS-SSIM↑ | RQSI_PSNR↓ |
|--------|-------|----------|-----------|
| $\pi^{\text{edge}}$ = [1,0,0,0,1] | **30.264** | **13.129** | **11.054** |
| [0,0,0,1,1] | 29.768 | 12.761 | 17.115 |

**Component ablation (CLIC2020-mobile)**:

| Configuration | Performance |
|---------------|-------------|
| $\phi^{\text{inter}} + \phi^{\text{intra}}$ (full) | Optimal |
| $\phi^{\text{inter}}$ only | Degraded at low bitrates (insufficient information preservation) |
| $\phi^{\text{intra}}$ only | Degraded at high bitrates (poor handling of quantization artifacts) |

**Cross-quality adaptation (without retraining)**:

| Compression Path | BD-Rate_P↓ | BD-Rate_M↓ | Note |
|------------------|-----------|-----------|------|
| 5→1 vs. 6→1 | -7.13% | -7.29% | Shorter path is superior |
| 4→1 vs. 5→1 | -7.36% | -7.02% | |
| 3→1 vs. 4→1 | -10.87% | -7.80% | Most significant improvement |

### Key Findings

1. **Edge quantization is consistently optimal**: Across all quantization frequencies and configurations, placing quantization at the front consistently outperforms other strategies, yielding up to 0.6 dB PSNR gain.
2. **Remarkable computational efficiency**: By avoiding pixel-domain recompression, FLOPs are reduced by up to 97.8%, as analysis/synthesis transforms constitute the most computationally intensive components while the transformation modules are lightweight.
3. **Cross-architecture generalization**: Validated across five different compression architectures, ranging from hyperprior to context-attention models.
4. **Training-free cross-quality adaptation**: Learned latent-space transforms can be flexibly combined to support different compression paths without retraining.

## Highlights & Insights

1. **Paradigm shift**: The reframing from "post-compression adaptation" to "in-compression adaptation" has far-reaching implications for distributed system design.
2. **Discovery of the edge quantization principle**: A theoretical explanation is provided via differential entropy analysis — early quantization enables subsequent modules to decorrelate redundancy more effectively.
3. **Extreme computational efficiency**: A 97.8% reduction in FLOPs implies that intermediate nodes require almost no computational resources, making this approach highly attractive for edge computing scenarios.
4. **Systematic experimental design**: Five architectures × three datasets × multiple quantization frequencies provide comprehensive coverage.
5. **Formalization of the policy space**: Casting quantization placement as an optimization problem over policy vectors lays the groundwork for future adaptive policy selection.

## Limitations & Future Work

1. **Limited to image compression**: The authors note that extension to video compression is planned, but the current work does not address it.
2. **Policy selection via enumeration**: As the number of levels increases, the policy space grows exponentially, necessitating adaptive policy selection mechanisms (e.g., reinforcement learning agents).
3. **Fixed transformation module architecture**: All levels share the same architecture; customization based on level-specific characteristics may yield further gains.
4. **Limited training dataset scale**: Transformation modules are trained solely on the DUTS dataset, which may constrain generalization.
5. **Real-world network conditions not considered**: The work does not address practical distributed environment challenges such as network latency and packet loss.

## Related Work & Insights

- **Progressive compression**: Element-level progressive transmission by Presta et al. (2025) → the limitations of passive truncation motivated the active transformation approach in this paper.
- **Successive compression**: SIC by Kim et al. (2022) → highlights the cumulative degradation caused by pixel-domain recompression.
- **Knowledge distillation**: The transformation module design is inspired by knowledge distillation, "distilling" high-quality latent representations into lower-quality levels.
- **Broader insight**: In distributed systems, operating in the abstract/compressed domain rather than reverting to the original domain is a principle generalizable to other distributed processing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Paradigm-level innovation; the discovery of the edge quantization principle is also highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Five architectures, three datasets, theoretical analysis, ablation studies, and efficiency analysis provide comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐ (Framework description is clear, though the dense notation raises the reading barrier)
- Value: ⭐⭐⭐⭐⭐ (Directly applicable engineering value for distributed scenarios such as 6G edge networks; 97.8% computational savings are highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](../../ICML2026/model_compression/hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](../../CVPR2026/model_compression/parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [\[AAAI 2026\] DynaQuant: Dynamic Mixed-Precision Quantization for Learned Image Compression](dynaquant_dynamic_mixed-precision_quantization_for_learned_i.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)

</div>

<!-- RELATED:END -->
