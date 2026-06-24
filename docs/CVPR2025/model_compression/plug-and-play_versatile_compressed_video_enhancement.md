---
title: >-
  [Paper Note] Plug-and-Play Versatile Compressed Video Enhancement
description: >-
  [CVPR 2025][Model Compression][Compressed video enhancement] This paper proposes a codec-aware compressed video enhancement framework. By reusing compression priors such as compression factors, motion vectors, and partition maps from the bitstream, a single model adaptively enhances videos across various compression levels, while also serving as a plug-and-play module to assist multiple downstream vision tasks.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Compressed video enhancement"
  - "codec-aware"
  - "dynamic networks"
  - "motion vectors"
  - "plug-and-play"
date: 2026-05-08
content_hash: b31e196961eca6a6
---

# Plug-and-Play Versatile Compressed Video Enhancement

**Conference**: CVPR 2025  
**arXiv**: [2504.15380](https://arxiv.org/abs/2504.15380)  
**Code**: [https://huimin-zeng.github.io/PnP-VCVE/](https://huimin-zeng.github.io/PnP-VCVE/)  
**Area**: Model Compression / Video Enhancement  
**Keywords**: Compressed video enhancement, codec-aware, dynamic networks, motion vectors, plug-and-play

## TL;DR

This paper proposes a codec-aware compressed video enhancement framework. By reusing compression priors such as compression factors, motion vectors, and partition maps from the bitstream, a single model adaptively enhances videos across various compression levels, while also serving as a plug-and-play module to assist multiple downstream vision tasks.

## Background & Motivation

**Background**: Video compression is a standard operation in data transmission. However, compression inevitably introduces artifacts that degrade visual quality, affecting the robustness of downstream tasks (such as object detection and semantic segmentation). Existing video enhancement methods are mainly categorized into in-loop filtering and post-processing.

**Limitations of Prior Work**: Existing post-processing enhancement methods suffer from three core issues. First, methods like the MFQE series and STDF train independent models for each compression level, failing to flexibly handle different or even unseen compression configurations. Second, while recent methods randomly mix inputs of different compression levels during training, this "compression-unaware" strategy offers limited generalization capability. Third, almost all methods focus solely on visual quality enhancement, ignoring the actual demands of assisting downstream tasks in real-world scenarios.

**Key Challenge**: Practical application scenarios (e.g., autonomous driving) require a versatile solution that can handle various compression levels with a single model and assist multiple downstream tasks without introducing computational bottlenecks. However, existing methods fail to strike a balance between flexibility, versatility, and efficiency.

**Goal**: To design a scheme that meets three criteria: (1) adaptively enhancing different compression levels with a single model; (2) assisting multiple downstream tasks in a plug-and-play manner; (3) introducing no computational bottlenecks.

**Key Insight**: The authors observe that the codec bitstream already contains rich compression prior information—CRF values reflect the degree of compression, motion vectors encode temporal relationships, and partition maps indicate regional complexity. This information is available for "free" at the decoder side but has been overlooked by existing methods.

**Core Idea**: Reuse the readily available information in the codec bitstream as conditions for dynamic networks, achieving compression-adaptive parameter tuning and space-adaptive regional enhancement.

## Method

### Overall Architecture

The overall framework consists of two subnetworks: the Compression-Aware Adaptation network (CAA) and the Bitstream-Aware Enhancement network (BAE). CAA acts as a "meta-network" that dynamically adjusts the parameters of the BAE network based on sequence-level and frame-level compression factors. The BAE network receives the adjusted parameters, aligns inter-frame details using motion vectors, and performs region-adaptive enhancement utilizing partition maps. The input consists of compressed video frames and corresponding bitstream information, and the output is the enhanced video frames.

### Key Designs

1. **Compression-Aware Adaptation Network (CAA) — Hierarchical Parameter Adaptation Mechanism**:

    - **Function**: Dynamically generates parameters for the enhancement network based on compression levels, allowing a single model to handle multiple compression configurations.
    - **Mechanism**: A hierarchical adaptation mechanism is designed. In the sequence-level adaptation phase, $N=6$ parallel expert layers (with identical structures but independent parameters) are preset to generate weighting coefficients $w_n$ conditioned on the sequence-level $CRF_s$, and the expert layer parameters are weighted and summed to obtain the sequence-adaptive parameters $f_{\theta_s} = \sum_{n=1}^{N} w_n f_{\theta_n}$. In the frame-level adaptation phase, the frame-level $CRF_i$ is utilized to predict auxiliary parameters $\triangle\theta_i$, which are added to the sequence-adaptive parameters to yield the frame-adaptive parameters $f_{\theta_i} = f_{\theta_s + \triangle\theta_i}$. The key is that $CRF_s$ remains constant throughout the sequence, so the sequence-adaptive parameters only need to be computed once and can be reused.
    - **Design Motivation**: Quality adjustment in video compression is inherently hierarchical—CRF controls the overall compression rate at the sequence level and dynamically adjusts at the frame level based on I/P/B frame types. Designing parameter adaptation to mimic this hierarchical structure is both natural and efficient. Experiments also show that when frame-level CRF is unavailable, substituting it with frame types (I/P/B) yields close results (PSNR drop < 0.03dB).

2. **Motion Vector Alignment (MV Alignment)**:

    - **Function**: Uses free motion vectors from the bitstream to aggregate neighboring frame information and provide temporal compensation for the current frame.
    - **Mechanism**: For each block in the current frame, the motion vector points to blocks with similar content in the past and future reference frames. Bilinear interpolation is used to warp reference features, which are then concatenated with the current frame in the channel dimension as input to the BAE network: $\hat{x_i} = [MV(h_i^p), MV(h_i^f), x_i]$.
    - **Design Motivation**: Although motion vectors are less precise than optical flow, they are "freely available" from the bitstream without extra computation. Compared to optical flow estimation methods (such as STDF's deformable convolution), motion vector alignment incurs almost zero extra overhead, making it highly suitable for real-time processing scenarios.

3. **Region-Aware Refinement**:

    - **Function**: Assigns independent convolutional filters to different regions based on regional complexity indicated by the partition map to achieve fine-grained enhancement.
    - **Mechanism**: Decouples the H.264 partition map into multiple binary masks (corresponding to 16×16, 8×16/16×8, and 8×8 block sizes), with each mask mapped to a set of sparse convolutional filters. The output is the sum of the frame-adaptive feature extraction and the region-refined features: $\hat{h_i} = f_{\theta_i} * h_i + \sum_{type=1}^{M} \mathcal{S}(M_i^{type}, h_i)$. Visualization shows that the refined features of different masks focus on different semantic regions (e.g., 8×8 focuses on static objects, 8×16 focuses on moving objects).
    - **Design Motivation**: The compression encoder has already partitioned the frame based on texture complexity—flat regions use large blocks, while detailed regions use small blocks. Directly reusing this partition information avoids the overhead of learning region masks, and the sparse convolution operates only on corresponding regions, resulting in high computational efficiency.

### Loss & Training

End-to-end training is performed using the Charbonnier penalty loss: $\mathcal{L} = \frac{1}{T}\sum_{i=1}^{T}\sqrt{\|y_i - \hat{y}_i\|^2 + \epsilon^2}$, where $\epsilon = 10^{-12}$. Training data utilizes a combination of the REDS and DAVIS training sets, processed with the H.264 compression standard, with CRF levels set to 15, 25, and 35.

## Key Experimental Results

### Main Results

Quality enhancement performance on the REDS4 dataset (PSNR dB):

| Method | Params/M | CRF15 | CRF25 | CRF35 | CRF18 (Unseen) | CRF28 (Unseen) | CRF38 (Unseen) |
|------|---------|-------|-------|-------|------------|------------|------------|
| Input | - | 41.04 | 34.92 | 29.25 | 39.12 | 33.18 | 27.69 |
| MFQE 2.0 | 1.64 | 40.95 | 34.83 | 29.22 | 38.97 | 33.13 | 27.67 |
| STDF | 1.27 | 41.15 | 35.23 | 29.74 | 39.28 | 33.58 | 28.11 |
| S2SVR | 7.43 | 41.96 | 35.61 | 29.87 | 39.88 | 33.87 | 28.19 |
| Metabit | 1.60 | 41.04 | 34.92 | 29.25 | 39.11 | 33.18 | 27.69 |
| **Ours** | **4.56** | **42.22** | **35.90** | **30.17** | **40.17** | **34.16** | **28.49** |

PSNR of assisting downstream $\times 4$ video super-resolution (BasicVSR++ baseline):

| Method | CRF15 | CRF25 | CRF35 |
|------|-------|-------|-------|
| BasicVSR++ | 29.61 | 26.19 | 23.38 |
| + S2SVR | 29.82 | 26.72 | 23.85 |
| + **Ours** | **29.92** | **26.87** | **24.00** |

### Ablation Study

| Configuration | CRF15 PSNR | Description |
|------|-----------|------|
| Full model (CRFi) | 42.22 | Full model, using CRF values at the frame level |
| Full model (slice type) | 42.24 | Substituting frame level with I/P/B types, difference is only < 0.03dB |
| Metabit (without hierarchical adaptation) | 41.04 | No enhancement effect |
| STDF (without codec information) | 41.15 | Only +0.11dB |

### Key Findings

- MFQE 2.0 and Metabit barely improve quality when trained on mixed data (even performing worse than the input on CRF15), indicating that simple training strategies that mix compression levels are completely ineffective.
- The hierarchical adaptation mechanism generalizes exceptionally well to unseen CRF configurations (+1.06dB on CRF18), whereas STDF/S2SVR only gain +0.16/+0.76dB.
- The degradation from replacing frame-level CRF with frame types (I/P/B) is negligible, which considerably lowers the barrier for practical deployment.
- This method requires only 61% of S2SVR's parameters and 16% of its FLOPs, achieving a throughput of 28 FPS.

## Highlights & Insights

- **Clever Reuse of Codec Information**: Motion vectors and partition maps are "free" outputs at the decoder side. This work transforms them from "overlooked byproducts" into valuable conditional signals. This concept is generalizable to any vision task dealing with compressed data.
- **Dynamic Adaptation in Parameter Space**: Unlike Mixture of Experts (MoE) which performs weighted fusion in the feature space (where computational complexity scales linearly with the number of experts), this method performs weighted fusion in the parameter space. During inference, it is equivalent to a single network with zero extra overhead.
- **Pragmatic Design**: The finding that frame types can replace frame CRFs implies that the method remains functional even in scenarios with restricted access to the full bitstream.

## Limitations & Future Work

- Validations are currently limited to the H.264 standard. Although the authors state H.265/H.266 offer similar priors, experimental verification is not provided.
- Motion vector alignment may lack precision in complex motion scenarios due to the limitations of block-level granularity.
- Evaluations on downstream tasks focus mainly on video-level tasks, with direct performance gains for image-level tasks (such as object detection) not yet considered.
- Compatibility with learned codecs (such as end-to-end video compression) has not been explored.

## Related Work & Insights

- **vs MFQE 2.0**: MFQE utilizes a BiLSTM to detect peak-quality frames for multi-frame enhancement but requires deploying separate models for each compression level. This work replaces it with a single adaptive model, offering far superior flexibility.
- **vs CVCP/CIAF**: These methods also exploit motion vectors and spatial priors, but focus exclusively on video super-resolution. The proposed framework is more general, supporting various downstream tasks.
- **vs Metabit**: Metabit also utilizes spatial priors but only processes I/P frames and lacks hierarchical adaptation. The hierarchical mechanism designed in this work covers all frame types.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of integrating codec information as dynamic network conditions is novel, though the individual module designs lean towards empirical engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering quality enhancement, three downstream tasks, generalization to unseen compression levels, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-argued motivation, though dense with mathematical notations.
- Value: ⭐⭐⭐⭐ High practical utility. The plug-and-play design facilitates easy deployment into various video analysis pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Plug-and-Play Fidelity Optimization for Diffusion Transformer Acceleration via Cumulative Error Minimization](../../ICLR2026/model_compression/plug-and-play_fidelity_optimization_for_diffusion_transformer_acceleration_via_c.md)
- [\[ICML 2026\] Plug-and-Play Spiking Operators: Breaking the Nonlinearity Bottleneck in Spiking Transformers](../../ICML2026/model_compression/plug-and-play_spiking_operators_breaking_the_nonlinearity_bottleneck_in_spiking_.md)
- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)
- [\[CVPR 2025\] AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)

</div>

<!-- RELATED:END -->
