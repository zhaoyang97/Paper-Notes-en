---
title: >-
  [Paper Note] SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity
description: >-
  [ICML2025][Segmentation][Spiking Neural Network] This paper proposes SpikeVideoFormer, the first spike-driven Transformer designed for video tasks. It utilizes Hamming attention instead of dot-product attention to accurately measure spike feature similarity, and combines joint space-time attention to maintain $\mathcal{O}(T)$ linear time complexity. It achieves state-of-the-art (SOTA) performance for SNNs across three video tasks while being 5-16 times more energy-efficient t…
tags:
  - "ICML2025"
  - "Segmentation"
  - "Spiking Neural Network"
  - "Video Transformer"
  - "Hamming Attention"
  - "Linear Time Complexity"
  - "Video Classification"
  - "Human Pose Tracking"
  - "Video Semantic Segmentation"
date: 2026-05-08
content_hash: 8ab66e79cb2ea9a2
---

# SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity

**Conference**: ICML2025  
**arXiv**: [2505.10352](https://arxiv.org/abs/2505.10352)  
**Code**: [JimmyZou/SpikeVideoFormer](https://github.com/JimmyZou/SpikeVideoFormer)  
**Area**: SNN Video / Spiking Neural Networks / Video Transformer  
**Keywords**: Spiking Neural Network, Video Transformer, Hamming Attention, Linear Time Complexity, Video Classification, Human Pose Tracking, Video Semantic Segmentation

## TL;DR

This paper proposes SpikeVideoFormer, the first spike-driven Transformer designed for video tasks. It utilizes Hamming attention instead of dot-product attention to accurately measure spike feature similarity, and combines joint space-time attention to maintain $\mathcal{O}(T)$ linear time complexity. It achieves state-of-the-art (SOTA) performance for SNNs across three video tasks while being 5-16 times more energy-efficient than ANNs.

## Background & Motivation

**Limitations of Existing SNNs**: Current SNN Transformers (e.g., SpikFormer, Meta-SpikeFormer) mainly focus on spatial feature modeling for single-image tasks, underutilizing the neuron-level temporal encoding capability of SNNs for processing video tasks.

**Defects of Dot-Product Attention**: Existing spike-driven attention mechanisms directly adopt the dot-product operation of ANNs as attention scores, which performs poorly on binary spike features. When zero elements exist in the spike query, the dot product ignores the corresponding information in the key, leading to identical similarity scores for highly different spike key vectors (feature collision).

**Unknown Space-Time Attention Design**: Space-time attention designs in ANNs (joint, hierarchical, factorized) cannot be directly applied to SNNs, necessitating the exploration of the optimal scheme tailored for spike-driven scenarios.

## Method

### Overall Architecture

Conv+ViT hybrid architecture with video input $T \times H \times W \times 3$:

1. **Temporal Spiking Encoder**: Inputs are temporally encoded into spikes via Leaky Integrate-and-Fire (LIF) neurons.
2. **Two Spike-driven CNN Blocks**: Depthwise separable convolution + channelwise convolution, downsampling to $T \times \frac{H}{8} \times \frac{W}{8} \times 4C$.
3. **Two Spike-driven Space-Time Transformers**: SDHA joint attention + Channel MLP, outputting $T \times \frac{H}{16} \times \frac{W}{16} \times 10C$.
4. **Task Heads**: Classification, regression, or segmentation heads.

### Key Designs: Spike-Driven Hamming Attention (SDHA)

**Theoretical Foundation** — Binary embedding extension of the Johnson-Lindenstrauss (JL) Lemma (Proposition 3.1):

There exists an approximation relationship between the normalized Hamming similarity $f_{\mathcal{H}}$ and the cosine similarity $f_{\mathcal{C}}$:

$$P\big(|f_{\mathcal{H}}(q_s, k_s) - g(f_{\mathcal{C}}(q, k))| \leq \delta\big) \geq 1 - 2e^{-\delta^2 D}$$

where $g(x) = 1 - \frac{1}{\pi}\arccos(x)$ is a monotonic continuous function. When the channel dimension $D$ is sufficiently large, the Hamming similarity can approximate the cosine similarity of traditional attention with high probability (maintaining rank consistency).

**Efficient Implementation of Hamming Similarity**:

$$f_{\mathcal{H}}(q_s, k_s) = \frac{1}{2} + \frac{1}{2D}(2q_s - \mathbf{1})^\top(2k_s - \mathbf{1})$$

- $(2q_s - \mathbf{1})$ maps $\{0,1\}$ to $\{-1,1\}$, which can be achieved via bit shifts without requiring multiplications.
- The scaling factor $\frac{1}{2D}$ is integrated into the threshold of the LIF neurons, avoiding extra multiplication during inference.
- Rearranging the computation order maintains $\mathcal{O}(ND^2)$ linear complexity.

**Final SDHA Formula**:

$$\text{SDHA} = \mathcal{SN}_{2D}\Big((2Q_s - \mathbf{1})\big[(2K_s - \mathbf{1})^\top V_s\big]\Big)$$

### Space-Time Attention Design

Comparison of three designs (input $B \times T \times N \times D$):

| Design | ANN Complexity | SNN Complexity | Parameters |
|------|-----------|-----------|--------|
| Joint | $\mathcal{O}(T^2N^2D)$ | $\mathcal{O}(TND^2)$ | $4D^2$ |
| Hierarchical | $\mathcal{O}(TN(T+N)D)$ | $\mathcal{O}(TND^2)$ | $8D^2$ |
| Factorized | $\mathcal{O}(TN(T+N)D)$ | $\mathcal{O}(TND^2)$ | $7D^2$ |

Key Finding: **Joint attention in SNN is both optimal and the most efficient** (fewest parameters, best performance), contrary to the strategy in ANNs where attention decomposition is required to reduce quadratic complexity. The linear attention of SNNs naturally resolves the complexity bottleneck of joint attention.

## Key Experimental Results

### Video Classification (Kinetics-400)

| Method | Spike | Parameters(M) | Energy (mJ) | Top-1(%) | Top-5(%) |
|------|-------|---------|----------|----------|----------|
| ViViT (ANN) | ✗ | 310.8 | 6651.6 | 80.6 | 94.7 |
| Swin-B (ANN) | ✗ | 88.1 | 1297.2 | 80.6 | 94.6 |
| Meta-SpikeFormer | ✓ | 55.9 | 396.4 | 75.5 | 90.1 |
| **SpikeVideoFormer** | ✓ | 55.9 | 412.1 | **79.8** | **94.0** |

Outperforms Meta-SpikeFormer by **+4.3%** in Top-1 accuracy; approaches Swin-B with only a 0.8% gap, while reducing energy consumption by **3×**.

### Human Pose Tracking (MMHPSD, T=32, Video)

| Method | Parameters(M) | Energy (mJ) | PA-MPJPE↓ |
|------|---------|----------|-----------|
| GLoT (ANN) | 40.5 | 4046.1 | 46.5 |
| Meta-SpikeFormer* | 55.8 | 387.2 | 54.5 |
| **SpikeVideoFormer** | 55.8 | 391.2 | **47.5** |

Reduces PA-MPJPE by **7.0mm** compared to SNN SOTA, approaching the best ANN model with only a 1.0mm gap, while reducing energy consumption by **10×**.

### Video Semantic Segmentation (CityScapes, Integer-LIF=4)

| Method | Parameters(M) | Energy (mJ) | mIoU(%) |
|------|---------|----------|---------|
| SegFormer (ANN) | 13.8 | 270.2 | 74.1 |
| Meta-SpikeFormer | 17.8 | 63.5 | 65.9 |
| **SpikeVideoFormer** | 17.8 | 65.3 | **73.1** |

Outperforms SNN SOTA by **+7.2%** mIoU, approaching SegFormer with only a 1% gap, while reducing energy consumption by **4×**.

### Key Findings from Ablation Study

| Variant | Pose PA-MPJPE | VSS mIoU |
|------|--------------|----------|
| Full Model | 39.8 | 73.1 |
| Hamming → Dot-product | 45.7 (+5.9) | 65.9 (-7.2) |
| Joint → Spatial-Only | 54.2 (+14.4) | 62.1 (-11.0) |
| Pre-train → Random | 53.8 (+14.0) | 61.3 (-11.8) |

Hamming attention makes one of the most significant contributions; removing temporal modeling leads to a drastic degradation in performance.

## Highlights & Insights

1. **Theory-Driven Design**: The validity of replacing the dot-product with Hamming similarity is rigorously derived from the JL Lemma, rather than being heuristically constructed.
2. **Counter-Intuitive Finding**: Joint attention is the optimal choice in SNNs (whereas ANNs require factorization to reduce complexity) because the linear attention of SNNs naturally tackles the complexity bottleneck.
3. **Cross-Task Generalization**: The same model achieves SOTA performance across three distinct tasks (classification, regression, and dense prediction), demonstrating strong universality.
4. **Efficiency Advantages Scaled with Sequence Length**: When $T = 8 \rightarrow 32$, energy consumption increases by only 4.1×, while the ANN counterpart increases by 8.3×, reflecting the advantage of $\mathcal{O}(T)$ complexity.
5. **Theoretical Guidance for Threshold Scaling**: $s = 1/2D$ is the theoretically derived optimal value, outperforming heuristically set fixed values in prior work.

## Limitations & Future Work

1. **Performance Gap in VSS**: There remains a significant gap compared to CFFM on the large-scale VSPW dataset (37.9 vs. 49.3 mIoU), largely due to the use of a simple segmentation head.
2. **Limited Inference Speed Advantage on GPUs**: The addition-only advantage of SNNs is physically manifested on neuromorphic hardware, leading to limited savings in inference time on current GPU platforms.
3. **Dependency on Pre-training**: Performance drops significantly without ImageNet pre-training, indicating that training from scratch capability needs further improvement.
4. **Limited to Vision Tasks**: The model has not yet been extended to more complex tasks such as video understanding (Video QA) or video generation.
5. **Additional Overhead of Integer-LIF**: Utilizing multi-bit spikes ({0, 1, 2, 3}) improves performance but deviates from strict binary constraints.

## Related Work & Insights

- **SpikFormer / Meta-SpikeFormer**: Pioneers in SNN Transformers using dot-product attention; this work exposes their theoretical limitations when applied directly to spike features.
- **ViViT / Video Swin**: Joint and factorized space-time attention designs of ANN video Transformers, which inspired the exploration of SNN space-time attention in this work.
- **JL Lemma (Jacques et al., 2013)**: The theoretical tool for distance preservation in binary embedding, providing the mathematical foundation for Hamming attention.

## Rating

- Novelty: ⭐⭐⭐⭐ — Hamming attention is supported by rigorous theoretical derivation, marking the first systematic exploration of SNN video Transformers.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three distinct video tasks, two input modalities, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations and abundant figures/tables.
- Value: ⭐⭐⭐⭐ — Establishes a new baseline for SNN applications in video domains, with clear efficiency advantages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SAMWise: Infusing Wisdom in SAM2 for Text-Driven Video Segmentation](../../CVPR2025/segmentation/samwise_infusing_wisdom_in_sam2_for_text-driven_video_segmentation.md)
- [\[ICCV 2025\] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation](../../ICCV2025/segmentation/inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)
- [\[CVPR 2026\] MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention](../../CVPR2026/segmentation/mixercseg_an_efficient_mixer_architecture_for_crack_segmentation_via_decoupled_m.md)
- [\[CVPR 2026\] Efficient Video Object Segmentation and Tracking with Recurrent Dynamic Submodel](../../CVPR2026/segmentation/efficient_video_object_segmentation_and_tracking_with_recurrent_dynamic_submodel.md)
- [\[CVPR 2025\] Rethinking Query-Based Transformer for Continual Image Segmentation](../../CVPR2025/segmentation/rethinking_query-based_transformer_for_continual_image_segmentation.md)

</div>

<!-- RELATED:END -->
