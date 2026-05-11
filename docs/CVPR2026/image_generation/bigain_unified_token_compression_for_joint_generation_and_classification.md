---
title: >-
  [Paper Note] BiGain: Unified Token Compression for Joint Generation and Classification
description: >-
  [CVPR2026][Image Generation][Diffusion model acceleration] BiGain proposes a frequency-aware token compression framework comprising two training-free operators: Laplacian-Gated Token Merging and Interpolation-Extrapolati…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "token compression"
  - "frequency-aware"
  - "joint generation-classification optimization"
  - "training-free"
date: 2026-05-08
content_hash: 2464824226efd47c
---

# BiGain: Unified Token Compression for Joint Generation and Classification

**Conference**: CVPR2026  
**arXiv**: [2603.12240](https://arxiv.org/abs/2603.12240)  
**Code**: [Greenoso/BiGain](https://github.com/Greenoso/BiGain)  
**Area**: Image Generation  
**Keywords**: Diffusion model acceleration, token compression, frequency-aware, joint generation-classification optimization, training-free

## TL;DR

BiGain proposes a frequency-aware token compression framework comprising two training-free operators: Laplacian-Gated Token Merging and Interpolation-Extrapolation KV Downsampling. It is the first to maintain generation quality while significantly improving discriminative classification performance in diffusion model acceleration.

## Background & Motivation

**Computational Bottleneck of Diffusion Models**: The sampling phase of diffusion models involves massive computation. Existing acceleration methods like token merging or downsampling (e.g., ToMe, ToDo) primarily focus on generation quality, neglecting the model's latent discriminative capabilities.

**Growing Demand for Dual-Purpose Models**: The same diffusion backbone can be used simultaneously for image generation and denoising likelihood-based classification (Diffusion Classifier), with broad applications in medical imaging, security, industrial inspection, and remote sensing.

**Acceleration Harms Classification More Than Generation**: Empirical observations show that naive token compression damages classification accuracy much earlier and more severely than generation quality—classification can collapse under extreme sparsity while generation remains acceptable.

**Compression Removes Key Classification Structures**: Traditional compression tends to remove high-frequency information such as edges, textures, and high-contrast boundaries that classification relies on. Even if the global appearance remains intact, classification performance drops significantly.

**Lack of Joint Optimization Perspective**: No previous framework has designed token compression strategies from a joint generation-classification perspective, leading to a gap where models "look good" but "classify poorly."

**Key Insight from Frequency Separation**: By mapping intermediate features to frequency-aware representations, high-frequency (edges/textures) and low-mid frequency (shape/semantics) can be decoupled, providing a design principle to serve both capabilities.

## Method

### Overall Architecture

BiGain is a **training-free, plug-and-play** framework containing two frequency-aware operators. It can be directly embedded into the inference pipelines of diffusion models like DiT and U-Net without fine-tuning. The core design principle is **Balanced Spectrum Preservation**: retaining high-frequency details that support classification while maintaining low-mid frequency semantics that support generation.

### Key Design 1: Laplacian-Gated Token Merging (L-GTM)

- Reshapes the token sequence into spatial $H \times W \times C$ form and computes local frequency magnitude $\mathbf{F} = \text{Reduce}_c(|\mathbf{X} * \mathbf{L}|)$ at each position using a Laplacian kernel $\mathbf{L}$.
- The Laplacian kernel, a discrete approximation of the second derivative, characterizes the degree of difference between a pixel and its neighborhood (high value = high frequency/edge, low value = smooth region).
- Within each grid, tokens with the lowest frequency magnitudes act as the **Target Set** $\mathcal{A}$ (low-frequency anchors), while the rest form the **Source Set** $\mathcal{B}$.
- Global bipartite matching selects the top $r\%$ source-target pairs with the highest similarity for equal-weight average merging.
- **Function**: Encourages merging in smooth regions while protecting high-frequency tokens (edges/textures), reducing attention cost from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N'^2 d)$.
- **Variant ABM**: Adaptive Block Merging, which performs pooling only on blocks where the maximum frequency magnitude is below a threshold $\tau$, suitable for high-resolution stages.

### Key Design 2: Interpolation-Extrapolation KV Downsampling (IE-KVD)

- Performs controllable interpolation/extrapolation downsampling on K and V, while Q remains at full resolution:
  $$\mathcal{D}_{\alpha,s}(\mathbf{Z})[i] = \alpha \cdot \mathbf{Z}[\text{nearest}(i)] + (1-\alpha) \cdot \frac{1}{|\mathcal{N}_s(i)|} \sum_{j \in \mathcal{N}_s(i)} \mathbf{Z}[j]$$
- $\alpha$ controls the balance between nearest-neighbor (preserving high frequency) and average pooling (preserving low frequency).
- **Reason for invariant Q**: Retains the fine-grained receptive field for each output token to stabilize generation quality while maintaining attention precision for discriminative cues.
- Reduces attention cost from $\mathcal{O}(N^2 d)$ to $\mathcal{O}(N \tilde{N} d)$.
- For classification, $\alpha = 0.9$ (biased toward nearest-neighbor/high frequency); for generation, $\alpha$ scales linearly from 0.8 to 1.2 (biased toward low frequency early on, high frequency later).

### Compatibility with Diffusion Classifiers

Both operators are timestep-local and deterministic, not relying on cross-timestep caching. They are fully compatible with the Monte Carlo paired difference estimation of Diffusion Classifiers—all classes share the same noise samples and compression strategy.

## Main Results

### Experimental Settings

- **Backbones**: Stable Diffusion v2.0 (U-Net) and DiT-XL/2 (Transformer)
- **Datasets**: ImageNet-1K, ImageNet-100, Oxford-IIIT Pets, COCO-2017
- **Metrics**: Classification Top-1 Acc / mAP; Generation FID

### Main Results 1: Token Merging (SD-2.0, Table 4)

| Dataset | Method | Acc @ 70% Merge ↑ | FID @ 70% Merge ↓ |
|---|---|---|---|
| Pets | ToMe | 65.76 | 38.35 |
| Pets | **BiGain-TM** | **74.63 (+8.87)** | **37.73 (-0.62)** |
| ImageNet-1K | ToMe | 37.35 | 18.42 |
| ImageNet-1K | **BiGain-TM** | **44.50 (+7.15)** | **18.08 (-0.34)** |
| COCO Acc@1 | ToMe | 57.32 | 29.00 |
| COCO Acc@1 | **BiGain-TM** | **61.44 (+4.12)** | **28.57 (-0.43)** |

At a 70% token merging ratio, BiGain-TM improves classification accuracy by 7.15% on ImageNet-1K while simultaneously improving FID by 0.34.

### Main Results 2: KV Downsampling (SD-2.0, Table 2)

| Dataset | Method | Acc @ 4× DS ↑ | FID @ 4× DS ↓ |
|---|---|---|---|
| Pets | ToDo | 77.46 | 31.48 |
| Pets | **BiGain-TD** | **78.03 (+0.57)** | **29.21 (-2.27)** |
| ImageNet-100 | ToDo | 48.70 | 15.63 |
| ImageNet-100 | **BiGain-TD** | **54.48 (+5.78)** | **15.46 (-0.17)** |

### Performance on DiT-XL/2 (Table 3 & 5)

- With 2× KV downsampling, BiGain-TD achieves **9.08%** higher classification accuracy than ToDo on ImageNet-100 (78.42 vs 69.34), with a 0.35 FID improvement.
- ToDo nearly collapses on DiT at 3× or higher factors (Acc drops to single digits, FID >190), whereas BiGain-TD maintains reasonable performance.
- Regarding token merging, BiGain-TM outperforms ToMe by **7.88%** in classification accuracy at a 70% merging ratio.

### Ablation Study & Key Findings

1. **Necessity of Frequency Awareness**: Removing Laplacian gating leads to a sharp drop in classification accuracy, validating the critical role of high-frequency protection for discriminative ability.
2. **Frequency Balance in KV Downsampling**: Generation tasks benefit from a linear schedule from low to high frequency ($\alpha: 0.8 \to 1.2$), whereas classification prefers a fixed $\alpha=0.9$ (biased toward high-frequency preservation).
3. **Comparison with Competing Methods** (Pets dataset, Table 1): At ~10% FLOPs reduction, BiGain-TM only loses 2.65% Acc (vs ToMe -8.07, SiTo -12.19, DiP-GO -4.50, MosaicDiff -3.65).
4. **Balanced Spectrum Preservation** is a reliable design criterion: Simultaneously retaining high-frequency details and low-mid frequency semantic content benefits both tasks.

## Highlights

- **First Dual-Objective Token Compression Framework**: Expands diffusion model acceleration from single-objective generation quality optimization to joint generation-classification optimization.
- **Elegant and Practical Frequency-Separation Insight**: Laplacian kernel calculation is simple and efficient, requires no learning, and is plug-and-play.
- **Cross-Architecture Generality**: Effective on both U-Net (SD-2.0) and DiT (DiT-XL/2).
- **Training-Free**: No fine-tuning or retraining required; directly embedded during inference.
- **Generalizable Design Criterion**: The principle of balanced spectrum preservation can guide the design of future compression methods.

## Limitations

- The Laplacian kernel is a fixed $3 \times 3$ kernel, which might not be the optimal frequency detector for high-frequency information at different scales.
- $\alpha$ parameters and merging ratios still require tuning for different models/datasets, lacking an adaptive mechanism.
- Performance was only verified under the Diffusion Classifier paradigm; it hasn't been extended to other discriminative protocols like linear probing or feature distillation.
- The ToDo baseline on DiT performs unusually poorly (collapsing at 3×), so the comparative gain might be overestimated.
- More complex scenarios like video diffusion models or 3D generation were not tested.

## Related Work & Insights

- **ToMe/ToMeSD**: Greedy token merging for Transformer and diffusion acceleration, optimizing only for generation quality.
- **ToDo**: Token downsampling via average pooling to reduce attention overhead, ignoring discriminative performance.
- **DiP-GO / Diff-Pruning**: Structured pruning methods that reduce computation through gradient or sub-network searches.
- **MosaicDiff / SiTo**: Other token reduction/pruning strategies focusing solely on generation fidelity.
- **Diffusion Classifier**: Utilizes class-wise denoising likelihood of diffusion models for classification; BiGain's compression makes this paradigm viable even under acceleration.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to propose a dual-objective perspective and frequency-aware compression principles with clear insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets × 2 backbones × 2 operators × multiple compression ratios, with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical chain from motivation to method and experiments; standardized formulas.
- Value: ⭐⭐⭐⭐ — Fills the gap where discriminative ability was neglected in diffusion acceleration; the design criteria have broad value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2026\] SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models](seacache_spectral-evolution-aware_cache_for_accelerating_diffusion_models.md)
- [\[CVPR 2026\] TAUE: Training-free Noise Transplant and Cultivation Diffusion Model](taue_training-free_noise_transplant_and_cultivation_diffusion_model.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)

</div>

<!-- RELATED:END -->
