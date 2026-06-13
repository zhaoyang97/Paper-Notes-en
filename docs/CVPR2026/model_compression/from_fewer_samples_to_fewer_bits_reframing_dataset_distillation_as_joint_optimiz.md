---
title: >-
  [Paper Note] From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness
description: >-
  [CVPR2026][Model Compression][Dataset distillation] This paper proposes QuADD, a framework that embeds a differentiable quantization module into the dataset distillation loop to jointly optimize synthetic data and quanti…
tags:
  - "CVPR2026"
  - "Model Compression"
  - "Dataset distillation"
  - "quantization-aware training"
  - "rate-distortion optimization"
  - "low-bit data compression"
  - "non-uniform quantization"
date: 2026-05-08
content_hash: 0310432d489ef697
---

# From Fewer Samples to Fewer Bits: Reframing Dataset Distillation as Joint Optimization of Precision and Compactness

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.02411](https://arxiv.org/abs/2603.02411)  
**Code**: Not released  
**Area**: Model Compression / Dataset Distillation
**Keywords**: Dataset distillation, quantization-aware training, rate-distortion optimization, low-bit data compression, non-uniform quantization

## TL;DR

This paper proposes QuADD, a framework that embeds a differentiable quantization module into the dataset distillation loop to jointly optimize synthetic data and quantization parameters, achieving Pareto-optimal compression of "fewer samples + lower precision" under a fixed bit budget.

## Background & Motivation

1. **Limitations of dataset distillation**: Existing DD methods primarily improve compactness by reducing the number of synthetic samples $M$ or data dimensionality $D$, while each data element is still stored at 32-bit full precision, neglecting bit-width $b$ as a controllable degree of freedom.
2. **Total bit-budget perspective**: The true storage cost of a dataset is $\text{Budget} = M \times D \times b$; optimizing only $M$ or $D$ without considering $b$ leaves significant storage efficiency unexploited.
3. **Drawbacks of post-quantization**: Quantizing after distillation leads to substantial accuracy degradation, as synthetic samples are not optimized for low-precision representation.
4. **Distributed and edge-computing requirements**: IoT devices and bandwidth-constrained distributed learning scenarios demand maximally compact data for transmission and storage, necessitating a shift from "fewer samples" to "fewer bits."
5. **Limitations of prior methods**: Color quantization approaches such as AutoPalette apply only to the image domain, introduce additional training complexity, and generalize poorly to other modalities.
6. **Challenge of co-optimizing quantization and distillation**: Quantization involves two non-differentiable operations—clipping and rounding—posing technical barriers to direct integration into gradient-based optimization loops.

## Method

### Overall Architecture

QuADD inserts a differentiable quantization layer $Q(\cdot)$ into the standard DD loop, replacing the objective of matching full-precision synthetic data $\mathcal{S}$ to real data $\mathcal{T}$ with matching quantized synthetic data $\mathcal{S}^q$ to $\mathcal{T}$:

$$\mathcal{S}^* = \arg\min_{\mathcal{S}} \mathbb{E}_{\theta \sim \Theta} [\mathcal{L}(\phi(\mathcal{T};\theta), \phi(\mathcal{S}^q;\theta))]$$

At each iteration, synthetic data is first passed through the quantization layer to obtain $\mathcal{S}^q = Q(\mathcal{S})$, the distillation loss is then computed, and gradients are back-propagated via the chain rule to both the synthetic data and quantizer parameters.

### Differentiable Quantization Layer Design

**Forward pass**:

- **Hard rounding**: Maps continuous values directly to the nearest codebook level; applicable to both uniform and non-uniform quantization.
- **Soft rounding**: Approximates the rounding operation with a continuous function (used only for the uniform quantization baseline).

**Backward pass**:

- **STE (Straight-Through Estimator)**: Treats the quantizer as an identity mapping within the clipping range, $\partial x^q / \partial x \approx \mathbf{1}(|x| \le \alpha)$.
- **Analytic surrogate gradients**: Derivatives of smooth approximation functions used for the soft quantizer.

**Uniform quantization baseline**: Codebook consists of equally spaced levels $Q^u(\alpha, b) = \alpha \times \{-1, \pm\frac{1}{2^{b-1}-1}, \ldots, 1\}$.

**Adaptive non-uniform quantization (APoT)**: Employs an Additive Powers-of-Two scheme, where each quantized value is expressed as a scaled sum of powers of two, allocating finer quantization granularity in regions of high data density. Only a single learnable parameter $\alpha$ (the clipping threshold) is introduced; the Reparameterized Clipping Function (RCF) ensures that $\alpha$ receives gradients from all samples.

### Quantization-Guided Initialization

A greedy selection strategy based on generalized graph cuts is used to initialize synthetic data: real data is first uniformly quantized to obtain $\mathcal{T}^q$, and samples that maximize the conditional gain $G^*(A|C)$ are iteratively selected, where sample similarity is measured by the cosine similarity of last-layer gradients.

### Training Procedure

Each iteration: sample a real-data mini-batch → quantize synthetic data → compute distillation loss → back-propagate gradients jointly to $\mathcal{S}$ and $Q$ → update both sets of parameters. The quantization layer is lightweight and modality-agnostic, introducing no additional computational overhead.

## Key Experimental Results

### Main Results: Accuracy Comparison Under Fixed Storage Budget

| Method | CIFAR-10 IPC10 | CIFAR-100 IPC10 | ImageNette IPC10 | Compression Ratio |
|--------|---------------|----------------|-----------------|-------------------|
| DATM (32-bit) | 65.7% | 47.3% | 67.8% | 1× |
| FreD | 57.3% | 34.9% | 66.2% | 9.6–12× |
| AutoPalette (7-bit) | 63.5% | 45.6% | 63.0% | 9.6× |
| **QuADD (9-bit)** | **65.4%** | **46.2%** | **67.0%** | **10.6×** |

At a 10.6× compression ratio, QuADD achieves accuracy within approximately 1% of the full-precision baseline, substantially outperforming AutoPalette and FreD.

### Cross-Domain Experiment: 3GPP Beam Management

- Full 32-bit data training: 89% accuracy
- Full 8-bit quantization: 87% accuracy, 4× compression
- DD without quantization: 77% accuracy, 46× compression
- **QuADD**: 81.9% accuracy at 36× compression; 77.5% accuracy at **183× compression**

### Ablation Study

- **Rate-distortion analysis**: Under a fixed budget, lower bit-width with more samples generally outperforms higher bit-width with fewer samples; optimal accuracy is concentrated at 2–3 bits/sub-pixel.
- **Cross-architecture generalization** (Table 2): Evaluated on AlexNet/VGG/ResNet, QuADD achieves 51.8% at IPC10 on ResNet, surpassing DATM (49.0%) and AutoPalette (50.1%).
- **Cross-distillation-method compatibility** (Table 3): QuADD is compatible with both TM and DM distillation frameworks, reaching 63.2% at IPC10 under TM, approaching the full-precision result of 65.2%.
- **Training efficiency**: QuADD reduces training time by 25–30% compared to DATM at medium-to-high IPC, and is substantially faster than AutoPalette and FreD.

## Highlights & Insights

- **Novel perspective**: Reframes the DD problem from "reducing samples" to "reducing bits," introducing a rate-distortion framework to analyze optimal allocation across $(M, b)$.
- **General-purpose framework**: The quantization module is modality-agnostic and distillation-method-agnostic, serving as a plug-and-play component compatible with TM, DM, DATM, and other DD methods.
- **High practical value**: Achieves over 10× storage compression with minimal accuracy loss, directly benefiting edge computing and distributed learning scenarios.
- **Lightweight implementation**: Adaptive quantization introduces only a single learnable parameter $\alpha$, incurring no additional training overhead.

## Limitations & Future Work

- Experiments are primarily conducted on CIFAR and small-scale datasets; results on large-scale benchmarks such as ImageNet are absent.
- The bit-width decomposition $n = b/k$ required by APoT non-uniform quantization constrains $b$ to be an integer multiple of $k$, limiting flexibility.
- Integration with recent latent-space DD methods (e.g., SRe2L, RDED) has not been validated.
- Cross-domain experiments cover only a single tabular data task; generalization to NLP, audio, and other modalities remains to be verified.
- The greedy selection strategy in quantization-guided initialization computes gradient similarities, which may be costly at large dataset scales.

## Related Work & Insights

| Method | Compression Dimension | Applicable Modality | Quantization Scheme | End-to-End |
|--------|-----------------------|---------------------|---------------------|-----------|
| FreD | Frequency-domain $D$ | Image | None | Yes |
| AutoPalette | Color bit-width $b$ | Image | Palette quantization | Yes |
| IDC/HaBa/SPEED | Parameterized dimension | Image | None | Yes |
| **QuADD** | **Bit precision $b$** | **Image + Tabular** | **Adaptive non-uniform** | **Yes** |

The key distinctions between QuADD and FreD/AutoPalette are: (1) the quantization module is modality-agnostic and not confined to the color space; (2) adaptive non-uniform quantization allocates quantization granularity according to the data distribution; and (3) $M$ and $b$ are jointly optimized from a rate-distortion perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of jointly optimizing quantization and distillation is original; the rate-distortion analysis offers meaningful insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Coverage across multiple datasets, architectures, distillation methods, and domains is comprehensive, though large-scale data experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, mathematical derivations are complete, and figures are informative.
- Value: ⭐⭐⭐⭐ — Directly applicable to resource-constrained scenarios; the framework demonstrates strong generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HierAmp: Coarse-to-Fine Autoregressive Amplification for Generative Dataset Distillation](hieramp_coarse-to-fine_autoregressive_amplification_for_generative_dataset_disti.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](../../NeurIPS2025/model_compression/beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)
- [\[CVPR 2026\] Fixed Anchors Are Not Enough: Dynamic Retrieval and Persistent Homology for Dataset Distillation](fixed_anchors_are_not_enough_dynamic_retrieval_and_persistent_homology_for_datas.md)

</div>

<!-- RELATED:END -->
