---
title: >-
  [Paper Note] Towards Lossless Implicit Neural Representation via Bit Plane Decomposition
description: >-
  [CVPR 2025][Audio & Speech][Implicit Neural Representation] Discovers that the model capacity upper bound of implicit neural representation (INR) grows exponentially with bit precision ($\mathcal{P}(f_\theta) \propto 2^n$), and proposes bit-plane decomposition—decomposing an n-bit signal into n independent 1-bit planes to train individual INRs, achieving lossless (BER=0) implicit neural representation of 16-bit images for the first time.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Implicit Neural Representation"
  - "Lossless Representation"
  - "Bit-plane Decomposition"
  - "Bit Bias"
  - "High-precision Signals"
date: 2026-05-08
content_hash: 496c36c806f42821
---

# Towards Lossless Implicit Neural Representation via Bit Plane Decomposition

**Conference**: CVPR 2025  
**arXiv**: [2502.21001](https://arxiv.org/abs/2502.21001)  
**Code**: [https://github.com/WooKyoungHan/LosslessINR](https://github.com/WooKyoungHan/LosslessINR)  
**Area**: Audio & Speech / Neural Representation  
**Keywords**: Implicit Neural Representation, Lossless Representation, Bit-plane Decomposition, Bit Bias, High-precision Signals

## TL;DR

Discovers that the model capacity upper bound of implicit neural representation (INR) grows exponentially with bit precision ($\mathcal{P}(f_\theta) \propto 2^n$), and proposes bit-plane decomposition—decomposing an n-bit signal into n independent 1-bit planes to train individual INRs, achieving lossless (BER=0) implicit neural representation of 16-bit images for the first time.

## Background & Motivation

### Background

**Background**: INRs (such as SIREN, FINER) map coordinates to signal values (image pixels/audio amplitudes) using neural networks to achieve continuous signal representation. However, existing methods already exhibit a noticeable bit error rate (BER) at 8-bit precision, and lossless representation of 16-bit high-precision signals remains unachievable.

**Limitations of Prior Work**: INRs exhibit a "bit bias" phenomenon—most significant bits (MSBs) are learned quickly and accurately, while least significant bits (LSBs) are learned slowly and imprecisely. This is analogous to spectral bias (low frequencies learned first, high frequencies harder to learn), but occurs in the bit dimension. Consequently, the model's BER increases exponentially with precision.

**Key Challenge**: A single network simultaneously models information from 16 bit-planes—the value of 1 in the most significant bit equals $2^{15}$ in the least significant bit. This massive scale discrepancy causes network capacity to be dominated by the MSBs.

**Key Insight**: Decompose an n-bit signal into n independent 1-bit planes, and train an independent INR for each plane. The theoretical upper bound of BER for a 1-bit signal is only $2^1$ instead of $2^n$.

**Core Idea**: Signal $\rightarrow$ Bit-plane decomposition $\rightarrow$ Independent INR per plane = Breaking the precision bottleneck to achieve lossless representation.

### Mechanism

**Goal**: ### Key Designs

1. **Bit-plane Decomposition**:

    - Function: Decomposing a high-precision signal into multiple low-precision sub-signals
    - Mechanism: An n-bit integer value can be decomposed into n binary planes $\{b_0, b_1, ..., b_{n-1}\}$, where each plane is a spatial function of 0/1 values.


## Method

### Key Designs

1. **Bit-plane Decomposition**:

    - Function: Decomposing a high-precision signal into multiple low-precision sub-signals
    - Mechanism: An n-bit integer value can be decomposed into n binary planes $\{b_0, b_1, ..., b_{n-1}\}$, where each plane is a spatial function of 0/1 values. An independent small INR is trained for each plane
    - Design Motivation: Theorem 1 proves that the model capacity upper bound is $\mathcal{P} \propto 2^n$. After decomposition, the complexity of each sub-problem is only $2^1$

2. **Discovery of the Bit Bias Phenomenon**:

    - Function: Revealing an overlooked source of bias in INR learning
    - Mechanism: Analogous to spectral bias (low frequencies learned first)—MSB is learned first, and LSB is learned later. This is because the error of MSB contributes $2^{2(n-1)}$ times more to the loss function than that of LSB, naturally biasing the gradient toward MSB
    - Design Motivation: Understanding the cause of the bias allows for designing the correct solution—decomposition rather than "scaling up the model"

### Loss & Training

MSE loss. Each bit-plane uses the target error bound $\epsilon(n) = \frac{1}{2(2^n-1)}$ as a precision reference. Any activation function, such as SIREN/FINER/Gaussian, can be used.

## Key Experimental Results

| Method | Precision | PSNR | BER |
|------|------|------|-----|
| SIREN (Original) | 16-bit | 47.04% | 0.128 |
| FINER (Original) | 16-bit | 40.64% | 0.164 |
| **Ours (Decomposition)** | 16-bit | **∞** | **0.000** |

### Ablation Study

- The bit bias where MSB converges first and LSB converges later consistently exists across all activation functions (ReLU/SIREN/FINER).
- After decomposition, each bit-plane converges rapidly (~5000 iterations), keeping the total computational cost manageable.
- Applications: Lossless compression, bit-depth expansion, ternary quantized INRs.

### Key Findings
- **Bit bias is the root cause of the accuracy bottleneck in INRs**—it is not due to insufficient model capacity, but rather learning being biased towards high-order bits.
- **Lossless representation can be achieved both theoretically and practically after decomposition**—achieving BER=0 for the first time.
- **16 small networks are more effective than one large network**—because the complexity of each sub-problem is exponentially reduced.

## Highlights & Insights
- **Discovery of bit bias**—a novel bias phenomenon analogous to spectral bias, providing insights for the entire INR field.
- **An extremely simple solution to a fundamental problem**—no changes to network architecture, activation functions, or training strategies; only changing the data representation.
- **A milestone for lossless INR**—lossless 16-bit representation has never been achieved prior to this work.

## Limitations & Future Work
- Storage and inference overhead of 16 independent networks.
- The method is specific to structured bit decomposition; other decompositions (e.g., frequency decomposition) remain unexplored.
- Mainly validated on images/audio; 3D scenes are not covered.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of bit bias and the decomposition solution are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple activation functions / multiple precisions / multiple applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a fundamental solution to the precision issue in INRs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[ACL 2026\] Privacy-preserving Prosody Representation Learning](../../ACL2026/audio_speech/privacy-preserving_prosody_representation_learning.md)
- [\[ICML 2026\] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](../../ICML2026/audio_speech/towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)
- [\[NeurIPS 2025\] Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost](../../NeurIPS2025/audio_speech/slimmable_nam_neural_amp_models_with_adjustable_runtime_computational_cost.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)

</div>

<!-- RELATED:END -->
