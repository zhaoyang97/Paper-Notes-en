---
title: >-
  [Paper Note] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization
description: >-
  [ICML 2025][Signal & Communication][Position Encoding] By extending each dimension in RoPE from a single frequency to a Fourier series representation and clipping undertrained low-frequency components, this work achieves reliable periodic extension of the attention mechanism, thereby significantly enhancing the length generalization capability of LLMs.
tags:
  - "ICML 2025"
  - "Signal & Communication"
  - "Position Encoding"
  - "Length Generalization"
  - "Rotary Position Embedding"
  - "Fourier Series"
  - "Spectral Analysis"
date: 2026-05-08
content_hash: 4018b87548d59036
---

# Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization

**Conference**: ICML 2025  
**arXiv**: [2412.17739](https://arxiv.org/abs/2412.17739)  
**Code**: [GitHub](https://github.com/TsinghuaC3I/Fourier-Position-Embedding)  
**Area**: Signal Communication  
**Keywords**: Position Encoding, Length Generalization, Rotary Position Embedding, Fourier Series, Spectral Analysis

## TL;DR

By extending each dimension in RoPE from a single frequency to a Fourier series representation and clipping undertrained low-frequency components, this work achieves reliable periodic extension of the attention mechanism, thereby significantly enhancing the length generalization capability of LLMs.

## Background & Motivation

**Background**: Large language models are typically trained on fixed context lengths, but practical applications often require processing sequences that far exceed the training length. RoPE (Rotary Position Embedding) endows attention patterns with periodicity through implicit NUDFT, theoretically allowing for length extrapolation.

**Limitations of Prior Work**: The periodicity of RoPE is severely corrupted in actual networks. Even though RoPE itself generates a single-frequency signal, its spectrum undergoes leakage and distortion after passing through linear transformations and activation functions, directly leading to the failure of periodicity.

**Key Challenge**: The gap between the theoretical periodicity of RoPE and the actual spectral damage in networks: linear layers cause spectral leakage (multi-frequency aliasing), activation functions generate harmonic distortions, and low-frequency components are undertrained due to finite training lengths.

**Goal**: To diagnose the root cause of the periodic failure of RoPE from the perspective of frequency domain analysis, and design a new position encoding scheme that maintains stable periodic behavior of attention when exceeding the training length.

**Key Insight**: Treating the position dependency in attention as a discrete signal, utilizing Fourier analysis tools to quantify spectral damage, and designing a restoration scheme based on this diagnosis.

**Core Idea**: Expanding the single-frequency signal of each dimension in RoPE into a Fourier series, while clipping insufficiently trained low-frequency components, to fundamentally restore the periodicity of attention.

## Method

### Overall Architecture

FoPE replaces the position encoding part in standard RoPE. In RoPE, each dimension corresponds to a single rotation frequency $\omega_m$, and the signal at position $n$ is $e^{i\omega_m n}$. FoPE extends this to $h_m(n) = H_m(n)(e^{i\omega_m n} + \sum_\omega a_\omega e^{i\omega n})$, meaning each dimension contains the primary frequency combined with its Fourier series. Concurrently, components with frequencies lower than $2\pi/N$ ($N$ is the training length) are clipped to ensure that all retained frequencies have sufficient periodic coverage within the training range. In implementation, this is parameterized via a weight matrix $W^F \in \mathbb{R}^{D \times (M - M_0)}$, where $M_0$ is the number of clipped low-frequency components, and $D$ and the initialization standard deviation $\sigma$ are configurable.

### Key Designs

1. **Fourier Series Expansion**:

    - **Function**: Repair spectral damage caused by linear layers and activation functions.
    - **Mechanism**: Since linear layers alias single-frequency signals into multi-frequency signals (spectral leakage) and activation functions generate harmonic frequencies (spectral distortion), a multi-frequency representation is directly introduced at the position encoding level. This allows the model to learn the correct spectral combinations to recover periodicity. $h_m(n)$ is modeled as a linear combination of the primary frequency $\omega_m$ and other frequencies.
    - **Design Motivation**: Theoretical analysis in the paper proves that after the linear transformation $W$ acts on the RoPE signal, the output becomes a superposition of multiple frequencies $\sum_m w_m e^{i\omega_m n}$, which is equivalent to the rows of a NUDFT matrix; the activation function $\sigma(\cdot)$ further introduces harmonics $k\omega_m$. Therefore, a single-frequency representation cannot sustain the spectral info actually required by the network.

2. **Clip Floor to Zero**:

    - **Function**: Eliminating the negative impact of insufficiently trained low-frequency components on extrapolation.
    - **Mechanism**: For components with frequency $\omega_m < 2\pi/N$, they cannot cover even a single full period within the training length $N$, meaning the model cannot learn their correct behavior. The contributions of these components are directly set to zero.
    - **Design Motivation**: Undertrained low-frequency signals exhibit unpredictable behavior during extrapolation, which destroys the periodicity of attention patterns. After clipping, all retained frequencies undergo at least one full period within the training range, ensuring sufficient learning.

3. **Zero-Frequency Mean Maintenance**:

    - **Function**: Ensuring that the DC component (zero frequency) of the Fourier series does not affect the periodic behavior of attention.
    - **Mechanism**: Explicitly constraining the zero-frequency component in the Fourier expansion so that the mean of the attention scores remains stable across the position dimension, avoiding shifts as the sequence length increases.
    - **Design Motivation**: The zero-frequency component corresponds to the mean shift of the signal. If unconstrained, the attention scores may systematically drift during long-sequence extrapolation, leading to a collapse of the probability distribution.

### Loss & Training

As a replacement module for position encoding, FoPE does not introduce additional loss functions. The training strategy is consistent with standard language models, utilizing the autoregressive cross-entropy loss. Key hyperparameters include the dimension $D$ of the Fourier coefficient matrix and the initialization standard deviation $\sigma$. FoPE can be combined with length extrapolation techniques like YARN (FoPE + YARN) to further improve extrapolation performance. In fine-tuning scenarios, such as tuning from a short context (e.g., 2k) to a target length (e.g., 4k), FoPE exhibits generalization capabilities significantly superior to RoPE.

## Key Experimental Results

### Main Results

| Model/Method | Training Length | Passkey 2x Accuracy | GovReport PPL (4-8k) | MultiNews PPL (4-8k) | TREC Acc (4-8k) |
|:--|:--|:--|:--|:--|:--|
| RoPE | 512 | ~0% | baseline | baseline | baseline |
| FoPE | 512 | ~95% | **+1.15** | **+1.87** | **+14** |
| RoPE + YARN | 2k→4k | Good | Moderate | Moderate | Moderate |
| FoPE + YARN | 2k→4k | **Best** | **Best** | **Best** | **Best** |

*Based on SmolLM-1.7B fine-tuning experiments. FoPE outperforms RoPE across all evaluated length ranges, with the gap being particularly pronounced when exceeding the training length.*

### Ablation Study

| Component | Passkey Accuracy Change | Perplexity Change | Description |
|:--|:--|:--|:--|
| Full FoPE (FS + CF) | Best | Best | Both components synergize |
| Fourier Series (FS) Only | Significant Improvement | Small Improvement | Largest contribution, repairs spectral damage |
| Clip Floor (CF) Only | Small Improvement | In-domain Improvement | Mainly helps in-domain stability |
| No Components (RoPE) | Baseline | Baseline | Severe degradation during extrapolation |

### Key Findings

- Fourier series expansion is the primary source of performance gains, indicating that spectral damage is the chief cause of RoPE's extrapolation failure.
- Clip floor is more effective within the domain (inside the training length), producing synergistic effects when combined with the Fourier series.
- The combined effect of FoPE and YARN far exceeds RoPE + YARN, demonstrating that the improvements of FoPE are orthogonal and complementary.
- On the Passkey Retrieval task, the accuracy of RoPE plummets to near 0% at 2x training length, whereas FoPE still maintains around 95%.

## Highlights & Insights

- Deeply analyzes the mechanism of RoPE's failure from a signal processing perspective, elevating empirical observations to theoretical understanding: it is not that the design of RoPE is flawed, but rather that subsequent network layers corrupt its periodicity.
- The solution maps elegantly to the problem diagnosis: spectral leakage $\rightarrow$ multi-frequency representation, under-training $\rightarrow$ frequency clipping.
- The method is plug-and-play, does not alter the model architecture, introduces no extra inference overhead, and exhibits high practicality.
- Orthogonal and complementary to existing methods like YARN, allowing for combined usage.

## Limitations & Future Work

- The paper mainly validates on small-to-medium scale models (SmolLM-1.7B); performance on larger models (such as 70B+) remains to be confirmed.
- The Fourier coefficient matrix introduces additional parameters; although the quantity is small, evaluation is required in extremely resource-constrained scenarios.
- The frequency clipping threshold $2\pi/N$ is fixed, and whether there exists a superior adaptive clipping strategy is worth exploring.
- The paper focuses on textual LLMs, and its efficacy on other models utilizing RoPE, such as Vision Transformers, remains unverified.

## Related Work & Insights

- **vs RoPE**: FoPE is a natural extension of RoPE, preserving the advantages of rotary embedding (relative position awareness, theoretical basis for extrapolation) while repairing the spectral damage in practical networks via the Fourier series.
- **vs YARN/NTK-aware**: These methods improve extrapolation by adjusting frequency bases or interpolation strategies, but do not fundamentally resolve the spectral damage issue. The direction of improvement for FoPE is orthogonal to theirs.
- **vs ALiBi**: ALiBi achieves position awareness via linear biases, independent of the rotary mechanism, but its performance on long sequences is typically inferior to the RoPE series.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Diagnoses the failure mechanism of RoPE and designs a fix from the perspective of spectral analysis, offering a novel angle and theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thoroughly validated on Passkey Retrieval and multiple downstream tasks, with clear ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear theoretical derivations with a complete logical chain from problem diagnosis to solution design.
- **Value**: ⭐⭐⭐⭐ A plug-and-play improvement to position encoding, carrying direct value for long-sequence applications of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rectifying Magnitude Neglect in Linear Attention](../../ICCV2025/signal_comm/rectifying_magnitude_neglect_in_linear_attention.md)
- [\[AAAI 2026\] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection](../../AAAI2026/signal_comm/balancing_multimodal_domain_generalization_via_gradient_modulation_and_projectio.md)
- [\[CVPR 2025\] Breaking the Low-Rank Dilemma of Linear Attention](../../CVPR2025/signal_comm/breaking_the_low-rank_dilemma_of_linear_attention.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)

</div>

<!-- RELATED:END -->
