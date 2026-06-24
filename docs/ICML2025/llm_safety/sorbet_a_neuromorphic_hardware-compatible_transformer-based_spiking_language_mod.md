---
title: >-
  [Paper Note] Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model
description: >-
  [ICML 2025][LLM Safety][spiking neural network] Proposes Sorbet, the first fully neuromorphic hardware-compatible Transformer-based spiking language model. By replacing traditional softmax and Layer Normalization with two key innovations—bit-shift-based PTsoftmax and Bit Shifting PowerNorm (BSPN)—it achieves performance comparable to BERT on the GLUE benchmark while reducing energy consumption by 27.16x.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "spiking neural network"
  - "neuromorphic hardware"
  - "transformer"
  - "energy efficiency"
  - "binary weight"
date: 2026-05-08
content_hash: a5433c0c6cb3c5ed
---

# Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model

**Conference**: ICML 2025  
**arXiv**: [2409.15298](https://arxiv.org/abs/2409.15298)  
**Code**: [github.com/Kaiwen-Tang/Sorbet](https://github.com/Kaiwen-Tang/Sorbet)  
**Area**: AI Safety / Model Efficiency / Spiking Neural Networks  
**Keywords**: spiking neural network, neuromorphic hardware, transformer, energy efficiency, binary weight

## TL;DR

Proposes Sorbet, the first fully neuromorphic hardware-compatible Transformer-based spiking language model. By replacing traditional softmax and Layer Normalization with two key innovations—bit-shift-based PTsoftmax and Bit Shifting PowerNorm (BSPN)—it achieves performance comparable to BERT on the GLUE benchmark while reducing energy consumption by 27.16x.

## Background & Motivation

- Running LLMs on edge devices raises privacy and energy efficiency requirements. Spiking Neural Networks (SNNs) are highly promising due to their event-driven, multiplication-free characteristics.
- Although existing Transformer-based SNNs (such as SpikeLM and SpikingBERT) replace matrix multiplication, they still rely on **softmax** and **Layer Normalization (LN)**—operations involving exponential calculation, division, and square root, which **cannot be implemented on neuromorphic hardware**.
- SpikFormer bypasses these issues by using convolution and Batch Normalization, but it is only suitable for vision tasks.
- **Key Challenge**: How to replace all operations in the Transformer with bit shifts and additions while maintaining competitive NLP performance?

## Method

### Overall Architecture

Sorbet is based on the BERT architecture and achieves full SNN compatibility through three steps:
1. Replacing softmax with **PTsoftmax**
2. Replacing Layer Normalization with **BSPN**
3. Quantizing all weights to 1-bit and activations to 4-bits, encoded via spiking neurons.

### Bit Shifting PowerNorm (BSPN)

Layer Normalization requires calculating the mean and variance (involving division and square roots). The design mechanism of BSPN is as follows:

**Step 1: Group Scaling**
- Compute the L1 norm of the input: $\mathcal{S} = \frac{1}{n}\sum_{i=1}^{n}|X_i|$
- Approximate the L1 norm to the nearest power of two: $k = \lceil\log_2(\mathcal{S})\rceil$
- **Perform division via right-shift operations**: $X \leftarrow X \gg k$

**Step 2: PowerNorm Normalization**
- Utilize the running variance $\psi^2$ (exponential moving average), directly normalizing with the stored $\psi$ during inference.
- The scaling factor $\gamma/\psi$ can be further quantized to powers of two.

**Theoretical Guarantees**:
- Theorem 4.2: BSPN maintains bounded gradients, $\|\partial\mathcal{L}_{BSPN}/\partial \tilde{X}_{:,i}\| \leq C$
- Lemma 4.3: $\Phi(X)$ is a 1-Lipschitz mapping.
- Lemma 4.4: The Lipschitz constant of BSPN is no larger than that of PowerNorm, and is typically smaller than BN.

### Power-of-Two Softmax (PTsoftmax)

Standard softmax involves exponentiation and division. The approximation mechanism of PTsoftmax is:

$$\text{PTsoftmax}(z_i) = \frac{2^{\lceil z_i \rceil}}{\sum_j 2^{\lceil z_j \rceil}} \approx 2^{\lceil z_i \rceil - k}$$

where $k = \lceil \log_2(\sum_j 2^{\lceil z_j \rceil}) \rceil$. Core operations:
- $2^{z_i}$ is implemented via **left-shifts**
- Division by $2^k$ is implemented via **right-shifts**
- Completely avoids exponentiation and division operations.

**Theoretical Guarantee** (Lemma 4.5): $\frac{1}{2\sqrt{2}} F_2(x_i) \leq \text{PTsoftmax}(x_i) \leq 2\sqrt{2} F_2(x_i)$, showing the approximation error is bounded within a constant factor.

### Loss & Training

A multi-step distillation strategy is adopted (Algorithm 3):
1. Quantize BERT to 1-bit weights and 4-bit activations.
2. Replace softmax with PTsoftmax $\rightarrow$ Perform distillation.
3. Replace LN with BSPN $\rightarrow$ Perform distillation.
4. Convert to SNN (via spiking neurons).

The loss function combines logits distillation (KL divergence) and intermediate activation distillation:
$$L = L_{\text{logits}} + L_{\text{reps}} = \text{KL}(p, q) + \sum_i \|r_i^s - r_i^t\|^2$$

## Key Experimental Results

### GLUE Benchmark

| Model | Size | QQP | MNLI-m | SST-2 | QNLI | RTE | MRPC | STS-B |
|------|:----:|:---:|:-----:|:----:|:----:|:---:|:----:|:-----:|
| BERT_base | 418M | 91.3 | 84.7 | 93.3 | 91.7 | 72.6 | 88.2 | 89.4 |
| BiT (1-bit) | 13.4M | 82.9 | 77.1 | 87.7 | 85.7 | 58.8 | 79.7 | 71.1 |
| SpikeLM | * | 87.9 | 76.0 | 86.5 | 84.9 | 65.3 | 78.7 | 84.3 |
| **Sorbet** | **13.4M** | **86.5** | **77.3** | **90.4** | **86.1** | **60.3** | **79.9** | **78.1** |

Sorbet achieves SOTA SNN results on four tasks, performing comparably to BiT, an ANN quantized model of the same size.

### Energy Efficiency Analysis

| Model | FP32 Energy (mJ) | FP16 | 1-Bit |
|------|:-----------:|:----:|:-----:|
| BERT | 51.41 | 15.21 | - |
| SpikeLM | 3.98 | 1.77 | - |
| **Sorbet** | - | - | **0.65** |

- **Saves 27.16x energy compared to BERT**, and 3.16x compared to SpikeLM.
- PTsoftmax is 27.62x more energy-efficient than standard softmax, while BSPN is 12.4x more energy-efficient than LN.
- The average spike firing rate is only 0.13-0.15, indicating a large number of neurons remain silent.

### Ablation Study

| Configuration | SST-2 Accuracy | Difference from Baseline ($\delta$) |
|------|:--------:|:---------:|
| Softmax + LN (4-bit) | 91.5 | - |
| PTsoftmax + LN | 90.8 | -0.7 |
| Softmax + BSPN | 91.2 | -0.3 |
| PTsoftmax + BSPN | 90.9 | -0.6 |

The performance degradation introduced individually by PTsoftmax and BSPN is minimal (<1%), with the primary accuracy drop originating from weight quantization and the spike generation process.

## Highlights & Insights

1. **First fully NLP-applicable neuromorphic-compatible model**: Solves the "last mile" problem of SNNs in the NLP field—namely, the substitution of softmax and LN.
2. **Elegant design of bit-shift approximations**: Approximates continuous operations using powers of two; theoretically bounded (constant factor error) while yielding minimal performance loss in practice.
3. **Solid theoretical foundation for BSPN**: Proves bounded gradients and non-increasing Lipschitz constants, ensuring training stability.
4. **Practical multi-step distillation strategy**: Progressively replaces components and applies distillation, avoiding catastrophic performance collapse caused by one-time replacements.

## Limitations & Future Work

- Lacks deployment and validation on actual neuromorphic chips (such as Intel Loihi), relying purely on evaluations via the Lava framework and Verilog simulations.
- The model scale is limited to the BERT-base level (13.4M after quantization), leaving larger-scale models unexplored.
- PTsoftmax does not strictly satisfy the normalization condition (the sum of probabilities is not strictly 1). Although the experimental impact is marginal, it is theoretically imperfect.
- A significant performance gap remains compared to current mainstream Large Language Models (such as DeepSeek and Llama), making it more suitable for edge inference scenarios.

## Related Work & Insights

- **Transformer SNNs**: Spikformer (Zhou et al., 2024) and Spike-driven Transformer (Yao et al., 2024) for computer vision; SpikeBERT (Lv et al., 2023) and SpikeGPT (Zhu et al.) for NLP, though they still rely on LN/softmax.
- **Quantized BERT**: BinaryBERT (Bai et al., 2021) and BiT (Liu et al., 2022) achieve binarization but retain complex operations.
- **Simplified Transformers**: I-BERT (Kim et al., 2021) approximates activation functions with integers but still requires multiplication and division operations.

## Rating

⭐⭐⭐⭐ — Resolves a well-defined and critical engineering and theoretical problem with elegant and simple designs backed by sufficient theoretical guarantees. However, it lacks physical hardware validation and exploration into larger model scales. It represents a significant milestone in the SNN+NLP direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[NeurIPS 2025\] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](../../NeurIPS2025/llm_safety/trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)
- [\[ICML 2025\] SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)
- [\[ICML 2025\] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)

</div>

<!-- RELATED:END -->
