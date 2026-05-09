---
title: >-
  [Paper Note] Boosting Entropy with Bell Box Quantization
description: >-
  [ICLR 2026][Model Compression][quantization-aware pre-training] This paper proposes Bell Box Quantization (BBQ), the first quantization method that simultaneously satisfies information-theoretic optimality (ITO) and compute-efficiency. The core insight is the domain-agnosticity of learning—the output domain of a quantizer need not coincide with its input domain. BBQ performs ITO quantization in the input domain to maximize entropy, then maps to hardware-acceleratable data types in the output domain, achieving comprehensive improvements over QuEST and LSQ in 1–4 bit QAPT settings.
tags:
  - ICLR 2026
  - Model Compression
  - quantization-aware pre-training
  - information-theoretically optimal quantization
  - compute-efficient data types
  - entropy maximization
  - low-precision inference
date: 2026-05-08
content_hash: cdd1f00dcc3df72f
---

# Boosting Entropy with Bell Box Quantization

**Conference**: ICLR 2026
**arXiv**: [2603.01599](https://arxiv.org/abs/2603.01599)
**Code**: [https://github.com/1733116199/bbq](https://github.com/1733116199/bbq)
**Area**: Model Compression / Quantization
**Keywords**: quantization-aware pre-training, information-theoretically optimal quantization, compute-efficient data types, entropy maximization, low-precision inference

## TL;DR
This paper proposes Bell Box Quantization (BBQ), the first quantization method that simultaneously satisfies information-theoretic optimality (ITO) and compute-efficiency. The core insight is the domain-agnosticity of learning—the output domain of a quantizer need not coincide with its input domain. BBQ performs ITO quantization in the input domain to maximize entropy, then maps to hardware-acceleratable data types in the output domain, achieving comprehensive improvements over QuEST and LSQ in 1–4 bit QAPT settings.

## Background & Motivation

**Background**: Quantization is a key technique for deploying DNNs on edge devices. Quantization-aware pre-training (QAPT) trains models from scratch at low precision, avoiding the additional overhead of full-precision pre-training followed by PTQ/QAFT. However, low-precision models have limited information capacity, making it difficult to fit large-scale data.

**Limitations of Prior Work**: Existing QAPT methods (e.g., QuEST, LSQ) employ compute-efficient data types such as INT4, but these types **do not satisfy ITO**—quantized values are used with unequal frequency, wasting the limited learning capacity. Conversely, existing ITO methods (e.g., NF4/NormalFloat) maximize entropy but require dequantization to full precision before computation, making them infeasible on energy-constrained edge devices.

**Key Challenge**: There exists a trade-off between ITO and compute-efficiency—ITO quantization values do not reside in hardware-supported data types and thus cannot leverage low-precision matrix multiplication; integer/floating-point types that are compute-efficient are not ITO for Gaussian-distributed weights.

**Goal**: Can ITO quantization be achieved without sacrificing compute-efficiency, enabling models to maximally exploit their limited learning capacity?

**Key Insight**: **Learning is domain-agnostic**—DNNs can learn from rotated images, frequency-domain data, and latent embeddings; as long as information is preserved, projecting data into a different domain does not impair learning.

**Core Idea**: The quantizer performs ITO quantization in the input domain to retain maximum information, then outputs to a distinct, compute-efficient domain that enables direct use of low-precision matrix multiplication.

## Method

### Overall Architecture
BBQ quantization proceeds in three steps: (a) Hadamard transform + RMS normalization → converts weights/activations from arbitrary distributions to the standard normal $N(0,1)$; (b) probability integral transform (PIT) → maps the normal distribution to a uniform distribution $U(0,1)$ via the Gaussian CDF $\Phi$; (c) uniform quantization → uniform quantization of uniformly distributed data is ITO by definition. The dequantization formula is a simple linear scaling $\hat{x} = \frac{\gamma}{2^{b-1}} q$, which can be accelerated using low-precision matrix multiplication.

Quantization formula: $q = \lfloor 2^b \Phi(v) \rfloor - 2^{b-1} - z$, where $v = \text{HT}(x) / \sigma$
Dequantization formula: $\hat{x} = \frac{\gamma}{2^{b-1}} q$

### Key Designs

1. **Hadamard Transform + RMS Normalization (Step 1a)**:

    - Function: Converts weights/activations from an unknown distribution to the standard normal distribution.
    - Mechanism: The input $x$ is first subjected to a Hadamard transform over every $H$ elements along the channel dimension (known to Gaussianize data), then divided by the RMS $\sigma$ to obtain unit-variance $v \sim N(0,1)$.
    - Design Motivation: ITO quantization requires knowledge of the data distribution. By Gaussianizing the unknown distribution via HT, the subsequent PIT can be executed exactly.

2. **Probability Integral Transform PIT (Step 1b)**:

    - Function: Maps normally distributed data to a uniform distribution.
    - Mechanism: The standard Gaussian CDF $\Phi$ is applied to $v \sim N(0,1)$, yielding $\Phi(v) \sim U(0,1)$. $\Phi$ replaces the clip function used in QuEST/LSQ.
    - Design Motivation: The clip function is piecewise linear, non-differentiable, and cannot distribute Gaussian data uniformly across quantization bins. By contrast, $\Phi$ is infinitely differentiable, smoother, and ensures all quantization values appear with equal probability (ITO). At inference time, the quantization boundaries $\Phi^{-1}(i/2^b)$ can be precomputed, and bin assignment can be performed via binary search requiring only $b$ floating-point comparisons, incurring negligible overhead.

3. **Uniform Quantization + Domain Transformation (Step 1c + Dequant)**:

    - Function: Maps ITO quantized values to compute-efficient data types (INT4/MX FP4).
    - Mechanism: Uniform quantization is applied to $U(0,1)$ to obtain $\lfloor 2^b \Phi(v) \rfloor$; a bias is subtracted to produce a signed integer $q$, which can be stored directly as INT4 or MX FP4. Dequantization is the simple linear transform $\hat{x} = \frac{\gamma}{2^{b-1}} q$, so the matrix product $\hat{X}\hat{W}$ can be completed in the low-precision domain and then linearly rescaled.
    - Design Motivation: This is the crux of the domain transformation—information-optimal quantization is performed in the input domain, while the output domain is hardware-friendly integers/floats.

4. **Learnable Scale Factor $\gamma$ and Initialization**:

    - Function: Controls the magnitude of the dequantized output $\hat{x}$.
    - Mechanism: The scale factor is decomposed as $s = \gamma / 2^{b-1}$, where $\gamma$ is independent of the bit-width $b$ and initialized to $\zeta^* \sigma_0$ (where $\zeta^*$ is derived by minimizing the expected quantization error), ensuring that the magnitude of $\hat{x}$ matches that of $x$ at the first iteration and preventing gradient explosion/vanishing.
    - Design Motivation: Naively substituting $\Phi$ for clip without proper initialization of $\gamma$ causes training divergence (perplexity spikes from 35.58 to 138.3 in ablation experiments).

5. **BBQ-Fast Variant**:

    - At inference time, an exponential moving average $E_{1/\sigma}$ replaces the on-the-fly computation of $1/\sigma$, eliminating cross-thread communication overhead for activation RMS and achieving identical perplexity at higher inference throughput.

### Loss & Training
- The Straight-Through Estimator (STE) is applied to the floor operation; all other operations are differentiable.
- Gradient scaling (divided by $\sqrt{d}$) is applied to $\gamma$; weight decay is not applied to $\gamma$.
- Channel-wise quantization is used for weights; per-tensor quantization is used for activations.

## Key Experimental Results

### Main Results
QAPT experiments are conducted on the LLaMA architecture using the C4 dataset, comparing BBQ, QuEST, and LSQ:

| Model Params | Training Tokens | Precision (bit) | BBQ Entropy/PPL | QuEST Entropy/PPL | LSQ Entropy/PPL |
|---|---|---|---|---|---|
| 95M | 3B | 4-bit | 3.93 / 25.51 | 3.61 / 26.37 | 3.59 / 27.46 |
| 95M | 3B | 3-bit | 2.96 / 26.55 | 2.78 / 29.04 | 2.74 / 30.27 |
| 95M | 3B | 2-bit | 1.97 / 31.34 | 1.92 / 35.58 | 1.69 / 36.58 |
| 95M | 3B | 1-bit | 1.00 / 49.22 | 1.00 / 67.78 | — |
| 200M | 10B | 4-bit | 3.93 / 18.79 | 3.61 / 19.06 | 2.73 / 1778 |
| 200M | 10B | 2-bit | 1.98 / 23.08 | 1.93 / 25.46 | 1.63 / 78.19 |
| 300M | 20B | 4-bit | 3.93 / 16.10 | 3.61 / 16.26 | — |
| 300M | 20B | 2-bit | 1.98 / 19.75 | 1.93 / 21.53 | — |

BBQ consistently achieves higher entropy and lower perplexity across all bit-widths. The advantage grows as precision decreases (2-bit: −4+ PPL; 1-bit: −18+ PPL). LSQ diverges on larger models.

### Ablation Study
Ablation on 2-bit LLaMA-95M (3B tokens):

| Configuration | PPL | Entropy | Note |
|---|---|---|---|
| BBQ (full) | 31.34 | 1.97 | Best |
| w/o HT | 35.79 | 1.98 | PPL +4.45 |
| w/o RMS | 35.93 | 1.98 | PPL +4.59 |
| QuEST (no PIT) | 35.58 | 1.92 | Baseline |
| + PIT w/o $\gamma$ init | 138.3 | 1.92 | Diverges! |
| + PIT + $\gamma$ init | 31.46 | 1.98 | PPL −4.12 |
| + Learnable $\gamma$ | 31.34 | 1.97 | Further −0.12 |

### Key Findings
- Replacing clip with PIT ($\Phi$) is the most critical improvement, but must be paired with proper $\gamma$ initialization.
- BBQ achieves the theoretical maximum entropy (e.g., 1.97/2.0 bits at 2-bit), whereas QuEST has an empirical entropy ceiling of approximately 1.93 bits.
- Inference speed: on an RTX 5090, BBQ is 40% faster than FP16 and 48% faster than NF4 (NF4 is slower than FP16 during prefill).
- The overhead of the BBQ quantization kernel is only 1/10 of the time saved by low-precision matrix multiplication.

## Highlights & Insights
- **Domain-agnosticity insight**: This is the central "aha" moment of the paper—a quantizer need not perform quantization and dequantization in the same domain. This simple yet profound observation breaks the perceived trade-off between ITO and compute-efficiency.
- **Replacing clip with $\Phi$**: The Gaussian CDF simultaneously serves as a smooth activation function (analogous to GELU vs. ReLU) and an information-optimal binning function, achieving two goals at once.
- **Elegant inference implementation**: At inference time, $\Phi$ + floor is jointly realized as a binary search over precomputed boundaries, requiring only $b$ comparisons and adding virtually no latency when fused into the quantization kernel.
- **Transferability of the domain transformation trick**: Whenever a task's optimization objective is domain-agnostic (e.g., neural network training), one may consider performing transformations in the information-preserving optimal domain while executing computation in the compute-efficient domain.

## Limitations & Future Work
- **Applicable only to QAPT**: Because $x$ and $\hat{x}$ reside in different domains, BBQ cannot guarantee a bounded quantization error $\|x - \hat{x}\|$, and is therefore not suitable for PTQ or short-duration QAFT.
- **Dependence on the Gaussianity assumption of HT**: While HT$(x)$ closely approximates a Gaussian at the start of training, deviations may arise in later stages, causing PIT to be only approximately ITO. The authors suggest replacing $\Phi$ with a more accurate smooth empirical CDF.
- **Evaluated only on language models**: Experiments on vision models (ViT, ConvNet) and multimodal models are absent.
- **Potential extension**: Generalizing the domain transformation approach to QAFT may require a short "domain adaptation" phase to allow the model to adjust to the new domain.

## Related Work & Insights
- **vs. QuEST**: QuEST also employs HT for Gaussianization but uses clip + uniform quantization (non-ITO for Gaussian data), yielding an empirical entropy ceiling of ~1.93 bits/2 bits. BBQ eliminates this bottleneck via PIT.
- **vs. NF4/NormalFloat**: NF4 is ITO but requires dequantization to full precision for computation, making prefill slower than FP16. BBQ is the first method to be simultaneously ITO and compute-efficient.
- **vs. N2UQ**: N2UQ also performs domain transformation but assumes a uniform weight distribution and acts on weights only. BBQ applies ITO to both weights and activations without assuming a specific distributional form.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The domain-agnosticity insight is simple yet profound; combining ITO with compute-efficiency is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-model, multi-precision comparisons with inference profiling, though validation on vision models is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is derived with clarity; the exposition flows seamlessly from information theory to implementation.
- Value: ⭐⭐⭐⭐ Directly advances low-bit QAPT; the exploration of 1-bit models is particularly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/model_compression/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)
- [\[NeurIPS 2025\] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs](../../NeurIPS2025/model_compression/matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)
- [\[ICCV 2025\] Context Guided Transformer Entropy Modeling for Video Compression](../../ICCV2025/model_compression/context_guided_transformer_entropy_modeling_for_video_compression.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

</div>

<!-- RELATED:END -->
