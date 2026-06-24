---
title: >-
  [Paper Note] Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models
description: >-
  [ACL 2025][Model Compression][Ternary Quantization] This paper systematically studies the scaling laws of Ternary Language Models (TriLM) and finds that TriLM benefits significantly more from increasing training data than from scaling parameter size. Guided by this insight, the Spectra-1.1 model family (1B/2B/3B) is trained on 1.2T tokens. The authors also propose 1.6-bit and 2-bit weight packing schemes along with the TriRun GPU kernel, achieving up to an 8x inference accele…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Ternary Quantization"
  - "Scaling Laws"
  - "Inference Acceleration"
  - "Weight Packing"
  - "GPU Kernels"
date: 2026-05-08
content_hash: 83cd364a0194a0ed
---

# Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.23025](https://arxiv.org/abs/2506.23025)  
**Code**: To be open-sourced soon (Spectra-1.1 models + TriRun inference kernel, MIT license)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Ternary Quantization, Scaling Laws, Inference Acceleration, Weight Packing, GPU Kernels

## TL;DR

This paper systematically studies the scaling laws of Ternary Language Models (TriLM) and finds that TriLM benefits significantly more from increasing training data than from scaling parameter size. Guided by this insight, the Spectra-1.1 model family (1B/2B/3B) is trained on 1.2T tokens. The authors also propose 1.6-bit and 2-bit weight packing schemes along with the TriRun GPU kernel, achieving up to an 8x inference acceleration.

## Background & Motivation

**Background**: The inference efficiency of LLMs is constrained by memory bandwidth bottlenecks, as GPU computing power scales far quicker than memory capacity and bandwidth. Post-Training Quantization (PTQ) is the prevailing inference acceleration solution but usually supports at best 4-bit precision; lower bit-widths lead to severe performance degradation. Recent Quantization-Aware Training (QAT) methods show that ternary-weight (-1, 0, 1) models can approach the performance of full-precision models at larger parameter scales, exhibiting higher bit efficiency.

**Limitations of Prior Work**: Three critical gaps exist in the field of Ternary Language Models (TriLM). (1) Absence of Scaling Law analysis: previous work only verified the effects of parameter scaling without systematically studying the impact of training token count on TriLM performance, failing to guide optimal compute allocation. (2) Lack of inference acceleration: existing research on efficient inference focuses almost entirely on 4-bit quantization, leaving sub-4-bit (especially ternary) dedicated inference kernels and weight storage schemes virtually non-existent. (3) Dearth of strong open-source models: no extensively pre-trained, strong open-source TriLM model families are available for community research, constraining post-training methodology and application exploration.

**Key Challenge**: Ternary quantization offers extreme memory compression ratios (16x compared to FP16), yet there is currently neither theoretical guidance on how to allocate training compute (parameters vs. training tokens) nor engineering infrastructure to realize actual speedup during real-world inference for ternary models.

**Goal**: Address three sub-problems: (1) Establish the scaling laws of TriLM and train a strong open-source model family; (2) Design an efficient storage scheme for ternary weights (approaching the information-theoretic optimal limit of 1.585 bits/weight); (3) Develop a GPU inference kernel to achieve end-to-end acceleration.

**Key Insight**: The authors first discover a key characteristic of TriLM via systematic scaling law experiments: $\hat{L}(N,D) \approx 2.19 + 4.73/N^{0.32} + 5.18/D^{0.81}$, where the exponent for the data term (0.81) is significantly larger than that of the parameter term (0.32). This implies that increasing training data is far more effective than increasing parameter count. Grounded in this insight, they fix the parameter sizes and scale the training data from 300B to 1.2T tokens. On the inference side, exploiting the unique structure where ternary weights have only three possible values, they design highly efficient packing/unpacking schemes that bypass floating-point multiplications.

**Core Idea**: The scaling law exponent of TriLM reveals that "feeding more data" is far more effective than "making models larger." Consequently, the Spectra-1.1 model family is trained on 1.2T tokens, which, together with the dedicated TriRun GPU kernel, achieves up to a 5x end-to-end inference acceleration.

## Method

### Overall Architecture

This work is divided into two relatively independent parts. On the training side: employing Quantization-Aware Training (QAT) on a decoder-only Transformer architecture, where linear layer weights are constrained to $\{-1, 0, 1\}$ plus a shared floating-point scaling factor. Online ternarization is performed during the forward pass, while latent floating-point weights are updated during backward propagation. The authors train 20 models across 5 parameter scales $\times$ 4 data scales to fit the scaling laws, followed by training three final models (1B/2B/3B) on 1.2T tokens using the optimal strategy. On the inference side: proposing two weight packing schemes, 2-bit (TQ2) and 1.6-bit (TQ1), integrated into llama.cpp on CPUs, and developing the TriRun mixed-precision kernel (FP16 $\times$ INT2) on GPUs, leveraging asynchronous memory copies and Tensor Cores to accelerate matrix multiplication.

### Key Designs

1. **TriLM Scaling Law Analysis**:

    - Function: Quantify the relationship between the validation loss of ternary models, parameter size $N$, and training token count $D$ to guide compute resource allocation.
    - Mechanism: Train 20 TriLMs in a grid spanning 99M to 1.1B parameters and 20B to 150B tokens to fit a parameterized scaling law of the Chinchilla form, $\hat{L}(N,D) = E + A/N^\alpha + B/D^\beta$. The fitting results yield $\alpha=0.32$ and $\beta=0.81$, indicating that the decay exponent of the data term is 2.5 times that of the parameter term. Contrasting this with the scaling laws of full-precision models, where $\alpha=0.56, \beta=0.53$ (almost symmetric), TriLM clearly skews towards a "data-first" strategy.
    - Design Motivation: This discovery directly guides the training strategy for Spectra-1.1. Instead of pursuing larger parameter sizes, the models are trained thoroughly with 1.2T tokens at the 1B-3B scale. This yields substantial improvements across benchmarks like MMLU compared to Spectra 1.0 (300B tokens).

2. **Efficient Weight Packing Schemes (TQ2 and TQ1)**:

    - Function: Compress ternary weights from a naive 2-bit representation to a 1.6-bit representation, which is closer to the information-theoretic limit.
    - Mechanism: The TQ2 (2-bit) scheme maps each ternary weight $d_i \in \{-1,0,1\}$ to $d_i' = d_i + 1 \in \{0,1,2\}$, storing every 4 trits in 8 bits. It utilizes blocks of 256 elements with an FP16 scaling factor, yielding an effective bit-width of 2.0625 bits/weight. The TQ1 (1.6-bit) scheme exploits the numerical approximation $3^5 = 243 < 256 = 2^8$ to encode 5 trits into one 8-bit integer, achieving an effective bit-width of 1.6 bits/weight, extremely close to the theoretical optimum of $\log_2(3) \approx 1.585$. During decoding, multiplication is used to approximate division and modulo operations to suit SIMD parallelism.
    - Design Motivation: TQ2 is faster (decoding requires only shifts and masks), whereas TQ1 is more memory-efficient (saving approximately 20% memory). TQ1 is preferred under memory-constrained scenarios, while TQ2 is selected for compute-limited environments.

3. **TriRun GPU Inference Kernel**:

    - Function: Implement FP16 $\times$ INT2 mixed-precision matrix multiplication for efficient ternary model inference on GPUs.
    - Mechanism: Based on the 2-bit packing scheme, CUDA asynchronous memory copy (`cp.async`) is used to load FP16 inputs into shared memory while overlapping computation. INT2 weights are loaded via asynchronous copies with cache hints to minimize L2 cache thrashing. The unpacked FP16 weight fragments and input fragments undergo tile matrix multiplication via Tensor Core `mma` instructions. Intermediate results accumulated in FP32 registers preserve precision before being converted back to FP16 and written to global memory. A double-buffered pipeline and hierarchical reduction are also employed.
    - Design Motivation: The 2-bit representation of ternary weights scales down each weight to just 0.25 bytes. Given an L40 GPU's FLOPs/byte ratio of approximately 105, computation becomes the bottleneck rather than memory when the batch size exceeds ~13. TriRun is specifically optimized for these higher-batch scenarios.

### Loss & Training

The standard cross-entropy loss for causal language modeling is utilized during training. The Quantization-Aware Training strategy maintains latent floating-point weights, and applies sign-extraction and scaling factors to ternarize them during the forward pass. Using the AdamW optimizer, near-linear multi-GPU scaling is achieved on an AMD MI250X cluster (up to 2,048 GPUs).

## Key Experimental Results

### Main Results

Comparison of Spectra-1.1 (TriLM with 1.6-bit effective weight) vs. LLaMA-1 7B (FP16) benchmarks:

| Benchmark | Spectra-1.1 1B | Spectra-1.1 2B | Spectra-1.1 3B | LLaMA-1 7B |
|------|---------------|---------------|---------------|------------|
| ARC Challenge (acc_norm) | 36.43 | 39.69 | 42.58 | 44.80 |
| ARC Easy (acc_norm) | 62.54 | 67.42 | 71.93 | 72.81 |
| HellaSwag (acc_norm) | 56.61 | 61.37 | 66.28 | 76.21 |
| BoolQ (acc) | 62.57 | 56.70 | 66.15 | 75.11 |
| LAMBADA (acc) | 47.31 | 48.85 | 54.22 | 73.53 |

TriRun GPU inference speedup (compared to the PyTorch FP16 baseline):

| Scenario | 70B Model | 405B Model |
|------|---------|----------|
| Time to First Token (64 inputs) | 4.7× | — |
| Time per Output Token | 4.9× | — |
| Layer-level Speedup (batch 16-32) | ~5× | **7.98×** |
| End-to-end Generation Speedup | **4.9×** | — |

### Ablation Study

| Configuration | Description |
|------|------|
| Spectra 1.0 (300B tokens) vs 1.1 (1.2T) | Continuous MMLU improvements, validating the efficacy of data scaling |
| $\alpha=0.32$ (TriLM) vs $\alpha=0.56$ (Float) | Parameter-scaling gains for TriLM are only 57% of those for Float |
| $\beta=0.81$ (TriLM) vs $\beta=0.53$ (Float) | Data-scaling gains for TriLM are 153% of those for Float |
| TQ2 (2-bit) vs TQ1 (1.6-bit) | TQ2 is faster, while TQ1 is more memory-efficient |

### Key Findings

- **"Data-First" Property of TriLM**: The data exponent $\beta=0.81$ in the scaling law is substantially larger than the parameter exponent $\alpha=0.32$, contrasting sharply with the symmetric property of full-precision models ($\alpha \approx \beta \approx 0.5$). Intuitively, because of the limited expressive capacity of ternary weights, scaling parameters only expands model capacity marginally. In contrast, training with more data allows the constrained parameters to acquire more efficient representations.
- **3B TriLM vs 7B Float**: Spectra-1.1 3B approaches LLaMA-1 7B performance on tasks like ARC (42.58 vs. 44.80), but its memory footprint is only about 1/15th (ternary 3B $\approx$ 0.6GB vs. FP16 7B $\approx$ 14GB).
- **Inference Speedup Dominates in Large Models + Large Batches**: TriRun’s acceleration factor scales up as model and batch size increase (hitting 8x at 405B, batch size 32) since memory bandwidth bottlenecks become far more prominent in these regimes.
- **Single-GPU Execution of 70B TriLM**: TriRun enables a 70B ternary model to run on a single L40S GPU, whereas the FP16 counterpart requires 4 GPUs.

## Highlights & Insights

- **Counter-Intuitive Yet Theoretically Grounded Scaling Laws of TriLM**: The extremely low expressiveness of ternary weights causes marginal gains of parameter scaling to diminish rapidly. Conversely, data scaling, through more thorough training, empowers each ternary weight parameter to hold more information. This finding provides crucial guidance for the training strategies of all ultra-low bit-width models.
- **Ingenious Design of 1.6-bit Packing**: Exploiting the number-theoretic approximation $3^5 \approx 2^8$, 5 trits are losslessly compressed into 1 byte. Replacing division and modulo operations during decoding with iterative multiplication ensures compatibility with SIMD, balancing both theoretical optimality and engineering practicality.
- **Practical Deployment Value of TriRun**: Running a 70B model on a single GPU with 5x acceleration directly impacts the deployment of ultra-low bit-width models. This represents the first GPU inference kernel system explicitly built for sub-4-bit models.

## Limitations & Future Work

- **Omission of Bit-width $b$ in the Scaling Law**: The current scaling laws fit TriLM and FloatLM independently, without establishing a unified three-variable scaling law modeling $(N, D, b)$.
- **Limited Model Scales**: Spectra-1.1 only trains models at or below 3B. There remains a significant gap compared to 7B/13B full-precision models, necessitating more compute resources to validate performance at larger scales.
- **TriRun Supports Only 2-bit**: The GPU kernel for the more memory-efficient 1.6-bit packing scheme (TQ1) has not yet been developed due to the higher complexity of its unpacking operations.
- **Absence of Generative NLP Tasks in Downstream Evaluations**: All evaluated benchmarks are multiple-choice or classification tasks, lacking an assessment of text generation quality.

## Related Work & Insights

- **vs. Spectra 1.0 (Kaushal et al. 2024)**: Spectra 1.0 was trained on 300B tokens with no scaling law analysis; this work scales training to 1.2T tokens and establishes the first scaling laws specifically for TriLM.
- **vs. GPTQ/AWQ (PTQ Methods)**: Post-Training Quantization methods degrade severely below 4-bit, whereas TriLM trained via QAT retains viable performance at 1.58-bit effective precision.
- **vs. BitNet (Wang et al. 2023)**: BitNet proved the viability of 1-bit/ternary models but provided no concrete inference acceleration system; this work’s TriRun fills this engineering void.

## Rating
- Novelty: ⭐⭐⭐⭐ Establishing the scaling laws of TriLM and discovering its "data-first" property, alongside the 1.6-bit packing scheme and the TriRun GPU kernel, constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ The evaluation covers scaling law fitting, diverse benchmark evaluations, CPU/GPU inference acceleration, and validation across multiple hardware platforms.
- Writing Quality: ⭐⭐⭐⭐ Highly structured, mathematically rigorous in its scaling law derivations, and mathematically complete in its packing scheme analysis.
- Value: ⭐⭐⭐⭐⭐ Full-stack contribution (scaling laws + models + inference kernel + open source) that systematically propels the research and deployment of ultra-low bit-width models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](../../ICML2026/model_compression/model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2025\] An Efficient Matrix Multiplication Algorithm for Accelerating Inference in Binary and Ternary Neural Networks](../../ICML2025/model_compression/an_efficient_matrix_multiplication_algorithm_for_accelerating_inference_in_binar.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[NeurIPS 2025\] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization](../../NeurIPS2025/model_compression/paretoq_improving_scaling_laws_in_extremely_low-bit_llm_quantization.md)

</div>

<!-- RELATED:END -->
