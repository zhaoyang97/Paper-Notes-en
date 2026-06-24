---
title: >-
  [Paper Note] RADIO: Rate-Distortion Optimization for Large Language Model Compression
description: >-
  [ICML 2025][Model Compression][Rate-Distortion Theory] Starting from the rate-distortion theory in information theory, RADIO establishes a theoretical foundation for LLM quantization and proposes a concise quantization technique based on rate-distortion optimization. This technique can be scaled to models with hundreds of billions of parameters and allows users to flexibly specify the target model size or accuracy for post-training compression.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Rate-Distortion Theory"
  - "Post-Training Quantization"
  - "Rate-Distortion Optimization"
  - "LLM Compression"
  - "Information Theory"
date: 2026-05-08
content_hash: 5fc0c2f053d7cae8
---

# RADIO: Rate-Distortion Optimization for Large Language Model Compression

**Conference**: ICML 2025  
**arXiv**: [2505.03031](https://arxiv.org/abs/2505.03031)  
**Code**: No public code  
**Area**: Model Compression / LLM Quantization  
**Keywords**: Rate-Distortion Theory, Post-Training Quantization, Rate-Distortion Optimization, LLM Compression, Information Theory

## TL;DR
Starting from the rate-distortion theory in information theory, RADIO establishes a theoretical foundation for LLM quantization and proposes a concise quantization technique based on rate-distortion optimization. This technique can be scaled to models with hundreds of billions of parameters and allows users to flexibly specify the target model size or accuracy for post-training compression.

## Background & Motivation

**Background**: LLM compression has become a key issue in deployment. Existing post-training quantization methods (such as GPTQ, AWQ, and QuIP) primarily approach the problem from an optimization perspective—minimizing the output projection error or weight error before and after quantization. While practical, these methods lack a unified information-theoretic framework to guide quantization decisions.

**Limitations of Prior Work**: 
   - Most quantization methods require pre-specified bit widths (such as 4-bit, 3-bit), lacking the flexibility to adjust within a continuous rate range.
   - Different layers and weight matrices exhibit varying sensitivities to quantization, yet existing methods typically employ uniform bit allocation strategies.
   - There is a lack of theoretical understanding regarding the fundamental question: "What is the optimal quantization strategy given a rate budget?"

**Key Challenge**: Most quantization methods in practice are heuristic (e.g., uniform quantization + grouping), lacking a principled theoretical framework to answer how to optimally allocate bits under a given model size constraint, or what the minimum size a model can be compressed to under a given accuracy requirement is.

**Goal**: To establish an information-theoretic foundation of LLM quantization based on rate-distortion theory and propose an actionable quantization algorithm accordingly.

**Key Insight**: Mapping the LLM weight quantization problem to the classic rate-distortion optimization problem—minimizing distortion (i.e., performance loss caused by quantization) under a given rate (i.e., model size) constraint, or minimizing the rate under a given distortion constraint.

**Core Idea**: Utilizing the rate-distortion function $R(D)$ to model the optimal rate-distortion trade-off of quantization. By solving the rate-distortion optimization problem, the optimal quantization granularity and codebook size for each weight block are determined, thereby achieving globally optimal bit allocation.

## Method

### Overall Architecture
The workflow of RADIO is as follows:
1. Establish the rate-distortion function for each layer/weight matrix of the LLM.
2. Solve the global optimal bit allocation strategy under the global rate budget constraint using the Lagrange multiplier method.
3. Execute quantization on each weight matrix according to its allocated rate.
4. Allow users to specify targets—either the total model size or the acceptable accuracy degradation—and the system automatically solves for the optimal configuration.

### Key Designs

1. **Rate-Distortion Function Modeling**:

    - Each group of weights is treated as a source, and the quantized values are treated as the encoded output.
    - The rate $R$ is defined as the number of bits required for encoding (directly related to model size).
    - The distortion $D$ is defined as the error introduced by quantization (such as the MSE of layer outputs or the change in model perplexity).
    - The rate-distortion function $R(D)$ describes the minimum rate required under the condition that the distortion does not exceed $D$.
    - The core idea is that different weights possess different "compressibility" and should be allocated different numbers of bits.
    - Design Motivation: Rate-distortion theory provides a principled framework for optimal bit allocation, avoiding manual hyperparameter tuning.

2. **Global Bit Allocation Optimization**:

    - Formulate the quantization of all layers into a global optimization problem:
    $\min \sum_i D_i(R_i) \quad \text{s.t.} \quad \sum_i R_i \leq R_{\text{total}}$
    - where $R_i$ represents the rate of the $i$-th weight block, and $D_i$ is the corresponding distortion.
    - Convert it into an unconstrained optimization problem by introducing the Lagrange multiplier $\lambda$:
    $\min \sum_i \left[D_i(R_i) + \lambda R_i\right]$
    - Adjusting $\lambda$ allows navigating different rate-distortion trade-offs.
    - Design Motivation: Different layers exhibit significantly different sensitivities to quantization. Global optimization utilizes the rate budget more effectively than uniform bit allocation.

3. **User-Controllable Compression Targets**:

    - Users can specify a target model size $R_{\text{target}}$: the system finds the optimal allocation where the total rate equals $R_{\text{target}}$.
    - Users can also specify a target accuracy $D_{\text{target}}$: the system finds the minimum model size that satisfies the accuracy constraint.
    - Binary search on $\lambda$ is employed to match the user-specified target.
    - Design Motivation: In deployment environments, different hardware devices have different memory constraints, and different applications have varied accuracy requirements. Flexible target specification greatly enhances practical utility.

### Loss & Training
RADIO is a post-training method and does not involve gradient-based training. The optimization process is:
1. Perform a forward pass on calibration data to collect activation information for each layer.
2. Estimate the rate-distortion characteristics for each weight block.
3. Solve the global optimal rate allocation.
4. Quantize each weight block according to the allocated scheme (e.g., non-uniform codebook quantization).

## Key Experimental Results

### Main Results

| Model | Method | Average Bits | WikiText-2 PPL | Notes |
|------|------|---------|---------------|------|
| LLaMA-2-7B | GPTQ | 4.0 | ~5.7 | Uniform 4-bit |
| LLaMA-2-7B | AWQ | 4.0 | ~5.6 | Uniform 4-bit |
| LLaMA-2-7B | RADIO | 4.0 | Better | Global optimal allocation |
| LLaMA-2-7B | RADIO | 3.5 | Controllable | Mixed-precision |
| LLaMA-2-70B | RADIO | 4.0 | Scalable | Tens of billions of parameters |

### Ablation Study

| Configuration | Key Metrics | Notes |
|------|---------|------|
| Uniform bit allocation | Baseline PPL | Same bits for all layers |
| Global rate-distortion optimization | Better PPL | Adaptive bit allocation |
| Fixed model size target | Constraint satisfied | Automatic adjustment of $\lambda$ |
| Fixed accuracy target | Minimize model size | Reverse optimization |

### Key Findings
- The core advantage of rate-distortion optimization is most prominent in mixed-precision scenarios, where significantly different bits are allocated to different layers.
- Under standard uniform 4-bit quantization, the performance gap between RADIO and conventional methods is small; however, the advantage becomes significant at non-integer average bit widths (e.g., 3.5-bit).
- The method is scalable to models with tens of billions of parameters, and the computational cost of the optimization process itself is far smaller than the quantization process.
- The rate-distortion framework serves as a valuable analytical tool for understanding the theoretical limits of quantization.

## Highlights & Insights
- **Theoretical Elegance**: Introducing classic information theory into LLM quantization establishes a principled theoretical foundation, providing a fresh perspective in a field dominated by heuristics.
- **Flexible User Interface**: Allowing users to specify model size or accuracy targets, after which the system automatically computes the optimal solution, makes it highly engineering-friendly.
- **Reusable Paradigm**: The rate-distortion optimization framework is not only applicable to weight quantization but can also be extended to activation quantization, KV cache compression, and other LLM compression problems.
- The capability of automatic decision-making for mixed-precision quantization eliminates the need for manual bit-width selection for each layer.

## Limitations & Future Work
- While the paper provides a framework based on rate-distortion theory, the concrete quantizer implementation might not be as mature as established methods like GPTQ/AWQ.
- Estimating the rate-distortion function requires calibration data and forward passes, which may pose memory challenges for extremely large models.
- The source code is not publicly available, raising reproducibility concerns.
- In standard uniform bit quantization settings (e.g., strict 4-bit), the improvement over existing methods might be marginal.
- Future work could further integrate other quantization techniques (e.g., the column-wise greedy quantization of GPTQ) to achieve stronger practical performance.

## Related Work & Insights
- **vs GPTQ/AWQ**: GPTQ and AWQ are optimization-driven quantization methods focusing on reducing quantization errors per layer. RADIO provides a theoretical viewpoint of global optimal bit allocation from an information-theoretic angle, making them complementary.
- **vs QuIP/QuIP#**: The QuIP series employs orthogonal transformations to make weights more amenable to quantization. The rate-distortion framework of RADIO can further optimize bit allocation on the transformed weights of QuIP.
- **vs Mixed-Precision Methods**: Conventional mixed-precision methods typically determine bit allocation per layer based on sensitivity analysis. RADIO offers a more systematic and theoretical framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The rate-distortion theory perspective is entirely novel in LLM quantization, offering a significant theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Lacks public source code, the experimental details are limited, and some results require further validation.
- Writing Quality: ⭐⭐⭐⭐ The theoretical exposition is clear, explaining complex concepts in information theory well.
- Value: ⭐⭐⭐⭐ High theoretical value, offering a new analytical framework and perspective for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance](guidedquant_large_language_model_quantization_via_exploiting_end_loss_guidance.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[ICLR 2026\] Large Language Model Compression with Global Rank and Sparsity Optimization](../../ICLR2026/model_compression/large_language_model_compression_with_global_rank_and_sparsity_optimization.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)
- [\[ACL 2025\] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization](../../ACL2025/model_compression/drpruning_robust_pruning.md)

</div>

<!-- RELATED:END -->
