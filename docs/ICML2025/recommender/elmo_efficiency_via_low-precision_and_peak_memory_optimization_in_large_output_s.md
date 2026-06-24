---
title: >-
  [Paper Note] ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces
description: >-
  [ICML 2025][Recommender Systems][Extreme Multilabel Classification] The ELMO framework is proposed to reduce the training memory of XMC models with 3 million labels from 39.7 GiB to 6.6 GiB without losing classification accuracy, achieved via pure BFloat16/Float8 low-precision training combined with peak memory optimizations such as gradient fusion and chunking strategies.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "Extreme Multilabel Classification"
  - "Low-precision Training"
  - "Peak Memory Optimization"
  - "Float8"
  - "Large-scale Output Space"
date: 2026-05-08
content_hash: d6ab16310fc3f7ed
---

# ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces

**Conference**: ICML 2025  
**arXiv**: [2510.11168](https://arxiv.org/abs/2510.11168)  
**Code**: [xmc-aalto/elmo](https://github.com/xmc-aalto/elmo)  
**Area**: Recommendation Systems  
**Keywords**: Extreme Multilabel Classification, Low-precision Training, Peak Memory Optimization, Float8, Large-scale Output Space

## TL;DR

The ELMO framework is proposed to reduce the training memory of XMC models with 3 million labels from 39.7 GiB to 6.6 GiB without losing classification accuracy, achieved via pure BFloat16/Float8 low-precision training combined with peak memory optimizations such as gradient fusion and chunking strategies.

## Background & Motivation

Extreme Multilabel Classification (XMC) is widely used in scenarios such as large-scale tag recommendation, product recommendation, and Wikipedia tagging, where the label space can reach hundreds of thousands to millions. In this setting, the linear classification head (rather than the encoder) becomes the primary computational and memory bottleneck. For instance, with an embedding dimension of 768 and 3 million labels, the classifier weights alone require ~8 GiB, which bloats to ~32 GiB when including gradients and optimizer states.

Although the current state-of-the-art (SOTA) method Renee avoids part of the memory consumption of intermediate variables by skipping explicit loss computation, it still suffers from three major issues:

**Inefficient Mixed Precision**: FP16-FP32 mixed-precision training requires maintaining a copy of full-precision parameters and upcasting classifier gradients to FP32, causing severe memory waste.

**Peak Memory Accumulation**: The execution sequence of the computation graph causes memory-intensive operations to overlap at the same time, driving peak GPU memory up to 39.7 GiB.

**No Compression of Classifier Weights**: Neither Renee nor other label-shortlist methods compress the classifier layer weights to save memory.

Furthermore, FP16 exhibits inherent instability in XMC scenarios: a large label space leads to overflow during gradient accumulation, while small gradients face underflow. This co-existence of overflow and underflow makes standard loss scaling difficult to handle both simultaneously.

## Method

### Overall Architecture

The core idea of ELMO is to transition XMC model training from FP16-FP32 mixed precision to pure low precision (BF16 or FP8), coupled with a suite of peak memory optimization strategies. The overall scheme is divided into three progressive levels:

1. **Pure 16-bit Training** (BF16): Eliminating redundant memory of multi-precision copies.
2. **Architecture-level Memory Optimization**: Reducing peak memory via computation stream restructuring, chunking strategies, and gradient fusion.
3. **8-bit Training** (FP8 E4M3): Further compressing classifier weights and the encoder to FP8.

### Key Designs

#### 1. Pure BF16 Training + Precision Compensation

Directly casting weights from FP32 to BF16 causes optimizer updates to be cancelled (round-to-nearest wipes out updates smaller than half the distance between adjacent representable numbers). ELMO adopts two complementary strategies:

- **Classifier Weights → Stochastic Rounding**: Prioritizing memory efficiency, stochastic rounding serves as an unbiased estimator to prevent catastrophic rounding errors during small gradient accumulation.
- **Encoder Weights → Kahan Summation**: Tracking rounding errors with an extra compensation term $c$ and correcting them in subsequent additions:

$$
y \leftarrow v - c, \quad c \leftarrow ((s+y) - s) - y, \quad s \leftarrow s + y
$$

Although Kahan summation requires an extra compensation buffer, it eliminates the need for additional copies of high-precision weights. Since the encoder has a small parameter size, this overhead is negligible.

#### 2. Removing Classifier Momentum

Experiments show that the classifier layer does not require a momentum buffer, and pure SGD with a high learning rate is sufficient. This step directly saves ~8 GiB (with 3 million labels) of momentum storage.

#### 3. Computation Stream Restructuring + Chunking

The peak GPU memory of Renee stems from the simultaneous co-existence of classifier forward outputs, gradients, and weight copies in the GPU memory. The solution of ELMO is:

- **Decoupling Encoder and Classifier Updates**: First complete the encoder forward pass, and then process the classifier in chunks.
- **Label Chunking**: Divide the label space into $k$ equally sized chunks ($k = 3 \sim 8$ in experiments), and sequentially perform forward, backward, and parameter updates for each chunk, reducing the transient memory demand by $k$ times.
- Chunking has no significant impact on training latency.

#### 4. FP8 Classifier Training

Through sweep experiments over different combinations of exponent and mantissa bits, it is discovered that:

- **Weights**: 3 exponent bits are sufficient (the dynamic range of E4M3 is adequate). Performance drops when the mantissa is below 6 bits, but stochastic rounding can fully recover the performance.
- **Gradients**: Under E5M2, ~20% of the gradients are still zero, meaning they must be kept in BF16.
- **Inputs**: FP8 E4M3 is sufficient to cover the input range of the classifier.

Detailed workflow: Cast the BF16 classifier input to FP8 E4M3 $\rightarrow$ perform matrix multiplication with FP8 weights to obtain BF16 logits $\rightarrow$ compute gradients in BF16.

#### 5. Gradient Fusion Triton Kernel

ELMO fuses gradient computation and SGD updates into a single Triton kernel, executing entirely in SRAM:

1. Load classifier weights, logits, and inputs from HBM to SRAM.
2. Perform matmul in SRAM to compute gradients.
3. Directly update weights in SRAM using SGD + stochastic rounding.
4. Write back to HBM.

This completely eliminates the storage requirement for classifier gradients in GPU memory (HBM).

#### 6. FP8 Encoder (torchao)

Integrating the torchao framework to perform FP8 training on the Transformer encoder, further reducing activation memory (from 4.6 GiB in BF16 to 3 GiB).

### Loss & Training

- The loss function follows the design of Renee (BCE loss), where the explicit loss value is not calculated and only the logit gradients are needed.
- The classifier uses pure SGD with a high learning rate (no momentum), and the encoder uses AdamW + Kahan summation.
- Classifier weights use stochastic rounding (BF16/FP8), and encoder weights use Kahan summation compensation (BF16).
- No tensor scaling techniques are utilized.

## Key Experimental Results

### Main Results

Comparison with various SOTA methods on standard XMC datasets such as Wiki-500K, AmazonTitles-670K, Amazon-670K, and Amazon-3M:

| Dataset | Method | P@1 | P@3 | P@5 | Peak Memory (GiB) |
|--------|------|-----|-----|-----|---------------|
| Wiki-500K | LightXML | 76.19 | 57.22 | 44.12 | 15.72 |
| Wiki-500K | CascadeXML | 77.0 | 58.3 | 45.1 | 18.8 |
| AmazonTitles-670K | LightXML | 41.7 | 37.3 | 34.2 | 13.99 |
| AmazonTitles-670K | CascadeXML | 42.1 | 37.5 | 34.1 | 22.3 |
| Amazon-3M | Renee | — | — | — | 39.7 |
| Amazon-3M | ELMO-BF16 | Comparable | Comparable | Comparable | 10.3 |
| Amazon-3M | ELMO-FP8 | Comparable | Comparable | Comparable | **6.6** |

### Memory Comparison (3M labels, batch=128, BERT-base, embed=768)

| Configuration | Initial Memory | Peak Memory | Reduction vs Renee |
|------|-----------|---------|----------------|
| Renee (FP16-FP32 mixed) | 17.9 GiB | 39.7 GiB | — |
| ELMO-BF16 | 5.2 GiB | 10.3 GiB | **74%** |
| ELMO-FP8 | 3.2 GiB | 6.6 GiB | **83%** |

### Ablation Study

| Configuration | Key Metrics / Effects | Description |
|------|---------------|------|
| Removing momentum | Memory decreased by ~8 GiB, accuracy unchanged | Classifier does not need momentum |
| Stochastic rounding vs Round-to-nearest | Stochastic rounding recovers FP32 accuracy | Significant difference when mantissa < 6 bits |
| Number of chunks $k = 3 \sim 8$ | No significant impact on training latency | Memory decreases linearly |
| Gradient fused Triton kernel | Gradient memory $\rightarrow \approx 0$ | Executed entirely in SRAM |
| E4M3 vs E5M2 weights | E4M3 is sufficient | 3 exponent bits can cover the weight range |
| BF16 vs FP8 gradients | BF16 is required | Under FP8 E5M2, 20% of gradients are zero |
| torchao FP8 encoder | Activation 4.6 $\rightarrow$ 3 GiB | Introduces an extra 0.5 GiB buffer |

### Key Findings

1. **FP16 mixed precision is unstable in XMC**: A large label space leads to overflow during gradient accumulation, while small gradients face underflow, making loss scaling unable to accommodate both.
2. **The extended dynamic range of BF16 perfectly resolves the overflow issue**, and combining it with stochastic rounding/Kahan summation can compensate for precision loss.
3. **FP8 training does not require tensor scaling**: The native dynamic range of E4M3 is sufficient to cover classifier weights and inputs.
4. **Classifier gradients must remain in BF16**: In FP8, even with the E5M2 format, ~20% of gradients drop to zero.
5. **The key to peak memory optimization lies in computation stream restructuring**, rather than mere precision reduction.

## Highlights & Insights

- **Problem-Oriented System Design**: Starting from memory profiling, analyzing GPU memory bottlenecks layer by layer, and designing targeted solutions serves as a model of system optimization.
- **Triton Gradient Fusion Kernel**: Completing all gradient calculation and parameter updates in SRAM to thoroughly eliminate gradient storage is a highly elegant engineering innovation.
- **Mixed-Precision Strategy**: Instead of a one-size-fits-all accuracy, different optimal precisions are selected for weights (FP8), gradients (BF16), encoder (Kahan BF16), and classifier (SR BF16/FP8).
- **New Dataset with 8.6M Labels**: LF-Paper2Keywords-8.6M is currently the largest public XMC benchmark, which helps advance the field.
- **6x Memory Reduction without Accuracy Loss**: 6.6 GiB vs 39.7 GiB, meaning models with millions of labels can be trained on consumer-grade GPUs instead of requiring an A100-80G.

## Limitations & Future Work

1. **Only evaluated on the BERT-base encoder**: Performance under larger encoders (e.g., BERT-large, DeBERTa-v3) remains unknown.
2. **FP8 training depends on Hopper/Ada/Blackwell GPUs**: E4M3 hardware acceleration is limited to relatively new GPU architectures.
3. **Chunking strategy may limit parallelism**: Label chunking is processed serially, which can become a temporal bottleneck in extremely large label spaces.
4. **Integration with sparse training has not been explored**: Combining this method with dynamic sparse training (e.g., ELIAS) could potentially reduce memory even further.
5. **Portability of Triton kernels**: Custom Triton kernels increase maintenance costs and migration difficulty.

## Related Work & Insights

- **Renee** (Jain et al., 2023): The cornerstone of end-to-end XMC training, upon which ELMO performs memory optimization.
- **DEXML** (Gupta et al., 2024): Obviates the classifier via a dual-encoder setup, though at the expense of higher computing and memory overhead.
- **torchao**: A general-purpose FP8 training framework, which ELMO applies to the XMC encoder.
- **Insight**: The approach of low-precision training + custom fused kernels can be generalized to other large output space tasks (such as large vocabulary softmax in language models).

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | First to achieve pure FP8 training in XMC; the Triton gradient fusion is highly ingenious. |
| Technical Depth | 5 | From floating-point representation to kernel programming, covering multi-level technical stacks. |
| Experimental Thoroughness | 4 | Broad coverage across multiple datasets + ablations + new benchmarks, but experiments on large encoders are missing. |
| Value | 5 | The 6x memory reduction directly lowers hardware barriers; code is open-sourced. |
| Writing Quality | 4 | Clear logic; memory trace visualization is highly intuitive. |
| Overall Score | **4.4** | An outstanding example of system optimization work. |

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ACL 2026\] GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation](../../ACL2026/recommender/graphlora_structure-aware_low-rank_adaptation_for_large_language_model_recommend.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[ICLR 2026\] Low-pass Personalized Subgraph Federated Recommendation](../../ICLR2026/recommender/low-pass_personalized_subgraph_federated_recommendation.md)
- [\[AAAI 2026\] AutoPP: Towards Automated Product Poster Generation and Optimization](../../AAAI2026/recommender/autopp_towards_automated_product_poster_generation_and_optimization.md)

</div>

<!-- RELATED:END -->
