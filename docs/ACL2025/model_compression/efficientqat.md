---
title: >-
  [Paper Note] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models
description: >-
  [ACL 2025][Model Compression][quantization-aware training] EfficientQAT proposes a two-stage QAT framework: first performing Block-wise All-Parameter training (Block-AP) to provide a good initialization, and then executing End-to-End Quantization-Parameter fine-tuning (E2E-QP) to capture cross-block interactions. It achieves 2-bit quantization of Llama-2-70B in 41 hours on a single A100 GPU, with only a 3-point accuracy degradation.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "quantization-aware training"
  - "LLM compression"
  - "block-wise training"
  - "low-bit quantization"
  - "step size"
date: 2026-05-08
content_hash: e8ee043ed01e0ac3
---

# EfficientQAT: Efficient Quantization-Aware Training for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2407.11062](https://arxiv.org/abs/2407.11062)  
**Code**: [https://github.com/OpenGVLab/EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: quantization-aware training, LLM compression, block-wise training, low-bit quantization, step size

## TL;DR
EfficientQAT proposes a two-stage QAT framework: first performing Block-wise All-Parameter training (Block-AP) to provide a good initialization, and then executing End-to-End Quantization-Parameter fine-tuning (E2E-QP) to capture cross-block interactions. It achieves 2-bit quantization of Llama-2-70B in 41 hours on a single A100 GPU, with only a 3-point accuracy degradation.

## Background & Motivation

**Background**: LLM quantization methods fall into three categories: (1) PTQ (GPTQ/AWQ/OmniQuant) performs fast quantization via block-wise reconstruction, but suffers from significant accuracy loss at low-bit settings; (2) QAT (BitNet b1.58) trains all parameters end-to-end, achieving the best accuracy but requiring extremely high resources (training from scratch); (3) Q-PEFT (QLoRA/PEQA) freezes quantized weights and trains only a small number of parameters, failing to recover quantization loss at low-bit settings.

**Limitations of Prior Work**: (1) PTQ restricts the optimization space by training only rounding/clipping parameters and ignoring cross-block interactions; (2) Vanilla QAT requires complete training data and multi-GPU setups, which is infeasible for 70B models; (3) Q-PEFT contains too few trainable parameters (step size accounts for only ~1.6%), failing to recover quantization errors in ultra-low bit scenarios.

**Key Challenge**: The trade-off between all-parameter training (good accuracy but high overhead) versus limited-parameter training (high efficiency but poor accuracy); and block-wise training (memory-friendly but ignoring cross-block interactions) versus end-to-end training (captures interactions but causes memory explosion).

**Goal**: To achieve quantization accuracy close to vanilla QAT on a single GPU, especially in ultra-low-bit (2-bit/3-bit) scenarios.

**Key Insight**: Decomposing QAT into two complementary stages: block-wise all-parameter training to provide a good initialization, and end-to-end training of only step sizes to capture cross-block interactions.

**Core Idea**: By dividing QAT into a two-stage approach—block-wise all-parameter training (Block-AP) and end-to-end quantization-parameter fine-tuning (E2E-QP)—the method achieves both sufficient optimization space and high memory efficiency.

## Method

### Overall Architecture
Stage 1 (Block-AP): Following the order of transformer blocks, all parameters (weights $W$, step sizes $s$, and zero points $z$) are trained within each block using a reconstruction loss $\rightarrow$ outputting a quantized model with $W_q, s, z$. $\rightarrow$ Stage 2 (E2E-QP): The quantized weights $W_q$ are frozen, and only step sizes $s$ are trained end-to-end, requiring gradients for only ~1.6% of the parameters.

### Key Designs

1. **Block-AP (Block-wise All-Parameter Training)**:

    - **Function**: Simultaneously trains weights, step sizes, and zero points within each transformer block.
    - **Mechanism**: Standard uniform quantization is defined as $W_{int} = \text{clamp}(\lfloor W/s \rceil + z, 0, 2^N-1)$, and dequantization is $\hat{W} = (W_{int} - z) \cdot s$. By embedding quantization/dequantization into the computation graph, STE (Straight-Through Estimator) is used to optimize all parameters via gradient descent.
    - **Design Motivation**: Prior block-wise methods (OmniQuant/BRECQ/AutoRound) only train a subset of parameters (clipping/rounding/LoRA), which limits the optimization space. Block-AP directly trains all parameters without requiring complex designs, showing a particularly distinct advantage in 2-bit scenarios.
    - **Difference from Prior Works**: It is the first method to train all parameters within a block-wise reconstruction paradigm; prior methods restricted the update range to $(-1, +1)$ to prevent overfitting, whereas Block-AP imposes no such restriction.

2. **E2E-QP (End-to-End Quantization Parameter Training)**:

    - **Function**: Freezes the quantized weights $W_q$ output by Block-AP and trains only the step sizes $s$ end-to-end.
    - **Mechanism**: Only dequantization (Eq.2) is performed without active quantization (Eq.1). The gradient $\partial\hat{w}/\partial s = w_q - z$ is simple to compute; trainable parameters account for only ~1.6% (when group size = 64), leading to extremely low memory requirements.
    - **Design Motivation**: While Block-AP ignores cross-block interactions, E2E-QP allows step sizes of all blocks to co-optimize through an end-to-end objective function. Meanwhile, memory usage is extremely low—2-bit E2E-QP for a 70B model requires only 34.2GB.
    - **Flexibility**: Can be directly trained on target datasets (for continual pre-training or instruction tuning).

3. **Complementarity of the Two Stages**:

    - Block-AP provides high-quality initialization (large optimization space $\rightarrow$ low quantization error) but does not consider cross-block interactions.
    - E2E-QP performs lightweight fine-tuning based on this initialization (small parameter volume $\rightarrow$ no overfitting) to capture cross-block interactions.
    - Combining the two stages achieves both the accuracy of QAT and the efficiency of PTQ.

## Key Experimental Results

### Main Results: Llama-2 Zero-Shot Inference (Average Accuracy over 5 Tasks)

| Method | Bit | Llama-2-7B | Llama-2-13B | Llama-2-70B |
|------|------|-----------|------------|------------|
| FP16 | 16 | 64.86 | 67.81 | 72.41 |
| GPTQ | 3 | 62.48 | 66.18 | 71.47 |
| AWQ | 3 | 62.82 | 66.14 | 71.41 |
| OmniQuant | 3 | 62.42 | 66.18 | 71.07 |
| **EfficientQAT** | **3** | **64.02** | **67.28** | **71.76** |
| OmniQuant | 2 | 46.98 | 53.56 | 54.87 |
| AutoRound | 2 | 54.50 | 60.72 | 67.70 |
| **EfficientQAT** | **2** | **59.50** | **63.88** | **68.93** |
| AQLM (VQ) | 2 | 57.61 | 62.22 | 69.85 |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Block-AP + E2E-QP | Best | Full EfficientQAT |
| Block-AP only | Good | Lacks cross-block interactions |
| E2E-QP only (RTN Initialization) | Poor | RTN initialization is too poor to recover |
| Block-AP training rounding only | Suboptimal | Restricts optimization space |
| Block-AP training clipping only | Suboptimal | Same as above |
| E2E training s + z | ≈ E2E training s | Converting z to full-precision incurs extra overhead |

### Key Findings
- **Outstanding Advantage in 2-bit Scenarios**: Under 2-bit settings, EfficientQAT outperforms OmniQuant by around 12-14 points and AutoRound by around 5 points.
- **Almost Lossless at 3-bit**: Llama-2-70B at 3-bit obtains 71.76 vs. FP16 at 72.41, with only a 0.65-point drop.
- **Extremely High Training Efficiency**: 2-bit quantization for a 70B model requires only a single A100, taking 41 hours and 34.2GB memory.
- **All-Parameter Training > Sub-Parameter Training in Block-AP**: Simply and directly training all parameters is more effective than carefully designing rounding/clipping parameters, challenging the conventional belief that "restricting the optimization space is necessary to prevent overfitting."
- **Cross-Modal Universality**: It is equally effective on instruction-tuned LLMs and multimodal LLMs (LLaVA).

## Highlights & Insights
- **"Simplicity is Key" Design Philosophy**: The greatest innovation of Block-AP is actually "not designing anything" — directly training all parameters instead of carefully designing rounding/clipping parameters like previous works. This indicates that optimization space is more crucial than regularization in LLM quantization.
- **Elegance of the Two-Stage Decomposition**: Decomposing the "all-parameter + end-to-end" nature of QAT into "all-parameter + block-wise" and "few-parameters + end-to-end", keeping each stage highly efficient. The combination approaches the performance of vanilla QAT.
- **Conciseness of E2E-QP Training Only the Step Size**: Although step size accounts for only ~1.6% of the parameters, it effectively captures cross-block interactions. This shows that the end-to-end adjustment of quantization parameters yields higher leverage than weight fine-tuning.
- **Transferability to Q-PEFT Scenarios**: The E2E-QP stage of EfficientQAT can be directly trained on instruction fine-tuning data, unifying compression and fine-tuning.

## Limitations & Future Work
- Only uniform quantization was explored; it has not been combined with vector quantization (e.g., AQLM/QuIP#)—AQLM remains competitive at 2-bit.
- Block-AP still requires block-wise full-precision forward passes, meaning memory might still be insufficient for ultra-large models (700B+).
- Activation quantization (WAQ) has not been explored; the work only conducts weight quantization.
- Sensitivity analysis on the choice of calibration data and training hyperparameters is insufficient.

## Related Work & Insights
- **vs OmniQuant (Shao et al., 2023)**: OmniQuant trains clipping parameters block-wise, which restricts its optimization space. EfficientQAT's Block-AP trains all parameters, outperforming it by 12+ points at 2-bit.
- **vs BitNet b1.58 (Ma et al., 2024)**: BitNet belongs to vanilla QAT trained from scratch, whereas EfficientQAT is an efficient QAT on pre-existing models, making it more widely applicable.
- **vs PEQA (Kim et al., 2023)**: PEQA only trains step sizes end-to-end (lacks Block-AP initialization), making recovery difficult when starting from RTN. EfficientQAT's Block-AP provides a critical, high-quality initialization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The two-stage decomposition logic is clear. Although Block-AP's "all-parameter training" is simple, it is proposed and proven effective for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across scales from 7B to 70B, other low-bit scenarios (2/3/4-bit), covering base, instruction-following, and multimodal scenarios, with a comprehensive ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and precise, with detailed method descriptions.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical, achieving 2-bit QAT for a 70B model on a single GPU provides a genuinely deployable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)
- [\[ACL 2025\] PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](ptq161_low_bit_quantization.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
