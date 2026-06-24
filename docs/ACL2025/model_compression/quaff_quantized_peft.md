---
title: >-
  [Paper Note] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis
description: >-
  [ACL 2025][Model Compression][quantization] This paper proposes the Outlier Spatial Stability Hypothesis (OSSH)—that the spatial locations of activation outlier channels remain stable during fine-tuning—and designs the Quaff framework based on this hypothesis. By handling only a few persistent outlier channels using targeted momentum scaling, Quaff achievements a 1.73× latency reduction and a 30% memory saving, while also improving accuracy on GPQA by 0.6%.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "quantization"
  - "PEFT"
  - "activation outlier"
  - "weight-activation quantization"
  - "fine-tuning"
date: 2026-05-08
content_hash: 31087e53fe94106e
---

# Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis

**Conference**: ACL 2025  
**arXiv**: [2505.14742](https://arxiv.org/abs/2505.14742)  
**Code**: [https://github.com/Little0o0/Quaff.git](https://github.com/Little0o0/Quaff.git)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: quantization, PEFT, activation outlier, weight-activation quantization, fine-tuning

## TL;DR
This paper proposes the Outlier Spatial Stability Hypothesis (OSSH)—that the spatial locations of activation outlier channels remain stable during fine-tuning—and designs the Quaff framework based on this hypothesis. By handling only a few persistent outlier channels using targeted momentum scaling, Quaff achievements a 1.73× latency reduction and a 30% memory saving, while also improving accuracy on GPQA by 0.6%.

## Background & Motivation

**Background**: PEFT (such as LoRA) reduces the number of trainable parameters, but the computational and memory overhead of fine-tuning billion-scale models remains substantial. Quantization is a primary means to improve efficiency: weight-only quantization (WOQ) compresses only weights but introduces mixed-precision computation bottlenecks; weight-activation quantization (WAQ) compresses both to INT8, utilizing integer arithmetic to achieve a 4× speedup.

**Limitations of Prior Work**: LLMs exhibit "emergent" channel-level outliers where certain channel activation values are 100× larger than the average. Existing solutions face a trilemma: (1) static scaling predefines scaling factors on calibration data, but shifts in activation distribution during fine-tuning cause mismatch; (2) dynamic scaling adjusts in real-time but requires storing full-precision weights and repeated re-quantization, incurring excessive memory and computational costs; (3) rotation transformations replace scaling but introduce computational inefficiencies.

**Key Challenge**: Channel scaling couples the quantization of weights and activations—the scaled weights $\hat{W} = sW$ depend on real-time activation statistics, rendering independent quantization impossible.

**Goal**: Decouple the dependence between weight and activation quantization, making INT8 WAQ both efficient and accurate in fine-tuning scenarios.

**Key Insight**: It is observed that the spatial locations of outlier channels remain stable during fine-tuning (OSSH). Therefore, one only needs to pre-identify these channels and target them for processing.

**Core Idea**: Leverage the spatial stability of activation outlier channels to dynamically scale only a small fraction (<5%) of stable channels, thereby decoupling the weight-activation quantization dependency.

## Method

### Overall Architecture
Preprocessing phase: Identify outlier channels $O$ using calibration data $\rightarrow$ Quantize frozen weights $W_{int}$ + Keep full-precision weights $W_O$ for outlier channels $\rightarrow$ Fine-tuning phase: Inject PEFT parameters $\theta$ $\rightarrow$ Run INT8 forward propagation with momentum scaling on outlier channel activations at each step.

### Key Designs

1. **Outlier Spatial Stability Hypothesis (OSSH)**:

    - Function: Propose and validate the hypothesis that the spatial locations of activation outlier channels remain stable during fine-tuning.
    - Mechanism: A pre-defined set of 5% outlier channels can achieve a hit rate of $>90\%$ during the fine-tuning process.
    - Design Motivation: If outlier channel locations are stable, they can be identified beforehand, avoiding global runtime scans. This naturally stems from the preservation of pre-trained features—outlier channels encode high-level semantic primitives, and fine-tuning only performs task adaptation, which does not alter them.
    - Comparison with Prior Work: Prior works only observed channel stability in inference scenarios (fixed models); OSSH extends this to fine-tuning scenarios (where distributions shift).

2. **Decoupled WAQ Formulation**:

    - Function: Decompose channel scaling $Y = \hat{X}(sW)$ into static and dynamic components.
    - Mechanism: $Y = \hat{X}W + \hat{X}_{:,O}(s_O - 1)W_O$, where the main body $\hat{X}W$ can complete INT8 computation with pre-quantized weights, and the compensation term $\hat{x}\hat{w}$ involves only a small number of outlier channels.
    - Since $(s-1)$ is zero for non-outlier channels, the compensation term is highly sparse, keeping the computational overhead $<5\%$.

3. **Targeted Momentum Scaling**:

    - Function: Compute scaling factors only for outlier channels, utilizing a momentum mechanism for smooth updates.
    - Mechanism: $s_t = \gamma s_{t-1} + (1-\gamma)\beta$, where $\beta_i = \max(1, \sqrt{\max(|X_{:,i}|)/\max(|W_i|)})$ for outlier channels, and $\beta_i = 1$ for non-outlier channels.
    - The momentum parameter $\gamma$ controls the update inertia, preventing overreaction to instantaneous activation fluctuations.
    - Compared with dynamic scaling, this reduces recomputation and memory overhead by 99%.

### Loss & Training
- Use standard STE (Straight-Through Estimator) for backpropagation.
- Compatible with four PEFT methods: LoRA, Prompt Tuning, P-tuning, and IA3.
- Outlier channel budget is controlled under 5% and adaptively allocated across layers (fewer for q_proj, more for down_proj).

## Key Experimental Results

### Main Results: GPQA Inference Benchmark (Phi-3 + LoRA)

| Method | Precision | Latency Ratio (vs FP32)↓ | Memory Ratio (vs FP32)↓ | GPQA Acc |
|------|------|-----------------|-----------------|----------|
| FP32 | FP32 | 1.0× | 1.0× | Baseline |
| SmoothQuant | W8A8 | ~0.65× | ~0.7× | Below Quaff |
| QuaRot | W8A8 | ~0.75× | ~0.75× | Below Quaff |
| LLM.int8() | W8A8 | Slower than FP32 | ~0.85× | Below Quaff |
| **Quaff** | W8A8 | **0.58×** | **0.70×** | **FP32+0.6%** |

### Ablation Study

| Configuration | Performance | Note |
|------|------|------|
| Quaff Complete | Best | Targeted Momentum Scaling + OSSH |
| w/o Momentum (Direct Scaling) | Obvious drop | Instantaneous fluctuations lead to instability |
| Static Scaling (SmoothQuant) | Drops by 2.1% | Distribution shift during fine-tuning causes mismatch |
| Global Dynamic Scaling | Similar but slower | Requires storing full-precision weights |
| Different OSSH Channel Ratios| 5% is best | Too few leads to insufficient coverage, too many increases overhead |

### Key Findings
- **Thorough validation of OSSH**: The 5% pre-defined channels maintain a hit rate of $>90\%$ across fine-tuning iterations, and this is consistent across different models (LLaMA-2, Phi-3, OPT).
- **Feasible on consumer GPUs**: Successfully ran LLaMA-2-7B fine-tuning on an RTX 2080 Super (8GB).
- **Cross-PEFT compatibility**: Quaff is effective across four methods: LoRA, Prompt Tuning, P-tuning, and IA3.
- **$(s-1)$ scaling is more stable than $s$**: It reduces weight sensitivity to the scaling factor, improving quantization stability.

## Highlights & Insights
- **Theoretical elegance of the OSSH hypothesis**: By observing the simple property that "outlier channel locations remain invariant", the method cleverly decouples the weight-activation coupling bottleneck in WAQ. This hypothesis is both intuitive and backed by experimental evidence.
- **Ingenuity of the formulation decomposition**: Decomposing $\hat{X}(sW)$ into $\hat{X}W + \hat{x}\hat{w}$ and exploiting the sparsity of the compensation term allows handling outliers almost for free. This algebraic transformation can be transferred to other scenarios requiring dynamic-static separation.
- **Practical deployment value**: Enables fine-tuning on consumer-grade GPUs, contributing directly to "democratizing LLMs."
- **The $(s-1)$ trick**: For non-outlier channels, $s_i = 1$ makes $(s-1)_i = 0$, naturally achieving sparsification—a highly practical trick.

## Limitations & Future Work
- Only INT8 quantization is validated; whether OSSH still holds in more extreme low-bit (INT4/INT2) scenarios remains unexplored.
- The impact of calibration data selection on outlier channel identification is not fully discussed.
- The momentum parameter $\gamma$ requires manual tuning, and no adaptive adjustment scheme is provided.
- The theoretical explanation of OSSH is quite intuitive ("feature preservation + semantic consistency") and lacks rigorous mathematical proof.

## Related Work & Insights
- **vs SmoothQuant (Xiao et al., 2023)**: SmoothQuant uses static channel scaling, which is effective at inference but fails during fine-tuning. Quaff achieves "targeted dynamic scaling" via OSSH.
- **vs LLM.int8() (Dettmers et al., 2022)**: Keeps outlier channels in full precision, but introduces mixed-precision overhead. Quaff is more efficient through the compensation term approach.
- **vs QuaRot (Ashkboos et al., 2024)**: Eliminates outliers using rotation instead of scaling, but calculating rotation matrices introduces extra overhead. Quaff is lighter by only processing 5% of channels.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The OSSH hypothesis is novel and convincing, and the formulation decoupling is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 benchmarks, 3 models, and 4 PEFT methods, including deployment validation on consumer GPUs.
- Writing Quality: ⭐⭐⭐⭐ Clear logic progressing from problem $\rightarrow$ hypothesis $\rightarrow$ method $\rightarrow$ validation.
- Value: ⭐⭐⭐⭐⭐ Inherently balances theoretical contribution and practical deployment value; holding high significance for consumer GPU fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](trans_peft_transferable.md)
- [\[ACL 2025\] One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments](one_quantllm_for_all_fine-tuning_quantized_llms_once_for_efficient_deployments.md)
- [\[ACL 2025\] L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)

</div>

<!-- RELATED:END -->
