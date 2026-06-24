---
title: >-
  [Paper Note] OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance
description: >-
  [ICML 2025][Multimodal VLM][VLM Training Acceleration] To address the computation imbalance caused by data and model heterogeneity in large-scale vision-language model instruction-tuning, the OmniBal framework is proposed to systematically balance computational workloads across devices from three perspectives: data, model, and memory, achieving approximately $1.8\times$ training speedup on InternVL-Chat.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "VLM Training Acceleration"
  - "3D Parallelism"
  - "Computation Balance"
  - "Pipeline Parallelism"
  - "Dynamic Batching"
date: 2026-05-08
content_hash: ef13a825fea9c0d2
---

# OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance

**Conference**: ICML 2025  
**arXiv**: [2407.20761](https://arxiv.org/abs/2407.20761)  
**Code**: [github.com/ModelTC/OmniBal](https://github.com/ModelTC/OmniBal)  
**Area**: Multimodal/Vision-Language Models, Distributed Training, System Optimization  
**Keywords**: VLM Training Acceleration, 3D Parallelism, Computation Balance, Pipeline Parallelism, Dynamic Batching

## TL;DR

To address the computation imbalance caused by data and model heterogeneity in large-scale vision-language model instruction-tuning, the OmniBal framework is proposed to systematically balance computational workloads across devices from three perspectives: data, model, and memory, achieving approximately $1.8\times$ training speedup on InternVL-Chat.

## Background & Motivation

The training scale of VLMs continues to expand: compared to LLaVA-1.5, InternVL-Chat scales up the dataset from 665K to 5M, the image resolution from $336\times336$ to $3840\times2160$, and the vision encoder from ~300M ViT-L to ~6B InternViT.

However, during large-scale 3D parallel training of VLMs, severe **computation imbalance** issues occur:

**Data Imbalance**: VLM inputs contain variable-length text and a variable number of images, leading to dramatic fluctuations in mini-batch sizes (with input size standard deviation as high as $1.4K \pm 0.9K$ tokens).

**Model Imbalance**: The computational cost differs significantly between the transformer blocks of ViT and LLM (with forward time standard deviation reaching $85 \pm 93$ ms).

**Memory Imbalance**: Dynamic inputs cause GPU memory demands to fluctuate ($39 \pm 23$ G), forcing the use of the most aggressive recomputation strategies.

These issues are absent in LLM text pre-training (where inputs are fixed-length, packable, and models are homogeneous), representing challenges unique to VLM training.

## Method

### Overall Architecture

OmniBal addresses computation imbalance from three closely related aspects in a progressive manner: data $\rightarrow$ model $\rightarrow$ memory.

### 1. Balancing Dynamic Mini-Batches (Data Level)

Two metrics are defined to measure data imbalance:
- **Pad Ratio** (intra-device): $\text{PadRatio} = \frac{\sum_i^B(t_{max} - t_i)}{t_{max} \times B}$
- **Dist Ratio** (inter-device): $\text{DistRatio} = \frac{\sum_i^N(T_{max} - T_i)}{T_{max} \times N}$

An **ISF (Iterative Sampling and Filtering)** algorithm is proposed:
- **Sampling Phase**: Randomly assign samples to the current group until the number of images $I_v$ or text length $I_t$ reaches the predefined thresholds $Q_v, Q_t$.
- **Filtering Phase**: Remove groups that do not satisfy the lower bounds $Q'_v, Q'_t$.

Alternating iteration for $T$ times reduces Pad Ratio from 0.31 to 0, and Dist Ratio from 0.34 to 0.02.

### 2. Balancing Model Partitioning (Model Level)

Goal: Find the optimal pipeline partitioning strategy $P^* = \arg\min_{P_i} f(P_i)$.

Due to the heterogeneity of ViT and LLM, straightforward splits based on parameter count or layer count are ineffective. A search-based method is proposed:
- Profile the forward time $\text{FWD}(l_i)$ of each layer.
- Calculate an anchor partition $P^+$ using a greedy algorithm.
- Generate a candidate set within radius $r$ around $P^+$.
- Use two ranking metrics: forward time variance $\text{VAR(fwd\_time)}$ and communication volume $\text{SUM(comm)}$.

### 3. Balancing Adaptive Recomputation (Memory Level)

Benefiting from the stabilized computation load and memory demands achieved in the previous two steps:
1. Enable full recomputation and record the remaining GPU memory $M_r$ for each stage.
2. Manually disable recomputation for certain layers and record the memory change $\Delta M_r$.
3. Estimate the memory savings $M_v, M_t$ for each layer of ViT and LLM.
4. Adaptively select the optimal recomputation strategy for each stage based on the estimates.

## Key Experimental Results

### Main Results

| Model | Balanced? | Backend | GPU Days | Speedup |
|------|-------|---------|----------|--------|
| 6+20B | ✗ | DeepSpeed | 38.9 | 1.00× |
| 6+20B | ✓ | DeepSpeed | 25.3 | **1.54×** |
| 6+20B | ✗ | Megatron | 61.8 | 0.63× |
| 6+20B | ✓ | Megatron | 21.3 | **1.83×** |
| 6+34B | ✗ | DeepSpeed | 54.3 | 1.00× |
| 6+34B | ✓ | DeepSpeed | 35.5 | **1.53×** |
| 6+34B | ✗ | Megatron | 75.4 | 0.72× |
| 6+34B | ✓ | Megatron | 30.5 | **1.80×** |

### Model Performance Preservation

Balanced training does not compromise model performance, performing on par with or slightly better than the baseline on five benchmarks: MMBench, ChartQA, AI2D, MMVet, and MME.

### Ablation Study

- Data Balancing: ISF reduces GPU Days from 61.8 to 51.9 (data balancing only).
- Model Balancing: Further decreases to 29.0 with Balanced Model Partitioning.
- Memory Balancing: Finally drops to 21.3 with Adaptive Recomputation.

## Highlights & Insights

1. **Systematic Solution**: For the first time, the computation imbalance in VLM is comprehensively identified and resolved for instruction-tuning.
2. **Three-layer Progressive Design**: Data balancing enables feasible model partitioning, and both make memory profile practicable, forming a unified framework.
3. **Significant Speedup**: Over $1.8\times$ speedup without loss of performance, saving substantial GPU resources in practice.
4. **Strong Generalization**: Consistently effective across various model scales, datasets, and hardware configurations.

## Limitations & Future Work

- The thresholds $Q_v, Q_t$ of the ISF algorithm need to be determined based on dataset statistics.
- The search space may expand substantially under high pipeline parallelism degrees.
- Currently, the method is only validated on the InternVL-Chat architecture.

## Related Work & Insights

- 3D Parallelism (Megatron-LM, DeepSpeed ZeRO)
- Pipeline Parallelism Optimization (GPipe, PipeDream, AdaPipe)
- VLM Training (LLaVA, InternVL-Chat, BLIP)

## Rating

⭐⭐⭐⭐ — Solid engineering contribution; the $1.8\times$ speedup is of high practical value for large-scale VLM training. The methodology design is systematic, and the experiments are comprehensive. Although leaning towards system optimization with limited theoretical novelty, it offers outstanding practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Parrot: Multilingual Visual Instruction Tuning](parrot_multilingual_visual_instruction_tuning.md)
- [\[ICML 2025\] Understanding and Mitigating Miscalibration in Prompt Tuning for Vision-Language Models](understanding_and_mitigating_miscalibration_in_prompt_tuning_for_vision-language.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)
- [\[ICML 2025\] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM](streamline_without_sacrifice_--_squeeze_out_computation_redundancy_in_lmm.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
