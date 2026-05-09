---
title: >-
  [Paper Note] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge
description: >-
  [CVPR 2026][Image Generation][LoRA Quantization] This paper proposes the QUAD framework, which treats LoRA weights as runtime inputs rather than compiling them into the model graph. Combined with a distillation fine-tuning strategy that shares quantization parameters across LoRAs, QUAD enables a single compiled model to dynamically switch among multiple GenAI tasks on mobile NPUs, achieving 6× memory compression and 4× latency improvement.
tags:
  - CVPR 2026
  - Image Generation
  - LoRA Quantization
  - Edge Deployment
  - Knowledge Distillation
  - Diffusion Models
  - Runtime Task Switching
date: 2026-05-08
content_hash: dffe4be723b790d9
---

# Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge

**Conference**: CVPR 2026
**arXiv**: [2603.29535](https://arxiv.org/abs/2603.29535)
**Code**: None
**Area**: Image Generation
**Keywords**: LoRA Quantization, Edge Deployment, Knowledge Distillation, Diffusion Models, Runtime Task Switching

## TL;DR

This paper proposes the QUAD framework, which treats LoRA weights as runtime inputs rather than compiling them into the model graph. Combined with a distillation fine-tuning strategy that shares quantization parameters across LoRAs, QUAD enables a single compiled model to dynamically switch among multiple GenAI tasks on mobile NPUs, achieving 6× memory compression and 4× latency improvement.

## Background & Motivation

1. **State of the Field**: GenAI capabilities on mobile devices (image editing, object removal, text-guided transformation, etc.) are increasingly prevalent, typically based on diffusion models (e.g., Stable Diffusion 1.5) with LoRA for task-specific adaptation.

2. **Limitations of Prior Work**: Current mobile deployment pipelines **compile each LoRA separately**—merging LoRA weights into the base model before quantization and compilation—resulting in an independent model binary per task. $N$ tasks = $N$ copies of the base model + $N$ compiled graphs, causing ROM usage to grow linearly.

3. **Root Cause**: Independently trained LoRAs exhibit different weight distributions, leading to inconsistent quantization parameters (scale and zero-point) that cannot share a single static quantized inference graph. On hardware such as NPUs that require fixed quantization parameters, each LoRA must be compiled separately, precluding runtime switching.

4. **Paper Goals**: Design a unified deployment framework that (a) shares quantization parameters across multiple LoRAs; (b) supports dynamic injection of LoRA weights at runtime without recompilation; and (c) maintains generation quality under low-precision inference.

5. **Starting Point**: Restructure the model graph construction—changing LoRA weights from compile-time embeddings to runtime input tensors, then fine-tuning via knowledge distillation to align all LoRAs with a unified quantization configuration.

6. **Core Idea**: LoRA as Input + sensitivity-analysis-guided shared quantization parameters + knowledge distillation fine-tuning = single-graph multi-task edge deployment.

## Method

### Overall Architecture

The system proceeds in three stages: (1) **Architecture modification**: each LoRA-augmented linear layer is restructured into a dynamic injection layer that accepts $A$ and $B$ matrices as inputs, and compiled into a frozen graph; (2) **QUAD**: determining shared quantization parameters across LoRAs and fine-tuning all LoRAs via knowledge distillation to achieve optimal accuracy under these parameters; (3) **Compilation and deployment**: graph optimization (scale folding, constant folding, dead code elimination) → conversion to NPU format → runtime LoRA loading and binding.

### Key Designs

1. **LoRA as Input (Architecture Restructuring)**:

    - **Function**: Transforms LoRA weights from static compile-time embeddings to dynamic runtime inputs.
    - **Mechanism**: For each LoRA-augmented linear layer $y = Wx + \alpha A(Bx)$, the matrices $A$ and $B$ are exposed as additional input nodes in the model graph. Only a single frozen base model binary needs to be compiled; different tasks are switched at inference time by loading the corresponding LoRA weights. LoRA weights are stored as buffers in RAM and bound to model inputs via a lightweight API.
    - **Design Motivation**: Eliminates the redundancy of maintaining one model copy per LoRA. With 10 LoRAs (120 MB each) and a 1.4 GB base model, the conventional approach requires 15 GB; this method requires only 2.6 GB (6× compression).

2. **QUAD — Unified Quantization Parameter Determination**:

    - **Function**: Determines shareable quantization scales and zero-points across all LoRAs.
    - **Mechanism**: A two-step strategy — (a) Compute a Quantization Sensitivity Score (QSS) for each LoRA: $QSS = E_x[D(f(x;w) \| f(x;\tilde{w}))]$, where $D$ is the JS divergence and $f(x;w)$, $f(x;\tilde{w})$ are the full-precision and quantized outputs, respectively. The LoRA with the highest QSS (most sensitive) serves as the anchor, and its quantization parameters are used as the global shared parameters. (b) If all LoRAs exhibit similar sensitivity, a Unified-LoRA fallback strategy merges all LoRA weight distributions to compute global quantization parameters.
    - **Design Motivation**: Selecting the most sensitive LoRA as the anchor ensures that the shared parameters are favorable for the hardest-to-quantize task; other LoRAs are adapted to these parameters via subsequent distillation fine-tuning.

3. **QUAD — Knowledge Distillation Fine-tuning**:

    - **Function**: Adapts non-anchor LoRAs to the shared quantization parameters.
    - **Mechanism**: A QuantSim model is constructed by encoding LoRA-2's weights using the quantization parameters of LoRA-1 (the anchor) via PTQ. The full-precision model serves as the teacher and the QuantSim model as the student. LoRA parameters are optimized to minimize the reconstruction loss between teacher and student outputs, combined with the original LVM training objective. Through iterative optimization, LoRA weights are adapted to satisfy the constraints of the shared quantization parameters.
    - **Design Motivation**: Directly applying another LoRA's quantization parameters causes severe accuracy degradation (FID spikes from 5.53 to 599 under full INT8). Knowledge distillation enables weight distributions to be adapted to the target quantization configuration while preserving task performance.

### Loss & Training

The knowledge distillation loss combines a teacher–student output reconstruction term with the original LVM training objective (e.g., DDPM denoising loss). The quantization configuration used is W8A16 (INT8 weights, INT16 activations). Experiments show that further compressing activations to INT8 causes significant accuracy degradation (FID increases from 12.2 to 599 under W8A8).

## Key Experimental Results

### Main Results

FP32 vs. INT8 quantization accuracy comparison:

| Use Case | Metric | Value |
|----------|--------|-------|
| Prompt-guided image transformation | $sim_d$ (cosine direction similarity) | 0.9428 |
| Prompt-guided image transformation | $sim_{image}$ (semantic similarity) | 0.881 |
| Prompt-guided image transformation | Structure loss | 0.045 |
| Object removal (after QUAD) | FID | 5.5287 |
| Object removal | SSIM | 0.94 |
| Object removal | PSNR | 33.04 |

On-device KPIs (SD1.5 1.1B model):

| Metric | Qualcomm (GS25) | LSI (GS25) | MediaTek (Tab S11) |
|--------|-----------------|------------|-------------------|
| End-to-end latency (8 steps) | 8826ms / 3723ms | 12456ms / 4217ms | 15682ms / 5528ms |
| Shared model ROM | 1375MB | 1125MB | 1177MB |
| LoRA ROM | 119MB | 134MB / 104MB | 31MB / 87MB |
| Peak RAM | 1739MB | 1259MB | 1590MB |

4-use-case deployment (SD1.5 0.7B model, GS25, OLSS 8 steps):

| Use Case | UNet Execution (ms) | End-to-End (ms) | LoRA ROM (MB) |
|----------|---------------------|-----------------|---------------|
| Text-to-Image | 48 | 1052 | N/A |
| Sketch-to-Image | 48 | 1527 | 77 |
| Sticker Generation | 48 | 1080 | 77 |
| Portrait Studio | 51 | 1874 | 77 |

### Ablation Study

Mixed-precision quantization comparison (prompt-guided image transformation):

| Configuration (W8A8:W8A16) | FID | LPIPS | PSNR | SSIM |
|----------------------------|-----|-------|------|------|
| 0:100 (full W8A16) | **12.23** | **0.1083** | **32.71** | **0.9808** |
| 40:60 | 13.05 | 0.1086 | 32.68 | 0.9806 |
| 80:20 | 14.28 | 0.1125 | 31.41 | 0.9777 |
| 100:0 (full W8A8) | 599.07 | 0.699 | 5.44 | 0.2324 |

### Key Findings

- **LoRA weights are not highly sensitive to quantization precision**: INT8 quantization of LoRA itself yields only a 1.5× ROM reduction with negligible accuracy loss, indicating that the value distributions of low-rank adaptation matrices are relatively simple.
- **Activation quantization is the critical bottleneck**: FID spikes to 599 under full W8A8, resulting in complete quality collapse. W8A16 is the optimal operating point.
- **Memory savings scale linearly with the number of tasks**: The memory benefit is limited with two tasks, but grows from 15 GB to 2.6 GB (6× compression) with ten tasks, demonstrating that the framework's core value lies in multi-task scenarios.
- **Runtime switching latency is negligible**: Switching LoRAs requires only loading a weight buffer (~100–200 MB), which is approximately 1.5 seconds faster than reloading the entire model (1–2 GB).
- The approach is effective across three chip vendors (Qualcomm/LSI/MediaTek), confirming the chip-agnostic nature of the framework.

## Highlights & Insights

- **Conceptual shift of LoRA as Input**: Reframing LoRA from "part of the model" to "input to the model" is a conceptually simple yet fundamentally paradigm-shifting idea for multi-task deployment—analogous to the transition from static to dynamic linking in software engineering.
- **QSS-guided anchor selection**: Rather than arbitrarily selecting one LoRA's quantization parameters, the most sensitive LoRA is chosen as the shared baseline, ensuring that the hardest-to-quantize task does not suffer accuracy collapse. This "weakest-link" thinking is broadly valuable in system design.
- **OTA scalability**: New LoRAs can be pushed to devices via OTA updates and used directly without updating the base model—consistent with the app store update paradigm and offering strong productization potential.

## Limitations & Future Work

- Validation is limited to SD 1.5 (1.1B and 0.7B); scalability to larger models (SDXL/Flux) remains unexplored.
- The W8A8 accuracy collapse limits further optimization of latency and memory, necessitating more advanced activation quantization techniques.
- QUAD's knowledge distillation fine-tuning must be executed independently for each non-anchor LoRA, resulting in training costs that grow linearly with the number of LoRAs.
- Evaluation relies primarily on FID/SSIM; subjective perceptual quality assessments are absent.
- Experiments are predominantly conducted on Samsung Research's internal use cases and chip platforms; compatibility with the open-source LoRA ecosystem (e.g., Civitai) is not validated.

## Related Work & Insights

- **vs. QLoRA/QaLoRA**: These methods focus on quantization efficiency during training and do not address multi-LoRA switching at deployment. QUAD targets the inference deployment problem.
- **vs. MobileDiffusion**: Focuses on reducing diffusion inference cost but assumes a static model graph and does not support runtime task switching.
- **vs. conventional merge+compile**: Each LoRA is merged into the base model and compiled independently, causing linear memory growth. QUAD achieves constant storage overhead (relative to the number of tasks) through base model sharing.

## Rating

- **Novelty**: ⭐⭐⭐ The "LoRA as Input" idea is intuitive but exhibits engineering ingenuity; the QSS sensitivity analysis constitutes a methodological contribution, though the overall emphasis is on systems engineering.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers multiple chip platforms and use cases with comprehensive KPI analysis spanning accuracy, latency, and memory; however, the number of evaluated use cases is limited (primarily 2–4).
- **Writing Quality**: ⭐⭐⭐ Framework description is clear, but the paper structure feels somewhat fragmented, with partial redundancy across figures and tables.
- **Value**: ⭐⭐⭐⭐ Directly addresses a practical industrial pain point—multi-task deployment sharing a single model on mobile devices—with immediate applicability to mobile GenAI deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] All-in-One Slider for Attribute Manipulation in Diffusion Models](all_in_one_slider_attribute_manipulation.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)

<!-- RELATED:END -->
