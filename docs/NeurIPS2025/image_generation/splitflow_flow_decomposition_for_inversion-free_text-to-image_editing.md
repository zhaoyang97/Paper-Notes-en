---
title: >-
  [Paper Note] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing
description: >-
  [NeurIPS 2025][Image Generation][Rectified Flow] SplitFlow decomposes a target prompt into multiple sub-prompts, computes an independent editing flow for each, and combines them into a unified editing trajectory via latent trajectory projection and adaptive velocity field aggregation. This resolves gradient entanglement and achieves higher fidelity and editability in text-guided image editing without requiring inversion.
tags:
  - NeurIPS 2025
  - Image Generation
  - Rectified Flow
  - Inversion-Free Editing
  - Flow Decomposition
  - Gradient Conflict
  - Multi-Task Learning
date: 2026-05-08
content_hash: 72e7890c42e38186
---

# SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing

**Conference**: NeurIPS 2025
**arXiv**: [2510.25970](https://arxiv.org/abs/2510.25970)
**Code**: [https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow)
**Area**: Image Editing / Diffusion Models
**Keywords**: Rectified Flow, Inversion-Free Editing, Flow Decomposition, Gradient Conflict, Multi-Task Learning

## TL;DR

SplitFlow decomposes a target prompt into multiple sub-prompts, computes an independent editing flow for each, and combines them into a unified editing trajectory via latent trajectory projection and adaptive velocity field aggregation. This resolves gradient entanglement and achieves higher fidelity and editability in text-guided image editing without requiring inversion.

## Background & Motivation

Rectified-flow-based image editing methods have become mainstream: RF-Inversion, RF-Solver, FireFlow, and others map real images back to the latent space via ODE inversion before editing. However, the inversion process is imprecise, and accumulated errors lead to semantic drift and visual distortion. FlowEdit proposes an inversion-free approach that directly manipulates velocity field differences, but editing quality remains limited.

**Key Challenge**: When the target prompt contains multiple attributes (e.g., "a German Shepherd wearing black sunglasses with its mouth open jumping on the grass"), guiding a single editing trajectory over all attributes causes **gradient entanglement and directional conflicts**, resulting in insufficient editing or excessive distortion.

**Key Insight**: Inspired by gradient conflict resolution in multi-task learning, the paper decomposes a semantically complex target prompt into multiple sub-prompts, computes independent editing flows for each, and then adaptively aggregates them to simultaneously ensure editing diversity and global consistency.

## Method

### Overall Architecture

SplitFlow consists of two stages:
1. **Flow Decomposition**: The target prompt is decomposed into $N$ sub-prompts using an LLM, and an independent inversion-free editing flow is computed for each sub-prompt.
2. **Flow Composition**: Sub-flows are combined into a unified editing trajectory via Latent Trajectory Projection (LTP) and Velocity Field Aggregation (VFA).

### Key Designs

1. **LLM-Based Prompt Decomposition (Flow Decomposition Stage)**:

    - Mistral-7B is used as the prompt reasoning engine, taking an instruction, source prompt, and target prompt as input.
    - The semantic difference is automatically decomposed into $N$ sub-target prompts (typically $N \leq 3$).
    - Example: "German Shepherd jumping with open mouth wearing sunglasses" → {"dog jumping wearing sunglasses", "dog jumping with open mouth", "German Shepherd wearing sunglasses"}
    - An independent editing flow $x_t^{FE(i)}$ is defined for each sub-prompt using the inversion-free formulation from FlowEdit.
    - The decomposition stage runs from $\eta_{max}=33$ to $\eta_{dec}=28$ (5 steps).

2. **Latent Trajectory Projection (LTP)**:

    - Ensures global semantic consistency by projecting each sub-target latent representation onto the direction of the full target prompt's latent representation.
    - Computation: $x_t^{proj(i)} = \langle x_t^{FE(i)}, \hat{x}_t^{FE} \rangle \cdot \hat{x}_t^{FE}$ (inner product along the channel dimension).
    - Aggregated projected latents: $x_t^{proj} = \frac{1}{N} \sum x_t^{proj(i)}$
    - Maintains global consistency of the target trajectory while preserving local changes introduced by each sub-prompt.

3. **Velocity Field Aggregation (VFA)**:

    - Computes relative velocity vectors between sub-target flows: $g_i = v_\theta(x_t^{proj(i)}, t, \phi^{tgt(i)}) - v_\theta(x_t^{src}, t, \phi^{src})$
    - Measures directional consistency between sub-flows using cosine similarity.
    - Derives spatially adaptive weight maps $w \in \mathbb{R}^{N \times H \times W}$ via softmax.
    - Aggregation weights: $w_i(h,w) = \exp\!\left(\sum_{j \neq i} S_{ij}(h,w)\right) / \sum_k \exp\!\left(\sum_{j \neq k} S_{kj}(h,w)\right)$
    - Sub-flows with higher mutual consistency receive larger weights, suppressing redundant flows and emphasizing unique directions.
    - A mathematical proof shows that VFA's weighted aggregation is strictly superior to simple averaging in terms of semantic alignment.

### Loss & Training

- SplitFlow is a **zero-shot, training-free** method requiring no training.
- Built upon Stable Diffusion 3 / 3.5 rectified flow models.
- $T=50$ steps, $\eta_{max}=33$ (first one-third of steps skipped), $\eta_{dec}=28$ (decomposition lasts 5 steps).
- Source CFG = 3.5, target CFG = 13.5.
- Inference time: ~83 minutes for 700 PIE-Bench images (FlowEdit: 57 minutes, plus ~20 minutes for LLM decomposition).

## Key Experimental Results

### Main Results (PIE-Bench)

| Method | Model | Structure Dist↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|------|--------|-------|--------|-------|-------------|-------------|
| FlowEdit | SD3 | 27.24 | 22.13 | 105.46 | 83.48 | 26.83 | 23.67 |
| iRFDS | SD3 | 62.72 | 19.61 | 186.39 | 74.59 | 24.54 | 21.67 |
| FTEdit | SD3.5 | 18.17 | 26.62 | 80.55 | 91.50 | 25.74 | 22.27 |
| **SplitFlow** | SD3 | 25.96 | 22.45 | 102.14 | 83.91 | **26.96** | **23.83** |
| **SplitFlow†** | SD3 | 14.55 | 25.22 | 68.53 | 87.54 | 26.23 | 23.01 |
| **SplitFlow** | SD3.5 | **11.68** | **27.12** | **52.93** | **89.76** | 26.29 | 22.89 |

### Ablation Study

| Configuration | Structure Dist↓ | PSNR↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|--------|-------|-------------|-------------|
| FlowEdit Baseline | 27.24 | 22.13 | 26.83 | 23.67 |
| Simple Average (AVG) | 22.28 | 23.36 | 26.81 | 23.67 |
| + LTP | 26.22 | 22.37 | 26.93 | 23.82 |
| + LTP + VFA (SplitFlow) | **25.96** | **22.45** | **26.96** | **23.83** |

### Key Findings

- Even simple averaging of sub-flows substantially improves background preservation (PSNR: 22.13→23.36), but does not improve CLIP similarity.
- LTP significantly improves CLIP similarity (editing quality); VFA further balances fidelity and editability.
- SplitFlow† under the fidelity-enhanced setting outperforms FTEdit, which uses a stronger backbone.
- Consistent gains over the baseline across different LLMs (Mistral, Qwen2, LLaMA2) and prompt strategies demonstrate robustness.
- $\eta_{dec}=28$ is the optimal balance point; smaller $\eta_{dec}$ prolongs decomposition, improving editability at the cost of fidelity.
- Total inference steps are approximately 48 ($3 \times 5 + 33$), slightly higher than the baseline, with substantially improved quality.

## Highlights & Insights

- **Novel analogy from multi-task learning to image editing**: Velocity fields are treated as gradients and sub-prompts as different tasks; gradient projection and aggregation are introduced to resolve conflicts.
- **Mathematical proof that VFA is strictly superior to simple averaging**: The advantage is established via Gibbs' inequality and Jensen's inequality, showing that weighted aggregation achieves semantic alignment $\geq$ uniform averaging.
- **Training-free, zero-shot design**: Using an LLM for prompt decomposition is an elegant choice with trade-offs — it requires no data but introduces a dependency on the LLM.
- **Highly modular decomposition-aggregation framework**: The decomposition and aggregation stages can be tuned independently, enhancing methodological flexibility.

## Limitations & Future Work

- Inference time increases by approximately 45% compared to the baseline (83 vs. 57 minutes), plus 20 minutes for LLM decomposition.
- Prompt decomposition quality depends on LLM capability and prompt engineering.
- The LLM serves only as a proxy for decomposition; future work may explore VLM-based or optimization-based decomposition approaches.
- Distortion may still occur in extreme editing scenarios.

## Related Work & Insights

- Compared to FlowEdit (inversion-free direct editing), SplitFlow resolves gradient entanglement through flow decomposition.
- Draws inspiration from gradient conflict resolution methods in multi-task learning, including PCGrad and Nash-MTL.
- Implication for future image editing research: decomposing the editing process by semantics is an effective strategy for improving quality in complex editing tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](../../ICCV2025/image_generation/flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](efficient_rectified_flow_for_image_fusion.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](../../ICCV2025/image_generation/ale_attribute_leakage_free_editing.md)

<!-- RELATED:END -->
