---
title: >-
  [Paper Note] Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts
description: >-
  [AAAI 2026][LLM/NLP][image correction] From a unified distortion rectification perspective, this paper proposes the UniRect framework, which employs Residual Progressive TPS for geometric deformation correction and Residual Mamba Blocks for degradation compensation. UniRect jointly handles four tasks—portrait correction, wide-angle rectangling, stitching rectangling, and rotation correction—via Sparse MoE for four-in-one multi-task learning. It achieves PSNR gains of 3.82 dB on stitching rectangling and 0.87 dB on rotation correction.
tags:
  - AAAI 2026
  - LLM/NLP
  - image correction
  - image rectangling
  - unified rectification
  - Mamba
  - thin-plate spline
  - mixture-of-experts
date: 2026-05-08
content_hash: 32c3700325e21bf8
---

# Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts

**Conference**: AAAI 2026
**arXiv**: [2512.18718](https://arxiv.org/abs/2512.18718)
**Code**: [https://github.com/yyywxk/UniRect](https://github.com/yyywxk/UniRect)
**Area**: Image Correction / Image Rectangling
**Keywords**: image correction, image rectangling, unified rectification, Mamba, thin-plate spline, mixture-of-experts

## TL;DR

From a unified distortion rectification perspective, this paper proposes the UniRect framework, which employs Residual Progressive TPS for geometric deformation correction and Residual Mamba Blocks for degradation compensation. UniRect jointly handles four tasks—portrait correction, wide-angle rectangling, stitching rectangling, and rotation correction—via Sparse MoE for four-in-one multi-task learning. It achieves PSNR gains of 3.82 dB on stitching rectangling and 0.87 dB on rotation correction.

## Background & Motivation

Smartphone imaging systems involve multiple image correction tasks, yet existing methods adopt task-specific architectures lacking unification. This paper is the first to establish mathematical connections among four tasks from the perspective of physical distortion models:

| Task | Distortion Source | Corresponding Model | Core Challenge |
|------|------------------|--------------------|--------------------|
| Portrait Correction (T1) | Wide-angle lens distortion | Kannala-Brandt model | Facial shape recovery |
| Wide-Angle Rectangling (T2) | Irregular boundary after wide-angle correction | Brown-Conrady model | Non-rectangular boundary filling |
| Stitching Rectangling (T3) | Irregular boundary after panoramic stitching | Brown-Conrady variant + offset $T_0$ | Complex nonlinear deformation |
| Rotation Correction (T4) | Rotation angle during capture | Brown-Conrady + rotation matrix $R_\alpha$ | Subtle rotation perception |

Limitations of prior work: (1) each task requires a dedicated model, wasting resources on edge devices; (2) the coupling relationships among distortions across tasks are unexploited; (3) severe task competition exists during joint multi-task training (mixed learning degrades per-task performance).

## Method

### Overall Architecture

UniRect consists of two major modules: the **Deformation Module (DM)**, which estimates and corrects geometric distortion via RP-TPS, and the **Restoration Module (RM)**, which compensates for interpolation degradation during deformation using Residual Mamba Blocks. The input image $X_0^i$ and its corresponding visual prompt $M_0^i$ are jointly fed into the framework, with the prompt indicating the task type.

### Key Designs

1. **Unified Distortion Model**: A unified mathematical equation is established as $[x_d, y_d]^T = \frac{1}{r} \sum_{j=1}^N (k_j \theta^{2j-1} + k'_j r^{2j-1}) \mathcal{R}_\alpha [x, y]^T + T_0$, where the four tasks correspond to different distortion types via parameter specialization.
2. **RP-TPS (Residual Progressive TPS)**: Two-level control point predictors $\mathcal{C}_0, \mathcal{C}_1$ progressively predict control point offsets. A key design insight is that both sampling steps are performed from the original input $X_0^i$ (rather than intermediate results), effectively avoiding accumulated interpolation errors. The control point grid is $12\times10$.
3. **Residual Mamba Blocks (RMBs)**: Four RMBs (32 channels) constitute the RM, leveraging Mamba's sequential scanning to capture global geometric information and long-range dependencies. For T2/T3 tasks with irregular boundaries, partial convolution is applied to handle invalid regions.
4. **Visual Prompt Design**: T1 = face mask (focusing on facial distortion); T2/T3 = boundary mask (enhancing boundary awareness, participating in boundary loss computation); T4 = all-white image (focusing on global content for rotation perception).
5. **SMoEs (Sparse Mixture-of-Experts)**: Five UniRect expert networks combined with a ResNet18 gating network using Top-1 selection. The gating network automatically routes inputs to appropriate experts, resolving multi-task competition.

### Loss & Training

$$\mathcal{L}_{DM} = \sum_{j=0}^{1} \gamma^j (\mathcal{L}_a^j + \alpha_1 \mathcal{L}_b^j + \alpha_2 \mathcal{L}_p^j + \alpha_3 \mathcal{L}_g^j)$$

where $\mathcal{L}_a$ is the L1 appearance loss, $\mathcal{L}_b$ is the boundary loss (driving control points toward boundaries), $\mathcal{L}_p$ is the line/shape penalty (preventing over-correction), and $\mathcal{L}_g$ is the gradient loss (aligning local textures). Hyperparameters: $\gamma=0.9$, $\alpha_1=0.01$, $\alpha_2=1.0$, $\alpha_3=0.01$. The RM uses L1 + perceptual loss. Training is conducted on 4× V100 GPUs, batch size 4, 200 epochs, polynomial learning rate schedule (initial lr $1\times10^{-4}$).

## Key Experimental Results

### Main Results: Four-Task SOTA Comparison

| Task | Method | PSNR↑ | SSIM↑ | FID↓ | LPIPS↓ |
|------|--------|-------|-------|------|--------|
| Wide-Angle Rectangling (T2) | RecRecNet | 18.68 | 0.5450 | 19.01 | 0.1136 |
| | **UniRect** | **19.90** | **0.5721** | 27.02 | 0.1245 |
| Stitching Rectangling (T3) | Nie et al.'22 | 21.28 | 0.7141 | 21.77 | 0.1557 |
| | **UniRect** | **25.10** | **0.7526** | **19.59** | **0.1120** |
| Rotation Correction (T4) | CoupledTPS | 22.29 | 0.6790 | 7.90 | 0.1970 |
| | **UniRect** | **23.16** | **0.7179** | **6.55** | **0.0873** |

### Ablation Study: Multi-Task Learning Strategy Comparison

| Strategy | T1 ShapeACC↑ | T2 PSNR↑ | T3 PSNR↑ | T4 PSNR↑ | Params |
|----------|-------------|----------|----------|----------|--------|
| Mixed Learning | 97.223 | 13.07 | 15.74 | 21.73 | 357.9M |
| Sequential (1-4-3-2) | 97.401 | 15.23 | 16.94 | 21.51 | 357.9M |
| Sequential (3-2-4-1) | 97.409 | 17.54 | 23.55 | 15.10 | 357.9M |
| **UniRect (SMoEs)** | **97.390** | **19.90** | **25.07** | **23.16** | 369.6M |
| Single-Task Learning (upper bound) | 97.454 | 19.90 | 25.10 | 23.16 | 357.9M×4 |

### Key Findings

- Mixed learning yields only 15.74 dB on T3 vs. 25.10 dB in single-task training, indicating severe task competition.
- Sequential learning exhibits a "primacy effect": the first task trained consistently dominates performance.
- SMoEs nearly matches the single-task upper bound with only a 3.3% parameter increase (369.6M vs. 357.9M).
- RP-TPS contributes most to rotation correction: PSNR improves from 21.56 to 23.16 (+1.6 dB).
- Visual prompts are most effective for T1 (ShapeACC +1.13) and T3 (PSNR +1.28 dB).

## Highlights & Insights

- First to unify four image correction/rectangling tasks through a physical distortion model, with rigorous mathematical derivation.
- The RP-TPS design of "always sampling from the original input" elegantly avoids intermediate interpolation degradation.
- The effectiveness of the Mamba architecture in geometric correction is validated, outperforming CNN and Transformer baselines.
- The visual prompt mechanism is simple yet effective, naturally distinguishing tasks via different masks.
- SMoEs resolve multi-task competition at minimal overhead, with four-in-one performance approaching four-by-one.

## Limitations & Future Work

- Cannot simultaneously apply multiple corrections to a single image (due to dataset constraints where each image corresponds to only one distortion type).
- The gating network struggles to distinguish T1 from T4, as both exhibit similar data distributions without boundary changes.
- The FID metric for T2 (27.02) is inferior to RecRecNet (19.01), indicating room for improvement in perceptual quality.
- Extension to additional correction tasks (e.g., fisheye correction, perspective correction) remains unexplored.

## Related Work & Insights

| Direction | Representative Methods | Difference from Ours |
|-----------|----------------------|----------------------|
| Portrait Correction | Tan'21, Zhu'22 | Task-specific networks, not generalizable to other tasks |
| Image Rectangling | Nie'22, RecRecNet, MOWA | MOWA is most similar but covers fewer tasks and lacks unified mathematical modeling |
| Multi-Task Learning | Loss balancing, gradient manipulation | Effective for similar tasks but poor for tasks with large distortion variation |
| Visual Mamba | VMamba, MambaIR | First application of Mamba to geometric correction tasks |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Unified distortion modeling perspective and RP-TPS are highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive four-task comparison with multi-strategy and prompt ablations
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivation and clear architectural diagrams
- Value: ⭐⭐⭐⭐ Significant practical value for smartphone image processing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model](transmamba_a_sequence-level_hybrid_transformer-mamba_language_model.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](../../ICCV2025/llm_nlp/balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)

</div>

<!-- RELATED:END -->
