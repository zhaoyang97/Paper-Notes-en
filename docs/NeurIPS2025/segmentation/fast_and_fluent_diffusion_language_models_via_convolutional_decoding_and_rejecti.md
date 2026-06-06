---
title: >-
  [Paper Note] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning
description: >-
  [NeurIPS 2025][Segmentation][diffusion language model] By introducing convolutional decoding normalization (replacing hard semi-autoregressive chunking) and rule-based rejective fine-tuning (R2FT)…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "diffusion language model"
  - "convolutional decoding normalization"
  - "rejective fine-tuning"
  - "semi-autoregressive"
  - "time-gap expansion"
date: 2026-05-08
content_hash: 3135f9e15f67dd42
---

# Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning

**Conference**: NeurIPS 2025
**arXiv**: [2509.15188](https://arxiv.org/abs/2509.15188)  
**Code**: Not released  
**Area**: Image Segmentation
**Keywords**: diffusion language model, convolutional decoding normalization, rejective fine-tuning, semi-autoregressive, time-gap expansion

## TL;DR

By introducing convolutional decoding normalization (replacing hard semi-autoregressive chunking) and rule-based rejective fine-tuning (R2FT), the proposed method achieves generation quality at 128 inference steps comparable to 512+ steps, reaching state-of-the-art performance among diffusion language models (DLMs).

---

## Background & Motivation

**Rise of Diffusion Language Models**: DLMs such as MDLM, SEDD, and LLADA generate text through iterative denoising, inherently supporting bidirectional dependency modeling. Unlike autoregressive models (ARMs), DLMs can simultaneously attend to full context, offering theoretically stronger expressive capacity.

**Inference Speed Bottleneck**: Existing DLMs require 512–1024 steps to achieve acceptable generation quality, making their inference speed substantially slower than ARMs. Reducing the number of steps is a critical challenge for practical deployment of DLMs.

**Limitations of Semi-Autoregressive Methods**: Methods such as Block-diffusion accelerate inference by splitting sequences into fixed-size blocks for semi-autoregressive (SAR) decoding, but this introduces a *time-gap expansion* problem—the continuous-time conditioning assumed during training is violated under few-step inference, leading to severe quality degradation at block boundaries.

**Content Degeneration**: DLMs frequently generate high-prior tokens (e.g., "the" and "and" appearing in inappropriate positions) and repeated tokens. These are inherent degeneration patterns arising from the absence of autoregressive constraints in the diffusion process, and they worsen as the number of steps decreases.

**Lack of Systematic Analysis**: Prior work lacks a mathematical analysis of the step-quality trade-off in DLMs and does not distinguish between *window mismatch* and *content degeneration* as two independent problems.

**Mechanism**: Two orthogonal solutions are proposed—convolutional decoding normalization addresses window mismatch through continuous contraction rather than hard chunking; R2FT eliminates degenerate tokens via DPO-style post-training. Their combination achieves 512-step quality at 128 steps.

---

## Method

### Overall Architecture

Two orthogonal solutions are proposed for two independent problems in DLMs, which can be applied separately or in combination:
- **Convolutional Decoding Normalization**: Addresses the time-gap expansion problem in semi-autoregressive inference.
- **Rejective Fine-Tuning (R2FT)**: Eliminates degeneration patterns including high-prior token placement and token repetition.

### Key Design 1: Theoretical Analysis of Time-Gap Expansion

- **Problem Definition**: In semi-autoregressive decoding, the sequence is divided into $B$ blocks, with parallel denoising within each block. During training, the DLM assumes that the timestep $t$ varies continuously, whereas during SAR inference the effective timestep per block is $t_\beta = 1/S_\beta$ (where $S_\beta$ is the number of steps within the block). As the total step count decreases, $d \cdot t_\beta$ grows, violating the continuous-time assumption.
- **Consequence**: Pronounced quality drops appear at block boundaries, manifesting as incoherence, grammatical errors, and information loss.
- **Significance**: This theoretical analysis explains why naively reducing the number of steps causes sudden quality degradation in SAR-DLMs and provides mathematical guidance for the proposed solutions.

### Key Design 2: Convolutional Decoding Normalization

- **Core Idea**: Hard block switching is replaced by learnable position-dependent scaling. A scaling factor $s_i = \tanh(u_i)$ is defined, where $u_i$ is computed based on the number of already-decoded neighbors within a convolutional kernel, enabling a **continuous transition** from "fully masked" to "fully decoded."
- **Mechanism**: At each denoising step, positions with more decoded neighbors receive higher decoding confidence. The convolutional window sweeps progressively from left to right across the sequence, avoiding hard boundaries.
- **Training Integration**: Convolutional normalization is embedded directly into the DLM training pipeline, enabling the model to learn to operate under a continuous window, ensuring consistency between training and inference.
- **Advantage**: Continuous contraction eliminates the $d \cdot t_\beta$ discontinuity, so that 128-step inference no longer violates training assumptions.

### Key Design 3: Rejective Fine-Tuning (R2FT)

- **Objective**: Eliminate two DLM-specific degeneration patterns—high-prior token placement at distant positions and token repetition.
- **Negative Example Construction**: Degenerate samples are automatically identified through rules—negative examples are synthesized by concatenating prompt fragments, inserting high-frequency meaningless tokens, and copying adjacent tokens, requiring no manual annotation.
- **Training Objective**: A DPO-style preference learning objective is adopted, down-weighting degenerate tokens so that the model learns to produce diverse and semantically coherent content after fine-tuning.
- **Design Advantage**: R2FT does not alter the decoding window mechanism and is orthogonal to convolutional normalization; it can be used independently or in combination.

### Loss & Training

- **Base Training**: Standard masked diffusion language model objective (masked diffusion loss) with the convolutional decoding normalization layer incorporated.
- **Post-Training**: R2FT applies a DPO loss to fine-tune the pretrained model, with normal generations as positive examples and rule-synthesized degenerate samples as negative examples.
- **Inference**: 128-step convolutional decoding with standard top-$k$/nucleus sampling.

---

## Key Experimental Results

### Main Results: Open-Ended Text Generation

Evaluated on AlpacaEval with the LLADA-8B architecture using G-Eval automatic scoring:

| Method | Steps | G-Eval ↑ | Relative to 512-step Baseline |
|--------|-------|----------|-------------------------------|
| LLADA (original) | 512 | Baseline | — |
| LLADA (original) | 128 | Significant drop | Severe quality degradation |
| + SAR hard chunking | 128 | Partial recovery | Block boundary artifacts remain |
| + Convolutional normalization | 128 | Near 512-step quality | Window mismatch eliminated |
| + R2FT | 128 | Further improvement | Degenerate tokens eliminated |
| **+ Conv. norm. + R2FT** | **128** | **DLM SOTA** | **Matches or surpasses 512 steps** |

### Ablation Study

| Ablation Condition | G-Eval Change | Explanation |
|--------------------|---------------|-------------|
| Remove convolutional normalization | Significant drop | Block boundary artifacts return |
| Remove R2FT | Moderate drop | High-prior tokens and repetition increase |
| SAR hard chunking only | Worse than conv. norm. | Time-gap expansion unresolved |
| 64 steps | Quality still acceptable | Convolutional normalization is robust |
| 256 steps | Close to full 512-step quality | Diminishing marginal returns |

### Key Findings

- **128 steps match 512-step quality**: Approximately 4× speedup, representing the best step-quality trade-off in DLMs to date.
- **Convolutional normalization is the primary contribution**: Alone, it substantially closes the quality gap across step counts, validating the time-gap expansion theory.
- **R2FT provides complementary gains**: Further eliminates qualitative degeneration on top of convolutional normalization; the combination is optimal.
- **Degeneration pattern analysis**: High-prior tokens appear more frequently at positions far from the prompt; R2FT specifically down-weights these positions.

---

## Highlights & Insights

- **Theory-driven engineering**: The mathematical root cause of time-gap expansion is rigorously analyzed before the convolutional normalization solution is designed, avoiding blind trial-and-error.
- **Orthogonal improvements are composable**: Convolutional normalization addresses structural issues; R2FT addresses content issues. Both are independently effective and jointly optimal.
- **No additional annotation required**: Negative examples for R2FT are entirely synthesized through rules, requiring no human preference data.
- **Strong practical value**: High-quality generation at 128 steps significantly advances the feasibility of deploying DLMs in practice.

---

## Limitations & Future Work

- **Limited evaluation coverage**: Validation is primarily conducted on open-ended text generation; other important tasks such as code generation, mathematical reasoning, and summarization are not covered.
- **Bidirectional advantage underdemonstrated**: DLMs' theoretical advantage lies in bidirectional modeling, yet the experiments do not include scenarios specifically designed to verify this property.
- **Scalability**: Validation is limited to the 8B parameter scale; effectiveness on larger or smaller models remains unknown.
- **Gap with ARMs**: Even achieving SOTA within DLMs, an absolute quality and speed gap relative to comparably sized ARMs (e.g., LLaMA-3-8B) persists.

---

## Related Work & Insights

- **Diffusion Language Models**: MDLM and SEDD established the foundations of discrete diffusion language modeling; LLADA scaled DLMs to large-scale pretraining; Block-diffusion proposed semi-autoregressive acceleration but did not resolve time-gap expansion.
- **DLM Acceleration**: Prior work has explored step reduction and caching strategies, but primarily for continuous diffusion models in the image domain; acceleration research for discrete DLMs remains limited.
- **Preference Alignment**: Post-training methods such as DPO and RLHF are widely applied to ARMs; this paper transfers similar ideas to DLMs, proposing R2FT tailored to diffusion-specific degeneration patterns.
- **Text Generation Quality**: ARMs address repetition and meaningless output via RLHF/DPO; DLMs exhibit a distinct degeneration pattern (high-prior token placement) that requires dedicated solutions.

---

## Rating

| Dimension | Score | Explanation |
|-----------|-------|-------------|
| Novelty | ⭐⭐⭐⭐ | Both the theoretical analysis of time-gap expansion and the convolutional normalization solution are novel contributions. |
| Experimental Thoroughness | ⭐⭐⭐ | Core conclusions are well validated, but coverage of task types and model scales is limited. |
| Writing Quality | ⭐⭐⭐⭐ | Theoretical analysis is rigorous; the motivation–method–experiment logical chain is clear. |
| Value | ⭐⭐⭐⭐ | A 4× speedup is of significant importance for the practical deployment of DLMs. |

**Overall Rating**: ⭐⭐⭐⭐ — The paper provides in-depth theoretical analysis; the proposed convolutional normalization and R2FT are practically effective, representing an important contribution to DLM acceleration. Task coverage could be further expanded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model](stead_robust_provably_secure_linguistic_steganography_with_diffusion_language_mo.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[CVPR 2026\] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](../../CVPR2026/segmentation/concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[NeurIPS 2025\] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)

</div>

<!-- RELATED:END -->
