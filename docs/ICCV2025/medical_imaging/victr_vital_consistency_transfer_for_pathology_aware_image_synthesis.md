---
title: >-
  [Paper Note] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis
description: >-
  [Medical Imaging] > This paper proposes ViCTr, a two-stage framework that combines Rectified Flow with a Tweedie-corrected diffusion process to achieve high-fidelity pathology-aware medical image synthesis. The method re…
tags:
  - "Medical Imaging"
date: 2026-05-08
content_hash: b1454b25ecccbb19
---

# ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2505.04963](https://arxiv.org/abs/2505.04963)
- **Authors**: Onkar Susladkar, Gayatri Deshmukh, Yalcin Tur, Gorkem Durak, Ulas Bagci (Northwestern University, Stanford University, UIUC)
- **Code**: [GitHub](https://github.com/Onkarsus13/ViCTr-2D) / [Weights](https://huggingface.co/onkarsus13/ViCTr-2D)
- **Area**: Medical Imaging / Medical Image Synthesis
- **Keywords**: Medical image synthesis, Rectified Flow, Tweedie's Formula, pathology-aware, liver cirrhosis, data augmentation, LoRA

## TL;DR

> This paper proposes ViCTr, a two-stage framework that combines Rectified Flow with a Tweedie-corrected diffusion process to achieve high-fidelity pathology-aware medical image synthesis. The method reduces inference steps from 50 to 3–4 and, for the first time, enables graded-severity pathology synthesis for abdominal MRI.

## Background & Motivation

### Problem Definition
Medical image synthesis aims to generate anatomically realistic and pathologically diverse synthetic medical images for data augmentation, alleviating the scarcity of medical imaging data.

### Existing Challenges

**Data scarcity**: Privacy regulations, inter-institutional data fragmentation, and interoperability constraints result in severe shortages of medical imaging data.

**Insufficient anatomical fidelity**: Existing methods struggle to maintain anatomical accuracy while simultaneously modeling pathological features.

**Difficulty with diffuse pathologies**: Diffuse lesions such as liver cirrhosis involve subtle tissue changes across multiple organ systems, far more complex than focal lesions such as tumors.

**Low sampling efficiency**: Conventional diffusion models require more than 50 sampling steps, incurring substantial computational overhead.

**Lack of pathology control**: Existing methods cannot finely control the severity of synthesized pathologies.

### Core Motivation
- Rectified Flow provides near-linear sampling trajectories, reducing the required number of steps.
- Tweedie's Formula corrects sampling bias and improves initialization accuracy.
- The combination of both, together with two-stage training, yields efficient, high-fidelity, and pathology-controllable synthesis.

## Method

### Overall Architecture
ViCTr comprises two training stages:
1. **Stage 1 — Pre-training**: Establishes anatomical priors on the ATLAS-8k dataset.
2. **Stage 2 — Fine-tuning**: Adapts to downstream tasks (CT/MRI generation and pathology synthesis) via LoRA.

### Key Designs

#### 1. Rectified Flow Trajectory
Interpolation is defined as: $x_t = (1-t)x_0 + tx_1$, where $x_0 \sim p_0$ (noise) and $x_1 \sim p_{target}$ (target data).

Velocity model training:
$$\hat{\theta} = \arg\min_\theta \mathbb{E}_{t \sim \text{Uniform}(0,1)} \left[ \|(x_1 - x_0) - v_\theta(x_t, t)\|^2 \right]$$

One-step distillation: $\hat{\mathcal{T}}(x_0) = x_0 + v(x_0, 0)$

#### 2. Tweedie's Formula Correction
Tweedie correction is incorporated into the Rectified Flow ODE:
$$dx_t = v_{\hat{\theta}}(x_t, t)dt + (1 - \bar{\alpha}_t)\nabla_{x_t} \log p(x_t) dt$$

The additional score term $(1 - \bar{\alpha}_t)\nabla_{x_t} \log p(x_t)$ corrects sampling bias, directing $x_t$ more accurately toward the target distribution.

One-step sampling extension:
$$\hat{\mathcal{T}}(x_0) = x_0 + v(x_0, 0) + (1 - \bar{\alpha}_0)\nabla_{x_0} \log p(x_0)$$

#### 3. Stage 1 — ATLAS-8k Pre-training

- **Input**: CT image $X_I$, segmentation mask $X_S$, text prompt $X_p$
- **Encoding**: A frozen VAE encoder extracts latent representations $Z_o$, $Z_s$; a pre-trained text encoder extracts $Z_p$
- **Forward diffusion**: $P(Z_t|Z_o) = (1-t) \cdot Z_o + t \cdot \epsilon_{true}$
- **Reverse diffusion**: $P(Z_{t-1}|Z_t, Z_s, Z_p, t) = Z_t + \delta T \times \phi_\theta(Z_t, Z_s, Z_p, t)$
- **EWC (Elastic Weight Consolidation)**: Selectively unfreezes critical layers to maintain model stability.

Loss function:
$$L_{diff} = -|\phi_\theta(Z_t, Z_s, Z_p, t) - (\epsilon_{true} - Z_o)|^2$$

Composite loss: $L_{diff} + L_2 + L_{SSIM}$

#### 4. Stage 2 — LoRA Fine-tuning
- **Dual network**: $\phi_{base}$ (frozen, retains anatomical knowledge) + $\phi_{adapt}$ (LoRA trainable)
- **Consistency loss**: $L_{consistency}$ aligns intermediate outputs between the two networks
- **Temporal consistency**: $L_{temporal}$ ensures smooth transitions across reverse diffusion timesteps
- **Spatial consistency**: $L_{spatial}$ enforces alignment in output reconstruction

#### 5. Pathology Generation
- Dataset: CirrMRI600+ (T1/T2 MRI)
- Liver segmentation masks combined with text prompts specifying severity: "low", "mild", "severe"
- Progressive pathology control

### Support for Multiple Diffusion Backbones

| Diffusion Method | Denoiser | Text Encoder |
|----------|----------|-----------|
| Stable Diffusion | UNet | CLIP-B/16 |
| Pixart-alpha | DiT | T5-XXXL |
| SDXL | Dual UNet | CLIP-L/14 |
| Flux | MultiModal Transformer | T5-XXXL + CLIP |
| SD-3 | MultiModal Transformer | T5-XXXL + CLIP |

## Key Experimental Results

### Main Results — Synthesis Quality (FID/MFID)

| Backbone | BTCV(CT) Vanilla/ViCTr | AMOS(MRI) Vanilla/ViCTr | CirrMRI600+ Vanilla/ViCTr | Steps Vanilla→ViCTr |
|------|------|------|------|------|
| Stable Diffusion | 25.44/19.67 → **21.98/19.02** | 25.43/21.76 → **20.37/19.11** | 28.34/23.43 → **25.57/21.46** | 40→4 |
| SDXL | 23.47/18.21 → **20.33/17.44** | 24.11/20.23 → **19.44/18.45** | 27.34/22.11 → **24.02/20.76** | 30→4 |
| SD-3 | 19.07/16.22 → **17.37/16.02** | 22.32/19.76 → **18.02/19.08** | 24.49/21.78 → **21.28/19.34** | 50→3 |
| Pixart-alpha | 21.32/17.09 → **19.22/16.96** | 23.78/20.04 → **18.76/18.56** | 26.06/20.07 → **23.04/18.92** | 25→3 |
| **Flux** | 15.52/15.01 → **13.28/14.08** | 19.02/18.28 → **15.55/16.58** | 22.46/18.88 → **19.96/17.01** | 30→3 |

ViCTr + Flux achieves **MFID 17.01** on CirrMRI600+, 28% lower than the previous best.

### Segmentation Performance Improvement (mDSC%↑ / mHD95↓)

| Segmentation Model | Real Data Only | +Augmentation | +30% Vanilla Synthetic | +30% ViCTr Synthetic |
|----------|-----------|------|-------------|------------|
| **BTCV** |
| UNet | 76.72 | 78.45 | 79.32 | **81.22** |
| TransUNet | 85.52 | 87.01 | 87.54 | **89.78** |
| nnUNet | 80.48 | 82.54 | 83.37 | **85.19** |
| MedSegDiff | 87.91 | 88.65 | 89.78 | **91.92** |
| **CirrMRI600+** |
| UNet | 68.74 | 69.38 | 70.12 | **73.39** |
| nnUNet | 71.02 | 72.49 | 73.56 | **78.89** |
| MedSegDiff | 76.92 | 77.11 | 78.03 | **81.37** |

nnUNet trained with ViCTr synthetic data on CirrMRI600+ achieves a gain of **+7.87% mDSC** (+3.8% over vanilla synthetic data).

### Ablation Study

| Ablation | FID |
|--------|-----|
| w/o Stage-1 pre-training | 17.33 |
| w/o Tweedie correction | 18.78 |
| Using Reflow | 20.19 |
| Using Flow Straight and Fast | 21.37 |
| Using Distribution Matching Distillation | 22.33 |
| $L_{diff}$ only | 18.77 |
| $L_{diff} + L_{spatial}$ | 18.21 |
| $L_{diff} + L_{consistency}$ | 17.02 |
| $L_{diff} + L_{spatial} + L_{consistency}$ | **15.55** |
| LoRA r=8 | 18.46 |
| LoRA r=16 | 17.52 |
| LoRA r=32 | 16.66 |
| **LoRA r=64 (final)** | **15.55** |

### Key Findings
1. **Tweedie correction is critical**: Removing it raises FID from 15.55 to 18.78, indicating a significant degradation in distribution alignment.
2. **Pre-training is indispensable**: Omitting Stage 1 raises FID from 15.55 to 17.33.
3. **Consistency loss contributes most**: Among the three loss components, $L_{consistency}$ yields the largest gain (17.02 vs. 18.77).
4. **Higher LoRA rank is better**: r=64 performs best, as a higher-dimensional adaptation space provides finer-grained parameter updates.
5. **Inference efficiency**: Sampling steps are reduced from 50 to 3–4, and inference time drops from 18.98s to 2.78s (SD-3).
6. **Radiologist validation**: Three radiologists were unable to distinguish synthesized from real cirrhotic MRI images in a Visual Turing Test.

## Highlights & Insights

1. **First abdominal MRI pathology synthesis**: This is the first method to achieve graded-severity control in abdominal MRI pathology synthesis, filling an important gap in the field.
2. **Theoretical innovation**: Embedding Tweedie's Formula into the Rectified Flow framework is an original theoretical contribution, not a trivial combination of existing techniques.
3. **Extreme efficiency**: A reduction from 50 to 3 steps (10× speedup) makes clinical deployment feasible.
4. **Broad compatibility**: Support for 5 mainstream diffusion backbones demonstrates the generality of the approach.
5. **Dual validation**: The method is evaluated through both quantitative metrics (FID/MFID/mDSC) and qualitative radiologist assessment.
6. **Elegant two-stage paradigm**: Stage 1 establishes anatomical priors while Stage 2 applies LoRA-based pathology adaptation, elegantly balancing generality and task specificity.

## Limitations & Future Work

1. **2D slice-level synthesis**: The current method operates on 2D slices only; 3D volumetric consistency is not guaranteed.
2. **Limited pathology types**: Only liver cirrhosis is validated; other diffuse conditions (e.g., hepatic steatosis, fibrosis) remain untested.
3. **Resolution constraint**: A resolution of 256×256 may be insufficient to capture certain subtle pathological features.
4. **Training overhead**: Pre-training requires approximately 52 hours on 8×8 A100 GPUs, imposing substantial resource demands.
5. **VAE bottleneck**: Reliance on a frozen VAE encoder/decoder means that information loss therein propagates to the final synthesis quality.

## Related Work & Insights

### Related Work
- **Medical diffusion**: MedSegDiff, EMIT-Diff, DiNO-Diffusion
- **Rectified Flow**: ReFlow, Flow Straight and Fast, Distribution Matching Distillation
- **Consistency models**: Consistency Models
- **Medical data augmentation**: DiffuseMix, DreamDA, ControlPolypNet

### Insights
- The combination of Rectified Flow and Tweedie correction is generalizable to other settings requiring high-fidelity few-step sampling.
- The two-stage paradigm of "general pre-training + LoRA task-specific adaptation" is particularly effective in data-scarce domains.
- Evaluation of medical image synthesis should jointly consider generation quality (FID) and downstream task utility (segmentation mDSC).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](../../NeurIPS2025/medical_imaging/llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](../../NeurIPS2025/medical_imaging/position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] Scaling Laws and Pathologies of Single-Layer PINNs: Network Width and PDE Nonlinearity](../../NeurIPS2025/medical_imaging/scaling_laws_and_pathologies_of_single-layer_pinns_network_width_and_pde_nonline.md)
- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/medical_imaging/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)

</div>

<!-- RELATED:END -->
