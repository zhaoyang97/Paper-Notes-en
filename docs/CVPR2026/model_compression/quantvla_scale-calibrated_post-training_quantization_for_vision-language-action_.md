---
title: >-
  [Paper Note] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models
description: >-
  [CVPR 2026][Model Compression] This paper proposes QuantVLA, the first training-free post-training quantization (PTQ) framework for Vision-Language-Action (VLA) models. Through a selective quantization layout and two lightweight calibration mechanisms—Attention Temperature Matching (ATM) and Output Head Balancing (OHB)—QuantVLA achieves approximately 70% memory reduction at W4A8 precision while surpassing the task success rate of the full-precision baseline.
tags:
  - CVPR 2026
  - Model Compression
date: 2026-05-08
content_hash: 34a45094359de202
---

# QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models

**Conference**: CVPR 2026
**arXiv**: [2602.20309](https://arxiv.org/abs/2602.20309)
**Code**: None
**Area**: Model Compression

## TL;DR

This paper proposes QuantVLA, the first training-free post-training quantization (PTQ) framework for Vision-Language-Action (VLA) models. Through a selective quantization layout and two lightweight calibration mechanisms—Attention Temperature Matching (ATM) and Output Head Balancing (OHB)—QuantVLA achieves approximately 70% memory reduction at W4A8 precision while surpassing the task success rate of the full-precision baseline.

## Background & Motivation

1. **Deployment Bottleneck of VLA Models**: VLA models (e.g., π0.5, GR00T N1.5) unify visual perception, language understanding, and action generation, but as model scale increases, their computational and memory demands grow drastically, severely hindering deployment on embedded robotic platforms.
2. **Blind Spots of Existing Efficiency Methods**: Approaches such as EfficientVLA, VLA-Cache, and MoLe-VLA primarily optimize the visual encoder or language layers (via pruning, caching, or routing), but almost none directly quantize the Diffusion Transformer (DiT) action head—which is the primary contributor to computation and memory.
3. **Inadequacy of General PTQ Methods**: Post-training quantization methods designed for LLMs/VLMs, such as SmoothQuant and DuQuant, cannot handle the heterogeneous activation distributions arising from tight multimodal coupling in VLA models. Direct application causes severe performance degradation (e.g., DuQuant drops success rate from 97.1% to 76.3% on π0.5).
4. **Fragility of the DiT Action Head**: Quantization-induced scale drift alters the effective temperature of attention logits and the energy of residual streams; these systematic shifts accumulate across deep DiT layers through residual connections and LayerNorm, destabilizing action generation.
5. **First Systematic Analysis**: This paper presents the first systematic theoretical analysis of quantization sensitivity in VLA models, identifying two failure modes of cross-module drift (temperature shift and energy drift), and proposes targeted solutions accordingly.

## Method

### Preliminaries: Diffusion-Based VLA Models

A VLA system comprises three components: (1) a visual encoder (e.g., SigLIP2, DINOv2) that encodes RGB frames into image tokens; (2) a language backbone that encodes text instructions into text tokens; and (3) a DiT action head that iteratively denoises action latents conditioned on fused visual-language features $F_{\text{VL}}$, robot proprioception, and diffusion timestep $t$:

$$x_{t-1} = f_\theta(x_t, F_{\text{VL}}, t)$$

After $T$ denoising steps, the final $x_0$ is decoded into executable actions.

### Quantization Sensitivity Analysis

#### DuQuant Reparameterization

QuantVLA adopts DuQuant's invertible reparameterization as its backbone: for each linear layer, channel-wise smoothing (diagonal matrix $\Lambda$), block orthogonal rotations $\hat{R}_{(1)}, \hat{R}_{(2)}$, and zigzag channel permutations are applied to redistribute activation outliers.

#### Two Failure Modes

Through first-order error propagation analysis, the paper identifies two systematic drifts introduced by quantization:

**Temperature Drift**: Quantization error $\varepsilon_{\text{up}}$ propagates into Q and K, altering the variance of attention logits and effectively shifting the softmax temperature, causing the attention distribution to deviate from the teacher model:

$$\Delta L \approx \frac{1}{\sqrt{d}} \left( (\varepsilon_{\text{up}} W_q) K_T^\top + Q_T (\varepsilon_{\text{up}} W_k)^\top \right) + \Delta L_{\text{local}}$$

**Energy Drift**: After multi-head concatenation and output projection, the magnitude of attention outputs shifts systematically, altering the residual injection gain and the operating point of LayerNorm:

$$\Delta O \approx J_{\text{softmax}}(L_T) \Delta L \, V_T W_{o,T} + A_T \varepsilon_{\text{up}} W_v W_{o,T} + A_T V_T \delta W_o + \Delta O_{\text{local}}$$

### QuantVLA Framework

QuantVLA consists of three core components:

#### Component 1: Selective Quantization Layout

- **LLM**: All linear layers are quantized to W4A8 (4-bit weights, 8-bit activations).
- **DiT Action Head**: Only MLP layers are quantized; attention projections $W_q, W_k, W_v, W_o$ **remain in floating point**.
- **Design Rationale**: Attention projections are most sensitive to upstream distribution shifts and directly determine the stability of the softmax distribution and residual injection gain. Experiments show that quantizing all DiT layers causes a catastrophic drop in success rate (π0.5: from 97.1% to 71.6%), whereas quantizing only MLP layers maintains 95.4%.

#### Component 2: Attention Temperature Matching (ATM)

A per-head scalar $\alpha$ is used to align the logit distributions of the teacher and quantized models:

$$\alpha_{\text{raw}} = \frac{\text{Std}(L_T)}{\text{Std}(L_Q) + 10^{-6}}$$

$$\alpha = \text{clip}(\alpha_{\text{raw}}, \alpha_{\min}, \alpha_{\max})$$

The corrected quantized logits are $L_Q = L_T / \alpha$. After calibration, $\alpha$ is folded into the dequantization scaling factor, incurring zero additional inference overhead.

#### Component 3: Output Head Balancing (OHB)

A per-layer scalar $\beta$ matches the energy after output projection:

$$\beta_{\text{raw}}(l) = \frac{\text{RMS}(Z_{T,l})}{\text{RMS}(Z_{Q,l}) + 10^{-6}}$$

$$\beta(l) = \text{clip}(\beta_{\text{raw}}(l), \beta_{\min}, \beta_{\max})$$

After correction, $Z_Q = Z_l / \beta(l)$, restoring the residual stream injection gain and the LayerNorm operating point.

Both calibration scalars use a neutral band $\varepsilon = 0.03$ to filter negligible differences (set to 1 if $|\log \alpha| < \varepsilon$), with a clipping range of $\pm 0.4$. The entire process requires only a small amount of unlabeled calibration data and no retraining.

## Key Experimental Results

Evaluation is conducted on the LIBERO simulator across four task suites: Spatial (spatial relation reasoning), Object (object manipulation), Goal (instruction-goal alignment), and Long (long-horizon decomposition and error accumulation control).

### Table 1: Ablation on Selective Quantization Layout (without ATM/OHB)

| Model | Precision | Quantization Scope | Spatial | Object | Goal | Long | Avg | Memory (GB) |
|---|---|---|---|---|---|---|---|---|
| π0.5 | FP16 | None | 98.5% | 99.0% | 97.5% | 93.5% | 97.1% | 4.27 |
| π0.5 | W4A8 | LLM only | 98.0% | 98.5% | 97.5% | 92.0% | 96.5% | 1.58 |
| π0.5 | W4A8 | DiT only | 81.5% | 94.5% | 71.5% | 39.0% | 71.6% | 3.85 |
| π0.5 | W4A8 | LLM+DiT (all) | 86.0% | 97.5% | 71.5% | 50.0% | 76.3% | 1.17 |
| π0.5 | W4A8 | LLM+DiT (MLP) | 98.0% | 97.0% | 94.5% | 92.0% | 95.4% | 1.28 |
| GR00T N1.5 | FP16 | None | 92.0% | 92.0% | 86.0% | 76.0% | 86.5% | 2.02 |
| GR00T N1.5 | W4A8 | LLM+DiT (MLP) | 90.0% | 86.0% | 80.0% | 74.0% | 82.5% | 0.91 |

Key finding: Quantizing all DiT layers (including attention projections) causes catastrophic degradation (Long task: 93.5% → 39.0%), while quantizing MLP layers only stays close to the baseline.

### Table 2: Full QuantVLA Results

| Model | Method | Precision | Spatial | Object | Goal | Long | Avg | Memory (GB) | Memory Saved |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 | FP16 Baseline | FP16 | 98.5% | 99.0% | 97.5% | 93.5% | 97.1% | 4.27 | - |
| π0.5 | DuQuant (LLM+DiT) | W4A8 | 86.0% | 97.5% | 71.5% | 50.0% | 76.3% | 1.17 | 72.6% |
| π0.5 | QuantVLA (LLM) | W4A8 | 98.5% | 99.0% | 96.5% | 96.5% | 97.6% | 1.58 | 63.0% |
| π0.5 | **QuantVLA** | **W4A8** | **98.5%** | **98.0%** | **98.0%** | **96.0%** | **97.6%** | **1.28** | **70.0%** |
| GR00T N1.5 | FP16 Baseline | FP16 | 92.0% | 92.0% | 86.0% | 76.0% | 86.5% | 2.02 | - |
| GR00T N1.5 | DuQuant (LLM+DiT) | W4A8 | 66.0% | 70.0% | 68.0% | 76.0% | 70.0% | 0.74 | 63.4% |
| GR00T N1.5 | **QuantVLA** | **W4A8** | **96.0%** | **92.0%** | **90.0%** | **74.0%** | **88.0%** | **0.91** | **55.0%** |

On π0.5, QuantVLA achieves a 97.6% average success rate at W4A8 precision, **exceeding** the FP16 baseline of 97.1%, while reducing memory from 4.27 GB to 1.28 GB (70% savings). On GR00T N1.5, it similarly surpasses the baseline (88.0% vs. 86.5%) with 55% memory savings.

### Table 3: Different Quantization Precisions and Denoising Steps

| Setting | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| π0.5 FP16 | 98.5% | 99.0% | 97.5% | 93.5% | 97.1% |
| π0.5 W4A8 | 98.5% | 98.0% | 98.0% | 96.0% | 97.6% |
| π0.5 W4A4 | 98.5% | 98.5% | 93.5% | 90.5% | 95.3% |
| GR00T N1.5 8 steps | 96.0% | 92.0% | 90.0% | 74.0% | 88.0% |
| GR00T N1.5 16 steps | 96.0% | 94.0% | 84.0% | 80.0% | 88.5% |

Even under the more aggressive W4A4 precision, a 95.3% average success rate is maintained, demonstrating strong robustness.

## Highlights & Insights

- **First VLA PTQ Framework**: This is the first work to successfully apply post-training quantization to VLA models including the DiT action head, filling a critical gap in the field.
- **Theory-Driven Design**: First-order error propagation analysis rigorously identifies temperature drift and energy drift as the two primary failure mechanisms; the designs of ATM and OHB are grounded in this theoretical foundation rather than empirical tuning.
- **Quantized Model Surpasses Baseline**: π0.5 achieves 97.6% at W4A8 versus 97.1% at FP16; a similar trend is observed on GR00T N1.5—suggesting that quantization introduces a regularization effect.
- **Completely Training-Free**: Only a small amount of unlabeled calibration data is required. ATM and OHB scalars are folded into dequantization factors, incurring zero additional inference overhead while preserving the original architecture and operator scheduling.
- **Generality**: Validated across two representative VLA models (π0.5 and GR00T N1.5) under different precision settings (W4A8, W4A4) and denoising step configurations.

## Limitations & Future Work

- **Limited Evaluation Scenarios**: Experiments are conducted only on the LIBERO simulator and the Simpler benchmark; real-robot deployment has not been validated, and the sim-to-real transfer remains unknown.
- **DiT Attention Layers Not Quantized**: The selective layout keeps DiT attention projections in floating point, limiting further memory compression potential; how to quantize these layers without performance loss remains an open problem.
- **Calibration Data Dependency**: Although the calibration dataset is small and unlabeled, whether its distribution affects generalization has not been thoroughly analyzed.

## Rating

- ⭐⭐⭐⭐ **Novelty**: First successful application of PTQ to VLA models and the DiT action head, with clear theoretical analysis of failure mechanisms.
- ⭐⭐⭐⭐ **Practical Value**: Training-free, zero inference overhead, 70% memory savings—highly valuable for robotic deployment.
- ⭐⭐⭐ **Experimental Thoroughness**: Ablations across two models, multiple precisions, and configurations are reasonably complete, but real-robot validation is absent.
- ⭐⭐⭐⭐ **Writing Quality**: Theoretical analysis is rigorous; the logical chain from sensitivity analysis to method design is complete and clear.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[CVPR 2026\] LLaVA-LE: Large Language-and-Vision Assistant for Lunar Exploration](llava-le_large_language-and-vision_assistant_for_lunar_exploration.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)

<!-- RELATED:END -->
