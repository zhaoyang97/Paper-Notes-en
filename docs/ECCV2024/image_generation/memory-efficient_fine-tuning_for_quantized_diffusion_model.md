---
title: >-
  [Paper Note] Memory-Efficient Fine-Tuning for Quantized Diffusion Model
description: >-
  [ECCV 2024][Image Generation] Proposes TuneQDM, the first memory-efficient fine-tuning method for quantized diffusion models. By introducing multi-channel quantization scale updates and a timestep-aware scale strategy, it achieves personalized generation quality on a 4-bit quantized model close to that of the full-precision counterpart.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 849140ec22b2a1fb
---

# Memory-Efficient Fine-Tuning for Quantized Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2401.04339](https://arxiv.org/abs/2401.04339)  
**Area**: Image Generation

## TL;DR

Proposes TuneQDM, the first memory-efficient fine-tuning method for quantized diffusion models. By introducing multi-channel quantization scale updates and a timestep-aware scale strategy, it achieves personalized generation quality on a 4-bit quantized model close to that of the full-precision counterpart.

## Background & Motivation

As the size of diffusion models continues to grow (e.g., 2.6B parameters for SDXL, 5.5B parameters for DALL-E 2), directly fine-tuning full-precision models requires massive memory and computing resources. Although model quantization (such as Q-Diffusion) can compress parameters to 4-bit/8-bit, whether the quantized models can be directly fine-tuned for downstream tasks remains an unexplored question.

The authors construct a baseline scheme (Q-Diffusion quantization + PEQA fine-tuning + DreamBooth personalization) and identify two core issues:

**Loss of Inter-Channel Patterns (P1)**: During full-precision fine-tuning, weight updates exhibit distinct inter-channel patterns. However, the baseline only updates the intra-channel scale factors, failing to capture these patterns.

**Insufficient Timestep-Awareness (P2)**: Diffusion models play different roles at different denoising steps (content features vs. coarse-grained features vs. noise removal). However, the baseline lacks the capability to differentiate between various timesteps, resulting in either high subject fidelity but neglected text prompts, or a failure to satisfy either.

## Method

### Overall Architecture

While freezing the quantized integer weights, TuneQDM only trains the quantization scale parameters. It comprises two core techniques:
1. Multi-Channel-wise Scale Update (MCSU)
2. Timestep-Aware Scale Update (TAS)

### Key Designs

**1. Multi-Channel-wise Scale Update (MCSU)**

The weights in standard quantization are reconstructed as $\hat{W}_f = s \cdot (W_q - z)$. The PEQA baseline only updates the per-channel output scale factor $s_{out} \in \mathbb{R}^m$.

MCSU introduces an additional input-direction scale factor $s_{in} \in \mathbb{R}^n$, representing the quantized weights as a separable function:

$$W_{tuned} = (s_{out} + \Delta s_{out}) \cdot (W_q^* - z^*) \cdot (s_{in} + \Delta s_{in})$$

With only $(m+n)$ trainable parameters (vs. the full matrix size of $m \times n$), it successfully captures inter-channel weight update patterns.

**2. Timestep-Aware Scale Update (TAS)**

The total timesteps $T$ are uniformly divided into $n$ intervals, with a dedicated set of quantization scale parameters maintained independently for each interval:

$$S_n = \{s_1, s_2, ..., s_n\}, \quad \mathcal{I}_n = \{I_i | (\frac{i \times T}{n}, \frac{(i+1) \times T}{n})\}$$

During training, the scale parameters of the corresponding expert are selected for scale updates based on the current timestep. During inference, the integer weights remain fixed, and only the scale parameters are switched, resulting in minimal memory overhead.

**3. Overall Pipeline**

1. Initialize weights and quantization parameters from the Q-Diffusion quantized checkpoint.
2. Initialize $s_{in} \sim \mathcal{N}(1, 0.01)$ for each layer.
3. Freeze $W_q$ and only train $s_{out}$ and $s_{in}$.
4. Select the corresponding expert based on the timestep for updates during training.

### Loss & Training

Standard diffusion training loss from DreamBooth is utilized for personalized fine-tuning.

## Key Experimental Results

### Main Results

**Single-Subject Generation**

| Method | Bit-width (W) | Model Size | Parameter Count | DINO-I ↑ | CLIP-I ↑ | CLIP-T ↑ |
|------|---------|---------|--------|----------|----------|----------|
| Full-Precision | 32 | 3.20GB | 859M | 0.431 | 0.746 | 0.316 |
| Baseline | 4 | 0.40GB | 0.33M | 0.519 | 0.787 | 0.313 |
| **TuneQDM** | **4** | **0.40GB** | **0.62M** | **0.551 (+6.2%)** | **0.802 (+1.9%)** | **0.306** |
| Baseline | 8 | 0.80GB | 0.33M | 0.581 | 0.824 | 0.300 |
| **TuneQDM** | **8** | **0.80GB** | **0.62M** | **0.578** | **0.816** | **0.307 (+2.3%)** |

Under the 4-bit setting, TuneQDM improves DINO-I by 6.2%, achieving performance close to full-precision using an 8x compressed model.

**Unconditional Generation (CIFAR-10)**

| Method | Bit-width | Parameter Count | IS ↑ | FID ↓ |
|------|------|--------|------|-------|
| Full-Precision | 32 | 35.8M | 9.00 | 4.53 |
| QLoRA (r=32) | 8 | 8.64M | 9.03 | 4.30 |
| QLoRA (r=2) | 8 | 0.57M | 9.03 | 4.15 |
| Baseline | 8 | 0.03M | 8.96 | 4.39 |
| **TuneQDM** | **8** | **0.13M** | **9.17** | **3.80** |

TuneQDM outperforms QLoRA (r=32) with 8.64M parameters using only 0.13M parameters, reducing the FID from 4.30 to 3.80.

### Ablation Study

| Configuration | Bit-width | MCSU | No. of TAS Experts | IS ↑ | FID ↓ |
|------|------|------|-----------|------|-------|
| Full-Precision | 32 | - | - | 9.00 | 4.53 |
| Baseline | 8 | ✗ | 1 | 8.96 | 4.39 |
| + TAS only | 8 | ✗ | 2 | 9.19 | 4.24 |
| + MCSU only | 8 | ✓ | 1 | 8.97 | 4.33 |
| **TuneQDM** | **8** | **✓** | **2** | **9.17** | **3.80** |
| TuneQDM | 8 | ✓ | 4 | 9.02 | 4.15 |

MCSU + 2 expert TAS is the optimal configuration; 4 experts lead to overfitting and degraded performance.

### Key Findings

- Weight variation rate plots for full-precision fine-tuning reveal clear inter-channel patterns, which are completely lost in the baseline.
- Using 2 timestep experts is the optimal choice, which aligns with studies like P2 weighting.
- Only storing the scale parameters (~3MB/dataset) allows switching between different tasks by reusing the same quantized checkpoint.
- User studies indicate that TuneQDM significantly outperforms the baseline in prompt fidelity.
- A performance gap still exists in multi-subject generation compared to full-precision models, which remains a bottleneck for the current method.

## Highlights & Insights

1. **Pioneering Problem Definition**: This work is the first to study the fine-tuning of quantized diffusion models, holding significant relevance for industrial deployment.
2. **Deep Insight into Separable Scale Factors**: Inter-channel patterns are identified by visualizing weight variation ratios, and elegantly addressed using the outer product structure $s_{out} \cdot W_q \cdot s_{in}$.
3. **Extreme Parameter Efficiency**: Only 0.62M trainable parameters compared to the 859M full model, yielding a compression ratio of over 1000x.
4. **Plug-and-Play**: Once quantized, merely switching scale parameters allows adaptation to different downstream tasks without needing re-quantization.

## Limitations & Future Work

- Validated only on Stable Diffusion v1.5; the effectiveness on larger models such as SDXL remains unknown.
- Notable performance degradation still persists in multi-subject generation.
- Only fine-tuning after PTQ (Post-Training Quantization) is supported; QAT (Quantization-Aware Training) scenarios are not explored.
- Under the 4-bit setting, CLIP-T (text fidelity) declines slightly, exhibiting a trade-off between subject and prompt fidelity.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to define and solve the fine-tuning problem for quantized diffusion models, with a pioneering problem formulation.
- **Technical Depth**: ⭐⭐⭐⭐ — The separable scale design is elegant, and the timestep-aware strategy is theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes single/multi-subject + unconditional generation + user studies + ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation analysis (via weight variation visualization) is highly intuitive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [\[ECCV 2024\] ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)

</div>

<!-- RELATED:END -->
