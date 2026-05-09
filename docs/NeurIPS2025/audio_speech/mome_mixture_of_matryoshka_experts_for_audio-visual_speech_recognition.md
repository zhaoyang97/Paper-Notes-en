---
title: >-
  [Paper Note] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition
description: >-
  [NeurIPS 2025][Audio & Speech][audio-visual speech recognition] MoME integrates sparse MoE into the Matryoshka representation learning framework for LLM-based audio-visual speech recognition. Through a shared router, it enables cross-granularity knowledge transfer, supporting elastic inference at multiple compression rates under a single set of model weights, while achieving state-of-the-art performance on AVSR/ASR/VSR.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - audio-visual speech recognition
  - Matryoshka representation learning
  - Mixture-of-Experts
  - elastic inference
  - LLM
date: 2026-05-08
content_hash: 03a7e9f39a233de2
---

# MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition

**Conference**: NeurIPS 2025  
**arXiv**: [2510.04136](https://arxiv.org/abs/2510.04136)  
**Code**: N/A  
**Area**: Audio & Speech  
**Keywords**: audio-visual speech recognition, Matryoshka representation learning, Mixture-of-Experts, elastic inference, LLM

## TL;DR

MoME integrates sparse MoE into the Matryoshka representation learning framework for LLM-based audio-visual speech recognition. Through a shared router, it enables cross-granularity knowledge transfer, supporting elastic inference at multiple compression rates under a single set of model weights, while achieving state-of-the-art performance on AVSR/ASR/VSR.

## Background & Motivation

LLM-based AVSR faces a fundamental tension: **token hunger vs. computational cost**. The temporal resolution of audio-visual speech signals far exceeds that of text, resulting in an enormous number of input tokens. Existing token compression methods (concatenation, resampling, average pooling, etc.) require a fixed compression rate to be specified in advance, producing fixed-length output sequences that cannot dynamically balance accuracy and efficiency at inference time.

Matryoshka Representation Learning (MRL) trains a single model with multiple compression rates, enabling dynamic granularity adjustment at inference. However, existing Matryoshka approaches suffer from two major shortcomings:

**Independent training per granularity** — each resolution is treated as an independent problem, lacking cross-scale interaction, with severe information loss at high compression rates.

**Uniform monolithic representations** — all scales share the same monolithic network architecture, precluding specialization for different granularities.

The core idea of MoME is to leverage sparse MoE experts to achieve cross-granularity knowledge transfer: the same set of experts is activated similarly across different compression rates, allowing low-resolution sequences to reuse expert pathways trained on high-resolution sequences.

## Method

### Overall Architecture

Input audio/video → pretrained encoders (Whisper/AV-HuBERT) → Matryoshka token sequences at multiple compression rates → frozen LLM (Llama 3) with parallel MoME modules → autoregressive decoding to transcription text. All $G \times L$ audio-visual compression rate combinations are trained jointly; any compression rate can be selected at inference time.

### Key Designs

1. **MoME module structure**: Each MoME module contains $N_r$ routed experts and $N_s$ shared experts. Each expert follows a bottleneck design (linear down-projection → GELU → linear up-projection), with the bottleneck dimension compressible to as small as 1. The router is a linear layer that selects $K$ routed experts via top-k sparse gating:
    $\text{MoME}(\mathbf{H}_l^{ij}) = \sum_{n=1}^{N_s} E_n(\mathbf{H}_l^{ij}) + \sum_{n=N_s+1}^{N_s+N_r} g_n E_n(\mathbf{H}_l^{ij})$
   where $g_n$ is determined by top-k sparse gating.

2. **Shared router and cross-granularity alignment**: The central innovation lies in sharing the experts and router of the MoME module across all Matryoshka sequences. This means the router simultaneously processes high-resolution (information-rich) and low-resolution (heavily compressed) inputs during training, naturally learning to activate similar expert subsets at different granularities. Experiments (Figure 5) confirm this implicit alignment — expert activation distributions within the same layer are highly consistent across compression rates, while activation patterns differ significantly across layers, achieving layer-wise diversity.

3. **Shared experts**: Inspired by DeepSeekMoE and Llama 4, one or two always-active shared experts are introduced to capture global, cross-modal, scale-invariant knowledge. Ablation studies confirm that shared experts yield measurable WER improvements.

4. **Flexible insertion positions**: MoME modules can be inserted in parallel at three positions within each LLM layer: the MHSA module, the FFN module, or the entire Transformer layer. The LLM backbone is frozen; only the MoME modules are trained (parameter-efficient fine-tuning).

### Loss & Training

Multi-granularity average cross-entropy loss:
$$\mathcal{L}_{LM} = -\frac{1}{G \cdot L} \sum_{i=1}^{G}\sum_{j=1}^{L} \log p(\mathbf{Y}|\mathbf{Z}^{ij}) \cdot c_{ij}$$
where $c_{ij}=1$ denotes equal weighting across granularities. A load balancing loss $\mathcal{L}_B$ (coefficient 0.01) is added to prevent routing collapse. Training compression rates are audio $\{4, 16\}$ and video $\{2, 5\}$, yielding four combinations.

## Key Experimental Results

### Main Results (AVSR, WER%↓)

| Method | Active Params | LRS2 (4,2) | (4,5) | (16,2) | (16,5) | LRS3 (4,2) | (4,5) | (16,2) | (16,5) |
|--------|--------------|-----------|-------|--------|--------|-----------|-------|--------|--------|
| Llama-AVSR (independent) | 27.5M | 4.1 | 4.5 | 5.3 | 8.1 | 2.4 | 2.8 | 3.3 | 4.1 |
| Llama-MTSK SS | 27.5M | 3.4 | 4.7 | 4.8 | 6.4 | 2.3 | 2.2 | 3.3 | 3.6 |
| Llama-MTSK MSS | 55.0M | 3.6 | 4.8 | 6.1 | 9.0 | 2.4 | 2.4 | 3.2 | 3.5 |
| MoME-23/4-MHSA | **12.7M** | **2.9** | **3.0** | **4.2** | **4.3** | **1.8** | **1.7** | **2.9** | **2.9** |
| MoME-23/4-LAYER | **12.7M** | **2.7** | **2.7** | **4.2** | **4.2** | **1.5** | **1.8** | **3.1** | **3.2** |

MoME outperforms all baselines across all compression rates while using 2–4× fewer active parameters.

### Ablation Study (MoME-MHSA on LRS2)

| Routed Experts | Shared Experts | Bottleneck Size | Top-k | (4,2) | (4,5) | (16,2) | (16,5) |
|---------------|---------------|----------------|-------|-------|-------|--------|--------|
| 1 | 0 | 48 | / | 3.4 | 3.4 | 4.9 | 5.1 |
| 4 | 0 | 24 | 2 | 3.3 | 3.3 | 4.8 | 5.0 |
| 4 | **1** | 24 | 2 | **3.2** | **3.2** | **4.4** | **4.7** |
| 23 | 1 | 12 | 4 | **2.9** | **3.0** | **4.2** | **4.3** |
| 23 | 2 | 12 | 4 | 2.8 | 3.0 | 4.1 | 4.7 |

### Key Findings

- **Noise robustness**: At SNR = −5 dB, MoME (32.6% WER) substantially outperforms Llama-AVSR (41.8%) and Llama-MTSK (44.9%).
- **Extreme compression**: With bottleneck dimension reduced to 1 (0.9M active parameters), WER degrades only marginally (LRS3: 1.8 → 2.0).
- **Cross-modal token analysis** (Figure 4): Audio-visual tokens at different compression rates exhibit strong linear correlation; high-compression tokens approximately correspond to 2–3 low-compression tokens.
- **Computational efficiency**: At (16, 5) compression, TFLOPs are reduced by 8×, and inference time decreases from 12.75 s to 6.74 s on 23-second speech.

## Highlights & Insights

- The first framework to unify MoE and Matryoshka representation learning, cleverly leveraging sparse experts for cross-granularity knowledge transfer.
- The shared router design allows cross-scale alignment to emerge naturally as an implicit property rather than an explicit constraint.
- The analogy to the *shallow brain hypothesis* — deep LLM backbone plus parallel shallow MoME modules — is an insightful conceptual framing.
- A single set of model weights supporting elastic inference is a deployment-friendly design well suited for on-device applications.

## Limitations & Future Work

- Validation is limited to English speech recognition; generalizability to multilingual and multi-task settings remains unexplored.
- The optimal MoME insertion position (MHSA/FFN/LAYER) varies by dataset, and no automatic selection mechanism is proposed.
- No comparison with the adaptive compression strategy (speech-rate-based) of MMS-LLaMA in terms of flexibility.
- Performance degrades when the number of shared experts exceeds 2; the underlying cause is not thoroughly analyzed.

## Related Work & Insights

- **Key distinction from Llama-MTSK**: The latter employs multi-scale LoRA but treats each scale independently, whereas MoME achieves cross-scale coupling through a shared router.
- The shared expert design from DeepSeekMoE proves effective in the multimodal Matryoshka setting.
- **Inspiration**: The MoME framework is generalizable to other multimodal tasks such as vision-language, requiring only replacement of the encoder and compression strategy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first unification of MoE and MRL; elegant design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (LRS2 + LRS3, three tasks, detailed ablations and visualizations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rich experimental figures and tables)
- Value: ⭐⭐⭐⭐⭐ (elastic inference + SOTA performance; high value for on-device deployment)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](../../ICCV2025/audio_speech/vggsounder_audio-visual_evaluations_for_foundation_models.md)

<!-- RELATED:END -->
