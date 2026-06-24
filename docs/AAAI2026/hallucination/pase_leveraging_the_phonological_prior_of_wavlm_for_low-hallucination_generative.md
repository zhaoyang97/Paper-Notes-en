---
title: >-
  [Paper Note] PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement
description: >-
  [AAAI 2026][Hallucination Detection][Speech Enhancement] This paper proposes PASE, a framework that leverages robust phonological priors embedded in pretrained WavLM via Denoising Representation Distillation (DRD) to suppress linguistic hallucinations, while employing a dual-stream representation (high-level phonetic + low-level acoustic) to eliminate acoustic hallucinations, simultaneously achieving state-of-the-art performance in both perceptual quality and content fidelity…
tags:
  - "AAAI 2026"
  - "Hallucination Detection"
  - "Speech Enhancement"
  - "Hallucination Suppression"
  - "WavLM"
  - "Phonological Prior"
  - "Representation Distillation"
date: 2026-05-08
content_hash: f686dbcb246be3b0
---

# PASE: Leveraging the Phonological Prior of WavLM for Low-Hallucination Generative Speech Enhancement

**Conference**: AAAI 2026
**arXiv**: [2511.13300](https://arxiv.org/abs/2511.13300)  
**Code**: [https://github.com/cisco-open/pase](https://github.com/cisco-open/pase)  
**Area**: Hallucination Detection
**Keywords**: Speech Enhancement, Hallucination Suppression, WavLM, Phonological Prior, Representation Distillation

## TL;DR

This paper proposes PASE, a framework that leverages robust phonological priors embedded in pretrained WavLM via Denoising Representation Distillation (DRD) to suppress linguistic hallucinations, while employing a dual-stream representation (high-level phonetic + low-level acoustic) to eliminate acoustic hallucinations, simultaneously achieving state-of-the-art performance in both perceptual quality and content fidelity.

## Background & Motivation

### Hallucination in Generative Speech Enhancement

Speech enhancement (SE) aims to recover clean speech from noisy mixed signals. Generative models (GANs, diffusion, flow matching, language models) surpass conventional discriminative approaches in perceptual quality, yet suffer from a largely overlooked problem—**hallucination**: enhanced speech that is inconsistent with the original signal in linguistic content or speaker characteristics.

### Taxonomy of Two Hallucination Types

The authors are the first to **systematically categorize** hallucinations in speech enhancement into two types:

**Linguistic Hallucination**: Enhanced speech contains incorrect spoken content, arising from the model's **inability to constrain valid phonological structures**. This is the more fundamental challenge.

**Acoustic Hallucination**: Enhanced speech exhibits inconsistent speaker characteristics, arising from **the loss of fine-grained acoustic details**. This can be mitigated by supplementing acoustic cues.

### Fundamental Limitations of Prior Work

Both major paradigms suffer from critical flaws:

**Continuous Representation Mapping (CRM)**:
- Treats S3M representations as simple feature vector sequences, ignoring the contextual structure underlying their pseudo-linguistic properties
- Performance gains stem primarily from the strength of the representations themselves rather than from effective exploitation of their internal structure

**Discrete Language Modeling (DLM)**:
- Discretizes S3M representations into token sequences and models them autoregressively
- **Prior contamination risk**: Learning from noise-corrupted representations tends to produce linguistically inconsistent outputs
- Discretization **discards critical acoustic information** (pitch, timbre), precluding mitigation of acoustic hallucinations
- Noise differs from true masking—noise acts as a "soft mask" that merely distorts information, allowing the model to bypass contextual knowledge by reconstructing from local cues alone

### Core Argument

The authors introduce the concept of **Phonological Prior**: S3M models (e.g., WavLM) do not truly understand linguistic semantics; rather, they construct pseudo-linguistic properties that simulate language comprehension by learning statistical co-occurrence patterns of phonetic structures from large-scale audio data. This prior should be **directly leveraged** rather than **relearned from corrupted inputs**.

## Method

### Overall Architecture

PASE (Phonologically Anchored Speech Enhancer) consists of two core components:

1. **DeWavLM** (Denoising WavLM): WavLM fine-tuned via Denoising Representation Distillation to serve as a denoising expert
2. **Dual-Stream Vocoder**: Reconstructs enhanced waveforms from DeWavLM's high-level phonetic representations and low-level acoustic representations

### Key Designs

#### 1. **Denoising Representation Distillation (DRD)**

Core Idea: Rather than learning phonological priors from noisy inputs (which risks prior contamination), directly leverage the robust priors already present in pretrained WavLM.

**Method**:
- Instantiate two WavLM copies: a **frozen teacher** and a **trainable student**, both initialized from pretrained weights
- The student learns to map noisy input waveforms to clean representations:
    - Input: noisy speech waveform
    - Target: final-layer output produced by the teacher from the corresponding clean speech
    - Loss: MSE loss

**Key Findings**:
- Using the final layers of both student and teacher (L24→L24) yields optimal performance
- Using only the KD loss (without the original masked prediction loss) is most effective—the pure KD objective provides strong regularization, effectively preventing knowledge degradation and catastrophic forgetting
- Joint SSL+KD objectives are suboptimal, as SSL loss induces representation drift

**Catastrophic Forgetting**: The new denoising objective may overwrite the model's original phonological priors; however, experiments show:
- PNMI (phoneme discriminability) remains stable across all fine-tuning objectives
- The KD loss acts as a regularizer that pulls representations back toward the original manifold, achieving an RFS of 0.98

#### 2. **Dual-Stream Acoustic Conditioning Reconstruction**

Core Idea: Two complementary layers are selected as vocoder inputs—

- **Phonetic Representation**: Final Transformer layer output, rich in abstract, context-dependent phonetic content → ensures linguistic integrity
- **Acoustic Representation**: First Transformer layer output, preserving fine-grained acoustic details → retains speaker identity and prosody

The two representations are fused via simple **element-wise addition** (after linear projection), which achieves optimal performance—since the two representations are largely orthogonal, addition suffices for efficient fusion, and more complex strategies yield no additional benefit.

#### 3. **Vocoder Architecture**

- Backbone: Enhanced Vocos with integrated attention modules for improved contextual modeling
- Adversarial training: Multi-Period Discriminator (MPD) + Multi-Band Multi-Scale STFT Discriminator (MBMSD), with equal adversarial loss weights
- Loss weights: Reconstruction : Adversarial : Feature Matching = 15 : 2 : 1

### Loss & Training

- **DeWavLM training**: 100K steps, batch size 4, learning rate 1e-4, AdamW + cosine decay
- **Vocoder training**: 200K steps, batch size 12, learning rate 2e-4, DeWavLM frozen during training
- Hardware: 4 × NVIDIA RTX 4090
- Training data: ~2,000 hours of clean speech, SNR range $[-5, 15]$ dB

## Key Experimental Results

### Main Results

Comparison on the simulated LibriTTS test set:

| Model | Params (M) | DNSMOS↑ | UTMOS↑ | SBS↑ | LPS↑ | SpkSim↑ | WER (%)↓ |
|-------|-----------|---------|--------|------|------|---------|---------|
| Noisy | - | 1.33 | 1.44 | 0.62 | 0.63 | 0.77 | 14.35 |
| TF-GridNet | 2.77 | 3.04 | 2.62 | 0.85 | 0.90 | 0.80 | 9.93 |
| StoRM | 55.12 | 3.07 | 2.55 | 0.68 | 0.65 | 0.63 | 45.94 |
| LLaSE-G1 | 1895.63 | 3.16 | 3.17 | 0.74 | 0.71 | 0.42 | 36.58 |
| AES-V2 | - | 3.35 | 4.09 | 0.79 | 0.85 | 0.60 | 21.32 |
| **PASE (ours)** | **382.14** | **3.12** | **3.09** | **0.90** | **0.93** | **0.80** | **7.49** |

PASE achieves the lowest WER (7.49%) and highest SpkSim/LPS/SBS simultaneously at the lowest computational cost (21.42 G MACs/s), striking a comprehensive balance between perceptual quality and content fidelity.

### Ablation Study

Ablation on DRD objective functions:

| Configuration | DNSMOS↑ | UTMOS↑ | SpkSim↑ | WER (%)↓ | Notes |
|--------------|---------|--------|---------|---------|-------|
| w/o DRD | 1.55 | 1.39 | 0.46 | 32.33 | No fine-tuning baseline |
| SSL only | 2.64 | 1.96 | 0.39 | 15.38 | Masked prediction loss only |
| KD only | **3.26** | **3.42** | **0.57** | **7.62** | MSE distillation loss only |
| SSL+KD | 3.07 | 2.95 | 0.52 | 8.78 | Joint loss |

Investigation of phonological prior sources:

| DeWavLM Variant | DNSMOS↑ | WER (%)↓ | Notes |
|----------------|---------|---------|-------|
| Base (960h pretrained) | 3.32 | 15.49 | Small model + less data |
| Base+ (94Kh pretrained) | 3.30 | 13.34 | Small model + more data |
| Large (94Kh pretrained) | 3.26 | **7.62** | Large model + more data |
| Base-FS (trained from scratch) | 3.33 | 36.16 | Small model, no pretraining |
| Large-FS (trained from scratch) | 3.24 | 38.62 | Large model, no pretraining |

Ablation on acoustic conditioning schemes:

| Scheme | SpkSim↑ | WER (%)↓ | Notes |
|--------|---------|---------|-------|
| w/o condition | 0.57 | 7.62 | No acoustic conditioning |
| Add | **0.80** | 7.50 | Simple addition |
| Concat | 0.80 | 7.49 | Concatenation |
| Cross-Attention | 0.79 | 7.78 | Cross-attention |
| FiLM | 0.80 | 7.58 | FiLM modulation |

### Key Findings

1. **Pretrained priors are indispensable**: From-scratch (FS) models yield WERs of 36–38%, whereas pretrained models achieve only 7–15%
2. **Phonological priors originate from masked prediction objectives**: MRS (Masked Reconstruction Score) is highly correlated with denoising WER; the contextual reasoning capability cultivated by masked prediction is the fundamental source of the prior
3. **Data scale amplifies but does not establish priors**: Effective priors can be established with as little as 960h of data (Base WER 15.49% vs. Base-FS 36.16%), but the large model + large data combination yields the best results
4. **Simple additive fusion suffices**: High-level phonetic and low-level acoustic representations are largely orthogonal; addition is efficient and incurs no information loss
5. **Commercial solution AES-V2 achieves the best perceptual quality but 21% WER**: This highlights the persistent trade-off between perceptual quality and linguistic accuracy

## Highlights & Insights

- **Strong conceptual contribution**: The paper is the first to systematically distinguish linguistic and acoustic hallucinations in speech enhancement and trace each back to its root cause
- **Paradigm shift**: Moves from "learning priors from corrupted inputs" to "directly leveraging existing robust priors," avoiding prior contamination
- **In-depth prior source analysis**: Through carefully designed experiments using three metrics (PNMI, RFS, MRS), the paper reveals that phonological priors fundamentally originate from the contextual reasoning capability cultivated by masked prediction objectives
- **Highly practical**: Computational cost of only 21.42 G MACs/s, far lower than StoRM ($317.76 \times 30$) and FlowSE ($36.79 \times 32$), while achieving superior performance

## Limitations & Future Work

- UTMOS scores are relatively low on the DNS1 with-reverb subset, suggesting a possible domain mismatch between reverberant training and test conditions
- Reliance on WavLM-Large as the backbone results in a non-trivial parameter count (382M), posing challenges for edge deployment
- The low-level acoustic representation in the dual-stream design may carry residual noise, causing UTMOS to drop from 3.42 to 3.09
- Generalization to multilingual speech is not discussed
- Adversarial training of the vocoder may introduce artifacts; further validation under extreme conditions is needed

## Related Work & Insights

- **WavLM**'s layer-wise analysis—where lower layers encode acoustic/speaker information and higher layers encode abstract linguistic information—provides the theoretical basis for the proposed design
- **HuBERT**'s masked prediction training paradigm informs the analysis of phonological prior sources
- The shortcomings of prior work such as **DeVo/GenSE/SELM/LLaSE-G1** (prior contamination, information loss) directly motivate the paradigm shift proposed in this paper
- Insight: The strategy of directly exploiting pretrained knowledge rather than relearning it may be equally effective in other S3M-based tasks, such as automatic speech recognition and speaker verification

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Outstanding in both conceptual innovation (hallucination taxonomy + paradigm shift) and technical innovation (DRD + dual-stream design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-metric, in-depth ablations (distillation layers, objective functions, prior sources, fusion schemes)
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is precise, analysis is logically layered, and causal reasoning is rigorous
- Value: ⭐⭐⭐⭐⭐ — Achieves the best balance between perceptual quality and content fidelity, with open-source code and high practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models](../../CVPR2026/hallucination/svhalluc_benchmarking_speech-vision_hallucination_in_audio-visual_large_language.md)
- [\[ICML 2026\] Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision Language Models](../../ICML2026/hallucination/adaptive_residual-update_steering_for_low-overhead_hallucination_mitigation_in_l.md)
- [\[ICML 2026\] Capturing Gaze Shifts for Guidance: Cross-Modal Fusion Enhancement for VLM Hallucination Mitigation](../../ICML2026/hallucination/capturing_gaze_shifts_for_guidance_cross-modal_fusion_enhancement_for_vlm_halluc.md)
- [\[ICLR 2026\] Imitating the Truth: Attention-aware Truth-Guided Enhancement for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/imitating_the_truth_attention-aware_truth-guided_enhancement_for_hallucination_m.md)
- [\[ICLR 2026\] Leveraging Pretrained Knowledge at Inference Time: LoRA-Gated Contrastive Decoding for Multilingual Factual Language Generation in Adapted LLMs](../../ICLR2026/hallucination/leveraging_pretrained_knowledge_at_inference_time_lora-gated_contrastive_decodin.md)

</div>

<!-- RELATED:END -->
