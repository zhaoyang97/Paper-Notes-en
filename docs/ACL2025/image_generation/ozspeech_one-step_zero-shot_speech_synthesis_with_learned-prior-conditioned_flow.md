---
title: >-
  [Paper Note] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching
description: >-
  [ACL2025][Image Generation][Flow Matching] This paper proposes OZSpeech, the first zero-shot TTS system that combines Optimal Transport Conditional Flow Matching (OT-CFM) with a learned prior distribution to achieve one-step sampling. It significantly outperforms existing approaches in content accuracy (WER), inference speed, and model size.
tags:
  - "ACL2025"
  - "Image Generation"
  - "Flow Matching"
  - "Zero-Shot TTS"
  - "One-Step Sampling"
  - "Learned Prior"
  - "Neural Codec"
date: 2026-05-08
content_hash: f835f29857b76a04
---

# OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching

**Conference**: ACL2025  
**arXiv**: [2505.12800](https://arxiv.org/abs/2505.12800)  
**Code**: [OZSpeech Demo](https://ozspeech.github.io/OZSpeech_Web/)  
**Area**: Image Generation  
**Keywords**: Flow Matching, Zero-Shot TTS, One-Step Sampling, Learned Prior, Neural Codec  

## TL;DR

This paper proposes OZSpeech, the first zero-shot TTS system that combines Optimal Transport Conditional Flow Matching (OT-CFM) with a learned prior distribution to achieve one-step sampling. It significantly outperforms existing approaches in content accuracy (WER), inference speed, and model size.

## Background & Motivation

- **Challenges of Zero-Shot TTS**: Zero-shot TTS requires cloning the acoustic characteristics of unseen speakers from a prompt of a few seconds, which remains a core challenge in speech synthesis.
- **Limitations of Autoregressive Methods**: Autoregressive models such as VALL-E suffer from infinite repetition issues due to non-deterministic sampling, making them less reliable under high-precision scenarios.
- **Bottlenecks of Diffusion Models**: Diffusion models like E2 TTS can generate high-quality audio, but multi-step sampling incurs high computational costs, making it difficult to satisfy real-time application demands.
- **Inadequacy of Existing Acceleration Schemes**: Consistency Models require high training costs across the full range $t \in [0,1]$; Shortcut Models introduce extra constraints and are more resource-intensive.
- **Limitations of Traditional OT-CFM**: Traditional methods construct the output distribution starting from Gaussian noise. The large gap between noise and the target necessitates multi-step sampling to converge.
- **Core Motivation**: Is it possible to start from a learned prior that is closer to the target distribution, enabling flow matching to accomplish high-quality speech synthesis in just a single step?

## Method

### Overall Architecture

The overall pipeline of OZSpeech is divided into three core modules:

1. **Prior Codes Generator ($f_\psi$)**: Converts text (phonemes) into a sequence of prior codes, serving as the starting point for flow matching.
2. **OT-CFM Vector Field Estimator ($v_\theta$)**: Starting from the prior codes, it estimates the vector field toward the target distribution by incorporating the rhythm and acoustic details from the acoustic prompt.
3. **FACodec**: Decomposes the waveform into decoupled representations of speaker identity, prosody, content, and acoustic details, and finally decodes them back into the speech waveform.

Key Innovation: Instead of starting from Gaussian noise, the framework starts from a learned prior that is already close to the target distribution, thereby allowing flow matching to be completed in one single step.

### Prior Codes Generator

A hierarchical cascaded neural network is employed, where the generation of each code sequence depends on the preceding code sequences:

$$p(\mathbf{q}_{1:6}|\mathbf{p};\psi) = p(\mathbf{q}_1|\mathbf{p};f_\psi^1)\prod_{j=2}^{6}p(\mathbf{q}_j|\mathbf{q}_{j-1};f_\psi^j)$$

- The first layer generates content codes conditioned on phoneme embeddings.
- Subsequent layers sequentially generate prosody and acoustic detail codes.
- A Duration Predictor is used to align phonemes with the output code sequences, minimizing the log-scale duration prediction error using the MSE loss.

### One-Step OT-CFM Reconstruction

Core mathematical reformulation: Replacing the random time step $t$ in standard OT-CFM with a learnable, prior-dependent time variable $\tau$:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{\mathbf{x}_{pr},\mathbf{x}_1}\left\|\mathbf{v}_\theta(\mathbf{x}_{pr},\tau) - \frac{\mathbf{x}_1 - \mathbf{x}_{pr}}{1-\tau}\right\|^2$$

**Key Differences**:
- It does not access $\mathbf{x}_0$ (noise distribution) and does not force $\mathbf{x}_0$ to follow a normal distribution.
- When the prior distribution $\mathbf{x}_{pr}$ is close to the target distribution $\mathbf{x}_1$, both the number of sampling steps and the step size are significantly reduced.
- This eventually achieves one-step sampling (NFE=1).

### Folding Mechanism and Quantizer Encoding

- **Folding**: Folds the $6$ quantizer sequences along the hidden dimension ($\mathbb{R}^{6 \times L \times D} \rightarrow \mathbb{R}^{L \times D'}$), modeling all quantizers simultaneously to avoid the high computational cost of sequential processing.
- **Quantizer Encoding**: Adds a learnable identification embedding $\omega$ to each quantizer to prevent the model from confusing different quantizers within the same sequence.
- Gaussian noise is added to the prior codes during input to ensure robustness and diversity.

### Loss & Training

The total loss consists of four parts:

$$\mathcal{L}_{total} = \mathcal{L}_{prior} + \mathcal{L}_{dur} + \mathcal{L}_{CFM} + \mathcal{L}_{anchor}$$

- **$\mathcal{L}_{prior}$**: Minimizes the negative log-likelihood of the generated prior codes.
- **$\mathcal{L}_{dur}$**: The MSE loss of the duration predictor.
- **$\mathcal{L}_{CFM}$**: The flow matching vector field regression loss.
- **$\mathcal{L}_{anchor}$**: A regularization term to prevent embedding collapse, implemented by minimizing the negative log-likelihood between the estimated target and the ground-truth target.

## Experiments

### Experimental Settings

- **Training Data**: LibriTTS ($500$ hours of multi-speaker English audio)
- **Evaluation Data**: LibriSpeech test-clean
- **Evaluation Metrics**: UTMOS (speech quality), WER (content accuracy), SIM-O/SIM-R (speaker similarity), F0/Energy (prosody), NFE/RTF (latency)
- **Baselines**: F5-TTS, VoiceCraft, NaturalSpeech 2, VALL-E

### Main Results

| Model | Training Data | WER↓ | SIM-O↑ | UTMOS↑ | NFE↓ | RTF↓ |
|------|---------|------|--------|--------|------|------|
| F5-TTS | 95,000h | 0.24 | 0.53 | 3.76 | 32 | 0.70 |
| VoiceCraft | 9,000h | 0.18 | 0.51 | 3.55 | - | 1.70 |
| NaturalSpeech 2 | 585h | 0.09 | 0.31 | 2.38 | 200 | 1.66 |
| VALL-E | 500h | 0.19 | 0.40 | 3.68 | - | 0.86 |
| **OZSpeech** | **500h** | **0.05** | **0.40** | 3.15 | **1** | **0.26** |

(The above results are under the 3s prompt setting)

**Key Findings**:
- OZSpeech achieves comprehensive SOTA on WER, reducing it by 44% compared to the second-best method under the 5s prompt setting.
- Inference speed is approximately 3 times faster than the second fastest method, F5-TTS.
- The model size is only 29%-71% of other methods (with only 17%-43% of trainable parameters).
- With only 500 hours of training data, Ours outperforms F5-TTS trained on 95,000 hours in terms of WER.

### Ablation Study

**Prompt Strategy Comparison**:
- Arbitrary Segment (random segment selection) outperforms First Segment (fixed starting segment) across all metrics.
- First Segment tends to overfit, shifting the prompt to the beginning of the target.
- Arbitrary Segment hides the prompt's position, thereby improving generalization performance.

### Noise Tolerance Analysis

- In acoustic prompt scenarios with noise, the WER of other models rises sharply as SNR decreases.
- The WER of OZSpeech remains stable, demonstrating excellent noise tolerance.
- This proves the inherent robustness of the learned prior mechanism against noise.

## Highlights & Insights

1. **Theoretical Elegance**: By introducing a learned prior into the OT-CFM framework, one-step sampling is naturally realized through mathematical derivation without requiring an additional distillation stage.
2. **Extreme Efficiency**: NFE=1, RTF=0.26, which is two orders of magnitude fewer sampling steps compared to NaturalSpeech 2 (200 steps).
3. **Small Model, Strong Performance**: With only 145M trainable parameters, it outperforms all large model baselines in WER.
4. **Value of Disentangled Representation**: Using FACodec to disentangle speech into content/prosody/acoustic details/speaker enables precise control over each attribute.
5. **Low-Resource Friendly**: Traditional OT-CFM (e.g., F5-TTS) yields WER > 0.95 when trained on 500 hours of data, whereas Ours achieves SOTA with just 500 hours.
6. **Clever Folding Mechanism**: Simultaneously modeling 6 quantizers avoids sequential processing, significantly reducing computational overhead.

## Limitations & Future Work

1. **Slightly Lower UTMOS**: Due to the trade-off between acoustic and semantic representations in FACodec, the overall speech quality score is slightly inferior to some baselines.
2. **Non-Optimal SIM-O/SIM-R**: There is still a gap in speaker similarity metrics compared to the best performing methods.
3. **English-Only Evaluation**: All experiments are conducted on English datasets, and cross-lingual generalization remains unvalidated.
4. **Dependency on FACodec**: System performance is highly dependent on the pre-training quality of FACodec.
5. **Limited Training Data Scale**: Validated only on 500 hours of data; whether large-scale data can yield further improvements remains to be explored.

## Related Work

- **Autoregressive TTS**: The VALL-E series redefines TTS as a conditional codec language modeling task.
- **Diffusion/Flow Matching TTS**: Diffusion and flow matching-based methods such as E2 TTS, NaturalSpeech 2/3, and F5-TTS.
- **Distillation Acceleration**: Methods that reduce sampling steps, such as Consistency Models and Shortcut Models.
- **Neural Audio Codecs**: Discrete speech representation methods such as SoundStream, EnCodec, and FACodec.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐⭐ First to combine learned prior with OT-CFM for one-step zero-shot TTS.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation, including ablation and noise analyses.
- **Value**: ⭐⭐⭐⭐⭐ The advantageous combination of low latency, small model size, and low data requirement shows great potential for deployment.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations with well-founded motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](rvc_rhythm_voice_conversion.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](../../CVPR2026/image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[CVPR 2026\] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching](../../CVPR2026/image_generation/stable_mean_flow_lyapunov-inspired_one-step_flow_matching.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[CVPR 2026\] ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control](../../CVPR2026/image_generation/projflow_projection_sampling_with_flow_matching_for_zero-shot_exact_spatial_moti.md)

</div>

<!-- RELATED:END -->
