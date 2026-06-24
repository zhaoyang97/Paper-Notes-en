---
title: >-
  [Paper Note] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models
description: >-
  [ACL 2025][Audio & Speech][audio codec] This paper uncovers and quantitatively analyzes the Discrete Representation Inconsistency (DRI) issue in neural audio codecs—where identical audio segments are encoded into different discrete token sequences depending on context. Two constraint methods, slice consistency and perturbation consistency, are proposed to improve average consistency by 21-36% and reduce the Word Error Rate (WER) by 3.72% in VALL-E speech generation.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "audio codec"
  - "discrete representation consistency"
  - "speech generation"
  - "VALL-E"
  - "neural codec language models"
date: 2026-05-08
content_hash: 15fbd9f1332da2e4
---

# Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models

**Conference**: ACL 2025  
**arXiv**: [2409.19283](https://arxiv.org/abs/2409.19283)  
**Code**: [https://consistencyinneuralcodec.github.io](https://consistencyinneuralcodec.github.io)  
**Area**: Speech  
**Keywords**: audio codec, discrete representation consistency, speech generation, VALL-E, neural codec language models

## TL;DR
This paper uncovers and quantitatively analyzes the Discrete Representation Inconsistency (DRI) issue in neural audio codecs—where identical audio segments are encoded into different discrete token sequences depending on context. Two constraint methods, slice consistency and perturbation consistency, are proposed to improve average consistency by 21-36% and reduce the Word Error Rate (WER) by 3.72% in VALL-E speech generation.

## Background & Motivation
1. **Background**: Speech LLMs use neural audio codecs (such as EnCodec) to discretize continuous audio into token sequences, which are then generated using autoregressive models.
2. **Limitations of Prior Work**: Discrete audio tokens exhibit context dependency—the same audio segment is encoded into different token sequences depending on whether context is present or not (the DRI phenomenon), unlike deterministic text tokens. This leads to a many-to-one mapping problem, increasing the uncertainty for the language model to predict the next token, which causes omissions and repetitions in speech generation.
3. **Key Challenge**: The convolutional layers in the encoder introduce contextual information to improve compression efficiency and reconstruction quality, but they simultaneously render the discrete representations fragile and sensitive, wherein subtle signal variations cause drastic drifts in the entire sequence.
4. **Goal**: Enhance the context independence of discrete tokens while maintaining the original receptive field and reconstruction quality.
5. **Key Insight**: Quantitatively analyze the DRI phenomenon, finding that consistency is worse in deeper codebooks, and design constraint methods to balance quality and consistency.
6. **Core Idea**: Slice consistency (eliminating context impact) + perturbation consistency (enhancing phase robustness).

## Method

### Overall Architecture
Audio $\rightarrow$ Encoder (with convolutional layers) $\rightarrow$ Latent representation Z $\rightarrow$ RVQ quantization $\rightarrow$ Discrete tokens. DRI Analysis: Token consistency is compared by encoding full audio and sliced audio separately. Enhancement: Slice consistency and perturbation consistency constraints are incorporated during training.

### Key Designs

1. **Quantitative Analysis of DRI**:
    - **Function**: Quantitatively reveal the severity of the DRI issue in mainstream audio codecs.
    - **Mechanism**: Define consistency accuracy $Acc_{\text{consistency}} = \frac{1}{TN}\sum_t\sum_i \mathbb{I}(\text{RVQ}(Z^{\text{slice}})[t,i] = \text{RVQ}(Z)[t,i])$. Test six codecs, including EnCodec, HiFiCodec, and SpeechTokenizer, across different slice lengths and codebook code layers.
    - **Design Motivation**: Prior work only qualitatively observed inconsistency, lacking systematic quantitative analysis.

2. **Slice Consistency Constraint**:
    - **Function**: Force the encoder to produce consistent latent representations for the same audio segment regardless of the presence of context.
    - **Mechanism**: Randomly crop a segment from the complete audio, encode them into $Z^{\text{slice}}$ and the corresponding $Z$ respectively, and enforce consistency between them using MSE: $\mathcal{L}_{\text{slice}} = \frac{1}{T}\sum_t \text{MSE}(Z^{\text{slice}}[t], Z[t])$.
    - **Design Motivation**: The source of DRI is the introduction of contextual information by convolutional layers. Directly reducing kernel size degrades compression efficiency and reconstruction quality, whereas the MSE constraint reduces contextual impact while keeping the receptive field intact.

3. **Perturbation Consistency Constraint**:
    - **Function**: Enhance encoder robustness against imperceptible signal perturbations.
    - **Mechanism**: Apply a slight phase perturbation (imperceptible to human ears) to the raw audio, ensuring that the encoded representation remains consistent with the original: $\mathcal{L}_{\text{perception}} = \text{MSE}(Z^{\text{perception}}, Z)$. In actual implementation, both constraints are combined into a single loss.
    - **Design Motivation**: Although phase variations do not affect auditory perception, they lead to drastic shifts in discrete tokens, raising the learning difficulty for language models.

### Loss & Training
Total Loss = Reconstruction Loss + Adversarial Loss + Feature Matching Loss + RVQ Commit Loss + $\lambda_{\text{con}}$ Consistency Loss. $\lambda_{\text{con}}=10.0$. Based on the RVQ-GAN framework, trained for 350k steps using the Adam optimizer with a batch size of 384, audio truncated to 1.28s, and 16kHz sampling rate. The consistency constraint is applied only to the encoder latent space, without modifying the structures of the decoder and quantizer.

## Key Experimental Results

### Main Results (Consistency Improvement)

| Codebook Layers | Baseline EnCodec | Ours | Gain |
|------|-----------|------|------|
| Layer 1 | ~75% | ~96% | +21.47% |
| First 3 Layers | ~55% | ~84% | +29.17% |
| First 8 Layers | ~35% | ~71% | +36.29% |

### Main Results (Speech Generation - VALL-E)

| Method | WER↓ | Speaker Sim↑ | UTMOS↑ |
|------|------|-------------|--------|
| VALL-E (EnCodec) | 5.89 | 0.682 | 3.45 |
| **VALL-E (Ours)** | **2.17** | **0.738** | **3.62** |
| Gain | -3.72% | +5.68% | +0.17 |

### Ablation Study

| Configuration | Layer 1 Consistency | First 3 Layers Consistency | WER↓ | SIM↑ | UTMOS↑ |
|------|-----------|-----------|------|------|--------|
| Slice 20% + Perturbation | 76.75% | 90.66% | 1.84 | 83.71% | 4.31 |
| Perturbation Only (No Slice) | 7.03% | 16.20% | 2.24 | 77.09% | 4.15 |
| Slice 20% Only (No Perturbation) | 75.91% | 90.85% | 2.36 | 81.84% | 4.14 |
| No Consistency Constraint | 6.94% | 15.49% | 4.73 | 76.95% | 4.10 |
| Slice 40% + Perturbation | 64.74% | 85.44% | 1.90 | 82.81% | 4.27 |
| Slice 60% + Perturbation | 31.79% | 60.95% | 3.02 | 82.41% | 4.25 |

A 20% slice ratio is optimal—shorter audio segments contain less contextual information, mitigating context dependency more effectively.

### Key Findings
- The DRI phenomenon is ubiquitous across all mainstream audio codecs, and is more severe in deeper codebooks.
- Shallow-layer tokens align well with context-independent semantic information, whereas deep-layer tokens focus on fragile acoustic details.
- Consistency improvements are positively correlated with downstream speech generation performance—higher consistency leads to lower WER and stronger speaker similarity.
- The approach is equally effective on the large-scale MLS dataset (44k hours): WER drops from 1.84 to 1.37, and SIM increases from 83.71% to 84.14%, demonstrating scalability.

## Highlights & Insights
- **Importance of the DRI Problem**: Reveals a fundamental yet neglected issue in audio discretization, explaining part of the reasons behind omissions and repetitions in speech LLMs.
- **Simplicity and Effectiveness of Constraints**: Significant consistency and generation quality gains are achieved by adding just an MSE constraint.
- **Transferability to Other Discretization Methods**: Any discretization methods utilizing an encoder-quantizer architecture may suffer from similar problems and benefit from similar constraints.
- **Differentiated Analysis of Shallow vs. Deep Layers**: Shallow-layer tokens align well with context-independent semantic information (consistency ~75%), while deep-layer tokens focus on fragile acoustic details (consistency ~35%). This finding provides crucial guidance for designing hierarchical codec strategies.
- **Inspiration from an Information-Theoretic Perspective**: The many-to-one mapping problem caused by DRI essentially increases the conditional entropy for the language model to predict the next token. Enforcing consistency is equivalent to reducing this conditional entropy.

## Limitations & Future Work
- The improvement in consistency may come at the cost of the encoder's ability to utilize contextual information, highlighting a trade-off between quality and consistency.
- The method has only been validated on speech generation tasks, with other audio tasks like music and sound effects generation left unexplored.
- Setting $\lambda_{\text{con}}$ requires manual tuning, and different tasks might need different values.
- Perturbation consistency only considers phase perturbations, without investigating other types of imperceptible perturbations (such as minor amplitude changes).
- The DRI analysis on 6 codecs reveals that all methods suffer from this issue, but the impact of different architectures (causal vs. non-causal convolution) on consistency has not been thoroughly analyzed.

## Related Work & Insights
- **vs. EnCodec/DAC**: These methods focus on reconstruction quality but ignore representation consistency; this paper demonstrates that consistency is equally vital.
- **vs. SpeechTokenizer**: SpeechTokenizer enhances shallow-layer semantics through semantic distillation, but does not address deep-layer inconsistency.
- **vs. LLM-Codec**: LLM-Codec also noted the inconsistency of discrete tokens, but only documented it as an observation without offering a solution.

## Rating
- Novelty: ⭐⭐⭐⭐ Discovery and quantitative analysis of the DRI problem are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 6 codecs, small/large-scale data, and both reconstruction + generation.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth analysis, intuitive diagrams, and rigorous experimental design.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the fields of speech discretization and speech LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook](unicodec_unified_audio_codec_with_single_domain-adaptive_codebook.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](../../ICML2026/audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)
- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ACL 2025\] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)

</div>

<!-- RELATED:END -->
