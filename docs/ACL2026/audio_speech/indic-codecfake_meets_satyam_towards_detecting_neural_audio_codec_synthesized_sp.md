---
title: >-
  [Paper Note] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages
description: >-
  [ACL 2026][Audio & Speech][Speech Deepfake Detection] This paper constructs the first multi-Indic language CodecFake detection benchmark, ICF…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech Deepfake Detection"
  - "Neural Audio Codec"
  - "Indic Languages"
  - "Hyperbolic ALM"
  - "CodecFake"
date: 2026-05-08
content_hash: 73f78b2e653e2707
---

# Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages

**Conference**: ACL 2026  
**arXiv**: [2604.19949](https://arxiv.org/abs/2604.19949)  
**Code**: [https://helixometry.github.io/IndicFake/](https://helixometry.github.io/IndicFake/)  
**Area**: AI Security / Speech Security  
**Keywords**: Speech Deepfake Detection, Neural Audio Codec, Indic Languages, Hyperbolic ALM, CodecFake

## TL;DR
This paper constructs the first multi-Indic language CodecFake detection benchmark, ICF, and proposes SATYAM—a hyperbolic audio large language model. By aligning semantic and paralinguistic representations using Bhattacharyya distance in hyperbolic space and subsequently aligning them with prompts, SATYAM achieves 98.32% detection accuracy with only 3.75M trainable parameters.

## Background & Motivation

**Background**: Speech deepfake technology is evolving rapidly. Novel synthetic speech—CodecFake (CF)—driven by neural audio codecs (NAC) used in audio large language models (ALM), has emerged as a new threat. Initiatives like the ASVspoof series have advanced synthetic speech detection, and recent works utilize pretrained models (e.g., WavLM, Whisper) and ALMs for detection tasks.

**Limitations of Prior Work**: Existing CF detection datasets focus almost exclusively on English (some including Chinese), leaving the vulnerability of Indic language communities largely unexplored. Experiments show that SOTA CF detectors trained on English data fail significantly on Indic languages (AASIST accuracy drops from 94% to 48%). Zero-shot evaluations of SOTA ALMs on the ICF dataset also perform poorly (accuracy around 13%).

**Key Challenge**: India is the world's most populous country with immense linguistic diversity (Indo-Aryan, Dravidian, Austroasiatic, etc.), making it highly vulnerable to AI voice scams. However, there is a lack of targeted CF detection datasets and models. The phonetic diversity and prosodic variations of these languages make English-centric detectors difficult to generalize.

**Goal**: (1) Construct the first large-scale multi-Indic language CF dataset; (2) Systematically evaluate the generalization capabilities of SOTA detectors and ALMs; (3) Propose a specialized detection model.

**Key Insight**: The authors observe that semantic and paralinguistic features may exhibit a hierarchical structure (from coarse-grained semantics to fine-grained prosody). Hyperbolic space is naturally suited for modeling such hierarchical relationships. Furthermore, Bhattacharyya distance has proven effective for alignment in speech representation.

**Core Idea**: Build the ICF dataset to fill the data gap; propose SATYAM, which performs two-stage fusion in hyperbolic space using Bhattacharyya distance—first aligning semantic (Whisper) and paralinguistic (TRILLsson) representations, then aligning them with input conditional prompts—utilizing Qwen2-7B as a decoder to generate detection decisions.

## Method

### Overall Architecture
SATYAM models CF detection as a conditional generation task. Input speech is encoded by two frozen audio encoders (Whisper for semantic features and TRILLsson for paralinguistic features). These representations are projected via CNNs, gated, and mapped into hyperbolic space. Through two-stage Bhattacharyya distance alignment (audio-audio and audio-prompt), the fused representations are mapped back to Euclidean space and injected into a frozen Qwen2-7B decoder to generate "Real" or "Fake" labels.

### Key Designs

1. **Indic-CodecFake (ICF) Dataset Construction**:
    - **Function**: Provides the first large-scale multi-Indic language CodecFake detection benchmark.
    - **Mechanism**: Uses IndicSUPERB (12 Indic languages) as the source for real speech, resynthesized using 14 different NAC configurations (DAC, Encodec, SoundStream, SpeechTokenizer, FunCodec, AudioDec, SNAC, MIMI). It maintains original train/valid/test splits and includes Seen (training and testing on the same NACs) and Unseen (testing on NACs not seen during training) evaluation settings.
    - **Design Motivation**: Existing CF datasets cover only English/Chinese and fail to represent the phonetic diversity and prosodic features of Indic languages. The multi-NAC configuration ensures that the detector learns universal forgery features across different codecs.

2. **Hyperbolic Dual-Stage Fusion**:
    - **Function**: Aligns speech representations and text prompts across different modalities within a hierarchy-aware geometric space.
    - **Mechanism**: Whisper and TRILLsson representations are projected via CNN and sigmoid gating before being mapped to a hyperbolic space with curvature $-c$ via the exponential map $\exp_0^c(u) = \tanh(\sqrt{c}\|u\|) \frac{u}{\sqrt{c}\|u\|}$. In the first stage, semantic and paralinguistic representations are aligned by minimizing the hyperbolic Bhattacharyya distance $\mathcal{L}_{S\text{-}S} = D_B(h_w, h_t)$ and fused via Mobius addition $h_f = h_w \oplus_c h_t$. In the second stage, the fused audio representation is aligned with the conditional prompt representation using BD $\mathcal{L}_{S\text{-}T} = D_B(h_f, h_A)$ and aggregated via Mobius addition again.
    - **Design Motivation**: Hierarchical structures exist within semantic and paralinguistic cues, as well as in cross-modal (audio-text) relationships. Hyperbolic space is naturally suited for embedding hierarchies, and Bhattacharyya distance is effective for speech representation alignment.

3. **Lightweight Conditional Generative Detection**:
    - **Function**: Achieves end-to-end detection with minimal trainable parameters (~3.75M).
    - **Mechanism**: The fused hyperbolic representations are mapped back to Euclidean space via logarithmic mapping and passed through a projection layer to generate prefix condition tokens for the frozen Qwen2-7B decoder. One conditional prompt ("Analyze the speech for unnatural artifacts") guides feature extraction, while a decision prompt guides the output to "Real" or "Fake". Only the CNN layers, projection layers, and hyperbolic alignment modules are trained.
    - **Design Motivation**: Freezing the audio encoders and LLM decoder significantly reduces training costs. Previous research suggests audio encoders are the primary bottleneck in ALM performance, so performance is improved through stronger encoder fusion strategies rather than scaling the LLM.

### Loss & Training
The total loss is $\mathcal{L} = \lambda_1 \mathcal{L}_{S\text{-}S} + \lambda_2 \mathcal{L}_{S\text{-}T} + \lambda_3 \mathcal{L}_{LM}$, where $\lambda_1=1, \lambda_2=0.5, \lambda_3=1$. The model is trained using the AdamW optimizer with a learning rate of $1 \times 10^{-4}$ and a batch size of 32 for 5 epochs.

## Key Experimental Results

### Main Results

| Method | ICF Acc% | ICF EER% | CodecFake Acc% | CodecFake EER% |
| :--- | :--- | :--- | :--- | :--- |
| AASIST | 90.60 | 12.47 | 94.21 | 10.13 |
| MiO | 92.80 | 9.04 | 95.11 | 6.49 |
| Fine-tune Qwen2-audio | 93.19 | 8.34 | 95.55 | 5.60 |
| **SATYAM** | **98.32** | **3.27** | **99.11** | **1.94** |
| SATYAM (Qwen2-1.8B) | 97.14 | 4.53 | 98.32 | 2.11 |

### Ablation Study

| Configuration | ICF Acc% | ICF EER% |
| :--- | :--- | :--- |
| W + Qwen2-7B (Whisper only) | 92.98 | 8.61 |
| T + Qwen2-7B (TRILLsson only) | 93.21 | 8.09 |
| W+T Concatenation (Euclidean) | 93.28 | 7.94 |
| W+T Mobius Addition (Hyperbolic) | 94.01 | 7.02 |
| W+T Euclidean BD | 94.93 | 5.39 |
| W+T Hyperbolic BD (Audio-Prompt only) | 95.78 | 5.14 |
| W+T Hyperbolic BD (Audio-Audio only) | 96.11 | 5.02 |
| **SATYAM (Full)** | **98.32** | **3.27** |

### Key Findings
- AASIST trained on English CodecFake data saw its accuracy drop from 94% to 48% on ICF, confirming severe cross-language generalization issues.
- Zero-shot accuracy of SOTA ALMs is only around 13%, indicating current ALMs have extremely limited CF detection capabilities.
- The single encoder TRILLsson performs slightly better than Whisper, reflecting that paralinguistic features are primary cues for deepfake detection.
- Complete two-stage fusion with hyperbolic BD is significantly superior to any single-stage or Euclidean alternative, proving the necessity of hyperbolic geometry and two-stage alignment.
- Cross-language family transfer (Dravidian to Indo-Aryan and vice versa) yielded EERs below 8.5%, demonstrating robust generalization.
- Replacing Qwen2-7B with the lightweight 1.8B version resulted in only a slight performance drop, suggesting that audio encoder quality, not LLM size, is the bottleneck.

## Highlights & Insights
- Fills the gap in Indic language CF detection. The ICF dataset, covering 12 languages and 14 NAC configurations, serves as a valuable community benchmark.
- Bhattacharyya distance in hyperbolic space is an innovative fusion scheme. Generalizing BD from Euclidean to hyperbolic space can be transferred to other multimodal tasks requiring hierarchical representation alignment.
- Outperforming full-parameter fine-tuning with only 3.75M trainable parameters suggests that correct inductive biases and fusion strategies are more important than model scale.

## Limitations & Future Work
- Only evaluated on the Qwen2 LLM family, although prior research suggests LLM choice has an limited impact.
- Resynthesis via encoding-decoding may not fully represent real-world attack scenarios (e.g., joint NAC-TTS generation).
- Numerical stability of hyperbolic operations could be an issue during large-scale training.
- Has not yet explored adversarial attacks or defense scenarios on the ICF dataset.

## Related Work & Insights
- **vs CodecFake (Wu et al.)**: CodecFake only covers the English VCTK corpus; ICF extends this to 12 Indic languages. AASIST’s performance plunges on ICF compared to CodecFake.
- **vs MiO**: MiO is a SOTA multi-encoder fusion method reaching 92.8% on ICF. SATYAM improves this to 98.3% using the same encoders, showing that the fusion strategy—not the encoder itself—is the bottleneck.
- **vs Gu et al. (ALM Detection)**: Previous work evaluated ALMs for traditional deepfake detection but neglected CF. This paper is the first to systematically evaluate the zero-shot capability of ALMs for CF detection, proving they currently lack proficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ (ICF dataset fills a critical gap; hyperbolic BD fusion is a novel technical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Includes zero-shot, in-domain, cross-benchmark, cross-language family, unseen codecs, and noisy conditions)
- Writing Quality: ⭐⭐⭐ (Clear but somewhat verbose; tables involve many symbols requiring reference)
- Value: ⭐⭐⭐⭐ (Direct contribution to the multilingual deepfake detection community; methodology is extensible)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation](sn-wer_script-normalized_wer_for_multi-script_indic_asr_evaluation.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation](sn-wer_script-normalized_wer_for_multi-script_indic_asr_evaluation.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](../../ACL2025/audio_speech/audio_token_consistency.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)

</div>

<!-- RELATED:END -->
