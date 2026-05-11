---
title: >-
  [Paper Note] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages
description: >-
  [ACL 2026][LLM Safety][Speech Deepfake Detection] This paper introduces ICF, the first multi-Indic-language CodecFake detection benchmark…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Speech Deepfake Detection"
  - "Neural Audio Codec"
  - "Indic Languages"
  - "Hyperbolic ALM"
  - "CodecFake"
date: 2026-05-08
content_hash: 1a603301fa8e3286
---

# Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages

**Conference**: ACL 2026
**arXiv**: [2604.19949](https://arxiv.org/abs/2604.19949)
**Code**: [https://helixometry.github.io/IndicFake/](https://helixometry.github.io/IndicFake/)
**Area**: AI Security / Speech Security
**Keywords**: Speech Deepfake Detection, Neural Audio Codec, Indic Languages, Hyperbolic ALM, CodecFake

## TL;DR
This paper introduces ICF, the first multi-Indic-language CodecFake detection benchmark, and proposes SATYAM—a hyperbolic audio large language model that aligns semantic and paralinguistic representations via Bhattacharyya distance in hyperbolic space before aligning with a conditioning prompt. With only 3.75M trainable parameters, SATYAM achieves 98.32% detection accuracy.

## Background & Motivation

**Background**: Speech deepfake technology has advanced rapidly. CodecFake (CF)—a new class of synthesized speech driven by neural audio codec (NAC)-based audio large language models (ALMs)—has emerged as a growing threat. Prior efforts such as the ASVspoof series have advanced synthetic speech detection, and recent work has employed pretrained models (WavLM, Whisper, etc.) and ALMs for this purpose.

**Limitations of Prior Work**: Existing CF detection datasets are almost exclusively focused on English (at most including Chinese), leaving the vulnerability of Indic language communities virtually unexplored. Experiments confirm that state-of-the-art CF detectors trained on English data fail severely on Indic languages (AASIST accuracy drops from 94% to 48%). Zero-shot evaluation of SOTA ALMs on ICF also yields extremely poor performance (accuracy of approximately 13%).

**Key Challenge**: India is the world's most populous country with extraordinary linguistic diversity (Indo-European, Dravidian, Austro-Asiatic families, etc.) and faces high risk from AI-driven voice fraud, yet lacks targeted CF detection datasets and models. The phonemic diversity and prosodic variability of Indic languages prevent English-centric detectors from generalizing.

**Goal**: (1) Construct the first large-scale multi-Indic-language CF dataset; (2) systematically evaluate the generalization of SOTA detectors and ALMs; (3) propose a targeted detection model.

**Key Insight**: The authors observe that semantic and paralinguistic features may exhibit a hierarchical structure—from coarse-grained semantics to fine-grained prosody—and that hyperbolic space is naturally suited to modeling such hierarchies. Additionally, Bhattacharyya distance has been shown to be effective for speech representation alignment.

**Core Idea**: Build the ICF dataset to fill the data gap; propose SATYAM, which performs two-stage fusion in hyperbolic space via Bhattacharyya distance—first aligning semantic (Whisper) and paralinguistic (TRILLsson) representations, then aligning with an input conditioning prompt—using Qwen2-7B as the decoder to generate detection decisions.

## Method

### Overall Architecture
SATYAM frames CF detection as a conditional generation task. Input speech is encoded by two frozen audio encoders (Whisper for semantic representations, TRILLsson for paralinguistic representations). After CNN projection and gating, representations are mapped into hyperbolic space, aligned via two-stage Bhattacharyya distance (speech–speech and speech–prompt), and the fused representation is mapped back to Euclidean space and injected into a frozen Qwen2-7B decoder to generate "Real" or "Fake."

### Key Designs

1. **Indic-CodecFake (ICF) Dataset Construction**:

    - **Function**: Provides the first large-scale multi-Indic-language CodecFake detection benchmark.
    - **Mechanism**: IndicSUPERB (12 Indic languages) serves as the real speech source; 14 NAC configurations (DAC, Encodec, SoundStream, SpeechTokenizer, FunCodec, AudioDec, SNAC, MIMI) are used to perform encode–decode resynthesis. The original train/valid/test splits are preserved, and two evaluation settings are designed: Seen (same NAC group used in training and testing) and Unseen (test NACs not seen during training).
    - **Design Motivation**: Existing CF datasets cover only English/Chinese and cannot represent the phonemic diversity and prosodic characteristics of Indic languages. The multi-NAC configuration ensures that detectors must learn generalizable forgery features across codecs.

2. **Hyperbolic Dual-Stage Fusion**:

    - **Function**: Aligns speech representations of different modalities and text prompts in a hierarchy-aware geometric space.
    - **Mechanism**: Whisper and TRILLsson representations are CNN-projected and sigmoid-gated, then mapped into hyperbolic space of curvature $-c$ via the exponential map $\exp_0^c(u) = \tanh(\sqrt{c}\|u\|) \frac{u}{\sqrt{c}\|u\|}$. In the first stage, semantic and paralinguistic representations are aligned by minimizing the hyperbolic Bhattacharyya distance $\mathcal{L}_{S\text{-}S} = D_B(h_w, h_t)$ and fused via Möbius addition $h_f = h_w \oplus_c h_t$. In the second stage, the fused speech representation and the conditioning prompt representation are similarly aligned via $\mathcal{L}_{S\text{-}T} = D_B(h_f, h_A)$ and aggregated with Möbius addition.
    - **Design Motivation**: Semantic and paralinguistic cues exhibit a hierarchical structure; cross-modal (speech–text) hierarchical relationships are also well-established. Hyperbolic space is naturally suited to embedding hierarchical structures, and Bhattacharyya distance has demonstrated effectiveness in speech representation alignment.

3. **Lightweight Conditional Generation Detection**:

    - **Function**: Achieves end-to-end detection with very few trainable parameters (~3.75M).
    - **Mechanism**: The fused hyperbolic representation is mapped back to Euclidean space via the logarithmic map and projected to generate prefix conditioning tokens injected into the frozen Qwen2-7B decoder. One conditioning prompt ("Analyze the speech for unnatural artifacts") guides feature extraction, and one decision prompt guides the output of "Real" or "Fake." Only the CNN layers, projection layers, and hyperbolic alignment modules are trained.
    - **Design Motivation**: Freezing the audio encoders and LLM decoder substantially reduces training cost. Prior research shows that audio encoders are the primary performance bottleneck in ALMs; thus, performance is improved through stronger encoder fusion strategies rather than larger LLMs.

### Loss & Training
The total loss is $\mathcal{L} = \lambda_1 \mathcal{L}_{S\text{-}S} + \lambda_2 \mathcal{L}_{S\text{-}T} + \lambda_3 \mathcal{L}_{LM}$, with $\lambda_1=1, \lambda_2=0.5, \lambda_3=1$. AdamW optimizer is used with learning rate $1 \times 10^{-4}$, batch size 32, and 5 training epochs.

## Key Experimental Results

### Main Results

| Method | ICF Acc% | ICF EER% | CodecFake Acc% | CodecFake EER% |
|------|---------|---------|---------------|---------------|
| AASIST | 90.60 | 12.47 | 94.21 | 10.13 |
| MiO | 92.80 | 9.04 | 95.11 | 6.49 |
| Fine-tune Qwen2-audio | 93.19 | 8.34 | 95.55 | 5.60 |
| **SATYAM** | **98.32** | **3.27** | **99.11** | **1.94** |
| SATYAM (Qwen2-1.8B) | 97.14 | 4.53 | 98.32 | 2.11 |

### Ablation Study

| Configuration | ICF Acc% | ICF EER% |
|------|---------|---------|
| W + Qwen2-7B (Whisper only) | 92.98 | 8.61 |
| T + Qwen2-7B (TRILLsson only) | 93.21 | 8.09 |
| W+T Concatenation (Euclidean) | 93.28 | 7.94 |
| W+T Möbius Addition (Hyperbolic) | 94.01 | 7.02 |
| W+T Euclidean BD | 94.93 | 5.39 |
| W+T Hyperbolic BD (speech–prompt only) | 95.78 | 5.14 |
| W+T Hyperbolic BD (speech–speech only) | 96.11 | 5.02 |
| **SATYAM (Full)** | **98.32** | **3.27** |

### Key Findings
- AASIST trained on English CodecFake data drops from 94% accuracy to 48% on ICF, confirming severe cross-lingual generalization failure.
- Zero-shot CF detection accuracy of SOTA ALMs is approximately 13%, demonstrating that current ALMs have extremely limited capability for CF detection.
- The TRILLsson single-encoder slightly outperforms Whisper, reflecting that the primary cues for deepfake detection are paralinguistic features.
- The complete two-stage hyperbolic BD fusion significantly outperforms any single-stage or Euclidean alternative, demonstrating the necessity of hyperbolic geometry and dual-stage alignment.
- Cross-family transfer (Dravidian→Indo-European and Indo-European→Dravidian) achieves EER below 8.5%, demonstrating good generalization.
- Replacing Qwen2-7B with the lightweight Qwen2-1.8B incurs only marginal performance degradation, confirming that audio encoder quality is the true performance bottleneck.

## Highlights & Insights
- This work fills a critical gap in Indic-language CF detection. The ICF dataset, covering 12 languages and 14 NAC configurations, constitutes a valuable community benchmark.
- Bhattacharyya distance in hyperbolic space is a novel fusion approach. Extending BD from Euclidean to hyperbolic space is transferable to other multimodal tasks requiring hierarchical representation alignment.
- Achieving substantial improvements over full-parameter methods with only 3.75M trainable parameters demonstrates that appropriate inductive biases and fusion strategies matter more than model scale.

## Limitations & Future Work
- Only the Qwen2 LLM decoder family is considered, though the authors cite evidence that LLM choice has limited impact.
- The encode–decode resynthesis paradigm may not fully represent real-world attack scenarios (e.g., joint NAC-TTS generation).
- Hyperbolic operations may suffer from numerical stability issues, particularly in large-scale training.
- Adversarial attack and defense scenarios on ICF remain unexplored.

## Related Work & Insights
- **vs. CodecFake (Wu et al.)**: CodecFake covers only the English VCTK corpus; ICF extends the scope to 12 Indic languages. AASIST's 94% accuracy on CodecFake collapses to 48% on ICF.
- **vs. MiO**: MiO is the SOTA multi-encoder fusion method, achieving 92.8% on ICF. SATYAM improves this to 98.3% on the same encoders via hyperbolic alignment, demonstrating that the fusion strategy—rather than the encoders themselves—is the bottleneck.
- **vs. Gu et al. (ALM-based detection)**: Prior work evaluated ALMs for conventional deepfake detection but did not address CF detection. This paper is the first to systematically evaluate zero-shot ALM capability for CF detection, finding that current ALMs are wholly inadequate.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The ICF dataset fills an important gap; hyperbolic BD fusion constitutes a novel technical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The evaluation is highly comprehensive, covering zero-shot evaluation, in-domain training, cross-benchmark transfer, cross-family transfer, unseen codecs, and noisy conditions.
- **Writing Quality**: ⭐⭐⭐ Content is rich but organization is somewhat verbose; the numerous table symbols require frequent cross-referencing.
- **Value**: ⭐⭐⭐⭐ Offers direct contributions to the multilingual deepfake detection community; SATYAM's methodology also has broader applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Protecting Bystander Privacy via Selective Hearing in Audio LLMs](protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)
- [\[AAAI 2026\] Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](../../AAAI2026/llm_safety/ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](../../AAAI2026/llm_safety/catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)

</div>

<!-- RELATED:END -->
