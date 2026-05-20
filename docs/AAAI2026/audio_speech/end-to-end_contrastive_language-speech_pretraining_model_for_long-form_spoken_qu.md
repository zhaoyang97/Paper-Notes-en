---
title: >-
  [Paper Note] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
description: >-
  [AAAI2026][Audio & Speech][Spoken Question Answering] This paper proposes CLSR, an end-to-end contrastive language-speech retriever that converts acoustic representations into text-like representations before aligning th…
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Spoken Question Answering"
  - "Contrastive Learning"
  - "Retrieval-Augmented Generation"
  - "Speech-Text Alignment"
  - "CIF"
date: 2026-05-08
content_hash: 5d3eaea6b081a580
---

# End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering

**Conference**: AAAI2026
**arXiv**: [2511.09282](https://arxiv.org/abs/2511.09282)  
**Code**: [193746/CLSR](https://github.com/193746/CLSR)  
**Area**: Audio & Speech
**Keywords**: Spoken Question Answering, Contrastive Learning, Retrieval-Augmented Generation, Speech-Text Alignment, CIF

## TL;DR
This paper proposes CLSR, an end-to-end contrastive language-speech retriever that converts acoustic representations into text-like representations before aligning them with text, enabling efficient extraction of question-relevant segments from long-form audio to support RAG-based spoken question answering for downstream LALMs.

## Background & Motivation
- Most existing SQA (Spoken Question Answering) models can only handle short audio clips under one minute, whereas real-world scenarios (meetings, lectures, online discussions) typically involve audio exceeding ten minutes.
- Large Audio Language Models (LALMs), despite their strong speech understanding capabilities, suffer from slow inference and degraded accuracy on long-form audio.
- RAG has demonstrated significant effectiveness in long-context textual QA, naturally raising the question of whether RAG can be applied to speech to retrieve the most relevant segments from long audio.
- Existing speech retrievers (e.g., CLAP, SpeechDPR) exhibit insufficient performance—CLAP excels at audio-effect-to-text alignment rather than speech-content-to-text alignment, while SpeechDPR is limited by text-free training and data scarcity.

## Core Problem
How to build an end-to-end speech-text retriever that, without relying on cascaded ASR followed by text retrieval, achieves or surpasses pipeline-based retrieval accuracy while substantially reducing inference time and error rates for long-form spoken question answering.

## Method

### Overall Architecture
CLSR consists of two components:
1. **Left branch**: A CIF (Continuous Integrate-and-Fire)-based non-autoregressive attention encoder-decoder (AED) that takes speech input $X$ and outputs a token probability distribution $D$.
2. **Right branch**: A Transformer text encoder (frozen BGE-base) that receives either text-like embeddings or real text embeddings and produces sentence-level representations for contrastive learning.

### CIF Module
- A speech encoder (SAN-M architecture) extracts acoustic features $H^s$.
- CIF computes per-frame weights $\alpha_i \in [0,1]$ via convolution, accumulating them until the sum exceeds a threshold $\beta$, thereby mapping time steps to token counts and producing acoustic representations $E^a$.
- This step realizes soft monotonic alignment from frame-level to token-level representations.

### Sampler Training Optimization
- Training proceeds in two rounds: in the first round, $E^a$ is directly used to predict the token distribution, yielding ASR output $Y^{asr}$.
- In the second round, $Y^{asr}$ is compared against ground-truth $Y^{con}$; at erroneous token positions, the correct embeddings are substituted into $E^a$ at sampling rate $\lambda$, producing mixed features $E^s$.
- $E^s$ is used to re-predict token distribution $D'$, enhancing the decoder's contextual modeling capacity.

### VQ Adaptor (Vector Quantization Adaptor)
- Argmax is applied to the token probability distribution $D$ to obtain the highest-probability token index $q_i$.
- Temperature-scaled softmax ($\gamma=0.1$) combined with straight-through gradient estimation preserves gradient flow.
- The quantized one-hot matrix $Q^{st}$ is multiplied with the text encoder's embedding weight matrix $W^{te}$ to produce text-like embeddings $E^{Y'}$.
- Core idea: rather than directly aligning acoustic representations with text representations, VQ "translates" acoustic representations into approximate representations in the text space, where contrastive learning is then performed.

### Contrastive Learning & Loss Function
- The text-like embedding of the context and the text embedding of the question are fed into the text encoder; CLS token outputs are used as sentence-level representations.
- Cosine similarity combined with NLL loss is used for alignment training.
- Total loss: $\mathcal{L}_{total} = (1-\alpha-\beta)\mathcal{L}_{ASR} + \alpha\mathcal{L}_{MAE} + \beta\mathcal{L}_{NLL}$, where $\alpha=\beta=\frac{1}{3}$.

### Training Strategy
- **Pre-training stage**: Paraformer is pre-trained on LibriSpeech 460h for ASR; BGE is pre-trained on clean text pairs.
- **Joint training**: BGE is frozen; the ASR module and contrastive loss are jointly optimized.
- **Post-training**: The ASR module is frozen; BGE is fine-tuned for a few epochs to adapt to text-like representations.

## Key Experimental Results

### Datasets
Four datasets: Spoken-SQuAD, LibriSQA, SLUE-SQA-5 (real recordings), and DRCD (Chinese).

### Main Results (Spoken-SQuAD*)

| Model | Paradigm | WER↓ | Q→C R@1 | Q→C R@10 |
|-------|----------|------|---------|----------|
| CLAP | E2E | - | 2.93 | 14.84 |
| Whisper+BGE | Pipeline | 19.39 | 69.93 | 90.53 |
| **CLSR** | **E2E** | **15.14** | **70.03** | **90.68** |

- CLSR substantially outperforms CLAP across all four datasets (R@1 improves from ~3% to ~70%) and surpasses SpeechDPR.
- CLSR achieves performance comparable to or better than the Whisper+BGE pipeline while attaining lower WER (15.14 vs. 19.39).
- On LibriSQA, CLSR achieves R@1 = 85.04%, approaching the text-only BGE baseline of 86.91%.

### Ablation Study
- **Removing the VQ adaptor**: R@10 drops sharply from ~86% to ~44%, validating the central role of text-like representations.
- **Removing the Sampler**: WER increases from 15.01 to 16.18, with retrieval recall also declining.
- Pre-training both the ASR module and BGE each contributes significantly to final performance.
- WER ~16.75% constitutes a threshold beyond which retrieval performance degrades sharply.

### Long-form SQA Performance
Evaluated on Spoken Wikipedia (average audio length ~30 minutes):
- Without CLSR: EM = 18.00, F1 = 23.55, inference time = 7935s.
- **With CLSR: EM = 27.60, F1 = 35.10, inference time = 783s (10× speedup).**

## Highlights & Insights
1. **First application of RAG to SQA**, providing a systematic framework for long-form spoken question answering.
2. The **text-like representation bridging strategy** elegantly circumvents the difficulty of direct speech-text alignment by leveraging mature text-based contrastive learning models for high-quality cross-modal retrieval.
3. **No large-scale speech-text pre-training is required**; joint training on task data alone achieves performance on par with pipeline methods.
4. The straight-through estimator in the VQ adaptor ensures end-to-end training feasibility.

## Limitations & Future Work
- Evaluation is limited to TTS-synthesized speech and a small number of real recordings; robustness to noisy environments and multi-speaker scenarios remains unexplored.
- The current approach fixes long audio segmentation at 40-second chunks, lacking adaptive semantic segmentation strategies.
- BGE is frozen during joint training; gains from post-training are limited, and better unfreezing strategies warrant investigation.
- Comparisons with more recent speech foundation models (e.g., Whisper-v3, SeamlessM4T) are absent.
- The long-form audio experiment uses only 500 samples, which is relatively small in scale.

## Related Work & Insights

| Method | Characteristics | Limitations |
|--------|----------------|-------------|
| CLAP | Audio-text contrastive learning | Suited for sound-effect matching, not speech-content retrieval |
| SpeechDPR | Text-free training for speech retrieval | Data scarcity leads to poor performance (R@20 only 19.94) |
| Whisper+BGE | ASR-cascaded text retrieval | Dependent on ASR quality; error propagation; weak Chinese support |
| **CLSR** | **VQ bridging + joint training** | **E2E achieves pipeline-level performance; WER and retrieval jointly optimized** |

The text-like representation idea is transferable to other cross-modal retrieval tasks (e.g., translating video representations into the text space for video-text retrieval). The CIF + VQ combination can serve as a general-purpose speech-to-discrete-token frontend, replacing conventional discretization schemes. The long-form audio RAG framework can be integrated with streaming ASR to enable real-time meeting question answering systems.

## Rating
- Novelty: ⭐⭐⭐⭐ (First introduction of RAG into SQA; text-like representation bridging approach is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets + ablations + long-form audio validation, though the long-form experiment scale is limited)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete formulations, well-supported by figures)
- Value: ⭐⭐⭐⭐ (Provides a practical framework for long-form SQA; 10× inference speedup has strong application potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)

</div>

<!-- RELATED:END -->
