---
title: >-
  [Paper Note] DNCASR: End-to-End Training for Speaker-Attributed ASR
description: >-
  [ACL 2025][Audio & Speech][speaker-attributed ASR] DNCASR, an end-to-end trainable speaker-attributed ASR system, is proposed by linking a neural clustering decoder and an ASR decoder. Through joint training to generate speaker-attributed transcripts, it achieves a 9.0% relative reduction in cpWER on the AMI meeting dataset.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "speaker-attributed ASR"
  - "neural clustering"
  - "end-to-end training"
  - "multi-speaker meeting"
  - "overlapping speech"
date: 2026-05-08
content_hash: e3218a9a55080f86
---

# DNCASR: End-to-End Training for Speaker-Attributed ASR

**Conference**: ACL 2025  
**arXiv**: [2506.01916](https://arxiv.org/abs/2506.01916)  
**Code**: None  
**Area**: Speech  
**Keywords**: speaker-attributed ASR, neural clustering, end-to-end training, multi-speaker meeting, overlapping speech

## TL;DR

DNCASR, an end-to-end trainable speaker-attributed ASR system, is proposed by linking a neural clustering decoder and an ASR decoder. Through joint training to generate speaker-attributed transcripts, it achieves a 9.0% relative reduction in cpWER on the AMI meeting dataset.

## Background & Motivation

**Background**: Multi-speaker meeting transcription requires solving the "who spoke what" problem. Traditional methods cascade speaker diarization and ASR as two separate subsystems, consisting of three stages: VAD, speaker embedding extraction, and clustering.

**Limitations of Prior Work**: In cascaded systems, diarization and ASR are trained independently, neglecting mutual interactive information. Existing integrated methods (e.g., EEND) cannot process the full waveform of long meetings; speaker inventory-based methods require prior knowledge of speaker identities; parallel systems, despite using neural clustering, train the DNC and ASR modules independently, leading to misalignments between speaker indices and serialized ASR outputs.

**Key Challenge**: The DNC module processes window-level speaker embeddings of the entire meeting (global info), whereas the ASR module only processes the waveform of a single VAD segment (local info). This huge discrepancy in inputs makes joint training difficult. In parallel systems, DNC lacks the word-order information from ASR, which may lead to misassigning the speaker index of the second turn in the serialized output to the first turn.

**Goal**: Achieve end-to-end trainable speaker-attributed ASR, enabling the DNC module to utilize ASR hidden features for more accurate speaker index assignment, especially in overlapping speech segments.

**Key Insight**: Introduce a Link Cross Attention module in the DNC decoder to transfer the hidden features of the ASR decoder to DNC, enabling information flow, and design a two-stage joint fine-tuning strategy.

**Core Idea**: Realize end-to-end jointly trained speaker-attributed ASR by adding cross-attention links to the word-level features of the ASR decoder within the neural clustering decoder.

## Method

### Overall Architecture

DNCASR consists of two encoders and two linked decoders:
- **Wav Encoder**: Uses WavLM (wavlm-base-plus) to encode local waveform information, outputting $\bm{E_w}$
- **Spk Encoder**: Takes window-level speaker embeddings extracted by ECAPA-TDNN as input to encode global speaker features, outputting $\bm{E_s}$
- **ASR Decoder**: A standard Transformer decoder that generates serialized transcriptions with speaker change tokens (`<sc>`)
- **DNC Decoder**: A modified Transformer decoder containing two cross-attention modules: Spk Cross Attn and Link Cross Attn

### Key Designs

1. **Link Cross Attention Module**: A second cross-attention module is added to each block of the DNC decoder, where the query comes from the Spk Cross Attn output $\bm{S}_{\text{CA}}[i]$, and key/value from the Wav Cross Attn output $\bm{W}_{\text{CA}}$ of the corresponding block in the ASR decoder. A masking mechanism ensures each speaker index only attends to its corresponding turn's word features:

$$\bm{L}_{\text{CA}}[i] = \text{CA}(\bm{S}_{\text{CA}}[i], \bm{W}_{\text{CA}} \odot \text{mask}_l[i])$$

This allows DNC to capture high-resolution word-level information from ASR, aligning speaker indices with ASR-output speaker turns.

2. **Two-Stage Joint Fine-Tuning**:

    - **Stage 1**: DNC and ASR modules are jointly trained, with the loss function being the sum of two cross-entropy losses. The DNC training objective is the speaker indices from the first segment to the current segment, while the ASR objective is the word sequence of the current segment. In this stage, Link Cross Attn can only attend to the $\bm{W}_{\text{CA}}$ features of the current segment, while past segments use a learnable `<pad>` embedding.
    - **Stage 2**: The ASR module is frozen, and only the DNC decoder is fine-tuned. The $\bm{W}_{\text{CA}}$ features of all segments across the entire meeting are precomputed so that each speaker index can attend to its corresponding word-level features (no `<pad>` is needed anymore). The training objective is the speaker indices of the entire meeting.

3. **Constrained Diaconis Augmentation (CDA)**: Addressing the limited training data issue, this constrains the rotation angle of speaker embeddings on top of the original Diaconis augmentation to prevent performance degradation from excessive rotation. The scale is set randomly between 0 and 10.

### Loss & Training

- Pre-training stage: DNC and ASR are trained independently; DNC is pre-trained on segments containing only a single speaker.
- Joint training loss: $\mathcal{L} = \mathcal{L}_{\text{DNC}} + \mathcal{L}_{\text{ASR}}$, the sum of two cross-entropy losses.
- Serialized Output Training (SOT) is used to handle overlapping speech.
- First Speaker Segmentation (FSS) is applied to AMI data to generate single-speaker pre-training data.

## Key Experimental Results

### Main Results

**Synthetic Data Results (cpWER %)**:

| Model | WER | cpWER |
|------|-----|-------|
| Parallel | 3.5 | 13.4 |
| DNCASR (S1) | 3.5 | 9.5 |
| DNCASR (S2) | 3.5 | **8.7** |

**AMI-MDM cpWER Results (Dev/Eval %)**:

| Model | cpWER | cpWER-Multi |
|------|-------|-------------|
| Cascaded | 35.2/33.0 | 46.0/46.1 |
| Parallel | 34.8/34.6 | 49.8/49.2 |
| DNCASR (S1) | 33.2/34.7 | 47.3/49.5 |
| DNCASR (S2) | 31.3/32.1 | 43.4/44.8 |
| DNCASR (S2)+CDA | **30.7/31.5** | **42.5/44.1** |

### Ablation Study

**AMI Eval Results under Oracle Word Sequences (%)**:

| Model | DER | cpWER-All | cpWER-Single | cpWER-Multi |
|------|-----|-----------|-------------|-------------|
| DNCASR (S1) | 6.7 | 19.3 | 5.6 | 33.3 |
| DNCASR (S2) | 6.5 | 17.8 | 6.5 | 28.5 |
| DNCASR (S2)+CDA | **6.3** | **17.4** | 6.3 | **28.3** |

### Key Findings

- Compared to the Parallel system, DNCASR (S2)+CDA achieves a relative cpWER reduction of **11.8%** and **9.0%** on AMI Dev/Eval, respectively.
- The reduction in cpWER-Multi for multi-speaker segments is more significant: 14.7% on Dev and 10.4% on Eval, indicating the improvement mainly stems from overlapping speech processing.
- In synthetic data experiments, Stage 2 achieves a 35.1% relative cpWER reduction compared to the Parallel system.
- Oracle experiments demonstrate Stage 2 is slightly worse than Stage 1 on single-speaker segments, with improvements concentrated on multi-speaker segments (15.0% relative reduction).
- Wilcoxon signed-rank test yields a p-value < 1e-6, with 31 out of 34 meetings showing improvement, demonstrating strong statistical significance.

## Highlights & Insights

- **Exquisite Information Flow Design**: The masking mechanism of Link Cross Attention naturally aligns ASR word-level features with DNC speaker indices, resolving the alignment vulnerability in parallel systems.
- **Progressive Two-Stage Training Design**: Stage 1 processes the current segment (with `<pad>`), and Stage 2 processes the full meeting, gradually expanding the receptive field of DNC.
- **No Non-Neural Clustering Required**: Completely end-to-end, without relying on traditional methods like spectral clustering.

## Limitations & Future Work

1. It still depends on independent speaker embedding extraction modules (ECAPA-TDNN) and independent VAD, falling short of a fully end-to-end system.
2. Validated only on the AMI dataset; other multi-speaker datasets have not been tested.
3. The main experiments use the smaller WavLM-base-plus; the appendix shows switching to WavLM-large yields a further improvement of over 10%.
4. Limited training data (only 100 hours of AMI), without comparison to systems trained on large-scale supervised data.

## Related Work & Insights

- The EEND (Fujita et al., 2019) series of end-to-end diarization methods cannot process the full waveform of long meetings, which inspired the strategy of using speaker embeddings to replace waveforms.
- The serialized output format of SOT (Kanda et al., 2020b) is adopted, serving as a key element in handling overlapping speech.
- Combining DNCASR with stronger foundation models (such as Whisper) could be considered to boost ASR performance, thereby improving speaker attribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The Link Cross Attention design is novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic + real data, Oracle/non-Oracle settings, with complete statistical tests.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and detailed illustrations.
- **Value**: ⭐⭐⭐⭐ — Substantially advances the field of multi-speaker meeting transcription.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](../../ACL2026/audio_speech/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)

</div>

<!-- RELATED:END -->
