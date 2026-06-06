---
title: >-
  [Paper Note] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition
description: >-
  [ICLR 2026][Audio & Speech][unified speech recognition] This paper proposes USR 2.0, which replaces autoregressive pseudo-label generation with CTC-driven teacher forcing…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "unified speech recognition"
  - "CTC"
  - "pseudo-labelling"
  - "audio-visual speech recognition"
  - "out-of-distribution robustness"
date: 2026-05-08
content_hash: fdd6512a4794e18b
---

# Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition

**Conference**: ICLR 2026
**arXiv**: [2602.19316](https://arxiv.org/abs/2602.19316)  
**Code**: None (extends the USR framework)  
**Area**: Audio & Speech
**Keywords**: unified speech recognition, CTC, pseudo-labelling, audio-visual speech recognition, out-of-distribution robustness

## TL;DR

This paper proposes USR 2.0, which replaces autoregressive pseudo-label generation with CTC-driven teacher forcing, enabling attention pseudo-labels to be produced in a single forward pass. The approach achieves nearly 2× training speedup, enhances out-of-distribution robustness via joint CTC-attention prediction, and establishes state-of-the-art results on LRS3/LRS2/WildVSR across all three tasks (ASR/VSR/AVSR) within a single unified model.

## Background & Motivation

Unified Speech Recognition (USR) employs a single model to simultaneously perform ASR (audio), VSR (lip reading), and AVSR (audio-visual) by leveraging semi-supervised pseudo-labelling to achieve state-of-the-art performance. However, USR exhibits two critical bottlenecks:

**Costly autoregressive pseudo-labelling**: The attention branch requires one forward pass per token, whereas CTC decoding is approximately 40× faster than AR decoding.

**Decoupled supervision leads to out-of-distribution fragility**: The CTC and attention branches are trained independently, causing the attention decoder to produce cascading errors on long sequences, noisy inputs, or unseen domains; these errors are further self-reinforced via EMA.

Core observation: CTC is substantially more robust in out-of-distribution scenarios (due to monotonic alignment and conditional independence), while attention yields higher quality within the training distribution. The question is whether both advantages can be combined.

## Method

### Overall Architecture

USR 2.0 follows a student-teacher architecture:
- Shared Transformer encoder with modality-specific ResNet-18 front-ends
- Dual-branch design with a CTC head and an attention decoder
- Teacher is the EMA of the student ($\tau$: cosine schedule from 0.998 to 1)
- Semi-supervised training: annotated data uses ground-truth labels; unannotated data uses pseudo-labels

### Key Designs

#### 1. CTC-Driven Teacher Forcing

Conventional USR generates attention pseudo-labels autoregressively:
$$\tilde{y}_u^{Att} = \arg\max_{y_u} P_{Att}(y_u | \tilde{y}_{<u}^{Att}, x; \theta_T)$$

USR 2.0 instead first performs greedy CTC decoding followed by collapse-and-deduplication, and uses the result as decoder input to generate attention targets:
$$\tilde{y}^{CTC} = \text{collapse}(\tilde{y}_{1:L}), \quad \tilde{y}_u^{Att} = \arg\max_{y_u} P_{Att}(y_u | \tilde{y}_{<u}^{CTC}, x; \theta_T)$$

**Core Insight**: Although the outputs may lack global coherence, global coherence is unnecessary in the pseudo-labelling setting — the teacher and student operate under the same CTC prefix, so knowledge transfer remains effective. The student learns a stable mapping from coherent CTC prefixes to the teacher's conditionally valid next-token predictions.

**Alignment property**: Both types of pseudo-labels share the same length ($U_{CTC}$), allowing the student decoder to predict both in a **single forward pass** — inheriting CTC robustness while retaining the expressive capacity of attention.

#### 2. Mixed Sampling Strategy

CTC-driven teacher forcing introduces a train-inference mismatch (exposure bias). At each step, one of two modes is selected with probability 0.5:

**CTC-driven mode** (probability 0.5):
$$\mathcal{L}^{CTC,m} = \text{CTC}(\hat{y}^{CTC,m}, \tilde{y}^{CTC})$$
$$\mathcal{L}^{Att,m} = 0.5 \cdot \text{CE}(\hat{y}^{Att,m}, \tilde{y}^{Att}) + 0.5 \cdot \text{CE}(\hat{y}^{Att,m}, \tilde{y}^{CTC})$$

- The decoder is supervised by both attention and CTC pseudo-labels simultaneously.
- The CTC branch uses only CTC pseudo-labels (attention pseudo-labels may be incoherent in this mode).

**AR mode** (probability 0.5):
$$\mathcal{L}^{CTC,m} = 0.5 \cdot \text{CTC}(\hat{y}^{CTC,m}, \tilde{y}^{CTC}) + 0.5 \cdot \text{CTC}(\hat{y}^{CTC,m}, \tilde{y}^{Att})$$
$$\mathcal{L}^{Att,m} = \text{CE}(\hat{y}^{Att,m}, \tilde{y}^{Att})$$

- Standard AR decoding mitigates the train-inference mismatch.
- The CTC branch receives supervision from both pseudo-label types (attention pseudo-labels are coherent in this mode).

**Elegant design**: Under both modes, the CTC and attention branches provide mutual supervision signals, forming a **coupled rather than decoupled** training objective.

#### 3. Joint CTC-Attention Prediction

In CTC-driven mode, the two pseudo-label sequences are length-aligned, enabling the student decoder to predict both within a single forward pass:
- CTC pseudo-labels (robustness)
- Attention pseudo-labels (expressiveness)

The decoder naturally integrates the advantages of both branches during training.

### Loss & Training

- Joint CTC-attention training: CTC weight 0.1; attention CE with label smoothing 0.1
- Modality weights: visual 0.3; audio/audio-visual 0.7
- Unlabelled-to-labelled loss ratio: visual 0.97; audio/audio-visual 0.75
- Confidence filtering threshold: 0.8; sequence-level CTC confidence = mean token log-probability
- Inference: ESPnet joint decoding, beam=40, CTC weight 0.1
- Vocabulary: 1000-token SentencePiece
- Model scales: Base / Base+ / Large / Huge

## Key Experimental Results

### Main Results

**Table 1: In-distribution performance (LRS3 WER%, low-resource 30h)**

| Method | Unified Model | VSR↓ | ASR↓ | AVSR↓ |
|--------|--------------|------|------|-------|
| AV-HuBERT | ✗ | 51.8 | 4.9 | 4.7 |
| BRAVEn | ✗ | 43.4 | 4.0 | 4.0 |
| USR | ✓ | 36.0 | 3.2 | 3.0 |
| **USR 2.0** | **✓** | **36.2** | **3.0** | **2.9** |

**Table 2: Huge model final results (LRS3)**

| VSR | ASR | AVSR |
|-----|-----|------|
| **17.6** | **0.9** | **0.8** |

**Table 3: Out-of-distribution robustness (greedy decoding WER%)**

| Method | LibriSpeech | WildVSR | AVSpeech |
|--------|-------------|---------|----------|
| AV-HuBERT | 29.1 | 82.4 | 26.0 |
| USR | 25.3 | 80.0 | 34.7 |
| **USR 2.0** | **15.4** | **73.7** | **25.0** |

**Table 4: Noise robustness (LRS3 ASR, beam=30)**

| Method | 10dB | 5dB | 0dB | -5dB | Avg. |
|--------|------|-----|-----|------|------|
| USR | 5.8 | 14.3 | 48.5 | 104.4 | 43.3 |
| **USR 2.0** | **5.2** | **13.4** | **44.0** | **94.4** | **39.3** |

### Ablation Study

**Long-sequence robustness**:
- USR exhibits sharply degraded WER beyond 155 frames (exceeding the training distribution).
- USR 2.0 remains stable up to 600 frames.
- Increasing beam size narrows the gap but at the cost of significant latency and memory overhead.

**Beam size sensitivity**:
- USR 2.0 already performs strongly under greedy or small-beam decoding.
- USR requires beam ≥ 30 to approach the greedy performance of USR 2.0.

**Mixed sampling probability**: A fixed ratio of 0.5 performs comparably to adaptive scheduling; the simpler fixed scheme is adopted.

### Key Findings

1. **CTC-driven pseudo-labels do not require global coherence**: Teacher and student share the same CTC-prefix condition; local conditional correctness is sufficient.
2. **Coupled vs. decoupled supervision**: Coupled CTC-attention supervision is the key driver of robustness gains — the two branches mutually "correct" each other.
3. **Speedup as an indirect source of gains**: Faster pseudo-labelling enables scaling to larger models and more data, which underlies the success of the Huge model.
4. **Greedy decoding quality is central to semi-supervised training**: Pseudo-labels are generated at every training step, so greedy-decoding quality directly determines training effectiveness.

## Highlights & Insights

- **"Coherence is unnecessary for pseudo-labelling"**: Counter-intuitive yet logically consistent — under teacher forcing with a shared CTC prefix, "incoherently correct" predictions suffice.
- **Dual gains in efficiency and quality**: Rather than a trade-off, the proposed pseudo-labelling strategy simultaneously improves both dimensions.
- **Underappreciated value of CTC**: Although the conditional independence assumption limits sequence modelling capacity, CTC robustness is extremely valuable in semi-supervised and out-of-distribution settings.
- **Practical value of unified models**: A single model handling ASR/VSR/AVSR, with VSR at 17.6% and AVSR at 0.8%, holds strong deployment value.

## Limitations & Future Work

1. The mixed sampling frequency is fixed at 0.5; the optimal ratio may differ across training stages.
2. CTC still degrades under severe noise (−5dB), potentially propagating errors to the attention branch.
3. Inference still requires beam search (beam=40), offering limited latency improvement.
4. Evaluation is limited to English; cross-lingual applicability to tonal languages and others remains unexplored.
5. The 1000-token vocabulary may be insufficient for large-vocabulary tasks.

## Related Work & Insights

- **USR** is the direct predecessor; this work precisely diagnoses its decoupled supervision and AR bottlenecks.
- Self-supervised methods such as **AV-HuBERT** employ unified pre-training but split into separate models at fine-tuning.
- The exposure-bias mitigation concept from **Scheduled Sampling** is referenced but addresses a different type of bias.
- Broader implication: the insight that "pseudo-labels need not be perfect" may generalise to other teacher-student frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ — CTC-driven teacher forcing is a concise and effective contribution
- Technical Depth: ⭐⭐⭐⭐ — the loss design of the mixed sampling strategy is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — comprehensive coverage of ID/OOD/noise/long-sequence/beam/multi-scale settings
- Value: ⭐⭐⭐⭐⭐ — 2× faster training combined with unified model state-of-the-art

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](../../ACL2026/audio_speech/pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ICML 2026\] Attend to Anything: Foundation Model for Unified Human Attention Modeling](../../ICML2026/audio_speech/attend_to_anything_foundation_model_for_unified_human_attention_modeling.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](../../AAAI2026/audio_speech/cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ACL 2026\] UniVocal: Unified Speech-Singing Code-mixed Synthesis](../../ACL2026/audio_speech/univocal_unified_speech-singing_code-switching_synthesis.md)

</div>

<!-- RELATED:END -->
