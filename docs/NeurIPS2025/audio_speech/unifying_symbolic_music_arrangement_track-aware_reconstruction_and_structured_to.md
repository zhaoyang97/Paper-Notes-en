---
title: >-
  [Paper Note] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
description: >-
  [NeurIPS 2025][Audio & Speech][music arrangement] This paper proposes a unified symbolic music arrangement framework that employs a segment-level self-supervised reconstruction objective (decoupling content from instrume…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "music arrangement"
  - "symbolic music"
  - "multi-track music generation"
  - "music tokenization"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 88975676f0835409
---

# Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization

**Conference**: NeurIPS 2025
**arXiv**: [2408.15176](https://arxiv.org/abs/2408.15176)  
**Code**: [https://www.oulongshen.xyz/automatic_arrangement](https://www.oulongshen.xyz/automatic_arrangement)  
**Area**: Audio & Speech / Symbolic Music Generation
**Keywords**: music arrangement, symbolic music, multi-track music generation, music tokenization, self-supervised learning

## TL;DR
This paper proposes a unified symbolic music arrangement framework that employs a segment-level self-supervised reconstruction objective (decoupling content from instrument style) and a novel multi-track tokenization scheme REMI-z, enabling a single pretrained model to handle diverse arrangement tasks—including orchestral arrangement, piano reduction, and drum arrangement—while surpassing task-specific state-of-the-art methods across all three benchmarks.

## Background & Motivation

Music arrangement refers to the art of adapting a piece for different performance configurations, encompassing:
- **Reinterpretation**: Rewriting for different instruments (e.g., orchestral → jazz ensemble)
- **Simplification**: Reducing to a solo instrument (e.g., full band → piano)
- **Additive generation**: Adding new tracks (e.g., adding drums to a song)

Prior work designs independent models for each task, giving rise to three core problems:

**Lack of cross-task generalizability**: Each arrangement task relies on dedicated architectures and training protocols, precluding the sharing of musical knowledge.

**Data bottleneck**: End-to-end methods depend on parallel datasets (i.e., multiple arrangement versions of the same piece), which are extremely scarce.

**Tokenization limitations**: Existing strictly chronological schemes such as REMI+ interleave notes from different instruments, fragmenting content and impeding instrument-level control.

Core insight: All arrangement tasks share a common structure—generating new musical tracks based on existing ones, subject to content and instrument constraints. This can be realized through a unified self-supervised reconstruction objective and structured tokenization.

## Method

### Overall Architecture

A decoder-only Transformer with 80M parameters is employed. The model is first pretrained with standard next-token prediction on the large-scale Los Angeles MIDI dataset (405K files), then fine-tuned on the Slakh2100 dataset via a unified reconstruction objective to support multiple arrangement tasks. The input sequence follows the format `[condition]<SEP>[target]`, with cross-entropy loss computed only over the target subsequence.

### Key Designs

1. **Segment-level decoupled reconstruction objective**:

    - The input music is decomposed into three token streams:
        - **Instrument condition $I(\cdot)$**: Specifies the instruments used in the target segment, sorted by pitch register (treble first), enabling instrument and voice control.
        - **Content condition $C(\cdot)$**: A pure note sequence extracted from the source music with instrument markers removed, sorted chronologically and deduplicated, encoding *what* is played rather than *who* plays it or *how*.
        - **History condition**: The complete token sequence of the preceding segment, providing cross-segment consistency.
    - Fine-tuning objective: $\mathcal{L}(\theta) = -\log p_\theta(\mathcal{T}_{task}(y^{(t)}) | I(\mathcal{T}_{task}(y^{(t)})), C(\mathcal{S}_{task}(y^{(t)})), \mathcal{T}_{task}(y^{(t-1)}))$
    - Design motivation: During training, all three conditions are extracted from the same piece of music (self-supervised), eliminating the need for parallel data; at inference time, arbitrary instruments and content can be freely combined.

2. **REMI-z tokenization scheme**:

    - Core change: Global strict chronological ordering is abandoned in favor of intra-track continuity.
    - Structure: Each measure consists of multiple track sequences, each corresponding to one instrument; notes within a track are sorted by onset, and tracks are ordered from highest to lowest average pitch.
    - "Zigzag" encoding pattern: Notes belonging to the same instrument remain contiguous in the token sequence, uninterrupted by notes from other instruments.
    - Advantages:
        - Sequence length reduced by 32.9% (151.68 vs. 225.91 tokens/measure), lowering computational cost.
        - Clear track boundaries facilitate instrument-level control.
        - Content encoding for each instrument is unaffected by concurrent instruments, increasing pattern repetition in training data.
    - Per-measure Shannon entropy is reduced (29.43 vs. 41.68 bits/token), decreasing information redundancy.

3. **Task instantiation**:

    - **Orchestral arrangement**: $\mathcal{S}_{task} = \mathcal{T}_{task} = \text{identity}$ (drum tracks excluded); random deletion of non-melody tracks and removal of duration tokens during training encourage creative rewriting.
    - **Piano reduction**: $\mathcal{T}_{task}$ selects the piano track; $\mathcal{S}_{task} = \text{identity}$; segments are filtered to retain only those where the piano track covers >40% of the pitch range.
    - **Drum arrangement**: $\mathcal{S}_{task}$ extracts all pitched instrument tracks; $\mathcal{T}_{task}$ extracts the drum track; 4-measure segments are used to capture cross-bar drum patterns.

### Loss & Training
- Pretraining: Standard next-token prediction; 4× RTX A5000 GPUs, batch size 12, 1 epoch.
- Fine-tuning: Segment-level reconstruction objective; single A40 GPU, 3 epochs, AdamW with linear warmup.
- Key regularization: Key normalization (all songs transposed to C major/A minor); random track deletion (deletion count sampled from a Poisson distribution).

## Key Experimental Results

### Main Results

**Orchestral Arrangement (Objective)**

| Model | I-IoU↑ | VER↓ | Note F1↑ | Notei F1↑ | Mel F1↑ |
|------|------|------|----------|------|------|
| Transformer-VAE | 97.5 | 35.0 | 49.5 | 40.0 | 24.7 |
| REMI+ Transformer | 95.0 | 18.2 | 94.4 | 76.0 | 68.8 |
| REMI-z (w/o pretraining) | 99.5 | 9.9 | 97.8 | 77.5 | 77.8 |
| **Ours (+pretraining)** | **99.8** | **7.6** | **97.5** | **87.0** | **84.5** |

**Drum Arrangement (Subjective, 5-point scale)**

| Model | Compatibility | Coherence | Transition | Creativity | Musicality |
|------|------|------|----------|------|------|
| CA v2 | 3.82 | 4.05 | 2.86 | 2.58 | 3.19 |
| **Ours** | **3.91** | 4.03 | **3.77** | **3.27** | **3.57** |
| Ground Truth | 4.31 | 4.18 | 3.36 | 3.16 | 3.78 |

### Ablation Study

| Configuration | Notei F1 | Mel F1 | Notes |
|------|---------|------|------|
| Full model | 87.0 | 84.5 | — |
| w/o voice control | 84.3 (−2.7) | 81.5 (−3.0) | Voice information aids instrument role inference |
| w/o history condition | 77.4 (−9.6) | 79.4 (−5.1) | Historical context is critical for cross-segment consistency |
| w/o pretraining | 77.5 | 77.8 | Pretraining yields +9.5 Notei F1 improvement |

**Tokenization Comparison (Unconditional Generation)**

| Tokenization | tokens/measure | tokens/note | Note PPL↓ |
|------|---------|------|------|
| REMI+ | 225.91 | 4.03 | 116.20 |
| REMI-z | 151.68 | 2.77 | **84.11** |

### Key Findings
- REMI-z is statistically significantly better than REMI+ on all objective metrics ($p < 0.001$), validating the advantages of structured tokenization.
- Pretrained knowledge transfer primarily benefits instrument-level modeling (Notei F1 +9.5%) and melody preservation (Mel F1 +6.7%).
- In drum arrangement, the model approaches ground truth musicality scores (3.57 vs. 3.78).
- REMI-z not only outperforms REMI+ on arrangement tasks but also achieves lower note-level perplexity in unconditional generation.

## Highlights & Insights
- The unified self-supervised reconstruction objective is the central contribution—by decoupling content from style, a single model can handle diverse arrangement scenarios without requiring any parallel data.
- The REMI-z tokenization scheme addresses the long-standing content fragmentation problem in multi-track music modeling, with benefits extending to general symbolic music modeling.
- The pretrain-then-finetune paradigm achieves success in symbolic music comparable to that observed in NLP.
- An 80M-parameter model surpassing task-specific state-of-the-art methods demonstrates that methodological design matters more than model scale.

## Limitations & Future Work
- The model is limited to 80M parameters; scaling up could yield substantial improvements.
- The pretraining corpus (405K MIDI files) is relatively small; larger datasets may prove more effective.
- Velocity information is not modeled, limiting the expressiveness of performance dynamics.
- Duration tokens are removed from the content condition, potentially discarding important instrument-specific rhythmic patterns.
- Piano reduction uses the original piano track rather than a human-arranged target, which may not fully reflect authentic arrangement requirements.

## Related Work & Insights
- Q-Transformer (Zhao et al., 2023) guides arrangement with high-level descriptors but separates style modeling from content realization.
- Composer's Assistant 2 performs drum arrangement via infilling but does not preserve the core of the original piece.
- The REMI/REMI+ tokenization series serves as the direct foundation from which REMI-z is derived.
- The random track deletion strategy echoes the span infilling denoising approach of BART-like models.
- The success of the pretrain-then-finetune paradigm reinforces the viability of a "foundation model" approach in symbolic music.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the unified arrangement framework and the REMI-z tokenization scheme represent original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three tasks with both objective and subjective evaluations; detailed ablation and tokenization analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, precise task definitions, and intuitive figures.
- Value: ⭐⭐⭐⭐ Provides a practical and generalizable solution for symbolic music arrangement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)
- [\[NeurIPS 2025\] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation](megadance_mixture-of-experts_architecture_for_genre-aware_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
